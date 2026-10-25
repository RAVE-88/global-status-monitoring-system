"""Global Status score: a transparent 0-100 per-country status index.

ACADEMIC DERIVED SCORE — NOT THE PRODUCTION CII. The WorldMonitor dashboard
ships its own production crisis-index (CII) pipeline; this module is a
self-contained, fully explainable reimplementation built for the ML
laboratory report. It consumes the same raw signals the dashboard ingests
(GDELT-derived event rows and USGS earthquakes) but its weights, saturation
caps and bands are chosen for classroom transparency, not tuned for
operational use. Nothing here feeds back into the app.

Method (per country, over the fetched window)
---------------------------------------------
Event signals come from the collected article rows; the threat level /
category counts use the same keyword classifier the dashboard applies
(``classifier.classify_by_keyword`` re-run on each headline — deterministic
and identical to the labels stored in the dataset's derived columns).
Countries are attributed via the ``detected_countries`` gazetteer column
(re-deriving from the headline when that column is empty); events naming no
tracked country count toward the "Global" bucket.

Earthquakes are matched to countries with a committed coarse bounding-box
table (~40 major countries, below). A quake whose coordinates fall in no
box — or in a box of an untracked country — counts toward "Global" only.

Each country's score is a weighted sum of six normalized components
(each clamped to [0, 1], weights sum to 100):

    severity        30  worst threat level seen (critical=1.0 ... info=0.05)
    event_volume    20  event count, saturated at 15 events
    negativity      15  how negative the average headline sentiment is
    corroboration   10  max near-duplicate corroboration, saturated at 5
    seismic         15  0.5 * (quake count / 10) + 0.5 * (max magnitude / 8)
    news_velocity   10  events per hour, saturated at 2/hour

Score = clamp(round(sum), 0, 100), banded 0-24 Low, 25-49 Medium, 50-74 High,
75-100 Critical. Every component value is returned in the per-country
``components`` breakdown so any number in the report can be traced to its
inputs by hand.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .classifier import classify_by_keyword
from .gazetteer import REGION_BY_CODE, detect_countries
from .preprocess import load_raw_events

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# severity weight per threat level, already normalized to [0, 1]
LEVEL_SEVERITY = {
    "critical": 1.0,
    "high": 0.7,
    "medium": 0.4,
    "low": 0.15,
    "info": 0.05,
}

# component -> (weight out of 100, doc string)
COMPONENT_WEIGHTS: dict[str, float] = {
    "severity": 30.0,
    "event_volume": 20.0,
    "negativity": 15.0,
    "corroboration": 10.0,
    "seismic": 15.0,
    "news_velocity": 10.0,
}

EVENT_VOLUME_SATURATION = 15.0     # events in window -> component = 1
CORROBORATION_SATURATION = 5.0     # max corroboration count -> component = 1
VELOCITY_SATURATION = 2.0          # events/hour -> component = 1
QUAKE_COUNT_SATURATION = 10.0      # quakes in window -> half of seismic
MAX_MAGNITUDE_SCALE = 8.0          # magnitude 8+ -> other half of seismic

STATUS_BANDS = ("Low", "Medium", "High", "Critical")


def band_for_score(score: float) -> str:
    """0-24 Low, 25-49 Medium, 50-74 High, 75-100 Critical."""
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


# Coarse bounding boxes for ~40 major countries, (lat_min, lat_max,
# lon_min, lon_max). Deliberately generous rectangles so offshore quakes
# still land somewhere; known overlaps (e.g. India/Pakistan frontier) are
# accepted because attribution is only used for an academic score.
COUNTRY_BOUNDING_BOXES: dict[str, tuple[float, float, float, float]] = {
    "US": (24.0, 50.0, -125.0, -66.0),
    "CA": (48.0, 72.0, -130.0, -60.0),
    "MX": (14.0, 33.0, -118.0, -86.0),
    "BR": (-34.0, 6.0, -74.0, -33.0),
    "AR": (-55.0, -21.0, -74.0, -53.0),
    "CL": (-56.0, -17.0, -76.0, -66.0),
    "PE": (-19.0, 0.0, -82.0, -68.0),
    "CO": (-5.0, 13.0, -79.0, -66.0),
    "VE": (0.0, 13.0, -74.0, -59.0),
    "GB": (49.0, 61.0, -9.0, 3.0),
    "FR": (41.0, 52.0, -6.0, 10.0),
    "DE": (47.0, 56.0, 5.0, 16.0),
    "IT": (36.0, 47.0, 6.0, 19.0),
    "ES": (35.0, 45.0, -10.0, 5.0),
    "PL": (48.0, 55.0, 13.0, 25.0),
    "GR": (34.0, 42.0, 19.0, 29.0),
    "UA": (44.0, 53.0, 22.0, 41.0),
    "RU": (41.0, 82.0, 27.0, 180.0),
    "TR": (35.0, 43.0, 25.0, 45.0),
    "SY": (32.0, 38.0, 35.0, 43.0),
    "IQ": (29.0, 38.0, 38.0, 49.0),
    "IR": (24.0, 40.0, 43.0, 64.0),
    "IL": (29.0, 34.0, 34.0, 36.0),
    "PS": (31.0, 33.0, 34.0, 36.0),
    "LB": (33.0, 35.0, 35.0, 37.0),
    "YE": (12.0, 19.0, 42.0, 55.0),
    "SA": (16.0, 33.0, 34.0, 56.0),
    "EG": (21.0, 32.0, 24.0, 37.0),
    "ET": (3.0, 15.0, 33.0, 48.0),
    "NG": (4.0, 14.0, 2.0, 15.0),
    "KE": (-5.0, 6.0, 33.0, 43.0),
    "ZA": (-35.0, -22.0, 16.0, 33.0),
    "IN": (6.0, 36.0, 68.0, 90.0),
    "PK": (23.0, 37.0, 60.0, 75.0),
    "AF": (29.0, 39.0, 60.0, 76.0),
    "CN": (18.0, 54.0, 73.0, 135.0),
    "MM": (9.0, 29.0, 92.0, 102.0),
    "JP": (30.0, 46.0, 128.0, 146.0),
    "KR": (33.0, 39.0, 124.0, 132.0),
    "KP": (37.0, 43.0, 124.0, 131.0),
    "TW": (21.0, 26.0, 119.0, 123.0),
    "ID": (-11.0, 7.0, 95.0, 141.0),
    "PH": (4.0, 19.0, 116.0, 127.0),
    "AU": (-44.0, -10.0, 112.0, 154.0),
    "NZ": (-47.0, -34.0, 166.0, 179.0),
}

GLOBAL_BUCKET = "Global"


def load_quakes(path: Path | None = None) -> list[dict]:
    """Read the USGS quake table from disk."""
    if path is None:
        path = DATA_DIR / "gsms_quakes.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _event_countries(event: dict) -> list[str]:
    """ISO-2 codes for an event row; falls back to re-running the gazetteer."""
    raw = (event.get("detected_countries") or "").strip()
    if raw:
        return [code for code in raw.split(";") if code]
    return detect_countries(event.get("headline", ""))


def _quake_countries(quake: dict) -> list[str]:
    """Tracked countries whose coarse box contains the quake's coordinates."""
    try:
        latitude = float(quake.get("latitude"))
        longitude = float(quake.get("longitude"))
    except (TypeError, ValueError):
        return []
    matched: list[str] = []
    for code, (lat_min, lat_max, lon_min, lon_max) in COUNTRY_BOUNDING_BOXES.items():
        if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
            matched.append(code)
    return sorted(matched)


def _parse_utc(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def global_status(
    events: Sequence[dict], quakes: Sequence[dict]
) -> dict:
    """Compute the per-country Global Status score from real signals.

    ``events`` are the collected article rows (see ``preprocess.load_raw_events``);
    ``quakes`` are the USGS rows (see ``load_quakes``). Returns a structured
    dict: per-country score, band, component breakdown and raw aggregates,
    plus a Global bucket for events/quakes that match no tracked country.
    """
    events = list(events)
    quakes = list(quakes)

    timestamps = [
        moment
        for moment in (_parse_utc(event.get("published_utc", "")) for event in events)
        if moment is not None
    ]
    if timestamps:
        window_hours = max(
            (max(timestamps) - min(timestamps)).total_seconds() / 3600.0, 1.0
        )
    else:
        window_hours = 1.0

    quake_hits: Counter[str] = Counter()
    quake_max_mag: dict[str, float] = {}
    global_quakes = 0
    global_quake_max_mag = 0.0
    for quake in quakes:
        matched = _quake_countries(quake)
        magnitude = float(quake.get("magnitude") or 0.0)
        if not matched:
            global_quakes += 1
            global_quake_max_mag = max(global_quake_max_mag, magnitude)
            continue
        for code in matched:
            quake_hits[code] += 1
            quake_max_mag[code] = max(quake_max_mag.get(code, 0.0), magnitude)

    country_stats: dict[str, dict] = {}

    for event in events:
        codes = _event_countries(event)
        classification = classify_by_keyword(event.get("headline", ""))
        category = classification["category"]
        level = classification["level"]
        try:
            sentiment = float(event.get("sentiment_score", "") or 0.0)
        except (TypeError, ValueError):
            sentiment = 0.0
        try:
            corroboration = int(float(event.get("corroboration_count", "") or 0))
        except (TypeError, ValueError):
            corroboration = 0
        try:
            risk_score = float(event.get("risk_score", "") or 0.0)
        except (TypeError, ValueError):
            risk_score = 0.0
        targets = codes or [GLOBAL_BUCKET]
        for code in targets:
            stats = country_stats.setdefault(
                code,
                {
                    "event_count": 0,
                    "category_counts": Counter(),
                    "level_counts": Counter(),
                    "sentiments": [],
                    "max_corroboration": 0,
                    "risk_scores": [],
                    "headlines": [],
                },
            )
            stats["event_count"] += 1
            stats["category_counts"][category] += 1
            stats["level_counts"][level] += 1
            stats["sentiments"].append(sentiment)
            stats["max_corroboration"] = max(stats["max_corroboration"], corroboration)
            stats["risk_scores"].append(risk_score)
            if len(stats["headlines"]) < 3:
                stats["headlines"].append(event.get("headline", ""))

    # events naming no tracked country already sit in the Global bucket
    global_event_count = country_stats.get(GLOBAL_BUCKET, {}).get("event_count", 0)

    # quakes matching no tracked box fall into Global's seismic signal
    quake_hits[GLOBAL_BUCKET] += global_quakes
    quake_max_mag[GLOBAL_BUCKET] = max(
        quake_max_mag.get(GLOBAL_BUCKET, 0.0), global_quake_max_mag
    )

    countries: list[dict] = []
    for code, stats in country_stats.items():
        event_count = stats["event_count"]
        avg_sentiment = (
            sum(stats["sentiments"]) / len(stats["sentiments"])
            if stats["sentiments"] else 0.0
        )
        max_corroboration = stats["max_corroboration"]
        worst_level = max(
            stats["level_counts"],
            key=lambda lvl: LEVEL_SEVERITY.get(lvl, 0.05),
        ) if stats["level_counts"] else "info"
        velocity = event_count / window_hours

        components = {
            "severity": _clamp01(LEVEL_SEVERITY.get(worst_level, 0.05)),
            "event_volume": _clamp01(event_count / EVENT_VOLUME_SATURATION),
            "negativity": _clamp01(-avg_sentiment),
            "corroboration": _clamp01(max_corroboration / CORROBORATION_SATURATION),
            "seismic": _clamp01(
                0.5 * (quake_hits.get(code, 0) / QUAKE_COUNT_SATURATION)
                + 0.5 * (quake_max_mag.get(code, 0.0) / MAX_MAGNITUDE_SCALE)
            ),
            "news_velocity": _clamp01(velocity / VELOCITY_SATURATION),
        }
        score = round(sum(
            COMPONENT_WEIGHTS[name] * value for name, value in components.items()
        ))
        score = max(0, min(100, score))

        countries.append(
            {
                "country": code,
                "region": REGION_BY_CODE.get(code, "Global"),
                "score": score,
                "band": band_for_score(score),
                "components": {name: round(value, 4) for name, value in components.items()},
                "event_count": event_count,
                "counts_by_category": dict(
                    sorted(stats["category_counts"].items())
                ),
                "level_counts": dict(sorted(stats["level_counts"].items())),
                "worst_level": worst_level,
                "avg_sentiment": round(avg_sentiment, 4),
                "max_corroboration": max_corroboration,
                "avg_risk_score": round(
                    sum(stats["risk_scores"]) / len(stats["risk_scores"]), 2
                ) if stats["risk_scores"] else 0.0,
                "quake_count": quake_hits.get(code, 0),
                "max_magnitude": round(quake_max_mag.get(code, 0.0), 2),
                "news_velocity_per_hour": round(velocity, 4),
                "top_headlines": stats["headlines"],
            }
        )
    countries.sort(key=lambda row: (-row["score"], row["country"]))

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "score_definition": (
            "Academic derived score for the GSMS ML laboratory: transparent "
            "weighted sum of severity(30) + event_volume(20) + negativity(15) "
            "+ corroboration(10) + seismic(15) + news_velocity(10), each "
            "component clamped to [0,1]. NOT the dashboard's production CII; "
            "labels and weights are heuristics for coursework, not an "
            "operational alerting product."
        ),
        "bands": "0-24 Low, 25-49 Medium, 50-74 High, 75-100 Critical",
        "window_hours": round(window_hours, 2),
        "n_events": len(events),
        "n_quakes": len(quakes),
        "global_unmatched_events": global_event_count,
        "global_unmatched_quakes": global_quakes,
        "countries": countries,
    }
