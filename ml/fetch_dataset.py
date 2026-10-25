#!/usr/bin/env python3
"""Build the GSMS ML dataset from the dashboard's own keyless sources.

Pulls real articles from the GDELT DOC API v2 and real seismic events from the
USGS GeoJSON feeds — the same public, keyless feeds the WorldMonitor dashboard
ingests — then cleans and assembles ML-ready CSVs:

  data/gsms_events.csv   one row per fetched article (duplicates kept; the
                         duplicate-removal step of the ML pipeline removes
                         near-duplicate stories at training time)
  data/gsms_quakes.csv   USGS seismic events for the trailing week
  data/gsms_concept.csv  tidy categorical table for the Find-S / Candidate
                         Elimination laboratory, discretized from real events

Label provenance: every label (category, risk_score, risk_band) is a DERIVED
HEURISTIC computed by transparent rules over the headline text — see
gsms_ml/classifier.py and LEVEL_RISK_WEIGHTS. Nothing here is ground truth and
nothing is randomly fabricated; the raw text is real, attributed, and
re-fetchable.

Usage:
    python3 fetch_dataset.py [--timespan 3d] [--maxrecords 250] [--delay 1.5]

The committed snapshot was built on 2026-09-30; re-running refreshes it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gsms_ml.classifier import LEVEL_RISK_WEIGHTS, classify_by_keyword
from gsms_ml.gazetteer import detect_countries, region_for_codes, region_for_source_country
from gsms_ml.lexicon import headline_sentiment, tokenize

DATA_DIR = Path(__file__).resolve().parent / "data"

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USGS_ALL_WEEK = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson"
)

# Topical queries bias *what gets fetched*; labels are always computed from the
# headline content itself, never from the query used to find the article.
CATEGORY_QUERIES = [
    "conflict",
    "military",
    "protest",
    "disaster",
    "economic",
    "cyber",
    "health",
    "diplomacy",
    "infrastructure",
    "security",
    "environment",
]

REGION_QUERIES = [
    "sourcecountry:US",
    "sourcecountry:UK",
    "sourcecountry:IN",
    "sourcecountry:GM",  # Germany
    "sourcecountry:FR",
    "sourcecountry:BR",
    "sourcecountry:JA",  # Japan
    "sourcecountry:SF",  # South Africa
    "sourcecountry:AS",  # Australia
    "sourcecountry:NU",  # Nigeria
]

COLUMNS = [
    "event_id",
    "headline",
    "snippet",
    "source_domain",
    "source_country",
    "published_utc",
    "region",
    "detected_countries",
    "sentiment_score",
    "corroboration_count",
    "is_breaking",
    "text_len",
    "word_count",
    "upper_ratio",
    "velocity_per_hour",
    "risk_score",
    "risk_band",
]


def http_get_json(url: str, retries: int = 3, timeout: float = 30.0):
    request = urllib.request.Request(
        url, headers={"User-Agent": "gsms-ml-lab/1.0 (college research project)"}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except Exception as error:  # noqa: BLE001 - network best effort
            last_error = error
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last_error}")


def parse_seen_date(raw: str) -> str | None:
    """GDELT 'seendate' (e.g. 20260930T121500Z) -> ISO-8601 UTC string."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        ).isoformat()
    except ValueError:
        return None


def fetch_articles(timespan: str, maxrecords: int, delay: float) -> list[dict]:
    """Fetch artlist rows for all queries, deduplicating by URL."""
    articles: dict[str, dict] = {}
    queries = CATEGORY_QUERIES + REGION_QUERIES
    for index, query in enumerate(queries):
        params = {
            "query": f"{query} sourcelang:english",
            "mode": "artlist",
            "maxrecords": str(maxrecords),
            "format": "json",
            "timespan": timespan,
            "sort": "datedesc",
        }
        url = f"{GDELT_DOC_URL}?{urllib.parse.urlencode(params)}"
        try:
            payload = http_get_json(url)
        except RuntimeError as error:
            print(f"  ! skipping query {query!r}: {error}", file=sys.stderr)
            continue
        rows = payload.get("articles") or []
        added = 0
        for row in rows:
            url_key = (row.get("url") or "").strip()
            title = (row.get("title") or "").strip()
            if not url_key or not title:
                continue
            if url_key in articles:
                continue
            articles[url_key] = {
                "headline": title,
                "snippet": "",
                "source_domain": (row.get("domain") or "").strip(),
                "source_country": (row.get("sourcecountry") or "").strip(),
                "published_utc": parse_seen_date(row.get("seendate") or "")
                or "",
                "link": url_key,
            }
            added += 1
        print(
            f"  [{index + 1}/{len(queries)}] {query!r}: +{added} "
            f"(total {len(articles)})",
            file=sys.stderr,
        )
        time.sleep(delay)
    return list(articles.values())


def fetch_quakes() -> list[dict]:
    payload = http_get_json(USGS_ALL_WEEK)
    quakes: list[dict] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {}) or {}
        geometry = feature.get("geometry", {}) or {}
        coords = geometry.get("coordinates") or [None, None, None]
        lon, lat, depth = (coords + [None, None, None])[:3]
        if props.get("mag") is None or lat is None:
            continue
        quakes.append(
            {
                "quake_id": feature.get("id", ""),
                "time_utc": datetime.fromtimestamp(
                    (props.get("time") or 0) / 1000.0, tz=timezone.utc
                ).isoformat(),
                "magnitude": props["mag"],
                "place": (props.get("place") or "").strip(),
                "latitude": round(float(lat), 4),
                "longitude": round(float(lon), 4) if lon is not None else None,
                "depth_km": round(float(depth), 1) if depth is not None else None,
                "significance": props.get("sig", 0) or 0,
                "tsunami_flag": 1 if props.get("tsunami") else 0,
            }
        )
    return quakes


def assemble_articles(raw_articles: list[dict]) -> list[dict]:
    """Group near-duplicate headlines and compute derived features + labels."""
    # bucket by publication day so the O(n^2) pass stays small
    def day_key(article: dict) -> str:
        return (article["published_utc"] or "")[:10]

    groups: dict[str, list[dict]] = {}
    for article in raw_articles:
        groups.setdefault(day_key(article), []).append(article)

    for rows in groups.values():
        for index, row in enumerate(rows):
            if row.get("_group") is not None:
                continue
            tokens = set(tokenize(row["headline"]))
            if not tokens:
                row["_group"] = index
                continue
            members = [row]
            for other_index in range(index + 1, len(rows)):
                other = rows[other_index]
                if other.get("_group") is not None:
                    continue
                other_tokens = set(tokenize(other["headline"]))
                if not other_tokens:
                    continue
                intersection = len(tokens & other_tokens)
                union = len(tokens | other_tokens)
                if union and intersection / union >= 0.9:
                    other["_group"] = index
                    members.append(other)
            row["_group_size"] = len(members)

    assembled: list[dict] = []
    for article in raw_articles:
        codes = detect_countries(article["headline"])
        region = (
            region_for_source_country(article["source_country"])
            if article["source_country"]
            else region_for_codes(codes)
        )
        classification = classify_by_keyword(article["headline"])
        sentiment = headline_sentiment(article["headline"])
        words = tokenize(article["headline"])
        text = article["headline"]
        group_size = article.get("_group_size", 1)
        published = article["published_utc"]
        breaking = int(any(w in text.lower() for w in
                           ("breaking", "just in", "urgent", "developing")))
        risk = level_risk_score(
            classification["level"],
            corroboration=group_size,
            breaking=breaking,
            sentiment=sentiment,
            conflict_region=(region != "Global" and classification["category"]
                             in ("conflict", "military", "terrorism")),
        )
        assembled.append(
            {
                "event_id": hashlib.sha1(
                    (article["link"] or text).encode("utf-8")
                ).hexdigest()[:16],
                "headline": text,
                "snippet": article["snippet"],
                "source_domain": article["source_domain"],
                "source_country": article["source_country"],
                "published_utc": published,
                "region": region,
                "detected_countries": ";".join(codes),
                "sentiment_score": round(sentiment, 4),
                "corroboration_count": group_size,
                "is_breaking": breaking,
                "text_len": len(text),
                "word_count": len(words),
                "upper_ratio": round(
                    sum(1 for c in text if c.isupper()) / max(len(text), 1), 4
                ),
                "velocity_per_hour": round(group_size / max(hours_span(published), 1.0), 4),
                "risk_score": risk,
                "risk_band": band_for_score(risk),
            }
        )
    return assembled


def hours_span(published_iso: str) -> float:
    """Hours since publication (fetch window is days, so 1h floor)."""
    if not published_iso:
        return 1.0
    try:
        moment = datetime.fromisoformat(published_iso)
    except ValueError:
        return 1.0
    delta = datetime.now(timezone.utc) - moment
    return max(delta.total_seconds() / 3600.0, 1.0)


def level_risk_score(
    level: str,
    corroboration: int,
    breaking: int,
    sentiment: float,
    conflict_region: bool,
) -> int:
    """Transparent derived risk score (0-100) from heuristic features."""
    score = LEVEL_RISK_WEIGHTS.get(level, 2.0)
    if corroboration >= 3:
        score += 8
    elif corroboration == 2:
        score += 4
    if breaking:
        score += 6
    score += 4.0 * abs(sentiment)
    if conflict_region:
        score += 5
    return int(max(0, min(100, round(score))))


def band_for_score(score: int) -> str:
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timespan", default="3d")
    parser.add_argument("--maxrecords", type=int, default=250)
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()

    print("Fetching GDELT articles...", file=sys.stderr)
    raw_articles = fetch_articles(args.timespan, args.maxrecords, args.delay)
    if not raw_articles:
        print("No articles fetched; aborting without touching snapshots.",
              file=sys.stderr)
        return 1
    assembled = assemble_articles(raw_articles)
    write_csv(DATA_DIR / "gsms_events.csv", COLUMNS, assembled)
    print(f"  wrote {len(assembled)} articles -> gsms_events.csv", file=sys.stderr)

    print("Fetching USGS earthquakes...", file=sys.stderr)
    quakes = fetch_quakes()
    quake_columns = [
        "quake_id", "time_utc", "magnitude", "place", "latitude",
        "longitude", "depth_km", "significance", "tsunami_flag",
    ]
    write_csv(DATA_DIR / "gsms_quakes.csv", quake_columns, quakes)
    print(f"  wrote {len(quakes)} earthquakes -> gsms_quakes.csv", file=sys.stderr)

    # Tidy categorical table for concept learning: discretize real events.
    concept_rows = [discretize(row) for row in assembled]
    concept_columns = [
        "category", "region", "alert_level", "source_domain", "sentiment_class",
        "corroboration_class", "breaking", "high_risk",
    ]
    write_csv(DATA_DIR / "gsms_concept.csv", concept_columns, concept_rows)
    print(f"  wrote {len(concept_rows)} rows -> gsms_concept.csv", file=sys.stderr)

    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "articles": len(assembled),
        "quakes": len(quakes),
        "concept_rows": len(concept_rows),
    }
    (DATA_DIR / "build_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


def discretize(row: dict) -> dict:
    sentiment = row["sentiment_score"]
    if sentiment <= -0.35:
        sentiment_class = "negative"
    elif sentiment >= 0.35:
        sentiment_class = "positive"
    else:
        sentiment_class = "neutral"
    corroboration = row["corroboration_count"]
    if corroboration >= 5:
        corroboration_class = "high"
    elif corroboration >= 2:
        corroboration_class = "medium"
    else:
        corroboration_class = "low"
    return {
        "category": row["region"].lower().replace(" ", "_")
        if row["region"] != "Global"
        else "global",
        "region": row["region"],
        "alert_level": row["risk_band"],
        "source_domain": row["source_domain"],
        "sentiment_class": sentiment_class,
        "corroboration_class": corroboration_class,
        "breaking": row["is_breaking"],
        "high_risk": 1 if row["risk_score"] >= 50 else 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
