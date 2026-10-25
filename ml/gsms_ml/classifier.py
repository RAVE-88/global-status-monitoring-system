"""Python port of the dashboard's keyword threat classifier.

Faithful port of ``shared/threat-keyword-classifier.ts`` (the classifier the
browser applies to every headline), so a dataset row labelled in Python carries
exactly the ``level``/``category`` the dashboard would show for the same title.
Tables are transcribed 1:1; the regex semantics (whole-word for short
keywords, trailing boundary for the Iran compound set, plain substring for the
rest) are reproduced with compiled regexes. Confidence values mirror the
browser's cascade. Labels produced from this port are heuristic and derived —
they are the *inputs* the ML models learn from, never ground truth.
"""

from __future__ import annotations

import re

ThreatLevel = str  # critical | high | medium | low | info
EventCategory = str

CRITICAL_KEYWORDS: dict[str, EventCategory] = {
    "nuclear strike": "military", "nuclear attack": "military",
    "nuclear war": "military", "invasion": "conflict",
    "declaration of war": "conflict", "declares war": "conflict",
    "all-out war": "conflict", "full-scale war": "conflict",
    "martial law": "military", "coup": "military", "coup attempt": "military",
    "genocide": "conflict", "ethnic cleansing": "conflict",
    "chemical attack": "terrorism", "biological attack": "terrorism",
    "dirty bomb": "terrorism", "mass casualty": "conflict",
    "massive strikes": "military", "military strikes": "military",
    "retaliatory strikes": "military", "launches strikes": "military",
    "launch attacks on iran": "military", "launch attack on iran": "military",
    "attacks on iran": "military", "strikes on iran": "military",
    "strikes iran": "military", "bombs iran": "military",
    "attacks iran": "military", "attack on iran": "military",
    "attack iran": "military", "attacked iran": "military",
    "attack against iran": "military", "bombing iran": "military",
    "bombed iran": "military", "war with iran": "conflict",
    "war on iran": "conflict", "war against iran": "conflict",
    "iran retaliates": "military", "iran strikes": "military",
    "iran launches": "military", "iran attacks": "military",
    "pandemic declared": "health", "health emergency": "health",
    "nato article 5": "military", "evacuation order": "disaster",
    "meltdown": "disaster", "nuclear meltdown": "disaster",
    "major combat operations": "military", "declared war": "conflict",
}

HIGH_KEYWORDS: dict[str, EventCategory] = {
    "war": "conflict", "armed conflict": "conflict", "airstrike": "conflict",
    "airstrikes": "conflict", "air strike": "conflict",
    "air strikes": "conflict", "drone strike": "conflict",
    "drone strikes": "conflict", "strikes": "conflict", "missile": "military",
    "missile launch": "military", "missiles fired": "military",
    "troops deployed": "military", "military escalation": "military",
    "military operation": "military", "ground offensive": "military",
    "bombing": "conflict", "bombardment": "conflict", "shelling": "conflict",
    "casualties": "conflict", "killed in": "conflict", "hostage": "terrorism",
    "terrorist": "terrorism", "terror attack": "terrorism",
    "assassination": "crime", "cyber attack": "cyber", "ransomware": "cyber",
    "data breach": "cyber", "sanctions": "economic", "embargo": "economic",
    "earthquake": "disaster", "tsunami": "disaster", "hurricane": "disaster",
    "typhoon": "disaster", "strike on": "conflict", "strikes on": "conflict",
    "attack on": "conflict", "attack against": "conflict",
    "attacks on": "conflict", "launched attack": "conflict",
    "launched attacks": "conflict", "launches attack": "conflict",
    "launches attacks": "conflict", "explosions": "conflict",
    "military operations": "military", "combat operations": "military",
    "retaliatory strike": "military", "retaliatory attack": "military",
    "retaliatory attacks": "military", "preemptive strike": "military",
    "preemptive attack": "military", "preventive attack": "military",
    "preventative attack": "military", "military offensive": "military",
    "ballistic missile": "military", "cruise missile": "military",
    "air defense intercepted": "military", "forces struck": "conflict",
}

MEDIUM_KEYWORDS: dict[str, EventCategory] = {
    "protest": "protest", "protests": "protest", "riot": "protest",
    "riots": "protest", "unrest": "protest", "demonstration": "protest",
    "strike action": "protest", "military exercise": "military",
    "naval exercise": "military", "arms deal": "military",
    "weapons sale": "military", "diplomatic crisis": "diplomatic",
    "ambassador recalled": "diplomatic", "expel diplomats": "diplomatic",
    "trade war": "economic", "tariff": "economic", "recession": "economic",
    "inflation": "economic", "market crash": "economic", "flood": "disaster",
    "flooding": "disaster", "wildfire": "disaster", "volcano": "disaster",
    "eruption": "disaster", "outbreak": "health", "epidemic": "health",
    "infection spread": "health", "oil spill": "environmental",
    "pipeline explosion": "infrastructure", "blackout": "infrastructure",
    "power outage": "infrastructure", "internet outage": "infrastructure",
    "derailment": "infrastructure",
}

LOW_KEYWORDS: dict[str, EventCategory] = {
    "election": "diplomatic", "vote": "diplomatic", "referendum": "diplomatic",
    "summit": "diplomatic", "treaty": "diplomatic", "agreement": "diplomatic",
    "negotiation": "diplomatic", "talks": "diplomatic",
    "peacekeeping": "diplomatic", "humanitarian aid": "diplomatic",
    "ceasefire": "diplomatic", "peace treaty": "diplomatic",
    "climate change": "environmental", "emissions": "environmental",
    "pollution": "environmental", "deforestation": "environmental",
    "drought": "environmental", "vaccine": "health", "vaccination": "health",
    "disease": "health", "virus": "health", "public health": "health",
    "covid": "health", "interest rate": "economic", "gdp": "economic",
    "unemployment": "economic", "regulation": "economic",
}

TECH_HIGH_KEYWORDS: dict[str, EventCategory] = {
    "major outage": "infrastructure", "service down": "infrastructure",
    "global outage": "infrastructure", "zero-day": "cyber",
    "critical vulnerability": "cyber", "supply chain attack": "cyber",
    "mass layoff": "economic",
}

TECH_MEDIUM_KEYWORDS: dict[str, EventCategory] = {
    "outage": "infrastructure", "breach": "cyber", "hack": "cyber",
    "vulnerability": "cyber", "layoff": "economic", "layoffs": "economic",
    "antitrust": "economic", "monopoly": "economic", "ban": "economic",
    "shutdown": "infrastructure",
}

TECH_LOW_KEYWORDS: dict[str, EventCategory] = {
    "ipo": "economic", "funding": "economic", "acquisition": "economic",
    "merger": "economic", "launch": "tech", "release": "tech",
    "update": "tech", "partnership": "economic", "startup": "tech",
    "ai model": "tech", "open source": "tech",
}

EXCLUSIONS = [
    "protein", "couples", "relationship", "dating", "diet", "fitness",
    "recipe", "cooking", "shopping", "fashion", "celebrity", "movie",
    "tv show", "sports", "game", "concert", "festival", "wedding",
    "vacation", "travel tips", "life hack", "self-care", "wellness",
    "strikes deal", "strikes agreement", "strikes partnership",
]

SHORT_KEYWORDS = {"war", "coup", "ban", "vote", "riot", "riots", "hack",
                  "talks", "ipo", "gdp", "virus", "disease", "flood", "strikes"}

TRAILING_BOUNDARY_KEYWORDS = {
    "attack iran", "attacked iran", "attack on iran", "attack against iran",
    "attacks on iran", "launch attacks on iran", "launch attack on iran",
    "bombing iran", "bombed iran", "strikes iran", "attacks iran",
    "bombs iran", "war on iran", "war with iran", "war against iran",
    "iran retaliates", "iran strikes", "iran launches", "iran attacks",
}

ESCALATION_ACTIONS = re.compile(
    r"\b(attack|attacks|attacked|strike|strikes|struck|bomb|bombs|bombed|bombing"
    r"|shell|shelled|shelling|missile|missiles|intercept|intercepted|retaliates"
    r"|retaliating|retaliation|killed|casualties|offensive|invaded|invades)\b"
)
ESCALATION_TARGETS = re.compile(
    r"\b(iran|tehran|isfahan|tabriz|russia|moscow|china|beijing|taiwan|taipei"
    r"|north korea|pyongyang|nato|us base|us forces|american forces|us military)\b"
)

_regex_cache: dict[str, re.Pattern[str]] = {}


def _keyword_regex(kw: str) -> re.Pattern[str]:
    cached = _regex_cache.get(kw)
    if cached is None:
        escaped = re.escape(kw)
        if kw in SHORT_KEYWORDS:
            cached = re.compile(rf"\b{escaped}\b")
        elif kw in TRAILING_BOUNDARY_KEYWORDS:
            cached = re.compile(rf"{escaped}(?![\w-])")
        else:
            cached = re.compile(escaped)
        _regex_cache[kw] = cached
    return cached


def _match_keywords(title_lower: str, keywords: dict[str, EventCategory]):
    for kw, cat in keywords.items():
        if _keyword_regex(kw).search(title_lower):
            return kw, cat
    return None


def _should_escalate_to_critical(lower: str, match_cat: EventCategory) -> bool:
    if match_cat not in ("conflict", "military"):
        return False
    return bool(ESCALATION_ACTIONS.search(lower) and ESCALATION_TARGETS.search(lower))


def classify_by_keyword(title: str, variant: str = "full") -> dict:
    """Return {'level', 'category', 'confidence', 'source'} for a headline.

    Mirrors the browser cascade: exclusions → critical → high (with compound
    escalation) → tech tables (tech variant only) → medium → low → info.
    """
    lower = title.lower()

    if any(ex in lower for ex in EXCLUSIONS):
        return {"level": "info", "category": "general", "confidence": 0.3,
                "source": "keyword"}

    is_tech = variant == "tech"

    match = _match_keywords(lower, CRITICAL_KEYWORDS)
    if match:
        return {"level": "critical", "category": match[1], "confidence": 0.9,
                "source": "keyword"}

    match = _match_keywords(lower, HIGH_KEYWORDS)
    if match:
        if _should_escalate_to_critical(lower, match[1]):
            return {"level": "critical", "category": match[1],
                    "confidence": 0.85, "source": "keyword"}
        return {"level": "high", "category": match[1], "confidence": 0.8,
                "source": "keyword"}

    if is_tech:
        match = _match_keywords(lower, TECH_HIGH_KEYWORDS)
        if match:
            return {"level": "high", "category": match[1], "confidence": 0.75,
                    "source": "keyword"}

    match = _match_keywords(lower, MEDIUM_KEYWORDS)
    if match:
        return {"level": "medium", "category": match[1], "confidence": 0.7,
                "source": "keyword"}

    if is_tech:
        match = _match_keywords(lower, TECH_MEDIUM_KEYWORDS)
        if match:
            return {"level": "medium", "category": match[1], "confidence": 0.65,
                    "source": "keyword"}

    match = _match_keywords(lower, LOW_KEYWORDS)
    if match:
        return {"level": "low", "category": match[1], "confidence": 0.6,
                "source": "keyword"}

    if is_tech:
        match = _match_keywords(lower, TECH_LOW_KEYWORDS)
        if match:
            return {"level": "low", "category": match[1], "confidence": 0.55,
                    "source": "keyword"}

    return {"level": "info", "category": "general", "confidence": 0.3,
            "source": "keyword"}


# Transparent risk contribution per threat level (0–100 scale), shared by the
# dataset builder and the status engine.
LEVEL_RISK_WEIGHTS: dict[str, float] = {
    "critical": 60.0,
    "high": 40.0,
    "medium": 20.0,
    "low": 8.0,
    "info": 2.0,
}
