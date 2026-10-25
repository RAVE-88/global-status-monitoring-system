"""Compact, committed sentiment lexicon and keyword→event-category rules.

The sentiment lexicon is a small AFINN-style word→score table (integer scores in
[-5, 5]) covering vocabulary that dominates global-news headlines. It is a
derived heuristic feature, not an oracle: it is fixed once, committed here, and
applied deterministically.

Stems are used deliberately ("casualt", "devastat") so inflected forms match
without enumerating them; the matcher treats a lexicon entry as a token prefix.
"""

from __future__ import annotations

import math
import re

# AFINN-style integer scores, range [-5, 5]. Entries are matched as token
# prefixes, so "casualt" covers casualty/casualties.
SENTIMENT_LEXICON: dict[str, int] = {
    # strongly negative
    "catastrophic": -5, "genocide": -5, "massacre": -5, "atrocit": -4,
    "devastat": -4, "apocalyptic": -5, "war": -3, "deadliest": -4,
    "warfare": -3,
    # negative
    "attack": -2, "attacked": -2, "killed": -3, "death": -3,
    "deaths": -3, "died": -3, "casualt": -3, "wounded": -2, "injured": -2,
    "crisis": -2, "conflict": -2, "fighting": -2, "clash": -2,
    "strike": -1, "airstrike": -2, "bombing": -2, "bombed": -2,
    "bomb": -2, "missile": -1, "shelling": -2, "invasion": -3, "offensive": -1,
    "fear": -2, "threat": -2, "danger": -2, "warning": -1,
    "tension": -1, "escalation": -2, "escalat": -1,
    "collapse": -2, "crash": -2, "plunge": -2, "slump": -1, "fall": -1,
    "drop": -1, "loss": -2,
    "recession": -2, "inflation": -1, "unemployment": -2, "poverty": -2,
    "famine": -4, "drought": -2, "flood": -2, "wildfire": -2,
    "earthquake": -2, "tsunami": -3, "hurricane": -2, "typhoon": -2,
    "storm": -1, "outbreak": -2, "epidemic": -3, "pandemic": -3, "virus": -1,
    "disease": -1, "infection": -1, "victim": -2,
    "hostage": -3, "terror": -3,
    "ransomware": -2, "breach": -1, "hacked": -2, "cyberattack": -2,
    "outage": -1, "blackout": -1, "sanction": -1,
    "embargo": -2, "corrupt": -2, "protest": -1,
    "riot": -2, "unrest": -2, "coup": -3,
    "displaced": -2, "refugee": -1, "emergency": -1,
    "explosion": -2, "blast": -2, "assassin": -3, "arrest": -1,
    "detained": -1, "killing": -3, "brutal": -3, "violence": -3,
    "violent": -3, "worsen": -1, "shortage": -1,
    "hike": -1, "default": -2, "controversy": -1, "scandal": -2,
    "condemn": -1, "criticiz": -1, "warn": -1,
    "slams": -1, "crackdown": -2, "ban": -1,
    "blockade": -2, "siege": -3, "toll": -2, "raging": -2, "sinks": -1,
    "tumbles": -1, "slows": -1, "rebuke": -1, "dispute": -1,
    "deficit": -1, "debt": -1, "layoff": -2, "shutdown": -1,
    "derail": -1, "fatal": -3, "deadly": -3, "tragedy": -3,
    # positive
    "peace": 3, "ceasefire": 2, "agreement": 1, "deal": 1, "treaty": 1,
    "summit": 1, "talks": 1, "aid": 1, "humanitarian": 1, "recovery": 2,
    "recover": 1, "growth": 2, "grow": 1, "gains": 1, "rise": 1,
    "surge": 1, "boost": 1, "win": 2,
    "won": 2, "success": 2, "breakthrough": 3,
    "progress": 2, "improve": 2, "hope": 2,
    "rescue": 2, "relief": 1, "support": 1, "rebuild": 2,
    "reform": 1, "vaccine": 1, "cure": 3, "treatment": 1,
    "election": 1, "cooperation": 2, "partner": 1,
    "investment": 1, "innovation": 2,
    "launch": 1, "discovery": 2, "award": 2,
    "historic": 1, "milestone": 2, "stabil": 1, "steady": 1, "strong": 1,
    "robust": 2, "profit": 2, "surplus": 2, "expansion": 1,
    "freed": 2, "safe": 1, "safety": 1,
    "thrive": 3, "prosper": 3, "optimis": 2, "positive": 2,
    "welcom": 1, "applaud": 2, "praise": 2, "commend": 2, "resolve": 1,
    "mediat": 1, "restor": 1, "advance": 1, "achieve": 2,
}

# Negations flip the polarity of the next scored word (small, fixed window).
NEGATIONS = {"not", "no", "never", "without", "denies", "denied", "rejects",
             "rejected", "avoids", "avoided"}

# Diminishers/intensifiers scale the next scored word.
DEGREE_MULTIPLIERS = {"very": 1.5, "extremely": 2.0, "highly": 1.5,
                      "slightly": 0.5, "barely": 0.5, "massively": 2.0,
                      "sharply": 1.5, "severely": 1.75}

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens of a headline/snippet."""
    return _TOKEN_RE.findall(text.lower())


def headline_sentiment(text: str) -> float:
    """Lexicon sentiment in [-1, 1] for a headline/snippet.

    Deterministic, interpretable baseline: tokenizes on non-letters, sums
    AFINN-style scores with negation flipping and degree scaling, then
    soft-normalizes so multi-word texts don't dominate.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    total = 0.0
    hits = 0
    flipped = False
    degree = 1.0
    for tok in tokens:
        if tok in NEGATIONS:
            flipped = True
            continue
        if tok in DEGREE_MULTIPLIERS:
            degree = DEGREE_MULTIPLIERS[tok]
            continue
        score = _prefix_lexicon_lookup(tok)
        if score is None:
            continue
        value = score * degree
        if flipped:
            value = -value * 0.75
        total += value
        hits += 1
        flipped = False
        degree = 1.0
    if hits == 0:
        return 0.0
    return math.tanh(total / (2.0 * math.sqrt(hits)))


def _prefix_lexicon_lookup(token: str) -> int | None:
    """Return the score of the longest lexicon entry that prefixes `token`."""
    for end in range(len(token), 2, -1):
        score = SENTIMENT_LEXICON.get(token[:end])
        if score is not None:
            return score
    return None
