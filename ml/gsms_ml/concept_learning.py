"""Concept learning: Find-S and the Candidate Elimination algorithm.

Classic symbolic concept learning from Mitchell, *Machine Learning* (1997),
chapter 2, demonstrated on real (but discretized) GSMS event data.

Hypothesis language
-------------------
A hypothesis is a dict mapping attribute name -> constraint, where a
constraint is:

- a concrete attribute VALUE (e.g. ``"high"``) — the instance must match it,
- ``"?"`` — any value accepted (the fully general constraint),
- ``"∅"`` — no value accepted (the empty / most specific hypothesis).

``covers(hypothesis, example)`` implements that semantics; ``more_general``
implements the generalization ordering ``g ⊒ h`` (g covers every instance h
covers).

Find-S
------
Scans the examples, ignoring negatives, and maintains the most specific
hypothesis consistent with the positive examples seen so far: every attribute
where two positives disagree (or a later positive disagrees with the current
hypothesis) is generalized to ``"?"``. With no positive examples the result is
``{}`` — the empty hypothesis, which accepts nothing.

Candidate Elimination
---------------------
Maintains BOTH boundary sets of the version space: the specific boundary S
(initially all ``"∅"``) and the general boundary G (initially all ``"?"``),
updating them with each example exactly as in Mitchell Figure 2.3. To stay
robust on noisy real data, each update is applied *tentatively*: if the
example would empty the version space (S empty, G empty, or no remaining
s ⊑ g pair — the boundaries crossed), the example is rolled back and recorded
in ``notes`` instead of crashing. This is an explicit, documented deviation
from the textbook algorithm, which assumes consistent data.

Design note: the minimal specializations of a G member on a negative example
pin one ``"?"`` attribute to a concrete value. To keep G finite we draw those
values from the distinct values observed among the *positive* training
examples (a standard implementation choice; Mitchell leaves the value
enumeration to the hypothesis-language definition).
"""

from __future__ import annotations

ANY = "?"        # constraint: any value accepted
NONE = "∅"       # constraint: no value accepted (empty hypothesis)


def attributes_of(examples: list[dict], label_key: str) -> list[str]:
    """Ordered union of attribute names across examples, excluding the label."""
    attributes: list[str] = []
    for example in examples:
        for key in example:
            if key != label_key and key not in attributes:
                attributes.append(key)
    return attributes


def covers(hypothesis: dict, example: dict, attributes: list[str]) -> bool:
    """True if `hypothesis` classifies `example` as positive.

    '?' accepts anything; '∅' accepts nothing; a concrete value must equal
    the example's value for that attribute.
    """
    for attribute in attributes:
        constraint = hypothesis.get(attribute, ANY)
        if constraint == ANY:
            continue
        if constraint == NONE:
            return False
        if example.get(attribute) != constraint:
            return False
    return True


def more_general(g: dict, h: dict, attributes: list[str]) -> bool:
    """True if g ⊒ h: g covers every instance that h covers.

    Per attribute: '?' on g's side is always general enough; '∅' on h's side
    means h accepts nothing there so any g works; otherwise g must accept the
    same value (or be '?'). '∅' on g's side with a concrete h is not general
    enough.
    """
    for attribute in attributes:
        g_c = g.get(attribute, ANY)
        h_c = h.get(attribute, ANY)
        if g_c == ANY:
            continue
        if h_c == NONE:
            continue
        if g_c != h_c:
            return False
    return True


def _equal(h1: dict, h2: dict, attributes: list[str]) -> bool:
    return all(h1.get(a, ANY) == h2.get(a, ANY) for a in attributes)


def find_s(
    examples: list[dict], label_key: str, positive_value
) -> dict:
    """Most specific hypothesis consistent with the positive examples.

    Starts from the empty hypothesis ({} = '∅' everywhere, accepts nothing)
    and, for every positive example, generalizes each disagreeing attribute
    to '?'. Negative examples are ignored, per Mitchell §2.4.
    """
    attributes = attributes_of(examples, label_key)
    hypothesis: dict | None = None
    for example in examples:
        if example.get(label_key) != positive_value:
            continue
        if hypothesis is None:
            hypothesis = {attribute: example.get(attribute) for attribute in attributes}
        else:
            hypothesis = {
                attribute: (
                    hypothesis[attribute]
                    if hypothesis[attribute] == example.get(attribute)
                    else ANY
                )
                for attribute in attributes
            }
    return hypothesis if hypothesis is not None else {}


def find_s_trace(
    examples: list[dict], label_key: str, positive_value
) -> list[dict]:
    """Per-example snapshots of the Find-S hypothesis, for display.

    Returns one record per example: {"step", "label", "is_positive",
    "hypothesis"} where "hypothesis" is the Find-S state AFTER that example.
    """
    attributes = attributes_of(examples, label_key)
    hypothesis: dict | None = None
    trace: list[dict] = []
    for step, example in enumerate(examples):
        is_positive = example.get(label_key) == positive_value
        if is_positive:
            if hypothesis is None:
                hypothesis = {
                    attribute: example.get(attribute) for attribute in attributes
                }
            else:
                hypothesis = {
                    attribute: (
                        hypothesis[attribute]
                        if hypothesis[attribute] == example.get(attribute)
                        else ANY
                    )
                    for attribute in attributes
                }
        trace.append(
            {
                "step": step,
                "label": example.get(label_key),
                "is_positive": is_positive,
                "hypothesis": dict(hypothesis) if hypothesis is not None else {},
            }
        )
    return trace


def _initial_specific(attributes: list[str]) -> dict:
    return {attribute: NONE for attribute in attributes}


def _initial_general(attributes: list[str]) -> dict:
    return {attribute: ANY for attribute in attributes}


def _minimal_generalization(
    specific: dict, example: dict, attributes: list[str]
) -> dict:
    """Unique minimal generalization of `specific` covering `example`.

    A '∅' attribute accepts nothing, so relaxing it to the example's value is
    the minimal choice (Mitchell: S_1 <- d on the first positive). A concrete
    attribute that disagrees must be relaxed to '?': any other concrete value
    would stop covering the specific hypothesis's own instances.
    """
    generalized: dict = {}
    for attribute in attributes:
        current = specific[attribute]
        example_value = example.get(attribute)
        if current == NONE:
            generalized[attribute] = example_value
        elif current == example_value:
            generalized[attribute] = current
        else:
            generalized[attribute] = ANY
    return generalized


def _minimal_specializations(
    general: dict, example: dict, attributes: list[str],
    candidate_values: dict[str, list],
) -> list[dict]:
    """Minimal specializations of `general` that no longer cover `example`.

    Pins exactly one '?' attribute to a concrete value drawn from the values
    observed among positive examples (excluding the negative example's own
    value, which alone would not exclude it).
    """
    results: list[dict] = []
    for attribute in attributes:
        if general.get(attribute, ANY) != ANY:
            continue
        for value in candidate_values.get(attribute, []):
            if value == example.get(attribute):
                continue
            specialized = dict(general)
            specialized[attribute] = value
            results.append(specialized)
    return results


def _deduplicate(hypotheses: list[dict], attributes: list[str]) -> list[dict]:
    unique: list[dict] = []
    for hypothesis in hypotheses:
        if not any(_equal(hypothesis, kept, attributes) for kept in unique):
            unique.append(hypothesis)
    return unique


def _prune_more_general(boundary: list[dict], attributes: list[str]) -> list[dict]:
    """Remove any hypothesis more general than another in the same boundary."""
    pruned = [
        h
        for h in boundary
        if not any(
            other is not h and more_general(other, h, attributes)
            and not _equal(other, h, attributes)
            for other in boundary
        )
    ]
    return _deduplicate(pruned, attributes)


def _version_space_alive(
    specific: list[dict], general: list[dict], attributes: list[str]
) -> bool:
    """True iff both boundaries are non-empty and some s ⊑ g pair exists."""
    if not specific or not general:
        return False
    return any(
        more_general(g, s, attributes) for s in specific for g in general
    )


def candidate_elimination(
    examples: list[dict], label_key: str, positive_value
) -> dict:
    """Version-space boundaries S and G over the examples (Mitchell Fig. 2.3).

    Returns a dict with:

    - ``specific``: the S boundary (list of hypothesis dicts),
    - ``general``: the G boundary (list of hypothesis dicts),
    - ``version_space_size``: len(S) * len(G) — the number of boundary pairs
      delimiting the version space (the full hypothesis lattice between the
      boundaries is exponential and deliberately not enumerated),
    - ``history``: per-example snapshots {"step", "label", "is_positive",
      "specific", "general", "skipped"} after each example,
    - ``notes``: human-readable records of every noisy/conflicting example
      that was skipped instead of crashing,
    - ``skipped_examples``: how many examples were rolled back as noise.
    """
    attributes = attributes_of(examples, label_key)
    if not attributes:
        return {
            "specific": [],
            "general": [],
            "version_space_size": 0,
            "history": [],
            "notes": ["no attributes available for concept learning"],
            "skipped_examples": 0,
        }

    # values per attribute observed among positive examples — the finite
    # enumeration used to specialize G (see module docstring)
    candidate_values: dict[str, list] = {
        attribute: sorted(
            {
                example[attribute]
                for example in examples
                if example.get(label_key) == positive_value
                and attribute in example
            },
            key=str,
        )
        for attribute in attributes
    }

    specific_boundary = [_initial_specific(attributes)]
    general_boundary = [_initial_general(attributes)]
    history: list[dict] = []
    notes: list[str] = []
    skipped = 0

    def snapshot(step: int, example: dict, was_skipped: bool) -> None:
        history.append(
            {
                "step": step,
                "label": example.get(label_key),
                "is_positive": example.get(label_key) == positive_value,
                "specific": [dict(h) for h in specific_boundary],
                "general": [dict(h) for h in general_boundary],
                "skipped": was_skipped,
            }
        )

    for step, example in enumerate(examples):
        is_positive = example.get(label_key) == positive_value
        previous_specific = [dict(h) for h in specific_boundary]
        previous_general = [dict(h) for h in general_boundary]
        note: str | None = None

        if is_positive:
            # drop G members inconsistent with the positive
            general_boundary = [
                g for g in general_boundary if covers(g, example, attributes)
            ]
            # generalize S members that fail to cover the positive
            new_specific: list[dict] = []
            for s in specific_boundary:
                if covers(s, example, attributes):
                    new_specific.append(s)
                    continue
                generalized = _minimal_generalization(s, example, attributes)
                if any(more_general(g, generalized, attributes) for g in general_boundary):
                    new_specific.append(generalized)
                else:
                    note = (
                        f"step {step}: positive example not reachable from S "
                        f"within G (no general boundary covers the required "
                        f"generalization) — treated as noise"
                    )
            specific_boundary = _prune_more_general(new_specific, attributes)
        else:
            # drop S members that cover the negative
            specific_boundary = [
                s for s in specific_boundary
                if not covers(s, example, attributes)
            ]
            # specialize G members that cover the negative
            new_general: list[dict] = []
            for g in general_boundary:
                if not covers(g, example, attributes):
                    new_general.append(g)
                    continue
                valid_specializations = [
                    specialized
                    for specialized in _minimal_specializations(
                        g, example, attributes, candidate_values
                    )
                    if not covers(specialized, example, attributes)
                    # a G member must stay general enough to cover the S
                    # boundary: keep h iff h ⊒ some s
                    and any(more_general(specialized, s, attributes)
                            for s in specific_boundary)
                ]
                if valid_specializations:
                    new_general.extend(valid_specializations)
                else:
                    note = (
                        f"step {step}: negative example admits no valid "
                        f"specialization of G above S — treated as noise"
                    )
            general_boundary = _prune_more_general(new_general, attributes)

        # noise guard: never let the example empty or cross the version space
        if not _version_space_alive(specific_boundary, general_boundary, attributes):
            specific_boundary = previous_specific
            general_boundary = previous_general
            skipped += 1
            if note is None:
                note = (
                    f"step {step}: example would empty/cross the version "
                    f"space (S and G boundaries inconsistent) — skipped as noise"
                )
            note = note + " [rolled back]"
            notes.append(note)
            snapshot(step, example, True)
            continue
        if note is not None:
            notes.append(note)
        snapshot(step, example, False)

    return {
        "specific": specific_boundary,
        "general": general_boundary,
        "version_space_size": len(specific_boundary) * len(general_boundary),
        "history": history,
        "notes": notes,
        "skipped_examples": skipped,
    }
