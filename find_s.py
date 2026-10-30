"""Find-S: the most-specific-hypothesis concept learner (UNIT II, from scratch).

Pure Python — no sklearn. Hypotheses are tuples over the attribute columns;
``?`` matches anything, ``""`` (empty) contradicts everything. Find-S scans
the positive examples and generalises the hypothesis to the most specific
tuple consistent with every positive example seen.
"""
from __future__ import annotations

from typing import Iterable

MATCH_ANY = "?"
EMPTY = ""  # most specific: matches nothing


def matches(hypothesis: tuple, example: tuple) -> bool:
    return all(h == MATCH_ANY or h == e for h, e in zip(hypothesis, example))


def is_more_general(a: tuple, b: tuple) -> bool:
    """True if hypothesis ``a`` is strictly more general than ``b``."""
    return a != b and all(h == MATCH_ANY or h == v for h, v in zip(a, b))


def find_s(examples: Iterable[tuple[tuple, bool]],
           trace: list[str] | None = None) -> tuple[tuple, list[str]]:
    """Return (most specific hypothesis consistent with all positives, trace).

    ``examples`` yields ((attr...), label) pairs where label True = positive.
    """
    hypothesis: tuple | None = None
    if trace is None:
        trace = []
    n_seen = 0
    n_attrs = 0
    for attrs, positive in examples:
        n_seen += 1
        attrs = tuple(attrs)
        n_attrs = len(attrs)
        if positive:
            if hypothesis is None:
                hypothesis = attrs
            elif not matches(hypothesis, attrs):
                hypothesis = tuple(
                    old if old == new else MATCH_ANY
                    for old, new in zip(hypothesis, attrs)
                )
        trace.append(f"example {n_seen} ({'positive' if positive else 'negative'}): "
                     f"h = {format_hypothesis(hypothesis, len(attrs))}")
    if hypothesis is None:
        hypothesis = tuple([EMPTY] * n_attrs)
    return hypothesis, trace


def format_hypothesis(h: tuple | None, n_attrs: int) -> str:
    if h is None:
        return "<none positive seen>"
    return "<" + ", ".join(h) + ">"
