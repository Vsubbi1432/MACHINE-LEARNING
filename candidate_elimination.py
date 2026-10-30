"""Candidate Elimination: version-space boundary sets (UNIT II, from scratch).

Pure Python. Maintains the specific boundary S (initially the all-empty,
most-specific hypothesis) and the general boundary G (initially all "?"),
generalising S on positives and specialising G on negatives. Specialisation
uses the attribute domains, so G members stay ground hypotheses. The version
space is exactly the hypotheses between the two boundaries.
"""
from __future__ import annotations

from itertools import product
from typing import Iterable

MATCH_ANY = "?"
EMPTY = ""  # most specific: matches nothing


def covers(g: tuple, h: tuple) -> bool:
    """True if g matches everything h matches (g more general or equal)."""
    return all(a == MATCH_ANY or a == b for a, b in zip(g, h))


def matches(h: tuple, example: tuple) -> bool:
    return covers(h, example)


def _minimize_general(hyps: list[tuple]) -> list[tuple]:
    """Drop members that duplicate or are covered by another member."""
    out: list[tuple] = []
    for h in hyps:
        if h in out:
            continue
        if not any(o != h and covers(o, h) for o in hyps):
            out.append(h)
    return out


def _maximize(hyps: list[tuple]) -> list[tuple]:
    """Keep only maximal members (drop those covered by another member)."""
    out: list[tuple] = []
    for h in hyps:
        if h in out:
            continue
        if not any(o != h and covers(o, h) for o in hyps):
            out.append(h)
    return out


def candidate_elimination(examples: Iterable[tuple[tuple, bool]],
                          domains: dict[int, list[str]],
                          max_trace: int = 12) -> dict:
    """Run Candidate Elimination and return the boundary-set summary."""
    n = len(domains)
    S: list[tuple] = [tuple([EMPTY] * n)]
    G: list[tuple] = [tuple([MATCH_ANY] * n)]
    trace: list[str] = []
    n_pos = n_neg = 0

    for step, (attrs, positive) in enumerate(examples, start=1):
        attrs = tuple(attrs)
        if positive:
            n_pos += 1
            G = [g for g in G if matches(g, attrs)]
            if not any(matches(s, attrs) for s in S):
                candidates = []
                for s in S:
                    generalized = tuple(
                        a if old == EMPTY else (old if old in (a, MATCH_ANY) else MATCH_ANY)
                        for old, a in zip(s, attrs)
                    )
                    candidates.append(generalized)
                S = _minimize_general(candidates)
            S = [s for s in S if any(covers(g, s) for g in G)]
        else:
            n_neg += 1
            S = [s for s in S if not matches(s, attrs)]
            new_G: list[tuple] = []
            for g in G:
                if not matches(g, attrs):
                    new_G.append(g)
                    continue
                for i in range(n):
                    if g[i] != MATCH_ANY:
                        continue
                    for v in domains[i]:
                        if v == attrs[i]:
                            continue
                        candidate = g[:i] + (v,) + g[i + 1:]
                        if not matches(candidate, attrs) and any(
                                covers(candidate, s) for s in S):
                            new_G.append(candidate)
            G = _maximize(new_G)
            G = [g for g in G if any(covers(g, s) for s in S)]

        trace.append(f"step {step} ({'positive' if positive else 'negative'}): "
                     f"|S|={len(S)} |G|={len(G)}")
        # Note: We do not break early; we continue processing all examples.
        # If the version space collapses (S or G empty), it will remain empty.

    return {
        "n_train": n_pos + n_neg,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "S": S,
        "G": G,
        "n_specific": len(S),
        "n_general": len(G),
        "version_space_size": _version_space_size(S, G, domains),
        "trace": trace[:max_trace],
    }


def _version_space_size(S: list[tuple], G: list[tuple],
                        domains: dict[int, list[str]]) -> int:
    """Count hypotheses strictly between the boundaries (small spaces only)."""
    columns = [(MATCH_ANY,) + tuple(domains[i]) for i in range(len(domains))]
    space = list(product(*columns))
    if len(space) > 500_000:  # pragma: no cover - guard for wide domains
        return -1
    return sum(
        1 for h in space
        if any(covers(g, h) for g in G) and any(covers(h, s) for s in S)
    )
