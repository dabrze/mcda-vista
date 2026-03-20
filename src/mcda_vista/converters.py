"""Converters from various numeric or symbolic representations to :class:`Relation`.

Ported from ``conversion_handling.py`` — same mathematical logic, but
functions return :class:`Relation` enum values, carry type-hints, and use
clean names.
"""

from __future__ import annotations

from mcda_vista.relation import Relation

__all__ = [
    "ROUND_DIGITS",
    "relation_from_named",
    "relation_from_named_no_pminus",
    "relation_from_credibilities",
    "relation_from_aggregates",
    "relation_from_ranks",
]

ROUND_DIGITS: int = 9
"""Number of decimal digits used when rounding before comparisons."""

# Lookup order mirrors the original integer encoding:
# index 0 → "-" (ERROR), 1 → "P-" (WORSE), 2 → "I" (INDIFFERENT),
# 3 → "P+" (BETTER), 4 → "R" (INCOMPARABLE).
_NAMED_RELATIONS: list[str] = ["-", "P-", "I", "P+", "R"]


def relation_from_named(name: str) -> Relation:
    """Convert a symbolic relation name to a :class:`Relation`.

    Parameters
    ----------
    name : str
        One of ``"-"``, ``"P-"``, ``"I"``, ``"P+"``, ``"R"``.

    Returns
    -------
    Relation
        The corresponding enum member, or :attr:`Relation.ERROR` when
        *name* is not recognised.

    Examples
    --------
    >>> relation_from_named("P+")
    <Relation.BETTER: 3>
    >>> relation_from_named("unknown")
    <Relation.ERROR: 0>
    """
    try:
        return Relation(_NAMED_RELATIONS.index(name))
    except ValueError:
        return Relation.ERROR


def relation_from_named_no_pminus(name: str) -> Relation:
    """Convert a symbolic name, treating ``"-"`` as :attr:`Relation.WORSE`.

    Behaves like :func:`relation_from_named` except that an unrecognised
    name (which would normally map to :attr:`Relation.ERROR`) is promoted
    to :attr:`Relation.WORSE`.

    Parameters
    ----------
    name : str
        One of ``"-"``, ``"P-"``, ``"I"``, ``"P+"``, ``"R"``.

    Returns
    -------
    Relation
        The corresponding enum member. :attr:`Relation.ERROR` is never
        returned; it is replaced by :attr:`Relation.WORSE`.

    Examples
    --------
    >>> relation_from_named_no_pminus("I")
    <Relation.INDIFFERENT: 2>
    >>> relation_from_named_no_pminus("unknown")
    <Relation.WORSE: 1>
    """
    rel = relation_from_named(name)
    if rel is Relation.ERROR:
        return Relation.WORSE
    return rel


def relation_from_credibilities(
    c12: float,
    c21: float,
    threshold: float,
) -> Relation:
    """Derive a relation from outranking credibility indices.

    Both credibility values and the *threshold* (λ) are rounded to
    :data:`ROUND_DIGITS` decimals before the comparison to avoid
    floating-point artefacts.

    Parameters
    ----------
    c12 : float
        Credibility that alternative 1 outranks alternative 2.
    c21 : float
        Credibility that alternative 2 outranks alternative 1.
    threshold : float
        Cutting level (λ).

    Returns
    -------
    Relation
        * :attr:`~Relation.INDIFFERENT` — both credibilities ≥ λ.
        * :attr:`~Relation.BETTER` — only *c12* ≥ λ.
        * :attr:`~Relation.WORSE` — only *c21* ≥ λ.
        * :attr:`~Relation.INCOMPARABLE` — neither credibility ≥ λ.

    Examples
    --------
    >>> relation_from_credibilities(0.8, 0.3, 0.6)
    <Relation.BETTER: 3>
    """
    c12 = round(c12, ROUND_DIGITS)
    c21 = round(c21, ROUND_DIGITS)
    threshold = round(threshold, ROUND_DIGITS)

    if c12 >= threshold:
        if c21 >= threshold:
            return Relation.INDIFFERENT
        return Relation.BETTER
    if c21 >= threshold:
        return Relation.WORSE
    return Relation.INCOMPARABLE


def relation_from_aggregates(
    a1: float,
    a2: float,
    delta: float,
) -> Relation:
    """Derive a relation by comparing two aggregate scores.

    Values are rounded to :data:`ROUND_DIGITS` decimals before comparison.

    Parameters
    ----------
    a1 : float
        Aggregate score for alternative 1.
    a2 : float
        Aggregate score for alternative 2.
    delta : float
        Indifference threshold — when ``|a1 − a2| ≤ delta`` the
        alternatives are considered indifferent.

    Returns
    -------
    Relation
        * :attr:`~Relation.INDIFFERENT` — ``|a1 − a2| ≤ delta``.
        * :attr:`~Relation.BETTER` — ``a1 − a2 > delta``.
        * :attr:`~Relation.WORSE` — ``a2 − a1 > delta``.

    Examples
    --------
    >>> relation_from_aggregates(0.7, 0.3, 0.1)
    <Relation.WORSE: 1>
    """
    a1 = round(a1, ROUND_DIGITS)
    a2 = round(a2, ROUND_DIGITS)
    delta = round(delta, ROUND_DIGITS)

    if abs(a1 - a2) <= delta:
        return Relation.INDIFFERENT
    if a1 < a2:
        return Relation.BETTER
    return Relation.WORSE


def relation_from_ranks(r1: float, r2: float) -> Relation:
    """Derive a relation from numeric ranks.

    A *lower* rank means a *better* (preferred) alternative.

    Parameters
    ----------
    r1 : float
        Rank of alternative 1.
    r2 : float
        Rank of alternative 2.

    Returns
    -------
    Relation
        * :attr:`~Relation.WORSE` — ``r1 < r2`` (better rank).
        * :attr:`~Relation.BETTER` — ``r1 > r2`` (worse rank).
        * :attr:`~Relation.INDIFFERENT` — ``r1 == r2``.

    Examples
    --------
    >>> relation_from_ranks(1, 3)
    <Relation.WORSE: 1>
    >>> relation_from_ranks(3, 3)
    <Relation.INDIFFERENT: 2>
    """
    if r1 != r2:
        if r1 < r2:
            return Relation.WORSE
        return Relation.BETTER
    return Relation.INDIFFERENT
