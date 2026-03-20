"""Core relation type for the VISTA library.

Encodes pairwise comparison outcomes used throughout MCDA methods
(ELECTRE, PROMETHEE, etc.). Numeric values follow the original encoding:
0 = error/trouble, 1 = P− (worse), 2 = I (indifferent),
3 = P+ (better), 4 = R (incomparable).
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["Relation"]


class Relation(IntEnum):
    """Pairwise comparison outcome between two alternatives."""

    ERROR = 0
    WORSE = 1
    INDIFFERENT = 2
    BETTER = 3
    INCOMPARABLE = 4

    # -- readable helpers -------------------------------------------------

    @property
    def label(self) -> str:
        """Human-readable name for the relation."""
        return _LABELS[self]

    @property
    def color(self) -> str:
        """Hex colour used when plotting this relation."""
        return _COLORS[self]

    # -- class-level utilities --------------------------------------------

    @classmethod
    def color_map(cls) -> dict[int, str]:
        """Return ``{value: color}`` mapping for every relation."""
        return {r.value: r.color for r in cls}

    @classmethod
    def from_name(cls, name: str) -> Relation:
        """Parse a short symbolic name into a :class:`Relation`.

        Accepted inputs (case-insensitive): ``"P+"``, ``"P-"``, ``"I"``,
        ``"R"``, ``"-"`` (mapped to :attr:`ERROR`, matching the original
        VISTA convention).

        Raises
        ------
        ValueError
            If *name* is not recognised.
        """
        try:
            return _NAME_MAP[name.strip().upper()]
        except KeyError:
            raise ValueError(
                f"Unknown relation name {name!r}. "
                f"Expected one of: {', '.join(_NAME_MAP)}"
            ) from None


# ── private lookup tables ────────────────────────────────────────────────

_LABELS: dict[Relation, str] = {
    Relation.BETTER: "Better",
    Relation.INDIFFERENT: "Indifferent",
    Relation.WORSE: "Worse",
    Relation.INCOMPARABLE: "Incomparable",
    Relation.ERROR: "Error",
}

_COLORS: dict[Relation, str] = {
    Relation.BETTER: "#69be28",
    Relation.INDIFFERENT: "#f0ab00",
    Relation.WORSE: "#cd202c",
    Relation.INCOMPARABLE: "#2a6ebb",
    Relation.ERROR: "#888888",
}

_NAME_MAP: dict[str, Relation] = {
    "P+": Relation.BETTER,
    "P-": Relation.WORSE,
    "I": Relation.INDIFFERENT,
    "R": Relation.INCOMPARABLE,
    "-": Relation.ERROR,
}
