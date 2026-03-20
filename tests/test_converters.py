"""Tests for converter functions (symbolic, credibility, aggregate, rank)."""
from __future__ import annotations

import pytest

from mcda_vista.converters import (
    relation_from_aggregates,
    relation_from_credibilities,
    relation_from_named,
    relation_from_named_no_pminus,
    relation_from_ranks,
)
from mcda_vista.relation import Relation


# ── relation_from_named ─────────────────────────────────────────────────


class TestRelationFromNamed:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("P+", Relation.BETTER),
            ("P-", Relation.WORSE),
            ("I", Relation.INDIFFERENT),
            ("R", Relation.INCOMPARABLE),
            ("-", Relation.ERROR),
        ],
    )
    def test_known_names(self, name, expected):
        assert relation_from_named(name) == expected

    def test_unknown_returns_error(self):
        assert relation_from_named("garbage") == Relation.ERROR

    def test_empty_string_returns_error(self):
        assert relation_from_named("") == Relation.ERROR


# ── relation_from_named_no_pminus ───────────────────────────────────────


class TestRelationFromNamedNoPminus:
    def test_dash_returns_worse(self):
        """'-' normally maps to ERROR, but here it should become WORSE."""
        assert relation_from_named_no_pminus("-") == Relation.WORSE

    def test_p_plus_returns_better(self):
        assert relation_from_named_no_pminus("P+") == Relation.BETTER

    def test_p_minus_returns_worse(self):
        assert relation_from_named_no_pminus("P-") == Relation.WORSE

    def test_i_returns_indifferent(self):
        assert relation_from_named_no_pminus("I") == Relation.INDIFFERENT

    def test_r_returns_incomparable(self):
        assert relation_from_named_no_pminus("R") == Relation.INCOMPARABLE

    def test_unknown_returns_worse(self):
        """Unknown names are promoted from ERROR to WORSE."""
        assert relation_from_named_no_pminus("garbage") == Relation.WORSE


# ── relation_from_credibilities ─────────────────────────────────────────


class TestRelationFromCredibilities:
    def test_better_when_c12_above_threshold(self):
        assert relation_from_credibilities(0.8, 0.3, 0.6) == Relation.BETTER

    def test_indifferent_when_both_above_threshold(self):
        assert relation_from_credibilities(0.8, 0.8, 0.6) == Relation.INDIFFERENT

    def test_worse_when_c21_above_threshold(self):
        assert relation_from_credibilities(0.3, 0.8, 0.6) == Relation.WORSE

    def test_incomparable_when_neither_above_threshold(self):
        assert relation_from_credibilities(0.3, 0.3, 0.6) == Relation.INCOMPARABLE

    def test_boundary_equal_to_threshold(self):
        # c12 == threshold and c21 < threshold → BETTER
        assert relation_from_credibilities(0.6, 0.3, 0.6) == Relation.BETTER

    def test_both_equal_to_threshold(self):
        assert relation_from_credibilities(0.6, 0.6, 0.6) == Relation.INDIFFERENT


# ── relation_from_aggregates ────────────────────────────────────────────


class TestRelationFromAggregates:
    def test_worse_when_a1_greater_than_a2(self):
        # a1=0.7, a2=0.3 → a1 > a2 → WORSE (per the code: a1 < a2 → BETTER)
        assert relation_from_aggregates(0.7, 0.3, 0.1) == Relation.WORSE

    def test_better_when_a1_less_than_a2(self):
        assert relation_from_aggregates(0.3, 0.7, 0.1) == Relation.BETTER

    def test_indifferent_when_within_delta(self):
        assert relation_from_aggregates(0.5, 0.5, 0.1) == Relation.INDIFFERENT

    def test_indifferent_at_delta_boundary(self):
        # |0.5 - 0.4| = 0.1 == delta → INDIFFERENT
        assert relation_from_aggregates(0.5, 0.4, 0.1) == Relation.INDIFFERENT


# ── relation_from_ranks ─────────────────────────────────────────────────


class TestRelationFromRanks:
    def test_worse_when_r1_lower(self):
        """Lower rank = better alternative, so r1<r2 → WORSE (a1 outranks a2)."""
        assert relation_from_ranks(1, 3) == Relation.WORSE

    def test_better_when_r1_higher(self):
        assert relation_from_ranks(3, 1) == Relation.BETTER

    def test_indifferent_when_equal(self):
        assert relation_from_ranks(2, 2) == Relation.INDIFFERENT

    def test_float_ranks(self):
        assert relation_from_ranks(1.5, 2.5) == Relation.WORSE
