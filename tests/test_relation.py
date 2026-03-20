"""Tests for the Relation enum."""
from __future__ import annotations

from enum import IntEnum

import pytest

from mcda_vista.relation import Relation


class TestRelationValues:
    """All five members exist with the documented integer encoding."""

    def test_error_is_0(self):
        assert Relation.ERROR == 0

    def test_worse_is_1(self):
        assert Relation.WORSE == 1

    def test_indifferent_is_2(self):
        assert Relation.INDIFFERENT == 2

    def test_better_is_3(self):
        assert Relation.BETTER == 3

    def test_incomparable_is_4(self):
        assert Relation.INCOMPARABLE == 4


class TestRelationLabel:
    """The .label property returns a human-readable string."""

    @pytest.mark.parametrize(
        "member, expected",
        [
            (Relation.ERROR, "Error"),
            (Relation.WORSE, "Worse"),
            (Relation.INDIFFERENT, "Indifferent"),
            (Relation.BETTER, "Better"),
            (Relation.INCOMPARABLE, "Incomparable"),
        ],
    )
    def test_label(self, member, expected):
        assert member.label == expected


class TestRelationColor:
    """The .color property returns a hex colour string."""

    @pytest.mark.parametrize("member", list(Relation))
    def test_color_is_hex_string(self, member):
        color = member.color
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7  # e.g. #69be28


class TestRelationColorMap:
    """color_map() returns {int: hex} for all five members."""

    def test_color_map_length(self):
        cmap = Relation.color_map()
        assert len(cmap) == 5

    def test_color_map_keys(self):
        cmap = Relation.color_map()
        assert set(cmap.keys()) == {0, 1, 2, 3, 4}

    def test_color_map_values_are_hex(self):
        for v in Relation.color_map().values():
            assert isinstance(v, str) and v.startswith("#")


class TestRelationFromName:
    """from_name() parses short symbolic labels."""

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
        assert Relation.from_name(name) == expected

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown relation name"):
            Relation.from_name("garbage")

    def test_unknown_empty_raises(self):
        with pytest.raises(ValueError):
            Relation.from_name("XYZ")


class TestRelationIsIntEnum:
    """Relation is an IntEnum and can be used as a plain int."""

    def test_is_intenum(self):
        assert issubclass(Relation, IntEnum)

    def test_usable_as_int(self):
        assert int(Relation.BETTER) == 3
        assert Relation.WORSE + 1 == 2
