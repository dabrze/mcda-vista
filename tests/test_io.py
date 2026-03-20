"""Tests for save/load round-trip (CSV and legacy formats)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from mcda_vista.io import load_legacy_relations, load_vista, save_vista


# ── CSV round-trip ──────────────────────────────────────────────────────


class TestSaveLoadCSV:
    def test_round_trip_grid_and_relations(self, tmp_path):
        grid = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        relations = np.array([1, 2, 3])

        save_vista(grid, relations, tmp_path / "test", format="csv")
        loaded = load_vista(tmp_path / "test")

        np.testing.assert_array_almost_equal(loaded["grid"], grid)
        np.testing.assert_array_equal(loaded["relations"], relations)

    def test_metadata_method_name(self, tmp_path):
        grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        relations = np.array([1, 3])

        save_vista(
            grid, relations, tmp_path / "meta",
            method_name="topsis",
            format="csv",
        )
        loaded = load_vista(tmp_path / "meta")
        assert loaded["method_name"] == "topsis"

    def test_metadata_reference(self, tmp_path):
        grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        relations = np.array([1, 3])
        ref = np.array([0.5, 0.5])

        save_vista(
            grid, relations, tmp_path / "ref",
            reference=ref,
            format="csv",
        )
        loaded = load_vista(tmp_path / "ref")
        assert loaded["reference"] is not None
        np.testing.assert_array_almost_equal(loaded["reference"], ref)

    def test_metadata_weights(self, tmp_path):
        grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        relations = np.array([1, 3])
        weights = np.array([2.0, 3.0])

        save_vista(
            grid, relations, tmp_path / "wt",
            weights=weights,
            format="csv",
        )
        loaded = load_vista(tmp_path / "wt")
        assert loaded["weights"] is not None
        np.testing.assert_array_almost_equal(loaded["weights"], weights)

    def test_metadata_params(self, tmp_path):
        grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        relations = np.array([1, 3])

        save_vista(
            grid, relations, tmp_path / "params",
            params={"threshold": 0.6},
            format="csv",
        )
        loaded = load_vista(tmp_path / "params")
        assert loaded["params"]["threshold"] == 0.6

    def test_csv_file_created(self, tmp_path):
        grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        relations = np.array([1, 3])

        csv_path = save_vista(grid, relations, tmp_path / "out", format="csv")
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

    def test_meta_json_created(self, tmp_path):
        grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        relations = np.array([1, 3])

        save_vista(grid, relations, tmp_path / "out", format="csv")
        meta_path = tmp_path / "out.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "n_criteria" in meta
        assert meta["n_criteria"] == 2

    def test_resolution_round_trips(self, tmp_path):
        grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        relations = np.array([1, 3])

        save_vista(
            grid, relations, tmp_path / "res",
            resolution=101,
            format="csv",
        )
        loaded = load_vista(tmp_path / "res")
        assert loaded["resolution"] == 101


# ── Legacy format ───────────────────────────────────────────────────────


class TestLegacyFormat:
    def test_save_and_load_legacy(self, tmp_path):
        grid = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        relations = np.array([1, 2, 3])

        path = save_vista(grid, relations, tmp_path / "legacy.txt", format="legacy_txt")
        assert path.exists()

        loaded = load_legacy_relations(path)
        np.testing.assert_array_equal(loaded, relations)

    def test_legacy_single_digit_encoding(self, tmp_path):
        relations = np.array([0, 1, 2, 3, 4])
        grid = np.zeros((5, 2))

        path = save_vista(grid, relations, tmp_path / "digits.txt", format="legacy_txt")
        loaded = load_legacy_relations(path)
        np.testing.assert_array_equal(loaded, relations)

    def test_legacy_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        loaded = load_legacy_relations(empty_file)
        assert len(loaded) == 0


# ── Error handling ──────────────────────────────────────────────────────


class TestIOErrors:
    def test_unknown_format_raises(self, tmp_path):
        grid = np.array([[0.0, 0.0]])
        relations = np.array([1])

        with pytest.raises(ValueError, match="Unknown format"):
            save_vista(grid, relations, tmp_path / "bad", format="parquet")

    def test_load_missing_meta_still_works(self, tmp_path):
        """Loading a CSV without a .meta.json should not crash."""
        grid = np.array([[0.0, 0.0], [1.0, 1.0]])
        relations = np.array([1, 3])

        save_vista(grid, relations, tmp_path / "nometa", format="csv")
        # Delete the meta file
        meta_path = tmp_path / "nometa.meta.json"
        meta_path.unlink()

        loaded = load_vista(tmp_path / "nometa")
        assert loaded["method_name"] == ""
        assert loaded["reference"] is None
