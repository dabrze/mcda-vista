"""Result I/O for VISTA analyses.

Supports saving and loading VISTA results in CSV (with companion
``.meta.json``) and legacy plain-text formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["save_vista", "load_vista", "load_legacy_relations"]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_vista(
    grid: np.ndarray,
    relations: np.ndarray,
    path: str | Path,
    *,
    reference: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    method_name: str = "",
    params: dict[str, Any] | None = None,
    resolution: int = 0,
    format: str = "csv",
) -> Path:
    """Persist a VISTA result to disk.

    Parameters
    ----------
    grid : np.ndarray
        Grid coordinates, shape ``(k, n)`` where *n* is the number of criteria.
    relations : np.ndarray
        Relation values for each grid point, shape ``(k,)`` or ``(k, 1)``.
    path : str | Path
        Destination file path (without extension — it is added automatically
        for CSV format, or used as-is for legacy format).
    reference : np.ndarray | None
        Reference alternative used in the analysis.
    weights : np.ndarray | None
        Criterion weights.
    method_name : str
        Name of the MCDA method that produced the result.
    params : dict | None
        Arbitrary method parameters to store as metadata.
    resolution : int
        Grid resolution (e.g. 101).
    format : ``"csv"`` | ``"legacy_txt"``
        Output format.

    Returns
    -------
    Path
        The path to the main output file that was written.

    Raises
    ------
    ValueError
        If *format* is not recognised.
    """
    path = Path(path)

    if format == "csv":
        return _save_csv(
            grid, relations, path,
            reference=reference,
            weights=weights,
            method_name=method_name,
            params=params,
            resolution=resolution,
        )
    if format in ("legacy_txt", "txt", "legacy"):
        return _save_legacy_txt(grid, relations, path)

    raise ValueError(f"Unknown format {format!r}; use 'csv' or 'legacy_txt'")


def _save_csv(
    grid: np.ndarray,
    relations: np.ndarray,
    path: Path,
    *,
    reference: np.ndarray | None,
    weights: np.ndarray | None,
    method_name: str,
    params: dict[str, Any] | None,
    resolution: int,
) -> Path:
    """Write CSV data file and companion .meta.json."""
    relations = np.asarray(relations).ravel()
    n_criteria = grid.shape[1]

    csv_path = path.with_suffix(".csv")
    meta_path = path.with_suffix(".meta.json")

    # Build CSV: criterion columns + relation column
    header_parts = [f"x{i}" for i in range(n_criteria)] + ["relation"]
    header = ",".join(header_parts)

    combined = np.column_stack([grid, relations])
    fmt = ",".join(["%1.6f"] * n_criteria + ["%1.0f"])
    np.savetxt(csv_path, combined, header=header, fmt=fmt, comments="")

    # Build metadata
    meta: dict[str, Any] = {
        "method_name": method_name,
        "resolution": resolution,
        "n_criteria": n_criteria,
        "n_points": int(grid.shape[0]),
    }
    if reference is not None:
        meta["reference"] = reference.tolist()
    if weights is not None:
        meta["weights"] = weights.tolist()
    if params:
        meta["params"] = params

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return csv_path


def _save_legacy_txt(
    grid: np.ndarray,
    relations: np.ndarray,
    path: Path,
) -> Path:
    """Write relation values as a single row (legacy ``save2txt_Xd`` style)."""
    relations = np.asarray(relations).ravel()
    path = Path(path)

    # Legacy format: relation values only, no newlines between values
    np.savetxt(path, relations.reshape(1, -1), delimiter="", newline="", fmt="%1.0f")

    return path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_vista(path: str | Path) -> dict[str, Any]:
    """Load a VISTA result previously saved with :func:`save_vista` (CSV format).

    Parameters
    ----------
    path : str | Path
        Path to the ``.csv`` file (the companion ``.meta.json`` is located
        automatically).

    Returns
    -------
    dict
        Dictionary with keys that mirror :class:`VistaResult` fields:

        * ``grid`` — ``np.ndarray``
        * ``relations`` — ``np.ndarray``
        * ``method_name`` — ``str``
        * ``params`` — ``dict``
        * ``reference`` — ``np.ndarray | None``
        * ``weights`` — ``np.ndarray | None``
        * ``resolution`` — ``int``
    """
    path = Path(path).with_suffix(".csv")
    meta_path = path.with_suffix(".meta.json")

    # Read CSV
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    grid = data[:, :-1]
    relations = data[:, -1].astype(int)

    # Read metadata
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    result: dict[str, Any] = {
        "grid": grid,
        "relations": relations,
        "method_name": meta.get("method_name", ""),
        "params": meta.get("params", {}),
        "resolution": meta.get("resolution", 0),
        "reference": (
            np.asarray(meta["reference"]) if "reference" in meta else None
        ),
        "weights": (
            np.asarray(meta["weights"]) if "weights" in meta else None
        ),
    }
    return result


def load_legacy_relations(
    path: str | Path,
    resolution: int = 101,
) -> np.ndarray:
    """Load relation values from a legacy single-row text file.

    Parameters
    ----------
    path : str | Path
        Path to the legacy ``.txt`` file produced by the original
        ``save2txt_Xd`` with ``XXX=None``.
    resolution : int
        Grid resolution used when the file was created (used to reshape the
        flat array when possible).

    Returns
    -------
    np.ndarray
        1-D integer array of relation values.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return np.array([], dtype=int)

    # Each character is a single-digit relation code
    values = np.array([int(ch) for ch in text if ch.isdigit()], dtype=int)
    return values
