#!/usr/bin/env python3
"""Export fuzzy COPRAS VISTA protocol data for R plotting.

Generates all VISTA results needed for a protocol run of fuzzy COPRAS
and writes them to ``tmp/fuzzy_copras_protocol/`` as CSV files (via
``save_vista()``) plus a ``protocol_results.json`` summary.

Usage:
    python -m experiments.export_fuzzy_copras_protocol
    python -m experiments.export_fuzzy_copras_protocol --resolution 51
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mcda_vista.core import VistaResult, generate_vista
from mcda_vista.fuzzy_utils import fuzzy_copras
from mcda_vista.io import save_vista
from mcda_vista.protocol import (
    DEFAULT_THIRD_ALTERNATIVES,
    check_diagonal_preference,
    check_dominance,
    check_preference_ratio,
    check_radial_preference,
    check_self_indifference,
    check_third_alternative_stability,
)

# ── Defaults ────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("tmp/fuzzy_copras_protocol")

BASELINE_PARAMS: dict[str, Any] = {
    "spread": 0.10,
    "skew": 0.0,
    "delta": 0.10,
}

SPREAD_VALUES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
SKEW_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
SKEW_FIXED_SPREAD = 0.30


def _safe_label(value: float) -> str:
    """Format a float for use in filenames (no dots → use 'p' as separator)."""
    sign = "m" if value < 0 else ""
    return f"{sign}{abs(value):.2f}".replace(".", "p")


# ── Helpers ─────────────────────────────────────────────────────────────


def _jsonable(obj: Any) -> Any:
    """Convert numpy types to JSON-serialisable Python types."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        return str(obj)
    return obj


def _save(result: VistaResult, name: str, out_dir: Path) -> Path:
    """Save a VistaResult to CSV and return the path."""
    return save_vista(
        grid=result.grid,
        relations=result.relations,
        path=out_dir / name,
        reference=result.reference,
        weights=result.weights,
        method_name=result.method_name,
        params=result.params,
        resolution=result.resolution,
    )


def _check_to_dict(check: Any) -> dict[str, Any]:
    """Serialise a CheckResult to a plain dict."""
    return {
        "name": check.name,
        "passed": check.passed,
        "message": check.message,
        "detail": _jsonable(check.detail),
    }


# ── Main ────────────────────────────────────────────────────────────────


def export_protocol(
    resolution: int = 101,
    output_dir: Path = OUTPUT_DIR,
    n_rays: int = 36,
) -> None:
    """Generate all VISTAs and run protocol checks, then export."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # ── 1. Baseline VISTA ───────────────────────────────────────────
    print("Generating baseline vista ...", flush=True)
    baseline = generate_vista(
        method=fuzzy_copras,
        resolution=resolution,
        progress=True,
        **BASELINE_PARAMS,
    )
    _save(baseline, "baseline", output_dir)
    print(f"  → baseline.csv ({baseline.grid.shape[0]} points)")

    # ── 2. Protocol checks 1–5 ─────────────────────────────────────
    print("Running protocol checks 1–5 ...", flush=True)
    chk_dominance = check_dominance(baseline)
    chk_self_indiff = check_self_indifference(baseline)
    chk_diagonal = check_diagonal_preference(baseline)
    chk_radial = check_radial_preference(baseline, n_rays=n_rays)
    chk_ratio = check_preference_ratio(baseline)

    checks_data = [
        _check_to_dict(chk_dominance),
        _check_to_dict(chk_self_indiff),
        _check_to_dict(chk_diagonal),
        _check_to_dict(chk_radial),
        _check_to_dict(chk_ratio),
    ]
    for cd in checks_data:
        icon = "✓" if cd["passed"] else ("✗" if cd["passed"] is False else "●")
        print(f"  {icon} {cd['name']}: {cd['message']}")

    # ── 3. Third-alternative stability (check 6) ───────────────────
    print("Generating third-alternative vistas ...", flush=True)
    third_results: list[VistaResult] = []
    third_labels: list[str] = []
    for ta in DEFAULT_THIRD_ALTERNATIVES:
        label = f"third_{_safe_label(ta[0])}_{_safe_label(ta[1])}"
        r = generate_vista(
            method=fuzzy_copras,
            resolution=resolution,
            third_alternative=ta,
            progress=True,
            **BASELINE_PARAMS,
        )
        _save(r, label, output_dir)
        third_results.append(r)
        third_labels.append(label)
        print(f"  → {label}.csv")

    chk_stability = check_third_alternative_stability(baseline, third_results)
    checks_data.append(_check_to_dict(chk_stability))
    icon = "✓" if chk_stability.passed else "✗"
    print(f"  {icon} {chk_stability.name}: {chk_stability.message}")

    # ── 4. Spread series ───────────────────────────────────────────
    print("Generating spread series ...", flush=True)
    spread_labels: list[str] = []
    for spread in SPREAD_VALUES:
        label = f"spread_{_safe_label(spread)}"
        r = generate_vista(
            method=fuzzy_copras,
            resolution=resolution,
            progress=True,
            spread=spread,
            skew=0.0,
            delta=BASELINE_PARAMS["delta"],
        )
        _save(r, label, output_dir)
        spread_labels.append(label)
        print(f"  → {label}.csv")

    # ── 5. Skew series ─────────────────────────────────────────────
    print("Generating skew series ...", flush=True)
    skew_labels: list[str] = []
    for skew in SKEW_VALUES:
        label = f"skew_{_safe_label(skew)}"
        r = generate_vista(
            method=fuzzy_copras,
            resolution=resolution,
            progress=True,
            spread=SKEW_FIXED_SPREAD,
            skew=skew,
            delta=BASELINE_PARAMS["delta"],
        )
        _save(r, label, output_dir)
        skew_labels.append(label)
        print(f"  → {label}.csv")

    # ── 6. Write protocol results JSON ─────────────────────────────
    elapsed = time.perf_counter() - t_start

    protocol_json: dict[str, Any] = {
        "method": "fuzzy_copras",
        "baseline_params": BASELINE_PARAMS,
        "resolution": resolution,
        "reference": [0.5, 0.5],
        "weights": [1.0, 1.0],
        "n_rays": n_rays,
        "checks": checks_data,
        "third_alternatives": {
            "positions": [ta for ta in DEFAULT_THIRD_ALTERNATIVES],
            "labels": third_labels,
        },
        "spread_series": {
            "values": SPREAD_VALUES,
            "fixed_skew": 0.0,
            "labels": spread_labels,
        },
        "skew_series": {
            "values": SKEW_VALUES,
            "fixed_spread": SKEW_FIXED_SPREAD,
            "labels": skew_labels,
        },
        "elapsed_seconds": round(elapsed, 2),
    }

    json_path = output_dir / "protocol_results.json"
    json_path.write_text(
        json.dumps(_jsonable(protocol_json), indent=2), encoding="utf-8"
    )
    print(f"\nProtocol results → {json_path}")
    print(f"Total time: {elapsed:.1f}s")


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fuzzy COPRAS VISTA protocol data for R plotting.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=101,
        help="Grid resolution per axis (default 101).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    export_protocol(
        resolution=args.resolution,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
