#!/usr/bin/env python3
"""Run fuzzy MCDA VISTA experiments from YAML configuration.

Usage:
    python -m experiments.run_fuzzy_experiment experiments/configs/fuzzy_spread.yaml
    python -m experiments.run_fuzzy_experiment experiments/configs/fuzzy_spread.yaml --plot
    python -m experiments.run_fuzzy_experiment experiments/configs/fuzzy_spread.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product as iter_product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from mcda_vista.core import generate_vista, VistaResult
from mcda_vista.fuzzy_utils import fuzzy_topsis, fuzzy_vikor, fuzzy_moora, fuzzy_waspas, fuzzy_edas, fuzzy_copras
from mcda_vista.io import save_vista
from mcda_vista.relation import Relation

# ---------------------------------------------------------------------------
# Callable lookup — bypasses the method registry entirely
# ---------------------------------------------------------------------------

FUZZY_METHODS: dict[str, Callable[..., Relation]] = {
    "fuzzy_topsis": fuzzy_topsis,
    "fuzzy_vikor": fuzzy_vikor,
    "fuzzy_moora": fuzzy_moora,
    "fuzzy_waspas": fuzzy_waspas,
    "fuzzy_edas": fuzzy_edas,
    "fuzzy_copras": fuzzy_copras,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment configuration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config (expected mapping): {path}")
    return cfg


# ---------------------------------------------------------------------------
# Job generation
# ---------------------------------------------------------------------------

def build_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the list of (method, spread, skew) jobs from the config.

    Iterates over ``cfg["methods"]`` and takes the Cartesian product of
    each method's ``spreads`` × ``skews`` lists.  Also appends jobs from
    the optional ``skew_experiment`` section.
    """
    jobs: list[dict[str, Any]] = []
    delta = cfg.get("delta", 0.10)

    for method_name, spec in cfg.get("methods", {}).items():
        if spec is None:
            spec = {}
        spreads = spec.get("spreads", [0.10])
        skews = spec.get("skews", [0.0])

        for spread, skew in iter_product(spreads, skews):
            label = f"{method_name}__spread={spread:g}__skew={skew:g}"
            jobs.append({
                "method_name": method_name,
                "spread": spread,
                "skew": skew,
                "delta": delta,
                "label": label,
            })

    # Optional skew sub-experiment
    skew_cfg = cfg.get("skew_experiment")
    if skew_cfg:
        method_name = skew_cfg["method"]
        spread = skew_cfg["spread"]
        for skew in skew_cfg.get("skews", []):
            label = f"{method_name}__skew_exp__spread={spread:g}__skew={skew:g}"
            jobs.append({
                "method_name": method_name,
                "spread": spread,
                "skew": skew,
                "delta": delta,
                "label": label,
            })

    return jobs


# ---------------------------------------------------------------------------
# Single-job execution
# ---------------------------------------------------------------------------

def _run_single(
    job: dict[str, Any],
    cfg: dict[str, Any],
    resolution: int,
    output_dir: Path,
    make_plots: bool,
) -> dict[str, Any]:
    """Execute a single fuzzy VISTA job and persist its outputs."""
    method_name = job["method_name"]
    method_fn = FUZZY_METHODS[method_name]

    n_criteria = cfg.get("n_criteria", 2)
    reference = cfg.get("reference")
    weights = cfg.get("weights")

    method_params: dict[str, Any] = {
        "spread": job["spread"],
        "skew": job["skew"],
        "delta": job["delta"],
    }

    t0 = time.perf_counter()
    result = generate_vista(
        method=method_fn,
        resolution=resolution,
        n_criteria=n_criteria,
        reference=reference,
        weights=weights,
        progress=True,
        **method_params,
    )
    elapsed = time.perf_counter() - t0

    label = job["label"]
    out_path = output_dir / label

    save_vista(
        grid=result.grid,
        relations=result.relations,
        path=out_path,
        reference=result.reference,
        weights=result.weights,
        method_name=result.method_name,
        params=result.params,
        resolution=result.resolution,
    )

    meta = {
        "label": label,
        "method": method_name,
        "resolution": resolution,
        "n_criteria": n_criteria,
        "reference": _jsonable(reference),
        "weights": _jsonable(weights),
        "method_params": _jsonable(method_params),
        "elapsed_seconds": round(elapsed, 3),
    }
    meta_path = out_path.parent / f"{out_path.name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if make_plots:
        _save_plot(result, out_path)

    return meta


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _save_plot(result: VistaResult, out_path: Path) -> None:
    """Save a PNG plot for a single VISTA result."""
    try:
        from mcda_vista.plotting import plot_vista
    except ImportError:
        print("  [warn] plotting unavailable (matplotlib not installed)")
        return

    fig = plot_vista(result, title=result.method_name)
    png_path = out_path.parent / f"{out_path.name}.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    make_plots: bool = False,
    resolution_override: int | None = None,
) -> None:
    """Run all fuzzy VISTA jobs described by *cfg*."""
    experiment_name = cfg.get("experiment", "unnamed")
    resolution = resolution_override or cfg.get("resolution", 101)

    jobs = build_jobs(cfg)
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {experiment_name}")
    print(f"Description: {cfg.get('description', '')}")
    print(f"Resolution : {resolution}")
    print(f"Jobs       : {len(jobs)}")
    print(f"Output     : {exp_dir}")
    print()

    if dry_run:
        for i, job in enumerate(jobs, 1):
            print(f"  [{i:3d}] {job['label']}")
            print(f"        spread={job['spread']:g}  skew={job['skew']:g}  delta={job['delta']:g}")
        print(f"\n  Total: {len(jobs)} job(s) — dry run, nothing generated.")
        return

    results_meta: list[dict[str, Any]] = []
    failed = 0

    for i, job in enumerate(jobs, 1):
        print(f"  [{i:3d}/{len(jobs)}] {job['label']} ... ", end="", flush=True)
        try:
            meta = _run_single(job, cfg, resolution, exp_dir, make_plots)
            print(f"done ({meta['elapsed_seconds']:.1f}s)")
            results_meta.append(meta)
        except Exception as exc:
            failed += 1
            print(f"FAILED ({type(exc).__name__}: {exc})")
            results_meta.append({
                "label": job["label"],
                "method": job["method_name"],
                "error": str(exc),
            })

    summary_path = exp_dir / "experiment_summary.json"
    summary = {
        "experiment": experiment_name,
        "description": cfg.get("description", ""),
        "resolution": resolution,
        "n_criteria": cfg.get("n_criteria", 2),
        "delta": cfg.get("delta", 0.10),
        "total_jobs": len(jobs),
        "failed_jobs": failed,
        "jobs": results_meta,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    status = "Done" if failed == 0 else f"Done with {failed} failure(s)"
    print(f"\n{status}. Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fuzzy MCDA VISTA experiments from YAML configs.",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to a YAML experiment configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/output",
        help="Root directory for experiment outputs (default: experiments/output/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without running them.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PNG plots for each VISTA result.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override the grid resolution from the config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)

    run_experiment(
        cfg,
        output_dir,
        dry_run=args.dry_run,
        make_plots=args.plot,
        resolution_override=args.resolution,
    )


if __name__ == "__main__":
    main()
