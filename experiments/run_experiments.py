#!/usr/bin/env python3
"""Run VISTA experiments from YAML configuration files.

Usage:
    python -m experiments.run_experiments configs/baseline.yaml
    python -m experiments.run_experiments configs/baseline.yaml --dry-run
    python -m experiments.run_experiments configs/baseline.yaml --plot --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product as iter_product
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mcda_vista.core import generate_vista
from mcda_vista.io import save_vista


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
# Job generation helpers
# ---------------------------------------------------------------------------


def _tag(params: dict[str, Any]) -> str:
    """Build a compact, filesystem-safe tag from parameters."""
    parts: list[str] = []
    for key, val in sorted(params.items()):
        if isinstance(val, (list, tuple)):
            val_str = "_".join(
                f"{v:g}" if isinstance(v, float) else str(v) for v in val
            )
        elif isinstance(val, float):
            val_str = f"{val:g}"
        else:
            val_str = str(val)
        parts.append(f"{key}={val_str}")
    return "__".join(parts) if parts else "default"


def _jobs_simple(
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate jobs for the *baseline* (simple) config format."""
    jobs: list[dict[str, Any]] = []
    methods: dict[str, dict] = cfg.get("methods", {})
    for method_name, method_params in methods.items():
        if method_params is None:
            method_params = {}
        jobs.append(
            {
                "method": method_name,
                "reference": cfg.get("reference"),
                "weights": cfg.get("weights"),
                "third_alternative": None,
                "method_params": dict(method_params),
                "label": method_name,
            }
        )
    return jobs


def _jobs_sweep(
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate jobs for sweep configs (weights / reference / third_alternative)."""
    sweep = cfg["sweep"]
    methods: dict[str, dict] = cfg.get("methods", {})
    jobs: list[dict[str, Any]] = []

    sweep_key = next(iter(sweep))  # weights | reference | third_alternative
    sweep_values = sweep[sweep_key]

    for sv in sweep_values:
        for method_name, method_params in methods.items():
            if method_params is None:
                method_params = {}
            job: dict[str, Any] = {
                "method": method_name,
                "reference": cfg.get("reference"),
                "weights": cfg.get("weights"),
                "third_alternative": None,
                "method_params": dict(method_params),
            }
            # Override the swept dimension
            job[sweep_key] = sv

            sv_tag = (
                "none"
                if sv is None
                else "_".join(f"{v:g}" if isinstance(v, float) else str(v) for v in sv)
            )
            job["label"] = f"{method_name}__{sweep_key}={sv_tag}"
            jobs.append(job)

    return jobs


def _jobs_method_sweeps(
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate jobs for the multi-param (method_sweeps) config format."""
    jobs: list[dict[str, Any]] = []
    method_sweeps: dict[str, dict] = cfg["method_sweeps"]

    for method_name, spec in method_sweeps.items():
        sweep_axes: dict[str, list] = spec.get("sweep", {})
        base_params: dict[str, Any] = dict(spec.get("base", {}))

        param_names = list(sweep_axes.keys())
        param_values = [sweep_axes[n] for n in param_names]

        for combo in iter_product(*param_values):
            mp = dict(base_params)
            combo_dict: dict[str, Any] = {}
            for name, val in zip(param_names, combo):
                mp[name] = val
                combo_dict[name] = val

            tag = _tag(combo_dict)
            jobs.append(
                {
                    "method": method_name,
                    "reference": cfg.get("reference"),
                    "weights": cfg.get("weights"),
                    "third_alternative": None,
                    "method_params": mp,
                    "label": f"{method_name}__{tag}",
                }
            )

    return jobs


def build_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch to the appropriate job builder based on config keys."""
    if "method_sweeps" in cfg:
        return _jobs_method_sweeps(cfg)
    if "sweep" in cfg:
        return _jobs_sweep(cfg)
    return _jobs_simple(cfg)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run_single(
    job: dict[str, Any],
    resolution: int,
    n_criteria: int,
    output_dir: Path,
    make_plots: bool,
) -> dict[str, Any]:
    """Execute a single VISTA generation job and persist its outputs."""
    t0 = time.perf_counter()
    result = generate_vista(
        method=job["method"],
        resolution=resolution,
        n_criteria=n_criteria,
        reference=job.get("reference"),
        weights=job.get("weights"),
        third_alternative=job.get("third_alternative"),
        progress=False,
        **job["method_params"],
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
        "method": job["method"],
        "resolution": resolution,
        "n_criteria": n_criteria,
        "reference": _jsonable(job.get("reference")),
        "weights": _jsonable(job.get("weights")),
        "third_alternative": _jsonable(job.get("third_alternative")),
        "method_params": _jsonable(job["method_params"]),
        "elapsed_seconds": round(elapsed, 3),
    }
    meta_path = out_path.parent / f"{out_path.name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if make_plots:
        _save_plot(result, out_path)

    return meta


def _save_plot(result: Any, out_path: Path) -> None:
    """Save a PNG plot for a single VISTA result."""
    try:
        from mcda_vista.plotting import plot_vista
    except ImportError:
        print("  [warn] plotting unavailable (matplotlib not installed)")
        return

    fig = plot_vista(result, title=result.method_name, point_size=7)
    png_path = out_path.parent / f"{out_path.name}.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


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


def run_experiment(
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    make_plots: bool = False,
    resolution_override: int | None = None,
) -> None:
    """Run all jobs described by *cfg*."""
    experiment_name = cfg.get("experiment", "unnamed")
    resolution = resolution_override or cfg.get("resolution", 101)
    n_criteria = cfg.get("n_criteria", 2)

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
            params_str = ", ".join(f"{k}={v}" for k, v in job["method_params"].items())
            if params_str:
                print(f"        params: {params_str}")
        print(f"\n  Total: {len(jobs)} job(s) — dry run, nothing generated.")
        return

    results_meta: list[dict[str, Any]] = []
    failed = 0
    for i, job in enumerate(jobs, 1):
        print(f"  [{i:3d}/{len(jobs)}] {job['label']} ... ", end="", flush=True)
        try:
            meta = _run_single(job, resolution, n_criteria, exp_dir, make_plots)
            print(f"done ({meta['elapsed_seconds']:.1f}s)")
            results_meta.append(meta)
        except Exception as exc:
            failed += 1
            print(f"FAILED ({type(exc).__name__}: {exc})")
            results_meta.append(
                {"label": job["label"], "method": job["method"], "error": str(exc)}
            )

    summary_path = exp_dir / "experiment_summary.json"
    summary = {
        "experiment": experiment_name,
        "description": cfg.get("description", ""),
        "resolution": resolution,
        "n_criteria": n_criteria,
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
        description="Run VISTA experiments from YAML configs.",
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
