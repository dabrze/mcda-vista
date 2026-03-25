"""VISTA protocol for evaluating MCDA ranking methods.

Implements the six-check protocol for assessing whether a ranking method
meets the decision maker's expectations.  Each check can be run
individually or as a full protocol via :func:`run_protocol`.

Quick start::

    from mcda_vista.protocol import run_protocol, plot_protocol_report

    report = run_protocol("topsis", resolution=101, delta=0.10)
    print(report.summary())
    fig = plot_protocol_report(report)
    fig.savefig("protocol_report.png", dpi=300)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mcda_vista.core import VistaResult, generate_vista
from mcda_vista.plotting import _draw_vista_on_ax, _shared_legend_handles
from mcda_vista.relation import Relation

__all__ = [
    "CheckResult",
    "ProtocolReport",
    "check_dominance",
    "check_self_indifference",
    "check_diagonal_preference",
    "check_radial_preference",
    "check_preference_ratio",
    "check_third_alternative_stability",
    "run_protocol",
    "plot_protocol_report",
    "DEFAULT_THIRD_ALTERNATIVES",
]


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Result of a single protocol check.

    Attributes
    ----------
    name : str
        Human-readable check name (e.g. ``"Dominance relation"``).
    passed : bool | None
        *True* if the check passed, *False* if it failed, *None* if
        the check was not applicable (e.g. wrong dimensionality).
    message : str
        One-line summary of the outcome.
    detail : dict
        Check-specific metrics (violation counts, ratios, etc.).
    vista_results : list[VistaResult]
        VISTA results used by or produced for this check.
    """

    name: str
    passed: bool | None
    message: str
    detail: dict = field(default_factory=dict)
    vista_results: list[VistaResult] = field(default_factory=list)


@dataclass
class ProtocolReport:
    """Complete protocol evaluation report for a method.

    Attributes
    ----------
    method_name : str
        Name of the evaluated method.
    params : dict
        Method-specific parameters used.
    checks : list[CheckResult]
        Results of the six protocol checks.
    baseline : VistaResult
        Baseline VISTA result (equal weights, midpoint reference).
    weight_sensitivity : list
        Optional weight-sensitivity results.  Each element is a
        ``(weights, vista_result, checks_1_to_5)`` tuple.
    metadata : dict
        Auxiliary data (e.g. ``elapsed_seconds``).
    """

    method_name: str
    params: dict
    checks: list[CheckResult]
    baseline: VistaResult
    weight_sensitivity: list[tuple[list[float], VistaResult, list[CheckResult]]] = field(
        default_factory=list
    )
    metadata: dict = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        """*True* if every check with a definitive result passed."""
        return all(c.passed for c in self.checks if c.passed is not None)

    def summary(self) -> str:
        """Human-readable summary string."""
        header = f"VISTA Protocol Report: {self.method_name}"
        lines = [header, "=" * len(header)]
        for c in self.checks:
            if c.passed is True:
                status = "✓ PASS"
            elif c.passed is False:
                is_num = c.detail.get("is_numerical_error", False)
                status = "✗? FAIL" if is_num else "✗ FAIL"
            else:
                # Informational check (e.g. preference ratio)
                balanced = c.detail.get("balanced")
                if balanced is True:
                    status = "● BALANCED"
                elif balanced is False:
                    status = "● IMBALANCED"
                else:
                    status = "? N/A"
            lines.append(f"  {status}  {c.name}: {c.message}")
        if self.weight_sensitivity:
            lines.append("")
            lines.append("Weight sensitivity:")
            for weights, _, checks in self.weight_sensitivity:
                w_str = ", ".join(f"{w:.2f}" for w in weights)
                passed = sum(1 for c in checks if c.passed)
                total = len(checks)
                lines.append(f"  weights=[{w_str}]: {passed}/{total} checks passed")
        return "\n".join(lines)


# ── Internal helpers ────────────────────────────────────────────────────


def _to_grid_2d(result: VistaResult) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a 2-criteria result into *(ticks, rel_grid)*.

    ``rel_grid[i, j]`` is the relation at ``(ticks[i], ticks[j])``.
    """
    res = result.resolution
    ticks = np.linspace(0.0, 1.0, res)
    rel_grid = result.relations.reshape(res, res)
    return ticks, rel_grid


def _nearest_tick_index(value: float, ticks: np.ndarray) -> int:
    return int(np.argmin(np.abs(ticks - value)))


def _count_transitions(
    sequence: Sequence[int] | np.ndarray[Any, Any],
    min_run: int = 1,
) -> int:
    """Count value changes in *sequence*.

    When *min_run* > 1, short runs (fewer than *min_run* consecutive
    identical values) are ignored, filtering grid-discretization noise
    at oblique boundary crossings. Accepts both Python sequences and
    NumPy slices derived from VISTA grids.
    """
    values = np.asarray(sequence, dtype=np.int64)
    if values.size < 2:
        return 0
    if min_run <= 1:
        return int(np.sum(values[1:] != values[:-1]))

    # Run-length encode
    runs: list[tuple[int, int]] = []
    current = int(values[0])
    length = 1
    for raw_value in values[1:]:
        v = int(raw_value)
        if v == current:
            length += 1
        else:
            runs.append((current, length))
            current = v
            length = 1
    runs.append((current, length))

    stable_values: list[int] = []
    for value, run_length in runs:
        if run_length < min_run:
            continue
        if stable_values and stable_values[-1] == value:
            continue
        stable_values.append(value)

    return max(0, len(stable_values) - 1)


def _ray_cells(
    ref_i: int,
    ref_j: int,
    angle: float,
    resolution: int,
) -> list[tuple[int, int]]:
    """Return distinct grid cells along a ray from *(ref_i, ref_j)*."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    step = 0.4
    cells: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None

    t = step
    max_diag = resolution * np.sqrt(2)
    while t <= max_diag:
        fi = ref_i + t * cos_a
        fj = ref_j + t * sin_a
        i = int(round(fi))
        j = int(round(fj))
        if i < 0 or i >= resolution or j < 0 or j >= resolution:
            break
        cell = (i, j)
        if cell != prev:
            cells.append(cell)
            prev = cell
        t += step

    return cells


# ── Individual checks ───────────────────────────────────────────────────


def check_dominance(result: VistaResult) -> CheckResult:
    """**Check 1 — Dominance relation.**

    The dominating cone (upper-right of reference) must contain no WORSE
    points and the dominated cone (lower-left) no BETTER points.
    """
    if result.n_criteria != 2:
        return CheckResult(
            name="Dominance relation",
            passed=None,
            message="Only supported for 2-criteria VISTA.",
        )

    ref = result.reference
    grid = result.grid
    rels = result.relations

    dominating = np.all(grid > ref, axis=1)
    dominated = np.all(grid < ref, axis=1)

    worse_in_dom = int(np.sum((rels == Relation.WORSE) & dominating))
    better_in_dom = int(np.sum((rels == Relation.BETTER) & dominated))
    n_dominating = int(np.sum(dominating))
    n_dominated = int(np.sum(dominated))

    violations = worse_in_dom + better_in_dom
    passed = violations == 0

    if passed:
        msg = (
            f"No violations ({n_dominating} dominating, "
            f"{n_dominated} dominated points)."
        )
    else:
        parts = []
        if worse_in_dom:
            parts.append(f"{worse_in_dom} Worse in dominating cone")
        if better_in_dom:
            parts.append(f"{better_in_dom} Better in dominated cone")
        msg = "; ".join(parts) + "."

    return CheckResult(
        name="Dominance relation",
        passed=passed,
        message=msg,
        detail={
            "worse_in_dominating": worse_in_dom,
            "better_in_dominated": better_in_dom,
            "dominating_total": n_dominating,
            "dominated_total": n_dominated,
        },
        vista_results=[result],
    )


def check_self_indifference(result: VistaResult) -> CheckResult:
    """**Check 2 — Self-indifference.**

    The grid point at the reference position must be INDIFFERENT.
    """
    ref = result.reference
    grid = result.grid

    distances = np.linalg.norm(grid - ref, axis=1)
    nearest_idx = int(np.argmin(distances))
    nearest_rel = Relation(int(result.relations[nearest_idx]))
    nearest_point = grid[nearest_idx]
    dist = float(distances[nearest_idx])

    passed = nearest_rel == Relation.INDIFFERENT
    is_numerical_error = nearest_rel == Relation.ERROR

    if passed:
        msg = (
            f"Reference point is Indifferent "
            f"(nearest grid point at distance {dist:.4f})."
        )
    elif is_numerical_error:
        msg = (
            f"Reference point is Error — likely a numerical issue "
            f"(nearest grid point at distance {dist:.4f})."
        )
    else:
        msg = (
            f"Reference point is {nearest_rel.label} "
            f"(expected Indifferent, nearest at distance {dist:.4f})."
        )

    return CheckResult(
        name="Self-indifference",
        passed=passed,
        message=msg,
        detail={
            "nearest_relation": nearest_rel.name,
            "nearest_point": nearest_point.tolist(),
            "distance": dist,
            "is_numerical_error": is_numerical_error,
        },
        vista_results=[result],
    )


def check_diagonal_preference(result: VistaResult) -> CheckResult:
    """**Check 3 — Diagonal preference change.**

    Along the main diagonal from the reference, the relation should
    change at most once in each direction (upper-right, lower-left).
    """
    if result.n_criteria != 2:
        return CheckResult(
            name="Diagonal preference change",
            passed=None,
            message="Only supported for 2-criteria VISTA.",
        )

    ticks, rel_grid = _to_grid_2d(result)
    res = result.resolution
    ref = result.reference

    diag_rels = np.array([int(rel_grid[k, k]) for k in range(res)])
    ref_k = _nearest_tick_index(ref[0], ticks)

    upper = diag_rels[ref_k:]
    upper_trans = _count_transitions(upper)

    lower = diag_rels[: ref_k + 1][::-1]
    lower_trans = _count_transitions(lower)

    passed = upper_trans <= 1 and lower_trans <= 1

    msg = (
        f"Upper-right: {upper_trans} transition(s), "
        f"lower-left: {lower_trans} transition(s)."
    )
    if not passed:
        msg += " Expected at most 1 in each direction."

    return CheckResult(
        name="Diagonal preference change",
        passed=passed,
        message=msg,
        detail={
            "upper_transitions": upper_trans,
            "lower_transitions": lower_trans,
            "diagonal_relations": diag_rels.tolist(),
            "ref_index": ref_k,
        },
        vista_results=[result],
    )


def check_radial_preference(
    result: VistaResult,
    n_rays: int = 36,
    min_run: int = 3,
) -> CheckResult:
    """**Check 4 — Radial preference change.**

    Along rays from the reference outward, the relation should change
    at most once per ray.  Short runs (< *min_run* cells) are ignored
    to filter grid-discretization noise at oblique boundaries.
    """
    if result.n_criteria != 2:
        return CheckResult(
            name="Radial preference change",
            passed=None,
            message="Only supported for 2-criteria VISTA.",
        )

    ticks, rel_grid = _to_grid_2d(result)
    res = result.resolution
    ref = result.reference

    ref_i = _nearest_tick_index(ref[0], ticks)
    ref_j = _nearest_tick_index(ref[1], ticks)

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ray_transitions: list[int] = []
    violating_angles: list[float] = []
    worst = 0

    for angle in angles:
        cells = _ray_cells(ref_i, ref_j, angle, res)
        if len(cells) < 2:
            ray_transitions.append(0)
            continue

        rels = [int(rel_grid[i, j]) for i, j in cells]
        n_trans = _count_transitions(rels, min_run=min_run)
        ray_transitions.append(n_trans)

        if n_trans > 1:
            violating_angles.append(float(np.degrees(angle)))
        worst = max(worst, n_trans)

    passed = worst <= 1

    if passed:
        msg = f"All {n_rays} rays have at most 1 transition."
    else:
        msg = (
            f"{len(violating_angles)}/{n_rays} rays have >1 transition "
            f"(worst: {worst})."
        )

    return CheckResult(
        name="Radial preference change",
        passed=passed,
        message=msg,
        detail={
            "n_rays": n_rays,
            "ray_transitions": ray_transitions,
            "violating_angles_deg": violating_angles,
            "worst_transitions": worst,
        },
        vista_results=[result],
    )


def check_preference_ratio(
    result: VistaResult,
    ratio_bounds: tuple[float, float] = (0.8, 1.2),
) -> CheckResult:
    """**Check 5 — Preference ratio.**

    Report the proportion of BETTER to WORSE points.  This is an
    informational check: *passed* is *None* (neither pass nor fail).
    The ratio is classified as **Balanced** or **Imbalanced**.
    """
    rels = result.relations
    total = len(rels)

    counts: dict[str, int] = {}
    for rel in Relation:
        counts[rel.name] = int(np.sum(rels == rel.value))

    n_better = counts["BETTER"]
    n_worse = counts["WORSE"]

    if n_worse > 0:
        ratio = n_better / n_worse
    elif n_better > 0:
        ratio = float("inf")
    else:
        ratio = 1.0

    lo, hi = ratio_bounds
    balanced = lo <= ratio <= hi

    pct_better = n_better / total * 100
    pct_worse = n_worse / total * 100

    if balanced:
        msg = f"Balanced ({pct_worse:.1f}% worse, {pct_better:.1f}% better than reference)."
    else:
        msg = f"Imbalanced ({pct_worse:.1f}% worse, {pct_better:.1f}% better than reference)."

    return CheckResult(
        name="Preference ratio",
        passed=None,
        message=msg,
        detail={
            "ratio": ratio,
            "balanced": balanced,
            "counts": counts,
            "percentages": {k: v / total * 100 for k, v in counts.items()},
            "ratio_bounds": list(ratio_bounds),
        },
        vista_results=[result],
    )


def check_third_alternative_stability(
    baseline: VistaResult,
    results_with_third: list[VistaResult],
    threshold: float = 0.05,
) -> CheckResult:
    """**Check 6 — Third alternative stability.**

    Compare VISTA results with added third alternatives against the
    baseline.  Methods with large changes are considered unstable.
    """
    total = len(baseline.relations)
    changes: dict[str, float] = {}
    max_change = 0.0

    for result in results_with_third:
        changed = int(np.sum(baseline.relations != result.relations))
        pct = changed / total
        label = (
            str(result.third_alternative.tolist())
            if result.third_alternative is not None
            else "?"
        )
        changes[label] = pct
        max_change = max(max_change, pct)

    passed = max_change <= threshold

    parts = [f"{k}: {v:.1%}" for k, v in changes.items()]
    changes_str = ", ".join(parts)

    if passed:
        msg = f"Max change {max_change:.1%} ≤ {threshold:.0%}. Changes: {changes_str}."
    else:
        msg = f"Max change {max_change:.1%} > {threshold:.0%}. Changes: {changes_str}."

    return CheckResult(
        name="Third alternative stability",
        passed=passed,
        message=msg,
        detail={
            "changes": changes,
            "max_change": max_change,
            "threshold": threshold,
        },
        vista_results=[baseline] + results_with_third,
    )


# ── Protocol runner ─────────────────────────────────────────────────────


DEFAULT_THIRD_ALTERNATIVES: list[list[float]] = [
    [0.50, 0.50],
    [0.25, 0.50],
    [0.50, 0.75],
    [0.25, 0.25],
    [0.25, 0.75],
]


def run_protocol(
    method: Callable[..., Relation] | str,
    resolution: int = 101,
    third_alternatives: list[list[float]] | None = None,
    extra_weights: list[list[float]] | None = None,
    stability_threshold: float = 0.05,
    ratio_bounds: tuple[float, float] = (0.8, 1.2),
    n_rays: int = 36,
    progress: bool = True,
    **method_params: Any,
) -> ProtocolReport:
    """Run the full six-check VISTA protocol on a method.

    The protocol evaluates the method using equal criteria weights and
    the midpoint ``[0.5, …]`` as reference, as specified in the VISTA
    protocol definition.

    Parameters
    ----------
    method : Callable | str
        MCDA method callable or registered name string.
    resolution : int
        Grid resolution per axis (default 101).
    third_alternatives : list or None
        Third-alternative positions for check 6.  Uses
        :data:`DEFAULT_THIRD_ALTERNATIVES` when *None*.
    extra_weights : list or None
        Optional additional weight vectors.  For each, checks 1–5 are
        re-run and stored in the report's ``weight_sensitivity`` field.
    stability_threshold : float
        Maximum allowed change fraction for check 6 (default 5 %).
    ratio_bounds : tuple
        Acceptable Better/Worse ratio range for check 5.
    n_rays : int
        Number of radial rays for check 4.
    progress : bool
        Show ``tqdm`` progress bars during VISTA generation.
    **method_params
        Forwarded to the MCDA method.

    Returns
    -------
    ProtocolReport
    """
    t_start = time.perf_counter()

    if third_alternatives is None:
        third_alternatives = DEFAULT_THIRD_ALTERNATIVES

    # ── baseline VISTA (equal weights, midpoint reference) ───────────
    baseline = generate_vista(
        method,
        resolution=resolution,
        progress=progress,
        **method_params,
    )

    # ── checks 1–5 on baseline ──────────────────────────────────────
    checks: list[CheckResult] = [
        check_dominance(baseline),
        check_self_indifference(baseline),
        check_diagonal_preference(baseline),
        check_radial_preference(baseline, n_rays=n_rays),
        check_preference_ratio(baseline, ratio_bounds=ratio_bounds),
    ]

    # ── check 6: third alternative stability ────────────────────────
    third_results: list[VistaResult] = []
    for ta in third_alternatives:
        r = generate_vista(
            method,
            resolution=resolution,
            third_alternative=ta,
            progress=progress,
            **method_params,
        )
        third_results.append(r)

    checks.append(
        check_third_alternative_stability(
            baseline, third_results, threshold=stability_threshold
        )
    )

    # ── optional weight sensitivity ─────────────────────────────────
    weight_sensitivity: list[tuple[list[float], VistaResult, list[CheckResult]]] = []
    if extra_weights is not None:
        for w in extra_weights:
            w_result = generate_vista(
                method,
                resolution=resolution,
                weights=w,
                progress=progress,
                **method_params,
            )
            w_checks = [
                check_dominance(w_result),
                check_self_indifference(w_result),
                check_diagonal_preference(w_result),
                check_radial_preference(w_result, n_rays=n_rays),
                check_preference_ratio(w_result, ratio_bounds=ratio_bounds),
            ]
            weight_sensitivity.append((w, w_result, w_checks))

    elapsed = time.perf_counter() - t_start

    return ProtocolReport(
        method_name=baseline.method_name,
        params=method_params,
        checks=checks,
        baseline=baseline,
        weight_sensitivity=weight_sensitivity,
        metadata={"elapsed_seconds": elapsed},
    )


# ── Plotting ────────────────────────────────────────────────────────────

_CHECK_COLORS = {True: "#69be28", False: "#cd202c", None: "#555555"}


def _check_icon(check: CheckResult) -> str:
    if check.passed is True:
        return "✓"
    if check.passed is False:
        if check.detail.get("is_numerical_error"):
            return "✗?"
        return "✗"
    # Informational (passed is None)
    balanced = check.detail.get("balanced")
    if balanced is True:
        return "●"
    elif balanced is False:
        return "●"
    return "?"


def _check_color(check: CheckResult) -> str:
    if check.passed is True:
        return "#69be28"
    if check.passed is False:
        if check.detail.get("is_numerical_error"):
            return "#e67e22"  # orange for numerical errors
        return "#cd202c"
    # Informational
    balanced = check.detail.get("balanced")
    if balanced is True:
        return "#69be28"
    elif balanced is False:
        return "#e67e22"
    return "#888888"


def _annotate_dominance(ax: Axes, check: CheckResult, result: VistaResult) -> None:
    ref = result.reference
    ax.add_patch(
        mpatches.Rectangle(
            (ref[0], ref[1]),
            1.0 - ref[0],
            1.0 - ref[1],
            linewidth=0,
            facecolor="#69be28",
            alpha=0.08,
            zorder=0,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (0, 0),
            ref[0],
            ref[1],
            linewidth=0,
            facecolor="#cd202c",
            alpha=0.08,
            zorder=0,
        )
    )


def _annotate_diagonal(ax: Axes, check: CheckResult, result: VistaResult) -> None:
    ax.plot(
        [0, 1], [0, 1],
        color="#555555", linewidth=0.8, linestyle="-.", alpha=0.7, zorder=4,
    )


def _annotate_radial(ax: Axes, check: CheckResult, result: VistaResult) -> None:
    ref = result.reference
    n_rays = check.detail.get("n_rays", 36)
    violating = set(check.detail.get("violating_angles_deg", []))
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    for angle in angles:
        deg = float(np.degrees(angle))
        is_violating = any(abs(deg - v) < 1.0 for v in violating)
        color = "#cd202c" if is_violating else "#555555"
        alpha = 0.5 if is_violating else 0.15
        lw = 0.8 if is_violating else 0.3

        dx = np.cos(angle) * 1.5
        dy = np.sin(angle) * 1.5
        ax.plot(
            [ref[0], ref[0] + dx],
            [ref[1], ref[1] + dy],
            color=color, linewidth=lw, alpha=alpha, zorder=1,
        )


def _annotate_ratio(ax: Axes, check: CheckResult, result: VistaResult) -> None:
    percentages = check.detail.get("percentages", {})
    balanced = check.detail.get("balanced", True)

    pct_worse = percentages.get("WORSE", 0)
    pct_better = percentages.get("BETTER", 0)
    pct_indiff = percentages.get("INDIFFERENT", 0)
    pct_incompat = percentages.get("INCOMPARABLE", 0)

    label = "Balanced" if balanced else "Imbalanced"
    text_lines = [
        label,
        f"  Worse:  {pct_worse:.1f}%",
        f"  Better: {pct_better:.1f}%",
    ]
    if pct_indiff > 0:
        text_lines.append(f"  Indiff: {pct_indiff:.1f}%")
    if pct_incompat > 0:
        text_lines.append(f"  Incom:  {pct_incompat:.1f}%")

    ax.text(
        0.02, 0.98, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=6,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        zorder=6,
    )


def _annotate_stability(ax: Axes, check: CheckResult, result: VistaResult) -> None:
    changes = check.detail.get("changes", {})
    threshold = check.detail.get("threshold", 0.05)

    text_lines = [f"Threshold: {threshold:.0%}"]
    for label, pct in changes.items():
        text_lines.append(f"  {label}: {pct:.1%}")

    ax.text(
        0.02, 0.98, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=6,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        zorder=6,
    )


_ANNOTATORS: dict[str, Any] = {
    "Dominance relation": _annotate_dominance,
    "Diagonal preference change": _annotate_diagonal,
    "Radial preference change": _annotate_radial,
    "Preference ratio": _annotate_ratio,
    "Third alternative stability": _annotate_stability,
}


def plot_protocol_report(
    report: ProtocolReport,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Render a multi-panel summary figure for a protocol report.

    The figure contains one subplot per check (2 × 3 grid), each showing
    the baseline VISTA with check-specific annotations and a pass / fail
    badge in the title.

    Parameters
    ----------
    report : ProtocolReport
        A completed protocol report from :func:`run_protocol`.
    figsize : tuple or None
        Figure size in inches; auto-computed if *None*.

    Returns
    -------
    Figure
    """
    n_checks = len(report.checks)
    ncols = 3
    nrows = max(1, int(np.ceil(n_checks / ncols)))

    if figsize is None:
        figsize = (4.5 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, check in enumerate(report.checks):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        _draw_vista_on_ax(ax, report.baseline)

        annotator = _ANNOTATORS.get(check.name)
        if annotator is not None:
            annotator(ax, check, report.baseline)

        icon = _check_icon(check)
        color = _check_color(check)
        ax.set_title(
            f"{icon} {check.name}",
            fontsize="small", fontweight="bold", color=color,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    for idx in range(n_checks, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.legend(
        handles=_shared_legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize="small",
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"VISTA Protocol: {report.method_name}",
        fontweight="bold", fontsize="large",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06, top=0.93)

    return fig
