"""Quantitative counterparts of the VISTA exploratory checklist."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from mcda_vista.core import VistaGenerator
from mcda_vista.relation import Relation

__all__ = ["DEFAULT_THIRD_ALTERNATIVES", "DIAGONAL_ANGLES", "quantify_checklist"]


DEFAULT_THIRD_ALTERNATIVES: tuple[tuple[float, float], ...] = (
    (0.50, 0.50),
    (0.25, 0.50),
    (0.50, 0.75),
    (0.25, 0.25),
    (0.25, 0.75),
)

DIAGONAL_ANGLES: tuple[float, float] = (np.pi / 4, 5 * np.pi / 4)
"""Directions of the main diagonal: toward the upper-right, then lower-left corner."""


def _context_value(
    context: Mapping[str, Any], key: str, default: Any,
) -> Any:
    return context.get(key, default)


def _count_transitions(relations: Sequence[int] | np.ndarray) -> int:
    """Count changes between consecutive relations, verbatim."""
    values = np.asarray(relations, dtype=np.int64)
    return int(np.count_nonzero(values[1:] != values[:-1])) if values.size > 1 else 0


def _count_colour_transitions(relations: Sequence[int] | np.ndarray) -> int:
    """Count relation changes ignoring undefined (ERROR) samples.

    :attr:`Relation.ERROR` marks a point where the method is numerically
    undefined rather than a fifth colour, so a red → ERROR → green walk is
    one transition, not two.  Counting it as a colour would let a method
    with a degenerate reference point (TOPSIS, see
    :func:`mcda_vista.methods.base.is_self_comparison`) score the
    well-behaved value of 2 on the diagonal while never showing an
    indifference band at all.
    """
    values = np.asarray(relations, dtype=np.int64)
    defined = values[values != int(Relation.ERROR)]
    return _count_transitions(defined)


def _ray_endpoint(reference: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return the intersection of a ray from *reference* with the unit square."""
    distances: list[float] = []
    for coordinate, delta in zip(reference, direction):
        if delta > 0:
            distances.append((1.0 - coordinate) / delta)
        elif delta < 0:
            distances.append(-coordinate / delta)
    return reference + min(distances) * direction


def _ray_samples(
    reference: np.ndarray, n_samples: int, angles: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return ``(len(angles), n_samples, 2)`` sample points along rays from *reference*."""
    directions = np.asarray(angles, dtype=float)
    rays = np.empty((directions.size, n_samples, reference.size), dtype=np.float64)
    for index, angle in enumerate(directions):
        direction = np.array([np.cos(angle), np.sin(angle)])
        rays[index] = np.linspace(reference, _ray_endpoint(reference, direction), n_samples)
    return rays


def _ray_transitions(
    generator: VistaGenerator,
    reference: np.ndarray,
    n_samples: int,
    angles: Sequence[float] | np.ndarray,
    identical_indifferent: bool,
    method_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Count relation changes along rays emanating from the reference point.

    The method is queried at every sample point rather than read off the sweep
    raster: snapping to grid cells makes a ray running nearly tangent to a
    stair-stepped boundary flip-flop between cells, which inflates the counts
    into double digits for perfectly smooth methods.

    The diagonal and radial checks differ only in the angles passed here, so
    they are guaranteed to measure the same quantity.
    """
    points = _ray_samples(reference, n_samples, angles)
    relations = generator.evaluate_points(
        points.reshape(-1, reference.size),
        identical_indifferent=identical_indifferent,
        **method_params,
    ).reshape(len(points), n_samples)
    per_ray = [_count_colour_transitions(row) for row in relations]
    per_ray_raw = [_count_transitions(row) for row in relations]
    return {
        "max": max(per_ray, default=0),
        "mean": float(np.mean(per_ray)) if per_ray else 0.0,
        "max_raw": max(per_ray_raw, default=0),
        "per_ray": per_ray,
    }


def quantify_checklist(
    method: Callable[..., Relation] | str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute numeric metrics for the six-step 2D VISTA checklist.

    Parameters
    ----------
    method
        Registered method name or a VISTA-compatible comparison callable.
    context
        Mapping containing optional ``resolution``, ``reference``, ``weights``,
        ``third_alternatives``, ``n_rays``, ``identical_indifferent``, and
        method-specific parameters in ``method_params``.  Values not supplied
        use the paper defaults.

    Returns
    -------
    dict
        Metric names mapped to values.

        ``dom_viol`` is the fraction of grid points inside a strict dominance
        cone whose relation is an outright reversal.  Weakening a preference to
        indifference or incomparability is tolerated; ERROR is not, since it is
        the absence of a verdict rather than a weaker one.

        ``self_indiff`` is *True*, *False*, or *None* when the relation at the
        reference is undefined (ERROR).

        ``diag_trans`` and ``ray_trans_max`` are both "largest number of
        relation changes encountered moving away from the reference", differing
        only in the directions considered: the diagonal uses the two
        :data:`DIAGONAL_ANGLES` (toward the upper-right and lower-left corner,
        where at most one transition is expected of a well-behaved method),
        the radial check uses ``n_rays`` evenly spaced directions.  Both query
        the method at each sample point rather than snapping to the sweep grid,
        so neither picks up rasterisation staircase artefacts.  Transition
        counts ignore ERROR samples; ``*_raw`` variants retain the verbatim
        counts.
    """
    resolution = int(_context_value(context, "resolution", 101))
    reference = np.asarray(_context_value(context, "reference", [0.5, 0.5]), dtype=float)
    weights = np.asarray(_context_value(context, "weights", [0.5, 0.5]), dtype=float)
    third_alternatives = _context_value(
        context, "third_alternatives", DEFAULT_THIRD_ALTERNATIVES,
    )
    n_rays = int(_context_value(context, "n_rays", 360))
    method_params = dict(_context_value(context, "method_params", {}))
    identical_indifferent = bool(_context_value(context, "identical_indifferent", False))

    if reference.shape != (2,) or weights.shape != (2,):
        raise ValueError("quantify_checklist requires two-dimensional reference and weights")

    generator = VistaGenerator(
        method,
        n_criteria=2,
        resolution=resolution,
        reference=reference.tolist(),
        weights=weights.tolist(),
    )
    baseline = generator.generate(
        progress=False,
        identical_indifferent=identical_indifferent,
        **method_params,
    )
    grid = baseline.grid
    relations = baseline.relations
    dominating = np.all(grid > reference, axis=1)
    dominated = np.all(grid < reference, axis=1)
    cone_mask = dominating | dominated
    correct = (
        (dominating & (relations == Relation.BETTER))
        | (dominated & (relations == Relation.WORSE))
    )
    # Weakening a preference to indifference or incomparability inside a cone is
    # tolerated; only an outright reversal counts.  ERROR is a violation — it is
    # the absence of a verdict, not a weaker one.
    tolerated = (relations == Relation.INDIFFERENT) | (relations == Relation.INCOMPARABLE)
    cone_total = int(np.count_nonzero(cone_mask))
    violations = int(np.count_nonzero(cone_mask & ~correct & ~tolerated))

    distances = np.linalg.norm(grid - reference, axis=1)
    reference_index = int(np.argmin(distances))
    reference_relation = Relation(int(relations[reference_index]))

    diagonal = _ray_transitions(
        generator, reference, resolution, DIAGONAL_ANGLES,
        identical_indifferent, method_params,
    )
    radial = _ray_transitions(
        generator,
        reference,
        resolution,
        np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False),
        identical_indifferent,
        method_params,
    )

    non_reference = np.ones(len(relations), dtype=bool)
    non_reference[reference_index] = False
    denominator = int(np.count_nonzero(non_reference))
    region_counts = {
        "better": int(np.count_nonzero((relations == Relation.BETTER) & non_reference)),
        "worse": int(np.count_nonzero((relations == Relation.WORSE) & non_reference)),
        "indiff": int(np.count_nonzero((relations == Relation.INDIFFERENT) & non_reference)),
        "incomp": int(np.count_nonzero((relations == Relation.INCOMPARABLE) & non_reference)),
        "error": int(np.count_nonzero((relations == Relation.ERROR) & non_reference)),
    }

    iia_changes: dict[str, float] = {}
    for alternative in third_alternatives:
        point = np.asarray(alternative, dtype=float)
        if point.shape != (2,):
            raise ValueError("third_alternatives must contain two-dimensional points")
        with_third = generator.generate(
            third_alternative=point.tolist(),
            progress=False,
            identical_indifferent=identical_indifferent,
            **method_params,
        )
        iia_changes[str(point.tolist())] = float(
            np.mean(with_third.relations != baseline.relations),
        )

    return {
        "method": baseline.method_name,
        "resolution": resolution,
        "dom_viol": violations / cone_total if cone_total else 0.0,
        "dom_viol_count": violations,
        "dom_cone_total": cone_total,
        "self_indiff": (
            None
            if reference_relation is Relation.ERROR
            else reference_relation == Relation.INDIFFERENT
        ),
        "self_relation": reference_relation.name,
        "diag_trans": diagonal["max"],
        "diag_trans_raw": diagonal["max_raw"],
        "diag_trans_per_ray": diagonal["per_ray"],
        "ray_trans_max": radial["max"],
        "ray_trans_mean": radial["mean"],
        "ray_trans_max_raw": radial["max_raw"],
        "ray_transitions": radial["per_ray"],
        "n_error": int(np.count_nonzero(relations == Relation.ERROR)),
        **{f"pct_{key}": value / denominator * 100 for key, value in region_counts.items()},
        "region_denominator": denominator,
        "iia_mean": float(np.mean(list(iia_changes.values()))) if iia_changes else 0.0,
        "iia_max": max(iia_changes.values(), default=0.0),
        "iia_changes": iia_changes,
    }
