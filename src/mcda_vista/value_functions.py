"""Piecewise-linear value (utility) functions for MCDA.

This module provides helpers for generating and evaluating piecewise-linear
value functions defined by a sequence of breakpoints in the [0, 1] × [0, 1]
normalised space.  Each breakpoint is an ``(x, y)`` pair where *x* is the
normalised criterion value and *y* is the corresponding utility.

Typical workflow
----------------
1. Generate breakpoints with :func:`generate_breakpoints`.
2. Evaluate individual criterion values with :func:`utility_of_value` or
   :func:`weighted_utility_of_value`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "Breakpoints",
    "verify_breakpoints",
    "generate_values_from_01",
    "generate_breakpoints_gain",
    "generate_breakpoints",
    "utility_of_value",
    "weighted_utility_of_value",
]

Breakpoints = list[tuple[float, float]]
"""Type alias for a list of ``(x, y)`` breakpoint pairs."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def verify_breakpoints(breakpoints: Sequence[tuple[float, float]]) -> Breakpoints:
    """Validate and return a list of breakpoints.

    Parameters
    ----------
    breakpoints : Sequence[tuple[float, float]]
        Sequence of ``(x, y)`` pairs defining the piecewise-linear value
        function.  Must satisfy:

        * At least two breakpoints.
        * Each breakpoint is a 2-element tuple.
        * First breakpoint starts at ``x = 0`` and last at ``x = 1``.
        * ``x`` values are strictly increasing.
        * First breakpoint has ``y = 0`` and last has ``y = 1``.
        * ``y`` values are non-decreasing.

    Returns
    -------
    Breakpoints
        The validated breakpoints as a :pydata:`Breakpoints` list.

    Raises
    ------
    ValueError
        If any of the above constraints are violated.
    """
    K = len(breakpoints)
    if K < 2:
        raise ValueError(
            f"At least 2 breakpoints are required, got {K}."
        )

    for k, bp in enumerate(breakpoints):
        if len(bp) != 2:
            raise ValueError(
                f"Breakpoint {k} must have exactly 2 elements, got {len(bp)}."
            )

    if breakpoints[0][0] != 0.0 or breakpoints[-1][0] != 1.0:
        raise ValueError(
            "The x-coordinate of the first breakpoint must be 0.0 and the "
            f"last must be 1.0; got {breakpoints[0][0]} and "
            f"{breakpoints[-1][0]}."
        )

    for k in range(K - 1):
        if breakpoints[k][0] >= breakpoints[k + 1][0]:
            raise ValueError(
                f"Breakpoint x-coordinates must be strictly increasing; "
                f"breakpoints[{k}] x={breakpoints[k][0]} is not less than "
                f"breakpoints[{k + 1}] x={breakpoints[k + 1][0]}."
            )

    if breakpoints[0][1] != 0.0 or breakpoints[-1][1] != 1.0:
        raise ValueError(
            "The y-coordinate of the first breakpoint must be 0.0 and the "
            f"last must be 1.0; got {breakpoints[0][1]} and "
            f"{breakpoints[-1][1]}."
        )

    for k in range(K - 1):
        if breakpoints[k][1] > breakpoints[k + 1][1]:
            raise ValueError(
                f"Breakpoint y-coordinates must be non-decreasing; "
                f"breakpoints[{k}] y={breakpoints[k][1]} is greater than "
                f"breakpoints[{k + 1}] y={breakpoints[k + 1][1]}."
            )

    return [(bp[0], bp[1]) for bp in breakpoints]


# ---------------------------------------------------------------------------
# Breakpoint generation
# ---------------------------------------------------------------------------

def generate_values_from_01(
    K: int,
    form: tuple[str, ...] | list,
    seed: int | None = None,
) -> list[float]:
    """Generate *K* values in [0, 1] according to a specified shape.

    Parameters
    ----------
    K : int
        Number of values to generate (must be >= 2).
    form : tuple or list
        Shape descriptor.  The first element is a single-character code:

        * ``'L'`` – linear (identity).
        * ``'X'`` – convex (circular arc, slow start).
        * ``'V'`` – concave (circular arc, fast start).
        * ``'R'`` – random (sorted uniform samples).
        * ``'S'`` – sigmoidal; ``form[1]`` controls steepness.
        * ``'P'`` – power; ``form[1]`` is the exponent.
    seed : int or None, optional
        Random seed used only for form ``'R'``.  When *None* (default) the
        random values are not reproducible.

    Returns
    -------
    list[float]
        A sorted list of *K* floats starting at 0.0 and ending at 1.0.

    Raises
    ------
    ValueError
        If *K* < 2 or the form code is unrecognised.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}.")

    valid_codes = {"L", "X", "V", "R", "S", "P"}
    if form[0] not in valid_codes:
        raise ValueError(
            f"Unrecognised form code '{form[0]}'; expected one of {valid_codes}."
        )

    if form[0] == "L":
        v = [j / (K - 1) for j in range(K)]
    elif form[0] == "X":
        v = [1 - (1 - (j / (K - 1)) ** 2) ** 0.5 for j in range(K)]
    elif form[0] == "V":
        v = list(reversed(
            [(1 - (j / (K - 1)) ** 2) ** 0.5 for j in range(K)]
        ))
    elif form[0] == "R":
        rng = np.random.default_rng(seed)
        v = sorted(
            [0.0] + [float(rng.random()) for _ in range(K - 2)] + [1.0]
        )
    elif form[0] == "S":
        v = (
            [0.0]
            + [
                1 / (1 + np.exp(-(j / (K - 1) - 0.5) * form[1]))
                for j in range(1, K - 1)
            ]
            + [1.0]
        )
    elif form[0] == "P":
        v = [(j / (K - 1)) ** form[1] for j in range(K)]

    return v


def generate_breakpoints_gain(
    K: int,
    formX: tuple[str, ...] | list,
    formY: tuple[str, ...] | list,
    seed: int | None = None,
) -> Breakpoints:
    """Generate gain-type breakpoints from independent x/y shape descriptors.

    Parameters
    ----------
    K : int
        Number of breakpoints.
    formX : tuple or list
        Shape descriptor for the x-coordinates (see
        :func:`generate_values_from_01`).
    formY : tuple or list
        Shape descriptor for the y-coordinates.
    seed : int or None, optional
        Random seed forwarded to :func:`generate_values_from_01`.

    Returns
    -------
    Breakpoints
        List of ``(x, y)`` pairs.
    """
    x = generate_values_from_01(K, formX, seed=seed)
    y = generate_values_from_01(K, formY, seed=seed)
    return [(x[j], y[j]) for j in range(K)]


def generate_breakpoints(
    mode: str,
    K: int,
    formX: tuple[str, ...] | list,
    formY: tuple[str, ...] | list,
    seed: int | None = None,
) -> Breakpoints:
    """Generate and validate breakpoints for a gain or cost criterion.

    Parameters
    ----------
    mode : str
        ``'gain'`` for a benefit criterion (higher is better) or ``'cost'``
        for a cost criterion (lower is better).
    K : int
        Number of breakpoints.
    formX : tuple or list
        Shape descriptor for x-coordinates.
    formY : tuple or list
        Shape descriptor for y-coordinates.
    seed : int or None, optional
        Random seed forwarded to :func:`generate_values_from_01`.

    Returns
    -------
    Breakpoints
        Validated breakpoints.

    Raises
    ------
    ValueError
        If *mode* is not ``'gain'`` or ``'cost'``, or if the resulting
        breakpoints fail validation.
    """
    if mode not in ("gain", "cost"):
        raise ValueError(f"mode must be 'gain' or 'cost', got '{mode}'.")

    bpts = generate_breakpoints_gain(K, formX, formY, seed=seed)

    if mode == "cost":
        bpts = [(bpt[0], 1.0 - bpt[1]) for bpt in bpts]

    return verify_breakpoints(bpts)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def utility_of_value(
    value: float,
    breakpoints: Sequence[tuple[float, float]],
) -> float:
    """Compute the utility for a normalised criterion *value*.

    The utility is obtained by linearly interpolating between the two
    breakpoints that bracket *value*.

    Parameters
    ----------
    value : float
        Normalised criterion value (should lie within the breakpoint
        x-range).
    breakpoints : Sequence[tuple[float, float]]
        Breakpoint pairs ``(x, y)`` defining the piecewise-linear function.

    Returns
    -------
    float
        Interpolated utility.

    Raises
    ------
    ValueError
        If *value* does not fall within any breakpoint interval.
    """
    for k in range(len(breakpoints) - 1):
        bpL = breakpoints[k]
        bpR = breakpoints[k + 1]
        if bpL[0] <= value <= bpR[0]:
            lambd = (value - bpL[0]) / (bpR[0] - bpL[0])
            return (1 - lambd) * bpL[1] + lambd * bpR[1]

    raise ValueError(
        f"Value {value} is outside the breakpoint x-range "
        f"[{breakpoints[0][0]}, {breakpoints[-1][0]}]."
    )


def weighted_utility_of_value(
    value: float,
    weight: float,
    breakpoints: Sequence[tuple[float, float]],
) -> float:
    """Compute the weighted utility for a normalised criterion *value*.

    Parameters
    ----------
    value : float
        Normalised criterion value.
    weight : float
        Criterion weight.
    breakpoints : Sequence[tuple[float, float]]
        Breakpoint pairs defining the piecewise-linear value function.

    Returns
    -------
    float
        ``weight * utility_of_value(value, breakpoints)``.
    """
    return weight * utility_of_value(value, breakpoints)
