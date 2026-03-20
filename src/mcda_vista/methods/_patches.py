"""Monkey-patches for the pyDecision library.

VISTA requires three small behavioural changes to pyDecision functions that
cannot be achieved via the public API.  Rather than asking users to manually
edit their installed copy (as documented in ``zmiany-w-pyDecision.txt``), we
apply the patches programmatically at import time.

Patched functions
-----------------
1. **ELECTRE III** (``pyDecision.algorithm.e_iii``):
   - ``qualification``, ``destilation_descending``, ``destilation_ascending``,
     and ``electre_iii`` gain ``alpha`` / ``beta`` parameters so the
     λ-threshold formula ``λ_s = alpha * λ_max + beta`` is configurable
     (the original hard-codes ``0.30 − 0.15 * λ_max``).

2. **PROMETHEE I** (``pyDecision.algorithm.p_i``):
   - ``promethee_i`` returns ``(cp_matrix, flow_plus, flow_minus)`` instead
     of just ``cp_matrix``.

3. **REGIME** (``pyDecision.algorithm.regime``):
   - ``regime_method`` skips the ``po_ranking(cp_matrix)`` visualisation
     call that opens a Matplotlib window.

The :func:`apply_patches` function is idempotent — calling it more than once
is harmless.  It is invoked automatically when this module is first imported.
"""

from __future__ import annotations

import warnings

__all__ = ["apply_patches"]

_patched: bool = False

# Expected pyDecision version against which the patches were developed.
_EXPECTED_PYDECISION_VERSION = "4.6.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_pydecision_version() -> None:
    """Emit a warning if pyDecision is not the expected version."""
    try:
        import pyDecision

        version = getattr(pyDecision, "__version__", None)
        if version is None:
            warnings.warn(
                "Could not detect pyDecision version; VISTA patches may not "
                "apply correctly.",
                stacklevel=3,
            )
        elif version != _EXPECTED_PYDECISION_VERSION:
            warnings.warn(
                f"pyDecision version {version} detected, but VISTA patches "
                f"were developed against {_EXPECTED_PYDECISION_VERSION}. "
                f"Patches will still be applied but may not work correctly.",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "pyDecision is not installed; VISTA patches cannot be applied.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Patch 1 — ELECTRE III: parameterise α / β
# ---------------------------------------------------------------------------


def _patch_electre_iii() -> None:
    """Replace ELECTRE III internals with α / β–aware versions."""
    import numpy as np
    from pyDecision.algorithm import e_iii as _mod

    # Keep references to unchanged helpers already defined in the module.
    _global_concordance_matrix = _mod.global_concordance_matrix
    _credibility_matrix = _mod.credibility_matrix
    _pre_order_matrix = _mod.pre_order_matrix
    _po_ranking = _mod.po_ranking

    # -- patched qualification ------------------------------------------------
    def qualification(credibility, alpha=-0.15, beta=0.30):
        lambda_max = np.max(credibility)
        lambda_s = alpha * lambda_max + beta
        lambda_L = credibility[credibility < (lambda_max - lambda_s)]
        if lambda_L.shape[0] > 0:
            lambda_L = lambda_L.max()
        else:
            lambda_L = 0
        matrix_d = np.zeros((credibility.shape[0], credibility.shape[0]))
        for i in range(credibility.shape[0]):
            for j in range(credibility.shape[0]):
                if i != j:
                    if (
                        credibility[i, j] > lambda_L
                        and credibility[i, j]
                        > credibility[j, i] + lambda_s
                    ):
                        matrix_d[i, j] = 1.0
        rows = np.sum(matrix_d, axis=1)
        cols = np.sum(matrix_d, axis=0)
        qual = rows - cols
        return qual

    # -- patched destilation_descending ---------------------------------------
    def destilation_descending(credibility, alpha=-0.15, beta=0.30):
        alts = ["a" + str(alt) for alt in range(1, credibility.shape[0] + 1)]
        rank = []
        while len(alts) > 0:
            qual = qualification(credibility, alpha, beta)
            if np.where(qual == np.amax(qual))[0].shape[0] > 1:
                index = np.where(qual == np.amax(qual))[0]
                credibility_tie = credibility[index[:, None], index]
                qual_tie = qualification(credibility_tie, alpha, beta)
                while (
                    np.where(qual_tie == np.amax(qual_tie))[0].shape[0] > 1
                    and np.where(qual_tie == np.amax(qual_tie))[0].shape[0]
                    < np.where(qual == np.amax(qual))[0].shape[0]
                ):
                    qual = qualification(credibility_tie, alpha, beta)
                    index_tie = np.where(qual == np.amax(qual))[0]
                    credibility_tie = credibility_tie[
                        index_tie[:, None], index_tie
                    ]
                    qual_tie = qualification(credibility_tie, alpha, beta)
                    for i in range(index.shape[0] - 1, -1, -1):
                        if not np.isin(i, index_tie):
                            index = np.delete(index, i, axis=0)
                if np.where(qual_tie == np.amax(qual_tie))[0].shape[0] > 1:
                    ties = ""
                    for i in range(index.shape[0]):
                        ties = ties + alts[index[i]]
                        if i != index.shape[0] - 1:
                            ties = ties + "; "
                    rank.append(ties)
                    for i in range(index.shape[0] - 1, -1, -1):
                        del alts[index[i]]
                else:
                    index_tie = np.where(qual_tie == np.amax(qual_tie))[0].item()
                    index = index[index_tie]
                    rank.append(alts[index])
                    del alts[index]
            else:
                index = np.where(qual == np.amax(qual))[0].item()
                rank.append(alts[index])
                del alts[index]
            credibility = np.delete(credibility, index, axis=1)
            credibility = np.delete(credibility, index, axis=0)
        return rank

    # -- patched destilation_ascending ----------------------------------------
    def destilation_ascending(credibility, alpha=-0.15, beta=0.30):
        alts = ["a" + str(alt) for alt in range(1, credibility.shape[0] + 1)]
        rank = []
        while len(alts) > 0:
            qual = qualification(credibility, alpha, beta)
            if np.where(qual == np.amin(qual))[0].shape[0] > 1:
                index = np.where(qual == np.amin(qual))[0]
                credibility_tie = credibility[index[:, None], index]
                qual_tie = qualification(credibility_tie, alpha, beta)
                while (
                    np.where(qual_tie == np.amin(qual_tie))[0].shape[0] > 1
                    and np.where(qual_tie == np.amin(qual_tie))[0].shape[0]
                    < np.where(qual == np.amin(qual))[0].shape[0]
                ):
                    qual = qualification(credibility_tie, alpha, beta)
                    index_tie = np.where(qual == np.amin(qual))[0]
                    credibility_tie = credibility_tie[
                        index_tie[:, None], index_tie
                    ]
                    qual_tie = qualification(credibility_tie, alpha, beta)
                    for i in range(index.shape[0] - 1, -1, -1):
                        if not np.isin(i, index_tie):
                            index = np.delete(index, i, axis=0)
                if np.where(qual_tie == np.amin(qual_tie))[0].shape[0] > 1:
                    ties = ""
                    for i in range(index.shape[0]):
                        ties = ties + alts[index[i]]
                        if i != index.shape[0] - 1:
                            ties = ties + "; "
                    rank.append(ties)
                    for i in range(index.shape[0] - 1, -1, -1):
                        del alts[index[i]]
                else:
                    index_tie = np.where(qual_tie == np.amin(qual_tie))[0].item()
                    index = index[index_tie]
                    rank.append(alts[index])
                    del alts[index]
            else:
                index = np.where(qual == np.amin(qual))[0].item()
                rank.append(alts[index])
                del alts[index]
            credibility = np.delete(credibility, index, axis=1)
            credibility = np.delete(credibility, index, axis=0)
        rank = rank[::-1]
        return rank

    # -- patched electre_iii --------------------------------------------------
    def electre_iii(
        dataset, P, Q, V, W, alpha=-0.15, beta=0.30, graph=False
    ):
        alts = ["a" + str(alt) for alt in range(1, dataset.shape[0] + 1)]
        alts_D = [0] * dataset.shape[0]
        alts_A = [0] * dataset.shape[0]
        global_concordance = _global_concordance_matrix(
            dataset, P=P, Q=Q, W=W
        )
        credibility = _credibility_matrix(
            dataset, global_concordance, P=P, V=V
        )
        rank_D = destilation_descending(
            credibility=credibility, alpha=alpha, beta=beta
        )
        rank_A = destilation_ascending(
            credibility=credibility, alpha=alpha, beta=beta
        )
        rank_M = []
        for i in range(dataset.shape[0]):
            for j in range(len(rank_D)):
                if alts[i] in rank_D[j]:
                    alts_D[i] = j + 1
            for k in range(len(rank_A)):
                if alts[i] in rank_A[k]:
                    alts_A[i] = k + 1
        for i in range(len(alts)):
            rank_M.append("a" + str(i + 1))
        rank_M.sort()
        rank_P = _pre_order_matrix(
            rank_D, rank_A, number_of_alternatives=dataset.shape[0]
        )
        if graph:
            _po_ranking(rank_P)
        return global_concordance, credibility, rank_D, rank_A, rank_M, rank_P

    # Apply patches to the module.
    _mod.qualification = qualification
    _mod.destilation_descending = destilation_descending
    _mod.destilation_ascending = destilation_ascending
    _mod.electre_iii = electre_iii

    # Also patch the re-export in pyDecision.algorithm so that
    # ``from pyDecision.algorithm import electre_iii`` picks up the
    # patched version.
    import pyDecision.algorithm as _alg

    _alg.electre_iii = electre_iii


# ---------------------------------------------------------------------------
# Patch 2 — PROMETHEE I: return flows
# ---------------------------------------------------------------------------


def _patch_promethee_i() -> None:
    """Make ``promethee_i`` return ``(cp_matrix, flow_plus, flow_minus)``."""
    import numpy as np
    from pyDecision.algorithm import p_i as _mod

    _preference_degree = _mod.preference_degree
    _po_ranking = _mod.po_ranking

    def promethee_i(dataset, W, Q, S, P, F, graph=False):
        pd_matrix = _preference_degree(dataset, W, Q, S, P, F)
        flow_plus = np.sum(pd_matrix, axis=1) / (pd_matrix.shape[0] - 1)
        flow_minus = np.sum(pd_matrix, axis=0) / (pd_matrix.shape[0] - 1)
        cp_matrix = np.empty(
            (pd_matrix.shape[0], pd_matrix.shape[0]), dtype="U25"
        )
        cp_matrix.fill("-")
        for i in range(cp_matrix.shape[0]):
            for j in range(cp_matrix.shape[0]):
                if (
                    (
                        flow_plus[i] > flow_plus[j]
                        and flow_minus[i] < flow_minus[j]
                    )
                    or (
                        flow_plus[i] == flow_plus[j]
                        and flow_minus[i] < flow_minus[j]
                    )
                    or (
                        flow_plus[i] > flow_plus[j]
                        and flow_minus[i] == flow_minus[j]
                    )
                ):
                    cp_matrix[i, j] = "P+"
                if (
                    flow_plus[i] == flow_plus[j]
                    and flow_minus[i] == flow_minus[j]
                    and i != j
                ):
                    cp_matrix[i, j] = "I"
                if (
                    flow_plus[i] > flow_plus[j]
                    and flow_minus[i] > flow_minus[j]
                ) or (
                    flow_plus[i] < flow_plus[j]
                    and flow_minus[i] < flow_minus[j]
                ):
                    cp_matrix[i, j] = "R"
        if graph:
            _po_ranking(cp_matrix)
        return cp_matrix, flow_plus, flow_minus

    _mod.promethee_i = promethee_i

    import pyDecision.algorithm as _alg

    _alg.promethee_i = promethee_i


# ---------------------------------------------------------------------------
# Patch 3 — REGIME: skip po_ranking visualisation
# ---------------------------------------------------------------------------


def _patch_regime() -> None:
    """Replace ``regime_method`` so it does not call ``po_ranking``."""
    import numpy as np
    from pyDecision.algorithm import regime as _mod

    def regime_method(dataset, weights, criterion_type):
        X = np.copy(dataset) / 1.0
        weights = weights / np.sum(weights)
        g_ind = np.zeros((X.shape[0], X.shape[0]))
        for i in range(g_ind.shape[0]):
            for k in range(g_ind.shape[0]):
                for j in range(X.shape[1]):
                    if i != k:
                        if criterion_type[j] == "max":
                            if X[i, j] >= X[k, j]:
                                g_ind[i, k] += weights[j]
                            else:
                                g_ind[i, k] -= weights[j]
                        else:
                            if X[i, j] < X[k, j]:
                                g_ind[i, k] += weights[j]
                            else:
                                g_ind[i, k] -= weights[j]
        cp_matrix = np.empty((X.shape[0], X.shape[0]), dtype="U25")
        cp_matrix.fill("-")
        for i in range(cp_matrix.shape[0]):
            for j in range(cp_matrix.shape[0]):
                if i != j:
                    if g_ind[i, j] > 0:
                        cp_matrix[i, j] = "P+"
                    if g_ind[i, j] == 0 or g_ind[i, j] == g_ind[j, i]:
                        cp_matrix[i, j] = "I"
        # po_ranking(cp_matrix)  — deliberately skipped (opens Matplotlib GUI)
        return cp_matrix

    _mod.regime_method = regime_method

    import pyDecision.algorithm as _alg

    _alg.regime_method = regime_method


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def apply_patches() -> None:
    """Apply all monkey-patches to pyDecision.  Idempotent."""
    global _patched
    if _patched:
        return
    _check_pydecision_version()
    _patch_electre_iii()
    _patch_promethee_i()
    _patch_regime()
    _patched = True


# Auto-apply when this module is first imported.
apply_patches()
