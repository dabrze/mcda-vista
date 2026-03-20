# VISTA Package — Review & Validation Roadmap

A step-by-step guide for reviewing the `mcda-vista` package, from high-level
overview down to implementation details.

---

## Phase 1: Orientation (10–15 min)

Get a feel for the package without running anything.

- [ ] **Read `README.md`** — understand the purpose, API surface, and examples
- [ ] **Read `pyproject.toml`** — check dependencies, extras, metadata, entry points
- [ ] **Browse `src/mcda_vista/__init__.py`** — verify the public API matches README
- [ ] **Skim `src/mcda_vista/relation.py`** — the foundational enum used everywhere
- [ ] **Skim `src/mcda_vista/core.py`** — understand `VistaGenerator` and `generate_vista()`
- [ ] **Look at `docs/VISTA_overview.png`** and `docs/parameters_plot.png` — reference visuals

**What to look for:**
- Does the README clearly explain what the package does?
- Are dependencies reasonable?
- Is the public API small and coherent?

---

## Phase 2: Smoke Tests (10 min)

Verify the package works at all.

```powershell
# 1. Check import works
python -c "import mcda_vista; print(mcda_vista.__version__)"

# 2. Run the full test suite
python -m pytest tests/ -v --tb=short

# 3. Check available methods
python -c "from mcda_vista.methods import list_methods; print(list_methods())"
```

**What to look for:**
- Import succeeds without errors
- All 145+ tests pass
- Method names are listed (18 adapters)

---

## Phase 3: Minimal Working Example (15 min)

Run the simplest possible VISTA generation to build intuition.

```python
import numpy as np
from mcda_vista import generate_vista, plot_vista, Relation

# Two-criteria problem: reference point at (0.5, 0.5)
result = generate_vista(
    method="topsis",
    ref_point=np.array([0.5, 0.5]),
    weights=np.array([0.5, 0.5]),
    n_grid=21,  # 21×21 grid → 441 test points
    criteria_types=np.array([1, 1]),  # both criteria are benefit type
)

# Inspect the result object
print(type(result))           # VistaResult
print(result.relations.shape) # (21, 21)
print(np.unique(result.relations))  # which Relation values appear?

# Plot it
fig = plot_vista(result, title="TOPSIS — Equal Weights")
fig.savefig("topsis_quick.png", dpi=150)
print("Saved topsis_quick.png")
```

**What to look for:**
- The grid should show BETTER/WORSE/INDIFFERENT regions
- The reference point (centre) should be INDIFFERENT
- Colours should match: green=BETTER, red=WORSE, blue=INDIFFERENT

---

## Phase 4: Try Multiple Methods (15–20 min)

Compare behaviour across method families. This reveals whether adapters
are wired correctly.

```python
import numpy as np
from mcda_vista import generate_vista, plot_vista_grid

methods = ["topsis", "saw", "vikor", "promethee_ii", "electre_iii"]
ref = np.array([0.5, 0.5])
w   = np.array([0.5, 0.5])
ct  = np.array([1, 1])

results = {}
for m in methods:
    try:
        results[m] = generate_vista(
            method=m, ref_point=ref, weights=w,
            n_grid=21, criteria_types=ct,
        )
        print(f"✓ {m}")
    except Exception as e:
        print(f"✗ {m}: {e}")

# Grid comparison plot
fig = plot_vista_grid(list(results.values()), list(results.keys()), ncols=3)
fig.savefig("method_comparison.png", dpi=150, bbox_inches="tight")
print("Saved method_comparison.png")
```

**What to look for:**
- All methods succeed (no exceptions)
- Ranking methods (TOPSIS, SAW) produce smooth BETTER/WORSE boundaries
- Outranking methods (ELECTRE, PROMETHEE) may show INCOMPARABLE regions
- VIKOR should show a narrow INDIFFERENT band (it's more discriminating)

---

## Phase 5: Verify Experiments (15–20 min)

Run the reproducible experiment suite.

```powershell
# Dry-run first (shows what would be computed, no heavy work)
python experiments/run_experiments.py --config experiments/configs/baseline.yaml --dry-run

# Run the baseline experiment (generates output in experiments/output/)
python experiments/run_experiments.py --config experiments/configs/baseline.yaml --plot

# Try the multi-parameter sweep
python experiments/run_experiments.py --config experiments/configs/multi_param.yaml --plot
```

**What to look for:**
- YAML configs are readable and self-documenting
- Output files appear in `experiments/output/`
- Plots look reasonable (no all-black or all-white grids)
- `--dry-run` correctly reports what it would do without computing

Then inspect the configs themselves:
- [ ] Read `experiments/configs/baseline.yaml` — the simplest experiment
- [ ] Read `experiments/configs/weights.yaml` — weight sensitivity sweep
- [ ] Read `experiments/configs/third_alt.yaml` — third-alternative injection
- [ ] Check `experiments/README.md` for documentation

---

## Phase 6: Weight Sensitivity (10 min)

A key use-case: how does the VISTA change as weights shift?

```python
import numpy as np
from mcda_vista import generate_vista, plot_vista_grid

ref = np.array([0.5, 0.5])
ct  = np.array([1, 1])

weight_sets = [
    (np.array([0.2, 0.8]), "w=(0.2, 0.8)"),
    (np.array([0.5, 0.5]), "w=(0.5, 0.5)"),
    (np.array([0.8, 0.2]), "w=(0.8, 0.2)"),
]

results, titles = [], []
for w, label in weight_sets:
    r = generate_vista(method="topsis", ref_point=ref, weights=w,
                       n_grid=21, criteria_types=ct)
    results.append(r)
    titles.append(label)

fig = plot_vista_grid(results, titles, ncols=3)
fig.savefig("weight_sensitivity.png", dpi=150, bbox_inches="tight")
```

**What to look for:**
- The BETTER/WORSE boundary should tilt as weights change
- Heavy weight on criterion 1 → boundary more sensitive to criterion 1

---

## Phase 7: Plotting Module Deep Dive (10 min)

Inspect the visual output more carefully.

- [ ] **Read `src/mcda_vista/plotting.py`** — understand `_draw_vista_on_ax()`
- [ ] **Check colour mapping** — `Relation.color` should match the R code in `viz.Rmd`
- [ ] **Try `plot_vista_comparison()`** — side-by-side comparison of two methods:

```python
from mcda_vista import generate_vista, plot_vista_comparison
import numpy as np

ref = np.array([0.5, 0.5])
w = np.array([0.5, 0.5])
ct = np.array([1, 1])

r1 = generate_vista(method="topsis", ref_point=ref, weights=w, n_grid=21, criteria_types=ct)
r2 = generate_vista(method="saw",    ref_point=ref, weights=w, n_grid=21, criteria_types=ct)

fig = plot_vista_comparison(r1, r2, "TOPSIS", "SAW")
fig.savefig("topsis_vs_saw.png", dpi=150)
```

**What to look for:**
- Legend is present and correct
- Axis labels show criterion values (0.0 to 1.0)
- Reference point is marked
- Compare output visually with `docs/parameters_plot.png` for style consistency

---

## Phase 8: Code Quality Review (20–30 min)

Deeper dive into the implementation.

### 8a. Core engine
- [ ] Read `src/mcda_vista/core.py` carefully
- [ ] Trace how `VistaGenerator.generate()` builds the dataset for each grid point
- [ ] Verify row ordering: row 0 = reference, row 1 = test point, row 2+ = extras
- [ ] Check that `criteria_types` (benefit/cost) are properly passed to methods

### 8b. Converters
- [ ] Read `src/mcda_vista/converters.py`
- [ ] **Known issue:** `relation_from_aggregates` docstring describes BETTER/WORSE
  conditions backwards (code is correct, docstring is wrong)
- [ ] Verify each converter handles edge cases (equal values, missing data)

### 8c. Value functions
- [ ] Read `src/mcda_vista/value_functions.py`
- [ ] Check `generate_breakpoints()` and `utility_of_value()` logic
- [ ] Verify breakpoint interpolation matches UTA-style methods

### 8d. Method adapters
- [ ] Read 2–3 adapters of different types:
  - A ranking method: `src/mcda_vista/methods/topsis.py`
  - An outranking method: `src/mcda_vista/methods/electre.py`
  - A value-function method: `src/mcda_vista/methods/uta.py`
- [ ] Check that `evaluate()` correctly maps to pyDecision functions
- [ ] Verify `default_params()` returns sensible defaults
- [ ] Read `src/mcda_vista/methods/_patches.py` — understand the monkey-patches

### 8e. Packaging & CI
- [ ] Review `pyproject.toml` for completeness
- [ ] **Known issue:** no version constraints on dependencies
- [ ] **Known issue:** ruff config uses deprecated `select` key
- [ ] Review `.github/workflows/ci.yml` for correctness

---

## Phase 9: Dashboard (Optional, 10 min)

If Streamlit is installed:

```powershell
pip install streamlit
streamlit run src/mcda_vista/app/dashboard.py
```

**What to look for:**
- Sidebar lets you pick method, weights, grid size
- Plot updates interactively
- Comparison mode works (select two methods)
- No crashes on parameter changes

---

## Phase 10: Edge Cases & Stress Tests (Optional, 15 min)

Push the boundaries:

```python
import numpy as np
from mcda_vista import generate_vista

ref = np.array([0.5, 0.5])
w = np.array([0.5, 0.5])
ct = np.array([1, 1])

# 1. Very fine grid (slow but should work)
r = generate_vista(method="topsis", ref_point=ref, weights=w,
                   n_grid=101, criteria_types=ct)
print(f"Fine grid: {r.relations.shape}")  # (101, 101)

# 2. Asymmetric weights
r = generate_vista(method="topsis", ref_point=ref,
                   weights=np.array([0.99, 0.01]),
                   n_grid=21, criteria_types=ct)

# 3. Cost criterion
r = generate_vista(method="topsis", ref_point=ref, weights=w,
                   n_grid=21, criteria_types=np.array([1, -1]))  # c2 is cost

# 4. Boundary reference point
r = generate_vista(method="topsis", ref_point=np.array([0.0, 1.0]),
                   weights=w, n_grid=21, criteria_types=ct)
```

**What to look for:**
- Fine grid completes (may take ~30s for some methods)
- Extreme weights don't crash
- Cost criteria produce reversed patterns
- Boundary reference points don't cause division-by-zero

---

## Known Issues (from code review)

These are documented but not yet fixed:

| # | Issue | Severity | File |
|---|-------|----------|------|
| 1 | Docstring says BETTER/WORSE backwards | Low | `converters.py` |
| 2 | No version constraints on deps | Medium | `pyproject.toml` |
| 3 | Deprecated ruff config key | Low | `pyproject.toml` |
| 4 | Placeholder URL in metadata | Low | `pyproject.toml` |
| 5 | No integration tests for adapters | High | `tests/test_methods/` |
| 6 | Dataset augmentation not in experiment runner | Medium | `run_experiments.py` |

---

## Summary

| Phase | Focus | Time |
|-------|-------|------|
| 1 | Orientation — read docs | 10–15 min |
| 2 | Smoke tests — import & pytest | 10 min |
| 3 | Minimal example — single VISTA | 15 min |
| 4 | Multiple methods — grid comparison | 15–20 min |
| 5 | Experiments — YAML configs & runner | 15–20 min |
| 6 | Weight sensitivity — key use case | 10 min |
| 7 | Plotting deep dive | 10 min |
| 8 | Code quality review | 20–30 min |
| 9 | Dashboard (optional) | 10 min |
| 10 | Edge cases (optional) | 15 min |
| **Total** | | **~2–2.5 hours** |

Start with Phases 1–4 for a solid understanding. Phases 5–7 verify
research reproducibility. Phases 8–10 are for thorough code review.
