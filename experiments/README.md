# VISTA Experiments

Reproducible experiment suite for generating VISTA visualisations across
methods, parameters, weights, reference points, and third-alternative
scenarios.

## Quick start

```bash
# From the mcda-vista/ directory

# Preview what a config will generate (no computation)
python -m experiments.run_experiments experiments/configs/baseline.yaml --dry-run

# Run the baseline experiment
python -m experiments.run_experiments experiments/configs/baseline.yaml

# Run with plots and a custom output directory
python -m experiments.run_experiments experiments/configs/baseline.yaml \
    --plot --output-dir results/

# Override the grid resolution (e.g. for a quick test)
python -m experiments.run_experiments experiments/configs/weights.yaml --resolution 21
```

## Configuration files

| Config | Mode | Description |
|---|---|---|
| `configs/baseline.yaml` | single-param | Baseline VISTAs for **all** methods with default parameters |
| `configs/weights.yaml` | weights sweep | Influence of criteria weights on VISTAs |
| `configs/refpoints.yaml` | reference sweep | Influence of reference-point position on VISTAs |
| `configs/third_alt.yaml` | third-alt sweep | Influence of a third alternative on VISTAs |
| `configs/multi_param.yaml` | multi-param | Per-method parameter grid sweeps |

## Config formats

### Simple (baseline)

Each method is run once with the specified parameters:

```yaml
methods:
  saw:
    delta: 0.10
```

### Sweep (weights / refpoints / third_alt)

An outer loop iterates over the swept dimension; every method is run for
each value:

```yaml
sweep:
  weights:
    - [0.25, 0.75]
    - [0.50, 0.50]
```

### Method sweeps (multi_param)

Each method defines its own parameter grid.  A Cartesian product of the
`sweep` values is generated and merged with `base` parameters:

```yaml
method_sweeps:
  promethee_i:
    sweep: {q: [0.00, 0.10, 0.20], p: [0.00, 0.10, 0.20]}
    base: {f: "t4", s: 0.0}
```

## Outputs

Each experiment creates a sub-directory under `--output-dir` (default
`experiments/output/`) named after the `experiment` field in the config.

For every job the runner produces:

* **`<label>.csv`** — grid coordinates and relation values
* **`<label>_meta.json`** — method name, parameters, timing, etc.
* **`<label>.png`** — (only with `--plot`) VISTA scatter plot

A top-level **`experiment_summary.json`** lists all jobs and their metadata.

## CLI flags

| Flag | Description |
|---|---|
| `--dry-run` | List jobs without running them |
| `--output-dir DIR` | Root output directory (default `experiments/output/`) |
| `--plot` | Generate PNG plots for each result |
| `--resolution N` | Override the config's grid resolution |
