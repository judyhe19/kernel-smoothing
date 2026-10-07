# kernel-smoothing

Kernel Smoothing for Sparse Generalized Additive Models: RashomonSmoothing and its extension KernelSmoothing.

## Setup

Python 3.9, 3.10, or 3.11 (required by `fastsparsegams`).

```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 MKL_NUM_THREADS=8
```

Run everything from the repository root.

## Data format

One CSV with a binary target column and numeric feature columns. Every non-target column is used as a feature.
If the target is not already coded 0/1, pass `--label-map`, e.g. `--label-map Bad=1,Good=0`.

## KernelSmoothing

```bash
python run_kernel_smoothing.py --csv data.csv --target label --val-size 0.5 --output-dir results_ks --overwrite
```

Protocol: five 80/20 splits (`random_state` 0-4), FastSparse fit, Rashomon ellipsoid sampling and the
nonnegative-logistic reference on the 80% train split, then one budgeted Sobolev-objective solve per
loss budget in `--budgets` (default `0.001,...,0.15`). `--val-size` sets the selection subset drawn from
the train split (0.2 or 0.5 in the paper, depending on the dataset).

Useful options: `--folds 1,3`, `--budgets 0.005,0.01`, `--solver trust-constr` (interior-point rescue
when SLSQP stalls), `--ftol-smooth`, `--maxiter-smooth`, `--tolerant-anchor`, `--fastsparse-lambda`,
`--fastsparse-gamma`, `--fastsparse-max-support`, `--families`, `--widths` (restrict the kernel dictionary).

Outputs in `--output-dir`: `fold_metrics.csv` (one row per fold and budget, with `status` in
`ok` / `uncertified` / `infeasible` / `risk_anchor`), `summary_metrics.csv` and `budget_table.csv`
(per-budget means and population std of test accuracy, test AUC and smoothness), `sobolev_by_feature.csv`,
`shape_curves.csv.gz`, `component_weights.csv`, `transitions.csv.gz`, `config.json`, and the plots
`pareto.png`, `shape_functions.png`, `component_weights.png`. Smoothness is the total first-order Sobolev
norm over non-binary features (`total_sobolev_smoothable`).

If no single solver covers all folds, run SLSQP and trust-constr into two directories and merge the best
feasible row per fold and budget (certified over uncertified, then smoother):

```bash
python merge_solver_runs.py results_slsqp results_trustconstr results_merged --csv data.csv --target label
```

## RashomonSmoothing

```bash
python run_rashomon_smoothing.py --csv data.csv --target label --output-dir results_rs --seed 0
```

Same five splits and FastSparse fit; smooths the upper/lower/q75/q25 Rashomon bound step functions with
logistic ramps and refits the component weights by cross-validated constrained logistic regression.
`--seed` fixes the Rashomon sampling (omit it for unseeded sampling). Outputs: `fold_metrics.csv`,
`summary.csv`, `sobolev_by_feature.csv`, per-fold `w_samples_fold*.csv` / `w_orig_fold*.csv`, and the
bound step functions in `segments/`.

## Tables and figures

```bash
python compile_summary.py results_ks_a results_ks_b --labels a,b --out summary
python plot_shapes.py --results-dir results_ks --csv data.csv --features f1 f2 f3 --fold 1 --epsilon 0.005
python plot_shapes_full.py --results-dir results_ks --csv data.csv --fold 1 --epsilon 0.005
```

`compile_summary.py` stacks several KernelSmoothing runs and picks the headline budget per run (the
primary budget 0.005 when all folds are feasible there, otherwise the tightest fully covered budget).
`plot_shapes.py` draws a one-row strip of selected shape functions; `plot_shapes_full.py` draws all of
them. Both read `shape_curves.csv.gz` from the results directory and the CSV for the feature ranges.

## Tests

```bash
python -m unittest tests/test_adaptive_rashomon_joint.py
```
