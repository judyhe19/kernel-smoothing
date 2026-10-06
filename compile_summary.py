from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_kernel_smoothing import PRIMARY_EPSILON, SMOOTHNESS_COLUMN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out", default="summary")
    args = parser.parse_args()

    result_dirs = [Path(item) for item in args.result_dirs]
    labels = args.labels.split(",") if args.labels else [str(item) for item in result_dirs]
    if len(labels) != len(result_dirs):
        parser.error("--labels must have one entry per result directory")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stacked = []
    headline_rows = []
    coverage_notes = []
    for name, result_dir in zip(labels, result_dirs):
        table = pd.read_csv(result_dir / "budget_table.csv")
        table.insert(0, "dataset", name)
        stacked.append(table)

        metrics = pd.read_csv(result_dir / "fold_metrics.csv")
        budget_rows = metrics[
            ~metrics["is_feasible_anchor"].fillna(False)
            & metrics["status"].isin(["ok", "uncertified"])
        ]
        n_total_folds = metrics["fold"].nunique()
        usable_by_eps = budget_rows.groupby("epsilon")["fold"].nunique()
        max_cover = int(usable_by_eps.max()) if not usable_by_eps.empty else 0
        candidates = usable_by_eps[usable_by_eps == max_cover].index
        if any(np.isclose(candidates, PRIMARY_EPSILON)):
            headline_eps = PRIMARY_EPSILON
        else:
            headline_eps = float(min(candidates)) if len(candidates) else np.nan

        budget_table = table[table["method"].str.startswith("KernelSmoothing (eps=") & ~table["method"].str.contains("anchor")]
        match = (
            budget_table[budget_table["method"].str.contains(f"eps={headline_eps:g}")]
            if np.isfinite(headline_eps) else budget_table.iloc[0:0]
        )
        if not match.empty:
            row = match.iloc[0]
            headline_rows.append({
                "dataset": name, "method": f"KernelSmoothing (eps={headline_eps:g})",
                "train_accuracy": row.get("train_accuracy", np.nan), "train_auc": row.get("train_auc", np.nan),
                "test_accuracy": row["test_accuracy"], "test_auc": row["test_auc"],
                "smoothness": row[SMOOTHNESS_COLUMN], "n_folds": row["n_folds"],
            })
        coverage_notes.append({
            "dataset": name,
            "folds_run": n_total_folds,
            "headline_epsilon": headline_eps,
            "headline_fold_coverage": max_cover,
            "epsilons_with_full_coverage": ", ".join(
                f"{e:g}" for e in usable_by_eps[usable_by_eps == n_total_folds].index
            ),
        })

    pd.concat(stacked, ignore_index=True).to_csv(out / "budget_table_all.csv", index=False)
    headline = pd.DataFrame(headline_rows)
    headline.to_csv(out / "headline.csv", index=False)
    coverage = pd.DataFrame(coverage_notes)
    coverage.to_csv(out / "coverage.csv", index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 200):
        print(headline.to_string(index=False))
        print()
        print(coverage.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
