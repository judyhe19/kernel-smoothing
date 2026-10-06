from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_kernel_smoothing import (
    SMOOTHNESS_COLUMN,
    DatasetSpec,
    build_budget_table,
    load_dataset,
    parse_label_map,
    plot_pareto,
    plot_shape_functions,
    summarize_metrics,
)

STATUS_RANK = {"ok": 2, "uncertified": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary_run")
    parser.add_argument("secondary_run")
    parser.add_argument("out")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--label-map", default=None)
    args = parser.parse_args()

    sources = {"primary": Path(args.primary_run), "secondary": Path(args.secondary_run)}
    out = Path(args.out)

    metrics = {}
    for name, directory in sources.items():
        frame = pd.read_csv(directory / "fold_metrics.csv")
        frame["solver_source"] = name
        metrics[name] = frame

    choice: dict[tuple, tuple] = {}
    for name, frame in metrics.items():
        budget = frame[~frame["is_feasible_anchor"].fillna(False)]
        for _, row in budget.iterrows():
            key = (row["fold"], round(row["epsilon"], 10))
            rank = STATUS_RANK.get(row["status"], 0)
            smooth = row.get("total_sobolev_smoothable", np.nan)
            score = (rank, -smooth if np.isfinite(smooth) else -np.inf)
            if key not in choice or score > choice[key][0]:
                choice[key] = (score, name, row)

    merged_budget = pd.DataFrame([entry[2] for entry in choice.values()])
    anchors = metrics["primary"][metrics["primary"]["is_feasible_anchor"].fillna(False)]
    merged = pd.concat([anchors, merged_budget], ignore_index=True).sort_values(["fold", "epsilon"])
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "fold_metrics.csv", index=False)

    chosen_by_source = {name: set() for name in sources}
    for key, entry in choice.items():
        chosen_by_source[entry[1]].add(key)

    for filename in ("sobolev_by_feature.csv", "shape_curves.csv.gz", "component_weights.csv", "transitions.csv.gz"):
        parts = []
        for name, directory in sources.items():
            frame = pd.read_csv(directory / filename)
            anchor_mask = frame["is_feasible_anchor"].fillna(False) if "is_feasible_anchor" in frame else pd.Series(False, index=frame.index)
            if name == "primary":
                parts.append(frame[anchor_mask])
            budget = frame[~anchor_mask]
            if not budget.empty:
                keep = budget[[
                    (row["fold"], round(row["epsilon"], 10)) in chosen_by_source[name]
                    for _, row in budget.iterrows()
                ]]
                parts.append(keep)
        pd.concat(parts, ignore_index=True).to_csv(out / filename, index=False)

    summary = summarize_metrics(merged)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    table = build_budget_table(merged)
    table.to_csv(out / "budget_table.csv", index=False)
    plot_pareto(summary, out / "pareto.png")
    if args.csv and args.target:
        spec = DatasetSpec(
            name=Path(args.csv).stem, csv=args.csv, target=args.target, val_size=0.5,
            label_map=parse_label_map(args.label_map),
        )
        raw = load_dataset(spec)
        curves = pd.read_csv(out / "shape_curves.csv.gz")
        plot_shape_functions(curves, raw, out / "shape_functions.png")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 200):
        print(table[["method", "test_accuracy", "test_auc", SMOOTHNESS_COLUMN, "n_folds"]].to_string(index=False))
    print(f"wrote merged results to {out}")


if __name__ == "__main__":
    main()
