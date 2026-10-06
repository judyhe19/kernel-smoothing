from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_NAME = "KernelSmoothing"
PRIMARY_EPSILON = 0.005


def load_curves(results_dir: Path, epsilon: float) -> pd.DataFrame:
    path = results_dir / "shape_curves.csv.gz"
    curves = pd.read_csv(path)
    anchor = curves["is_feasible_anchor"].fillna(False).astype(bool)
    frame = curves[~anchor & np.isclose(curves["epsilon"], epsilon)]
    if frame.empty:
        raise SystemExit(f"no non-anchor curves at epsilon={epsilon} in {path}")
    return frame


def feature_curve(frame: pd.DataFrame, feature: str, fold: str, x_grid: np.ndarray) -> np.ndarray | None:
    sub = frame[frame["feature"] == feature]
    if sub.empty:
        return None
    if fold == "mean":
        rows = []
        for _, fold_frame in sub.groupby("fold"):
            fold_frame = fold_frame.sort_values("x")
            rows.append(np.interp(x_grid, fold_frame["x"], fold_frame["contribution"]))
        return np.vstack(rows).mean(axis=0)
    sub = sub[sub["fold"] == int(fold)].sort_values("x")
    if sub.empty:
        return None
    return np.interp(x_grid, sub["x"], sub["contribution"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--fold", default="1")
    parser.add_argument("--epsilon", type=float, default=PRIMARY_EPSILON)
    parser.add_argument("--title", default=f"{METHOD_NAME} Shape Functions")
    parser.add_argument("--wrap", type=int, default=11)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    raw = pd.read_csv(args.csv)
    frame = load_curves(results_dir, args.epsilon)
    fold_label = "mean" if args.fold == "mean" else f"fold{int(args.fold)}"
    stem = Path(args.output) if args.output else results_dir / f"shapes_{fold_label}"
    stem.parent.mkdir(parents=True, exist_ok=True)

    n = len(args.features)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, n, figsize=(1.55 * n + 0.9, 2.1), sharey=True)
    axes = np.atleast_1d(axes)
    absent = []
    for axis, feature in zip(axes, args.features):
        x_grid = np.linspace(float(raw[feature].min()), float(raw[feature].max()), 400)
        y = feature_curve(frame, feature, args.fold, x_grid)
        if y is None:
            absent.append(feature)
            axis.plot(x_grid, np.zeros_like(x_grid), color="0.6", linewidth=0.9)
        else:
            axis.plot(x_grid, y, color="black", linewidth=0.9)
        axis.set_xlabel("\n".join(textwrap.wrap(feature, args.wrap, break_long_words=True)))
        axis.xaxis.set_major_locator(plt.MaxNLocator(2))
        axis.tick_params(length=2.5)
    axes[0].set_ylabel("Coefficient")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    fig.suptitle(args.title, fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.png / .pdf  (epsilon={args.epsilon:g}, fold={args.fold})")
    if absent:
        print(f"drawn as flat zero (not in this fold's support): {', '.join(absent)}")


if __name__ == "__main__":
    main()
