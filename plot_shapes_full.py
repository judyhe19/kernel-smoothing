from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_shapes import METHOD_NAME, PRIMARY_EPSILON, load_curves

MAX_COLS = 6


def plot_results(results_dir: Path, csv: str, fold: int, epsilon: float, wrap: int, stem: Path) -> Path:
    raw = pd.read_csv(csv)
    frame = load_curves(results_dir, epsilon)
    frame = frame[frame["fold"] == fold]
    if frame.empty:
        raise SystemExit(f"no fold {fold} curves at epsilon={epsilon}")
    present = set(frame["feature"].unique())
    features = [c for c in raw.columns if c in present]
    n = len(features)
    ncols = min(MAX_COLS, n)
    nrows = int(np.ceil(n / ncols))
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.75 * ncols + 0.6, 2.35 * nrows + 0.5), squeeze=False)
    for idx, feature in enumerate(features):
        axis = axes[idx // ncols, idx % ncols]
        sub = frame[frame["feature"] == feature].sort_values("x")
        x_grid = np.linspace(float(raw[feature].min()), float(raw[feature].max()), 400)
        axis.plot(x_grid, np.interp(x_grid, sub["x"], sub["contribution"]), color="black", linewidth=0.9)
        axis.set_xlabel("\n".join(textwrap.wrap(feature, wrap, break_long_words=True)))
        axis.xaxis.set_major_locator(plt.MaxNLocator(3))
        axis.yaxis.set_major_locator(plt.MaxNLocator(4))
        axis.tick_params(length=2.5)
        if idx % ncols == 0:
            axis.set_ylabel("Coefficient")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    fig.suptitle(f"{METHOD_NAME} Shape Functions", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.png/.pdf  ({n} features, fold {fold}, eps={epsilon:g})")
    return stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=PRIMARY_EPSILON)
    parser.add_argument("--wrap", type=int, default=14)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    stem = Path(args.output) if args.output else results_dir / f"shapes_full_fold{args.fold}"
    plot_results(results_dir, args.csv, args.fold, args.epsilon, args.wrap, stem)


if __name__ == "__main__":
    main()
