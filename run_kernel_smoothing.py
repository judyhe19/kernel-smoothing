from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

if sys.version_info[:2] not in {(3, 9), (3, 10), (3, 11)}:
    raise RuntimeError("fastsparsegams requires Python 3.9, 3.10, or 3.11")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from adaptive_rashomon_joint import (
    AdaptiveSolution,
    JointAdaptiveConfig,
    build_joint_problem,
    component_weight_rows,
    evaluate_solution,
    initial_parameters,
    materialize_solution,
    solution_from_risk_anchor,
    solve_risk_anchor,
    solve_smooth_budget,
    transition_rows,
)
import rashomon_smoothing_core as poc
import fastsparsegams
from utils import one_hot_encoding

for _msg in (
    "invalid value encountered in matmul",
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
):
    warnings.filterwarnings("ignore", message=_msg)


DEFAULT_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15)
PRIMARY_EPSILON = 0.005


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv: str
    target: str
    val_size: float
    label_map: dict | None = None
    fastsparse_lambda: float = 3.0
    fastsparse_gamma: float = 1e-5
    fastsparse_max_support: int = 20
    fastsparse_max_bins: int | None = None


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    raw = pd.read_csv(spec.csv)
    if spec.label_map is not None:
        raw[spec.target] = raw[spec.target].map(spec.label_map)
        if raw[spec.target].isna().any():
            raise ValueError(f"Unmapped target labels in {spec.name}")
    if spec.target != "Outcome":
        if "Outcome" in raw.columns:
            raise ValueError(f"{spec.name} already has an Outcome column; refusing to rename {spec.target}")
        raw = raw.rename(columns={spec.target: "Outcome"})
    raw["Outcome"] = raw["Outcome"].astype(int)
    return raw


def fit_fastsparse_and_extract(raw_fit: pd.DataFrame, spec: DatasetSpec):
    binary, _ = one_hot_encoding(
        raw_fit.drop(columns=["Outcome"]),
        one_hot=False,
        max_bins=spec.fastsparse_max_bins,
    )
    y = raw_fit["Outcome"].to_numpy(dtype=int)
    model = fastsparsegams.fit(
        binary.values.astype(np.float64),
        y,
        penalty="L0L2",
        loss="Logistic",
        lambda_grid=np.array([[spec.fastsparse_lambda]]),
        num_gamma=None,
        num_lambda=None,
        gamma_max=spec.fastsparse_gamma,
        gamma_min=spec.fastsparse_gamma,
        max_support_size=spec.fastsparse_max_support,
    )
    coefficients = model.coeff().toarray().ravel()
    selected_indices = np.flatnonzero(np.abs(coefficients[1:]) > 1e-12)
    selected_weights = coefficients[1:][selected_indices]
    selected_headers = [binary.columns[i] for i in selected_indices]
    selected_design = binary.iloc[:, selected_indices].to_numpy(dtype=float)
    return {
        "intercept": float(coefficients[0]),
        "weights": selected_weights,
        "headers": selected_headers,
        "design": selected_design,
        "y": y,
    }


def prepare_fold(raw: pd.DataFrame, fold_idx: int, spec: DatasetSpec, n_samples: int):
    started = time.perf_counter()
    outer_train, test = train_test_split(raw, test_size=0.2, random_state=fold_idx)
    _, val = train_test_split(outer_train, test_size=spec.val_size, random_state=fold_idx)
    base_fit = outer_train.copy()
    fit = fit_fastsparse_and_extract(base_fit, spec)
    samples = poc.sample_rashomon_coefficients(fit, n_samples=n_samples, seed=7000 + fold_idx)
    specs, mass_maps = poc.build_bound_specs(base_fit, fit, samples)
    del mass_maps
    column_order = tuple(
        (feature, bound)
        for feature in sorted({key[0] for key in specs})
        for bound in poc.BOUND_KEYS
    )
    X = poc.raw_design(base_fit, specs, column_order)
    y = base_fit["Outcome"].to_numpy(dtype=int)
    best_C = poc.select_reference_C(X, y, seed=8000 + fold_idx)
    reference_weights, reference_intercept = poc.fit_nonnegative_logistic(X, y, best_C)
    reference_logits = {
        split: poc.raw_design(frame, specs, column_order) @ reference_weights + reference_intercept
        for split, frame in (("train", base_fit), ("val", val), ("test", test))
    }
    reference_loss = poc.binary_log_loss(val["Outcome"], reference_logits["val"])
    return {
        "fold": fold_idx + 1,
        "base_fit": base_fit,
        "val": val,
        "test": test,
        "specs": specs,
        "column_order": column_order,
        "reference_weights": reference_weights,
        "reference_intercept": reference_intercept,
        "reference_loss": reference_loss,
        "reference_C": best_C,
        "reference_logits": reference_logits,
        "selected_features": sorted({key[0] for key in specs}),
        "prepare_seconds": time.perf_counter() - started,
    }


def _fixed_features(problem) -> set:
    features = {feature for feature, _ in problem.column_order}
    return {
        feature for feature in features
        if all(problem.components[key].is_fixed for key in problem.column_order if key[0] == feature)
    }


def _solution_rows(prepared, problem, solution: AdaptiveSolution):
    splits = {"train": prepared["base_fit"], "val": prepared["val"], "test": prepared["test"]}
    row, sobolev, curves = evaluate_solution(problem, solution, splits)


    fixed_features = _fixed_features(problem)
    if sobolev:
        row["total_sobolev_smoothable"] = float(
            sum(value for feature, value in sobolev.items() if feature not in fixed_features)
        )
    row.update({
        "fold": prepared["fold"],
        "reference_C": prepared["reference_C"],
        "reference_val_loss": prepared["reference_loss"],
        "prepare_seconds": prepared["prepare_seconds"],
        "n_active_features": len(problem.feature_weight_totals),
        "n_components": problem.n_weights,
        "n_transitions": len(problem.groups),
        "n_theta": problem.n_theta,
    })
    for split, logits in prepared["reference_logits"].items():
        for metric, value in poc.metric_row(splits[split]["Outcome"], logits).items():
            row[f"reference_{split}_{metric}"] = value

    weights = component_weight_rows(problem, solution)
    transitions = transition_rows(problem, solution)
    for item in weights + transitions:
        item["fold"] = prepared["fold"]
    sobolev_rows = [
        {
            "fold": prepared["fold"], "epsilon": solution.epsilon, "feature": feature,
            "sobolev_norm": value, "fixed_feature": feature in fixed_features,
        }
        for feature, value in sobolev.items()
    ]
    curve_rows = [
        {
            "fold": prepared["fold"], "epsilon": solution.epsilon, "feature": feature,
            "x": float(x_value), "contribution": float(curve_value),
        }
        for feature, (x_grid, values) in curves.items()
        for x_value, curve_value in zip(x_grid, values)
    ]
    return row, weights, transitions, sobolev_rows, curve_rows


def run_fold(prepared, config: JointAdaptiveConfig, budgets, objective_kind: str = "roughness", solver: str = "slsqp", tolerant_anchor: bool = False):
    problem = build_joint_problem(
        prepared["base_fit"], prepared["val"], prepared["specs"], prepared["column_order"],
        prepared["reference_weights"], prepared["reference_intercept"], prepared["reference_loss"],
        config=config,
    )
    if not problem.groups:
        raise RuntimeError(f"Fold {prepared['fold']}: every component is fixed; nothing to smooth")

    if solver == "trust-constr" and objective_kind != "sobolev":
        raise ValueError("trust-constr is only wired for the sobolev objective")
    sobolev_objective = None
    if objective_kind == "sobolev":
        from adaptive_rashomon_joint_sobolev import (
            build_sobolev_objective,
            solve_sobolev_budget,
            solve_sobolev_budget_trust,
        )
        sobolev_objective = build_sobolev_objective(problem, prepared["base_fit"])

    def budget_solver(epsilon, anchor, warm):
        if sobolev_objective is not None:
            if solver == "trust-constr":
                return solve_sobolev_budget_trust(problem, sobolev_objective, "joint", epsilon, anchor, warm=warm)
            return solve_sobolev_budget(problem, sobolev_objective, "joint", epsilon, anchor, warm=warm)
        return solve_smooth_budget(problem, "joint", epsilon, anchor, warm=warm)


    roughness_scale = float(np.max(np.abs(problem.roughness_matrix))) if problem.n_theta else 0.0
    if roughness_scale > 1e4:
        problem.roughness_matrix = problem.roughness_matrix / roughness_scale
    else:
        roughness_scale = 1.0

    started = time.perf_counter()
    frozen_anchor = solve_risk_anchor(problem, "frozen")
    frozen_seconds = time.perf_counter() - started
    started = time.perf_counter()
    joint_anchor = solve_risk_anchor(problem, "joint", warm=frozen_anchor.params if frozen_anchor.success else None)
    joint_seconds = time.perf_counter() - started
    if (
        tolerant_anchor
        and not joint_anchor.success
        and joint_anchor.equality_residual <= config.constraint_tolerance
        and joint_anchor.epsilon_min <= 0.0
    ):


        joint_anchor.success = True
        joint_anchor.message += " [tolerant-anchor: iteration-capped but feasible; epsilon_min is an upper bound]"
    feasibility = [{
        "fold": prepared["fold"], "mode": "joint",
        "minimum_val_loss": joint_anchor.minimum_loss, "epsilon_min": joint_anchor.epsilon_min,
        "success": joint_anchor.success, "iterations": joint_anchor.iterations,
        "equality_residual": joint_anchor.equality_residual,
        "solve_seconds": joint_seconds, "frozen_warmup_seconds": frozen_seconds,
        "message": joint_anchor.message,
    }]

    output = {"metrics": [], "weights": [], "transitions": [], "sobolev": [], "curves": [], "feasibility": feasibility}
    joint_warm = joint_anchor.params if joint_anchor.success else None


    anchor_solution = solution_from_risk_anchor(problem, joint_anchor)
    anchor_solution.solve_seconds = joint_seconds
    values = _solution_rows(prepared, problem, anchor_solution)
    values[0]["is_feasible_anchor"] = True
    values[0]["roughness_scale"] = roughness_scale
    values[0]["smoothness_objective"] = values[0].get("smoothness_objective", np.nan) * roughness_scale
    for item in values[1] + values[2] + values[3] + values[4]:
        item["is_feasible_anchor"] = True
    for key, items in zip(("metrics", "weights", "transitions", "sobolev", "curves"),
                          ([values[0]], values[1], values[2], values[3], values[4])):
        output[key].extend(items)

    for epsilon in sorted(set(budgets)):
        started = time.perf_counter()
        solution = budget_solver(epsilon, joint_anchor, joint_warm)
        if solution.status == "solver_failed" and joint_anchor.success:


            retry = budget_solver(epsilon, joint_anchor, initial_parameters(problem, "joint"))
            if retry.status == "ok":
                solution = retry
                solution.message = f"{solution.message} (after warm-start retry)"
        solution.solve_seconds = time.perf_counter() - started
        if solution.params is not None:
            joint_warm = solution.params
            if solution.weights is None:
                materialize_solution(problem, solution)
        values = _solution_rows(prepared, problem, solution)
        values[0]["is_feasible_anchor"] = False
        values[0]["roughness_scale"] = roughness_scale
        values[0]["smoothness_objective"] = values[0].get("smoothness_objective", np.nan) * roughness_scale
        for item in values[1] + values[2] + values[3] + values[4]:
            item["is_feasible_anchor"] = False
        for key, items in zip(("metrics", "weights", "transitions", "sobolev", "curves"),
                              ([values[0]], values[1], values[2], values[3], values[4])):
            output[key].extend(items)
    return output


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if "is_feasible_anchor" in metrics:
        metrics = metrics[~metrics["is_feasible_anchor"].fillna(False)]
    usable = metrics[metrics["status"].isin(["ok", "uncertified"])].copy()
    rows = []
    for epsilon, frame in usable.groupby("epsilon", sort=True):
        row = {"epsilon": epsilon, "n_folds": len(frame), "certified_fraction": frame["certified"].mean()}
        for column in (
            "train_accuracy", "train_auc", "val_accuracy", "val_auc", "val_log_loss",
            "test_accuracy", "test_auc", "test_log_loss", "total_sobolev_norm",
            "total_sobolev_smoothable", "smoothness_objective", "risk_slack", "solve_seconds",
        ):
            if column in frame:
                row[f"{column}_mean"] = frame[column].mean()
                row[f"{column}_std"] = frame[column].std(ddof=1)
                row[f"{column}_std_pop"] = frame[column].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


SMOOTHNESS_COLUMN = "smoothness (total_sobolev)"


def build_budget_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anchor_mask = metrics["is_feasible_anchor"].fillna(False) if "is_feasible_anchor" in metrics else pd.Series(False, index=metrics.index)
    anchors = metrics[anchor_mask & metrics["status"].isin(["risk_anchor"])]
    if not anchors.empty and "test_accuracy" in anchors and "total_sobolev_smoothable" in anchors:
        rows.append({
            "method": f"KernelSmoothing (eps=eps_min anchor, mean eps_min={anchors['epsilon_min'].mean():.4g})",
            "train_accuracy": anchors["train_accuracy"].mean() if "train_accuracy" in anchors else np.nan,
            "train_auc": anchors["train_auc"].mean() if "train_auc" in anchors else np.nan,
            "test_accuracy": anchors["test_accuracy"].mean(), "test_accuracy_std_pop": anchors["test_accuracy"].std(ddof=0),
            "test_auc": anchors["test_auc"].mean(), "test_auc_std_pop": anchors["test_auc"].std(ddof=0),
            SMOOTHNESS_COLUMN: anchors["total_sobolev_smoothable"].mean(),
            "smoothness_std_pop": anchors["total_sobolev_smoothable"].std(ddof=0),
            "n_folds": len(anchors),
        })
    usable = metrics[~anchor_mask & metrics["status"].isin(["ok", "uncertified"])]
    for epsilon, frame in usable.groupby("epsilon", sort=True):
        rows.append({
            "method": f"KernelSmoothing (eps={epsilon:g})" + (" [primary]" if np.isclose(epsilon, PRIMARY_EPSILON) else ""),
            "train_accuracy": frame["train_accuracy"].mean() if "train_accuracy" in frame else np.nan,
            "train_auc": frame["train_auc"].mean() if "train_auc" in frame else np.nan,
            "test_accuracy": frame["test_accuracy"].mean(), "test_accuracy_std_pop": frame["test_accuracy"].std(ddof=0),
            "test_auc": frame["test_auc"].mean(), "test_auc_std_pop": frame["test_auc"].std(ddof=0),
            SMOOTHNESS_COLUMN: frame["total_sobolev_smoothable"].mean() if "total_sobolev_smoothable" in frame else np.nan,
            "smoothness_std_pop": frame["total_sobolev_smoothable"].std(ddof=0) if "total_sobolev_smoothable" in frame else np.nan,
            "n_folds": len(frame),
        })
    return pd.DataFrame(rows)


def plot_pareto(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    frame = summary.sort_values("epsilon")
    sobolev_column = (
        "total_sobolev_smoothable_mean"
        if "total_sobolev_smoothable_mean" in frame
        else "total_sobolev_norm_mean"
    )
    for axis, metric, label in ((axes[0], "test_auc_mean", "Test AUC"), (axes[1], "test_accuracy_mean", "Test accuracy")):
        axis.plot(frame[sobolev_column], frame[metric], "o-", color="#d1495b", label="KernelSmoothing (budget sweep)")
        for _, row in frame.iterrows():
            axis.annotate(f"{row.epsilon:g}", (row[sobolev_column], row[metric]),
                          xytext=(4, 3), textcoords="offset points", fontsize=8)
        axis.set_xlabel("Smoothness (total Sobolev, binary features excluded; smoother left)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("KernelSmoothing budget sweep")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _primary_or_anchor(frame: pd.DataFrame, epsilon: float) -> tuple[pd.DataFrame, str]:
    if frame.empty or "epsilon" not in frame:
        return pd.DataFrame(), "no solutions"
    anchor_mask = frame["is_feasible_anchor"].fillna(False) if "is_feasible_anchor" in frame else pd.Series(False, index=frame.index)
    primary = frame[~anchor_mask & np.isclose(frame["epsilon"], epsilon)]
    if not primary.empty:
        return primary, f"epsilon={epsilon:g}"
    return frame[anchor_mask], "epsilon=eps_min (minimum-risk anchor; fixed budgets infeasible)"


def plot_component_weights(weights: pd.DataFrame, output_path: Path, epsilon: float = PRIMARY_EPSILON) -> None:
    frame, label = _primary_or_anchor(weights, epsilon)
    if frame.empty:
        return
    summary = frame.groupby(["feature", "bound"], as_index=False)[["reference_share", "final_share"]].mean()
    summary["label"] = summary["feature"] + "\n" + summary["bound"]
    x = np.arange(len(summary))
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(12, len(summary) * 0.6), 5.2), constrained_layout=True)
    axis.bar(x - width / 2, summary["reference_share"], width, color="#777777", label="reference")
    axis.bar(x + width / 2, summary["final_share"], width, color="#009E73", label="KernelSmoothing")
    axis.set_xticks(x, summary["label"], rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Within-feature component-weight share")
    axis.set_title(f"Component weights, {label}")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_shape_functions(curves: pd.DataFrame, raw: pd.DataFrame, output_path: Path, epsilon: float = PRIMARY_EPSILON) -> None:
    frame, label = _primary_or_anchor(curves, epsilon)
    if frame.empty:
        return
    features = sorted(frame["feature"].unique())
    columns = 3
    rows = int(np.ceil(len(features) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows), squeeze=False, constrained_layout=True)
    for axis, feature in zip(axes.flat, features):
        x_grid = np.linspace(float(raw[feature].min()), float(raw[feature].max()), 400)
        fold_curves = []
        for _, fold_frame in frame[frame["feature"] == feature].groupby("fold"):
            fold_frame = fold_frame.sort_values("x")
            fold_curves.append(np.interp(x_grid, fold_frame["x"], fold_frame["contribution"]))
        values = np.vstack(fold_curves)
        axis.plot(x_grid, values.mean(axis=0), color="#d1495b", label="fold mean")
        axis.fill_between(x_grid, values.mean(axis=0) - values.std(axis=0),
                          values.mean(axis=0) + values.std(axis=0), color="#d1495b", alpha=0.12)
        axis.set_title(feature)
        axis.set_xlabel(feature)
        axis.set_ylabel("Centered log-odds contribution")
        axis.grid(alpha=0.18)
    for axis in axes.flat[len(features):]:
        axis.set_visible(False)
    if features:
        axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"KernelSmoothing shape functions, {label}")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(output_dir: Path, artifacts: dict, config_payload: dict, raw: pd.DataFrame) -> None:
    frames = {key: pd.DataFrame(values) for key, values in artifacts.items()}
    frames["metrics"].to_csv(output_dir / "fold_metrics.csv", index=False)
    frames["weights"].to_csv(output_dir / "component_weights.csv", index=False)
    frames["transitions"].to_csv(output_dir / "transitions.csv.gz", index=False)
    frames["sobolev"].to_csv(output_dir / "sobolev_by_feature.csv", index=False)
    frames["curves"].to_csv(output_dir / "shape_curves.csv.gz", index=False)
    frames["feasibility"].to_csv(output_dir / "feasibility.csv", index=False)
    summary = summarize_metrics(frames["metrics"])
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    table = build_budget_table(frames["metrics"])
    table.to_csv(output_dir / "budget_table.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    plot_pareto(summary, output_dir / "pareto.png")
    plot_component_weights(frames["weights"], output_dir / "component_weights.png")
    plot_shape_functions(frames["curves"], raw, output_dir / "shape_functions.png")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 200):
        print(table[["method", "test_accuracy", "test_auc", SMOOTHNESS_COLUMN, "n_folds"]].to_string(index=False))


def parse_csv_values(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def parse_label_map(value: str | None):
    if not value:
        return None
    mapping = {}
    for item in value.split(","):
        key, label = item.split("=")
        mapping[key.strip()] = int(label)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--val-size", type=float, default=0.5)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--rashomon-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ftol-smooth", type=float, default=None)
    parser.add_argument("--maxiter-smooth", type=int, default=None)
    parser.add_argument("--objective", choices=("roughness", "sobolev"), default="sobolev")
    parser.add_argument("--solver", choices=("slsqp", "trust-constr"), default="slsqp")
    parser.add_argument("--tolerant-anchor", action="store_true")
    parser.add_argument("--fastsparse-lambda", type=float, default=3.0)
    parser.add_argument("--fastsparse-gamma", type=float, default=1e-5)
    parser.add_argument("--fastsparse-max-support", type=int, default=20)
    parser.add_argument("--fastsparse-max-bins", type=int, default=None)
    parser.add_argument("--families", default=None)
    parser.add_argument("--widths", default=None)
    args = parser.parse_args()

    spec = DatasetSpec(
        name=Path(args.csv).stem,
        csv=args.csv,
        target=args.target,
        val_size=args.val_size,
        label_map=parse_label_map(args.label_map),
        fastsparse_lambda=args.fastsparse_lambda,
        fastsparse_gamma=args.fastsparse_gamma,
        fastsparse_max_support=args.fastsparse_max_support,
        fastsparse_max_bins=args.fastsparse_max_bins,
    )
    budgets = parse_csv_values(args.budgets, float)
    folds = parse_csv_values(args.folds, int)
    if any(fold not in range(1, 6) for fold in folds):
        parser.error("--folds must contain values from 1 through 5")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is nonempty: {output_dir}. Pass --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(spec)
    overrides = {}
    if args.objective == "sobolev":
        overrides["ftol_smooth"] = 1e-8
    if args.ftol_smooth is not None:
        overrides["ftol_smooth"] = args.ftol_smooth
    if args.maxiter_smooth is not None:
        overrides["maxiter_smooth"] = args.maxiter_smooth
    if args.families is not None:
        overrides["families"] = tuple(item.strip() for item in args.families.split(",") if item.strip())
    if args.widths is not None:
        overrides["widths"] = tuple(parse_csv_values(args.widths, float))
    config = JointAdaptiveConfig(**overrides)
    artifacts = {"metrics": [], "weights": [], "transitions": [], "sobolev": [], "curves": [], "feasibility": []}
    for fold in folds:
        prepared = prepare_fold(raw, fold - 1, spec, args.rashomon_samples)
        print(
            f"[{spec.name}] fold={fold} prepared in {prepared['prepare_seconds']:.1f}s "
            f"features={prepared['selected_features']} reference_C={prepared['reference_C']:g}",
            flush=True,
        )
        result = run_fold(prepared, config, budgets, objective_kind=args.objective, solver=args.solver, tolerant_anchor=args.tolerant_anchor)
        for key in artifacts:
            artifacts[key].extend(result[key])
        by_status = pd.Series([row["status"] for row in result["metrics"]]).value_counts().to_dict()
        epsilon_min = result["feasibility"][0]["epsilon_min"]
        print(f"[{spec.name}] fold={fold} done statuses={by_status} epsilon_min={epsilon_min:.5f}", flush=True)

    payload = {
        "dataset": asdict(spec),
        "config": asdict(config),
        "budgets": budgets,
        "folds": folds,
        "rashomon_samples": args.rashomon_samples,
        "primary_epsilon": PRIMARY_EPSILON,
        "mode": "joint",
        "objective": args.objective,
        "solver": args.solver,
        "protocol": (
            "train_test_split(test_size=0.2, random_state=fold-1); "
            f"val=second half of train_test_split(train, test_size={spec.val_size}, random_state=fold-1); "
            "FastSparse+bounds+reference on full 80% train"
        ),
    }
    write_outputs(output_dir, artifacts, payload, raw)
    print(f"[{spec.name}] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
