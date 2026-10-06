import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

import fastsparsegams
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from constrained_logistic import ConstrainedLogisticRegression
from step_smoothing_rash import ImprovedAlignedLogisticStepFunction, compute_sobolev_semi_norm
from utils import hessian, one_hot_encoding


def export_xlabel_to_csv(xlabel_dict, output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "x_value"])
        for key, values in xlabel_dict.items():
            for v in values:
                writer.writerow([key, v])


def load_xlabel_data(file_path):
    feature_x_values = {}
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            key, x_value = row
            if key not in feature_x_values:
                feature_x_values[key] = []
            feature_x_values[key].append(float(x_value))
    for key in feature_x_values:
        feature_x_values[key] = np.array(feature_x_values[key])
    return feature_x_values


def get_upper_lower_bounds(xlabel, w_orig, export_dir, w_samples):
    shape_coeffs = []
    for w_new in w_samples:
        shape_coeffs.append([])
        count = 0
        for key, values in xlabel.items():
            if key == "bias":
                continue
            l = count
            r = count + len(values)
            count += len(values)
            coefficients_steps = np.zeros(len(w_new[l:r]) + 2)
            coeff_reverse_cumsum = np.cumsum(w_new[l:r][::-1])[::-1]
            coefficients_steps[1:-1] = coeff_reverse_cumsum
            coefficients_steps[0] = coeff_reverse_cumsum[0]
            for value in coefficients_steps:
                shape_coeffs[-1].append(value)

    w_upper = []
    w_lower = []
    for i in range(len(shape_coeffs[0])):
        w_upper.append(np.max([shape_coeffs[j][i] for j in range(len(shape_coeffs))]))
        w_lower.append(np.min([shape_coeffs[j][i] for j in range(len(shape_coeffs))]))

    count = 0
    for key, values in xlabel.items():
        if key == "bias":
            continue
        l = count
        r = count + len(values)
        count += len(values)
        upper_segment = np.hstack((w_upper[l], w_upper[l:r]))
        lower_segment = np.hstack((w_lower[l], w_lower[l:r]))
        orig_segment = np.hstack((w_orig[l], w_orig[l:r]))
        np.savetxt(os.path.join(export_dir, f"{key}_upper.csv"), upper_segment, delimiter=",")
        np.savetxt(os.path.join(export_dir, f"{key}_lower.csv"), lower_segment, delimiter=",")
        np.savetxt(os.path.join(export_dir, f"{key}_orig.csv"), orig_segment, delimiter=",")


def prepare_step_function_data(x_values, y_values):
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    y_steps = np.zeros_like(y_values)
    if len(y_values) == 1:
        y_steps[0] = y_values[0]
    else:
        y_steps[:-1] = y_values[1:]
        y_steps[-1] = y_values[-1]
    return x_values, y_steps


def fit_feature_bounds(feature, feature_x_values, xlabel, w_samples, w_new, enforce_monotonic=True, monotonic_constraints=None):
    x_values = feature_x_values[feature]

    l, r = 0, 0
    count = 0
    found_feature = False
    for key, values in xlabel.items():
        if key == "bias":
            continue
        l = count
        r = count + len(values)
        count += len(values)
        if key == feature:
            found_feature = True
            break
    if not found_feature:
        raise ValueError(f"Feature {feature} not found in xlabel")

    shape_coeffs = []
    for weights in w_samples:
        shape_coeffs.append([])
        coefficients_steps = np.zeros(len(weights[l:r]) + 2)
        coeff_reverse_cumsum = np.cumsum(weights[l:r][::-1])[::-1]
        coefficients_steps[1:-1] = coeff_reverse_cumsum
        coefficients_steps[0] = coeff_reverse_cumsum[0]
        for value in coefficients_steps:
            shape_coeffs[-1].append(value)

    upper_bounds = []
    lower_bounds = []
    q75_bounds = []
    q25_bounds = []
    for i in range(len(shape_coeffs[0])):
        column_values = [shape_coeffs[j][i] for j in range(len(shape_coeffs))]
        upper_bounds.append(np.max(column_values))
        lower_bounds.append(np.min(column_values))
        q75_bounds.append(np.percentile(column_values, 75))
        q25_bounds.append(np.percentile(column_values, 25))

    orig_segment = np.zeros(len(w_new[l:r]) + 2)
    coeff_reverse_cumsum = np.cumsum(w_new[l:r][::-1])[::-1]
    orig_segment[1:-1] = coeff_reverse_cumsum
    orig_segment[0] = coeff_reverse_cumsum[0]

    min_len = len(x_values)
    if len(upper_bounds) != min_len or len(lower_bounds) != min_len or len(orig_segment) != min_len or len(q75_bounds) != min_len or len(q25_bounds) != min_len:
        min_len = min(len(x_values), len(upper_bounds), len(lower_bounds), len(orig_segment), len(q75_bounds), len(q25_bounds))
        x_values = x_values[:min_len]
        upper_bounds = upper_bounds[:min_len]
        lower_bounds = lower_bounds[:min_len]
        q75_bounds = q75_bounds[:min_len]
        q25_bounds = q25_bounds[:min_len]
        orig_segment = orig_segment[:min_len]

    if enforce_monotonic:
        try:
            constraint = None
            if monotonic_constraints is not None:
                if feature in monotonic_constraints:
                    constraint = monotonic_constraints[feature]

            def get_direction(data, specified_constraint):
                if specified_constraint == "increasing":
                    return True
                elif specified_constraint == "decreasing":
                    return False
                else:
                    slope, _ = np.polyfit(np.arange(len(data)), data, 1)
                    return slope > 0

            is_increasing_upper = get_direction(upper_bounds, constraint)
            iso_upper = IsotonicRegression(increasing=is_increasing_upper)
            upper_bounds = iso_upper.fit_transform(np.arange(len(upper_bounds)), upper_bounds)

            is_increasing_lower = get_direction(lower_bounds, constraint)
            iso_lower = IsotonicRegression(increasing=is_increasing_lower)
            lower_bounds = iso_lower.fit_transform(np.arange(len(lower_bounds)), lower_bounds)

            is_increasing_q75 = get_direction(q75_bounds, constraint)
            iso_q75 = IsotonicRegression(increasing=is_increasing_q75)
            q75_bounds = iso_q75.fit_transform(np.arange(len(q75_bounds)), q75_bounds)

            is_increasing_q25 = get_direction(q25_bounds, constraint)
            iso_q25 = IsotonicRegression(increasing=is_increasing_q25)
            q25_bounds = iso_q25.fit_transform(np.arange(len(q25_bounds)), q25_bounds)

            is_increasing_orig = get_direction(orig_segment, constraint)
            iso_orig = IsotonicRegression(increasing=is_increasing_orig)
            orig_segment = iso_orig.fit_transform(np.arange(len(orig_segment)), orig_segment)
        except Exception as e:
            print(f"Failed to enforce monotonicity for {feature}: {e}")

    x_steps_upper, y_steps_upper = prepare_step_function_data(x_values, upper_bounds)
    x_steps_lower, y_steps_lower = prepare_step_function_data(x_values, lower_bounds)
    x_steps_q75, y_steps_q75 = prepare_step_function_data(x_values, q75_bounds)
    x_steps_q25, y_steps_q25 = prepare_step_function_data(x_values, q25_bounds)

    x_range = (x_values.min(), x_values.max())
    x_pred = np.linspace(x_range[0], x_range[1], 1000)

    results = {
        "upper": {"x_true": x_steps_upper, "y_true": y_steps_upper, "x_pred": x_pred},
        "lower": {"x_true": x_steps_lower, "y_true": y_steps_lower, "x_pred": x_pred},
        "q75": {"x_true": x_steps_q75, "y_true": y_steps_q75, "x_pred": x_pred},
        "q25": {"x_true": x_steps_q25, "y_true": y_steps_q25, "x_pred": x_pred},
    }

    for key, bounds in (("upper", upper_bounds), ("lower", lower_bounds), ("q75", q75_bounds), ("q25", q25_bounds)):
        try:
            model = ImprovedAlignedLogisticStepFunction(x_values, bounds)
            results[key]["smooth_pred"] = model(x_pred)
            results[key]["model"] = model
        except Exception as e:
            print(f"Failed to fit multi-logistic for {key} bound of {feature}: {e}")
            results[key]["smooth_pred"] = np.zeros_like(x_pred)

    return results


def calculate_sobolev_norm(results_feature, weights_norm, x_min, x_max, n_points=1000):
    def weighted_func(x_vals):
        y_weighted = np.zeros_like(x_vals)
        keys = ["upper", "lower", "q75", "q25"]
        for i, key in enumerate(keys):
            model = results_feature.get(key, {}).get("model")
            if i < len(weights_norm):
                weight = weights_norm[i]
            else:
                weight = 0.0
            if model is not None:
                y_weighted += weight * model(x_vals)
        return y_weighted

    return compute_sobolev_semi_norm(weighted_func, x_min, x_max, n_points)


def train_coefficients(X_train, y_train, X_test, y_test, xlabel, w_samples, w_new, xlabel_path, monotonic_constraints=None):
    feature_x_values = load_xlabel_data(xlabel_path)
    for feature in feature_x_values.keys():
        if feature == "bias":
            continue
        if feature in X_train.columns:
            feature_x_values[feature] = np.insert(feature_x_values[feature], 0, np.min(X_train[feature]))
            feature_x_values[feature] = np.append(feature_x_values[feature], np.max(X_train[feature]))

    features = [f for f in feature_x_values.keys() if f != "bias"]
    results = {}
    for feature in features:
        if feature not in X_train.columns:
            continue
        try:
            results[feature] = fit_feature_bounds(
                feature, feature_x_values, xlabel, w_samples, w_new,
                enforce_monotonic=True, monotonic_constraints=monotonic_constraints,
            )
        except Exception as e:
            print(f"Error fitting bounds for {feature}: {e}")

    X_train_smoothed = []
    X_test_smoothed = []
    valid_features = []
    for feature in features:
        if feature not in results:
            continue
        res = results[feature]
        model_upper = res.get("upper", {}).get("model")
        model_lower = res.get("lower", {}).get("model")
        model_q75 = res.get("q75", {}).get("model")
        model_q25 = res.get("q25", {}).get("model")
        if model_upper and model_lower and model_q75 and model_q25:
            try:
                x_train_feat = X_train[feature].values
                x_test_feat = X_test[feature].values
                X_train_smoothed.append(model_upper(x_train_feat))
                X_train_smoothed.append(model_lower(x_train_feat))
                X_train_smoothed.append(model_q75(x_train_feat))
                X_train_smoothed.append(model_q25(x_train_feat))
                X_test_smoothed.append(model_upper(x_test_feat))
                X_test_smoothed.append(model_lower(x_test_feat))
                X_test_smoothed.append(model_q75(x_test_feat))
                X_test_smoothed.append(model_q25(x_test_feat))
                valid_features.append(feature)
            except Exception as e:
                print(f"Skipping {feature} for meta-model: {e}")

    if not valid_features:
        raise RuntimeError("No valid features for the meta-model")

    X_train_smoothed = np.array(X_train_smoothed).T
    X_test_smoothed = np.array(X_test_smoothed).T

    param_grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(ConstrainedLogisticRegression(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train_smoothed, y_train)
    meta_model = grid_search.best_estimator_

    train_acc = meta_model.score(X_train_smoothed, y_train)
    train_auc = roc_auc_score(y_train, meta_model.predict_proba(X_train_smoothed)[:, 1])

    coefs = meta_model.coef_[0]
    feature_weight_map = {}
    for i, feature in enumerate(valid_features):
        idx = i * 4
        feature_weight_map[feature] = (coefs[idx], coefs[idx + 1], coefs[idx + 2], coefs[idx + 3])

    feature_sobolev_norms = {}
    for feature in valid_features:
        w_upper, w_lower, w_q75, w_q25 = feature_weight_map[feature]
        x_vals = feature_x_values[feature]
        if len(x_vals) > 0:
            w_sum = w_upper + w_lower + w_q75 + w_q25
            if abs(w_sum) > 1e-6:
                weights_norm = [w_upper / w_sum, w_lower / w_sum, w_q75 / w_sum, w_q25 / w_sum]
            else:
                weights_norm = [0.25, 0.25, 0.25, 0.25]
            feature_sobolev_norms[feature] = calculate_sobolev_norm(results[feature], weights_norm, x_vals.min(), x_vals.max())

    test_predictions_binary = meta_model.predict(X_test_smoothed)
    test_predictions_proba = meta_model.predict_proba(X_test_smoothed)[:, 1]
    test_accuracy = accuracy_score(y_test, test_predictions_binary)
    test_auc = roc_auc_score(y_test, test_predictions_proba)

    return {
        "meta_model": meta_model,
        "results": results,
        "feature_weights": feature_weight_map,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "train_accuracy": train_acc,
        "train_auc": train_auc,
        "feature_sobolev_norms": feature_sobolev_norms,
    }


def sample_in_ellipsoid(H, w_orig, n_samples=100):
    d = H.shape[0]
    u = np.random.normal(size=(n_samples, d))
    u = u / (np.linalg.norm(u, axis=1).reshape(-1, 1))
    r = (np.random.random(size=n_samples)) ** (1 / d)
    x_ = u * r.reshape(-1, 1)
    lamb, V = np.linalg.eigh(H)
    a = np.sqrt(1 / lamb)
    dw_samples = ((a * V) @ x_.T).T
    return dw_samples + w_orig


def get_xlabel(header):
    xlabel = {}
    xlabel["bias"] = [0, 1]
    for k in header:
        f = k.split("<=")[0]
        if f not in xlabel:
            xlabel[f] = [float(k.split("<=")[1])]
        else:
            xlabel[f].append(float(k.split("<=")[1]))
    return xlabel


def parse_label_map(value):
    if not value:
        return None
    mapping = {}
    for item in value.split(","):
        key, label = item.split("=")
        mapping[key.strip()] = int(label)
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--output-dir", default="results_rashomon_smoothing")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rashomon-samples", type=int, default=1000)
    parser.add_argument("--fastsparse-lambda", type=float, default=3.0)
    parser.add_argument("--fastsparse-gamma", type=float, default=1e-5)
    parser.add_argument("--fastsparse-max-support", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    export_dir = output_dir / "segments"
    export_dir.mkdir(parents=True, exist_ok=True)
    xlabel_path = output_dir / "xlabel.csv"

    og_data = pd.read_csv(args.csv)
    label_map = parse_label_map(args.label_map)
    if label_map is not None:
        og_data[args.target] = og_data[args.target].map(label_map)
    og_y = og_data[args.target]
    total_start = perf_counter()
    data, counts = one_hot_encoding(og_data.drop(columns=[args.target]), one_hot=False)
    data[args.target] = og_y

    fold_rows = []
    total_sob_norms = {}
    for j in range(args.folds):
        if args.seed is not None:
            np.random.seed(args.seed + j)
        data_train, data_test = train_test_split(data, test_size=0.2, random_state=j)
        og_data_train, og_data_test = train_test_split(og_data, test_size=0.2, random_state=j)

        X_train = data_train.drop(columns=[args.target])
        y_train = data_train[args.target]

        og_X_train = og_data_train.drop(columns=[args.target])
        og_y_train = og_data_train[args.target]
        og_X_test = og_data_test.drop(columns=[args.target])
        og_y_test = og_data_test[args.target]

        model = fastsparsegams.fit(
            X_train.values.astype(np.float64),
            np.array(og_y_train).ravel(),
            penalty="L0L2",
            loss="Logistic",
            lambda_grid=np.array([[args.fastsparse_lambda]]),
            num_gamma=None,
            num_lambda=None,
            gamma_max=args.fastsparse_gamma,
            gamma_min=args.fastsparse_gamma,
            max_support_size=args.fastsparse_max_support,
        )

        w_new = []
        header_new = []
        for i, beta in enumerate(model.coeff().toarray().ravel()[1:]):
            if beta != 0:
                w_new.append(beta)
                header_new.append(X_train.columns[i + 1])
        w_new = np.asarray(w_new)

        X_new = data_train[header_new[1:]]
        X0 = np.ones((X_new.shape[0], 1))
        X_new = np.hstack((X0, X_new))

        lamb2 = 0.001
        sample_p = X_new.sum(0) / X_new.shape[0]
        H = hessian(w_new, X_new, y_train, lamb2, sample_p)
        w_samples = sample_in_ellipsoid(H, w_new, n_samples=args.rashomon_samples)

        np.savetxt(output_dir / f"w_samples_fold{j + 1}.csv", w_samples, delimiter=",")
        np.savetxt(output_dir / f"w_orig_fold{j + 1}.csv", w_new, delimiter=",")

        xlabel = get_xlabel(header_new)
        export_xlabel_to_csv(xlabel, xlabel_path)
        get_upper_lower_bounds(xlabel, w_new, export_dir, w_samples)

        fit = train_coefficients(og_X_train, og_y_train, og_X_test, og_y_test, xlabel, w_samples, w_new, xlabel_path)

        f_sob_norms = fit["feature_sobolev_norms"]
        for feat, val in f_sob_norms.items():
            total_sob_norms.setdefault(feat, []).append(val)

        row = {
            "fold": j + 1,
            "selected_features": ";".join(sorted(f_sob_norms)),
            "n_features": len(f_sob_norms),
            "train_accuracy": fit["train_accuracy"],
            "train_auc": fit["train_auc"],
            "test_accuracy": fit["test_accuracy"],
            "test_auc": fit["test_auc"],
            "total_sobolev": float(np.sum(list(f_sob_norms.values()))),
        }
        fold_rows.append(row)
        print(
            f"fold={j + 1} features={sorted(f_sob_norms)} test_accuracy={row['test_accuracy']:.4f} "
            f"test_auc={row['test_auc']:.4f} total_sobolev={row['total_sobolev']:.4f}",
            flush=True,
        )

    total_end = perf_counter()
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary = {}
    for column in ("train_accuracy", "train_auc", "test_accuracy", "test_auc", "total_sobolev"):
        summary[f"{column}_mean"] = folds[column].mean()
        summary[f"{column}_std_pop"] = folds[column].std(ddof=0)
    summary["n_folds"] = len(folds)
    summary["seconds"] = total_end - total_start
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    per_feature = pd.DataFrame(
        [{"feature": feat, "sobolev_mean": np.mean(norms), "n_folds": len(norms)} for feat, norms in total_sob_norms.items()]
    )
    per_feature.to_csv(output_dir / "sobolev_by_feature.csv", index=False)

    print(f"test_accuracy {summary['test_accuracy_mean']:.4f} +/- {summary['test_accuracy_std_pop']:.4f}")
    print(f"test_auc {summary['test_auc_mean']:.4f} +/- {summary['test_auc_std_pop']:.4f}")
    print(f"train_accuracy {summary['train_accuracy_mean']:.4f} +/- {summary['train_accuracy_std_pop']:.4f}")
    print(f"train_auc {summary['train_auc_mean']:.4f} +/- {summary['train_auc_std_pop']:.4f}")
    print(f"total_sobolev {summary['total_sobolev_mean']:.4f} +/- {summary['total_sobolev_std_pop']:.4f}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
