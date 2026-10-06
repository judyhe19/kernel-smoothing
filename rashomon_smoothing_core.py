from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import fastsparsegams
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
from scipy.special import betaln, betainc, betaincinv, expit, ndtr, ndtri
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from utils import one_hot_encoding


BOUND_KEYS = ("upper", "lower", "q75", "q25")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
PAPER_SOBOLEV_POINTS = 1000


@dataclass(frozen=True)
class KernelSpec:
    family: str
    width: float


@dataclass
class MassMap:
    x_grid: np.ndarray
    u_grid: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "MassMap":
        values = np.sort(np.asarray(x, dtype=float))
        unique, first, counts = np.unique(values, return_index=True, return_counts=True)
        midranks = (first + 0.5 * (counts - 1)) / max(len(values) - 1, 1)
        return cls(unique, np.clip(midranks, 0.0, 1.0))

    def __call__(self, x):
        return np.interp(np.asarray(x, dtype=float), self.x_grid, self.u_grid, left=0.0, right=1.0)


@dataclass
class BoundSpec:
    feature: str
    bound_key: str
    thresholds: np.ndarray
    interval_levels: np.ndarray

    @property
    def base(self) -> float:
        return float(self.interval_levels[0])

    @property
    def deltas(self) -> np.ndarray:
        return np.diff(self.interval_levels)

    def evaluate_step(self, x):
        x = np.asarray(x, dtype=float)

        interval = np.searchsorted(self.thresholds, x, side="left")
        return self.interval_levels[np.clip(interval, 0, len(self.interval_levels) - 1)]


@dataclass(frozen=True)
class TransitionGroup:
    feature: str
    bound_key: str
    transition_index: int
    location_x: float
    location_u: float
    delta: float
    start: int
    stop: int


@dataclass
class AdaptiveBoundFunction:
    mass_map: MassMap
    base: float
    locations_u: np.ndarray
    deltas: np.ndarray
    dictionary: tuple[KernelSpec, ...]
    mixtures: np.ndarray

    def __call__(self, x):
        x_arr = np.asarray(x, dtype=float)
        scalar = x_arr.ndim == 0
        u = np.atleast_1d(self.mass_map(x_arr))
        y = np.full_like(u, self.base, dtype=float)
        for q, (location, delta) in enumerate(zip(self.locations_u, self.deltas)):
            for k, kernel in enumerate(self.dictionary):
                values, _ = kernel_value_derivative(u, location, kernel)
                y += delta * self.mixtures[q, k] * values
        return y[0] if scalar else y


def kernel_dictionary() -> tuple[KernelSpec, ...]:

    families = ("logistic", "gaussian", "beta33", "beta24", "beta42")
    widths = (0.005, 0.02, 0.05, 0.10)
    return tuple(KernelSpec(family, width) for family in families for width in widths)


def _beta_parameters(family: str) -> tuple[float, float]:
    return {
        "beta33": (3.0, 3.0),
        "beta24": (2.0, 4.0),
        "beta42": (4.0, 2.0),
    }[family]


def kernel_value_derivative(u, location_u: float, spec: KernelSpec):
    u = np.asarray(u, dtype=float)
    width_10_90 = spec.width
    if spec.family == "logistic":
        scale_parameter = width_10_90 / (2.0 * np.log(9.0))
        raw = expit((u - location_u) / scale_parameter)
        deriv = raw * (1.0 - raw) / scale_parameter
        low = expit(-location_u / scale_parameter)
        high = expit((1.0 - location_u) / scale_parameter)
    elif spec.family == "gaussian":
        scale_parameter = width_10_90 / (2.0 * ndtri(0.9))
        z = (u - location_u) / scale_parameter
        raw = ndtr(z)
        deriv = np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * scale_parameter)
        low = ndtr(-location_u / scale_parameter)
        high = ndtr((1.0 - location_u) / scale_parameter)
    else:
        a, b = _beta_parameters(spec.family)
        q10, median, q90 = (float(betaincinv(a, b, q)) for q in (0.1, 0.5, 0.9))
        span = width_10_90 / (q90 - q10)
        left = location_u - median * span
        v = np.clip((u - left) / span, 0.0, 1.0)
        raw = betainc(a, b, v)
        deriv = np.zeros_like(v)
        interior = (v > 0.0) & (v < 1.0)
        vi = v[interior]
        log_pdf = (a - 1.0) * np.log(vi) + (b - 1.0) * np.log1p(-vi) - betaln(a, b)
        deriv[interior] = np.exp(log_pdf) / span
        low_v = np.clip(-left / span, 0.0, 1.0)
        high_v = np.clip((1.0 - left) / span, 0.0, 1.0)
        low, high = betainc(a, b, low_v), betainc(a, b, high_v)
    scale = max(float(high - low), 1e-12)
    return (raw - low) / scale, deriv / scale


def binary_log_loss(y, logits) -> float:
    y = np.asarray(y, dtype=float)
    logits = np.asarray(logits, dtype=float)
    return float(np.mean(np.logaddexp(0.0, logits) - y * logits))


def metric_row(y, logits) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    logits = np.asarray(logits, dtype=float)
    probabilities = expit(logits)
    return {
        "accuracy": float(accuracy_score(y, probabilities >= 0.5)),
        "auc": float(roc_auc_score(y, probabilities)),
        "log_loss": binary_log_loss(y, logits),
    }


def compute_paper_sobolev_norm(func, x_min, x_max, n_points=PAPER_SOBOLEV_POINTS):
    if x_max <= x_min:
        return 0.0
    x_grid = np.linspace(x_min, x_max, n_points)
    values = np.asarray(func(x_grid), dtype=float)
    if np.ptp(values) < 1e-10:
        return 0.0
    h = 1.0 / (n_points - 1)
    squared = values**2
    l2_squared = h * (squared[0] / 2.0 + np.sum(squared[1:-1]) + squared[-1] / 2.0)
    l2_norm = float(np.sqrt(l2_squared))
    if l2_norm < 1e-10:
        return 0.0
    normalized = values / l2_norm
    derivative = (normalized[2:] - normalized[:-2]) / (2.0 * h)
    return float(np.sqrt(np.sum(derivative**2) * h))


def fit_nonnegative_logistic(X, y, C: float, initial=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    x0 = np.zeros(p + 1) if initial is None else np.asarray(initial, dtype=float)

    def objective(params):
        weights, intercept = params[:-1], params[-1]
        logits = X @ weights + intercept
        return binary_log_loss(y, logits) + 0.5 * np.sum(weights**2) / (C * n)

    def gradient(params):
        weights, intercept = params[:-1], params[-1]
        residual = expit(X @ weights + intercept) - y
        grad_w = X.T @ residual / n + weights / (C * n)
        return np.r_[grad_w, np.mean(residual)]

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * p + [(None, None)],
        options={"maxiter": 3000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Reference meta-model failed: {result.message}")
    return result.x[:-1], float(result.x[-1])


def select_reference_C(X, y, seed: int) -> float:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = {C: [] for C in C_GRID}
    for train_idx, val_idx in splitter.split(X, y):
        for C in C_GRID:
            weights, intercept = fit_nonnegative_logistic(X[train_idx], y[train_idx], C)
            scores[C].append(binary_log_loss(y[val_idx], X[val_idx] @ weights + intercept))
    return min(C_GRID, key=lambda C: float(np.mean(scores[C])))


def fit_fastsparse_and_extract(raw_fit: pd.DataFrame):
    target = "Outcome"
    binary, _ = one_hot_encoding(raw_fit.drop(columns=[target]), one_hot=False)
    y = raw_fit[target].to_numpy(dtype=int)
    model = fastsparsegams.fit(
        binary.values.astype(np.float64),
        y,
        penalty="L0L2",
        loss="Logistic",
        lambda_grid=np.array([[3.0]]),
        num_gamma=None,
        num_lambda=None,
        gamma_max=1e-5,
        gamma_min=1e-5,
        max_support_size=20,
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


def sample_rashomon_coefficients(fit, n_samples: int, seed: int, ridge=0.001, radius=1.0):
    X = fit["design"]
    weights = fit["weights"]
    probabilities = expit(fit["intercept"] + X @ weights)
    curvature = probabilities * (1.0 - probabilities)
    hessian = (X.T * curvature) @ X / len(X)
    sample_mass = X.mean(axis=0)
    hessian += np.diag(2.0 * ridge * np.maximum(sample_mass, 1e-3))
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalues = np.maximum(eigenvalues, 1e-8)

    rng = np.random.default_rng(seed)
    dimension = len(weights)
    directions = rng.normal(size=(n_samples, dimension))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radii = rng.random(n_samples) ** (1.0 / max(dimension, 1))
    unit_ball = directions * radii[:, None]
    transform = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
    samples = weights + radius * (unit_ball @ transform.T)

    return np.vstack([weights, samples])


def parse_selected_headers(headers, weights):
    grouped: dict[str, list[tuple[float, int, float]]] = {}
    for global_idx, (header, weight) in enumerate(zip(headers, weights)):
        feature, threshold_text = header.rsplit("<=", 1)
        grouped.setdefault(feature, []).append((float(threshold_text), global_idx, float(weight)))
    for feature in grouped:
        grouped[feature].sort(key=lambda item: item[0])
    return grouped


def build_bound_specs(raw_fit, fit, samples):
    grouped = parse_selected_headers(fit["headers"], fit["weights"])
    specs: dict[tuple[str, str], BoundSpec] = {}
    mass_maps: dict[str, MassMap] = {}
    for feature, entries in grouped.items():
        thresholds = np.asarray([entry[0] for entry in entries], dtype=float)
        indices = np.asarray([entry[1] for entry in entries], dtype=int)
        center_weights = fit["weights"][indices]
        feature_samples = samples[:, indices]


        sampled_levels = np.cumsum(feature_samples[:, ::-1], axis=1)[:, ::-1]
        sampled_levels = np.column_stack([sampled_levels, np.zeros(len(samples))])
        center_levels = np.r_[np.cumsum(center_weights[::-1])[::-1], 0.0]
        increasing = bool(center_levels[-1] >= center_levels[0])

        summaries = {
            "upper": np.max(sampled_levels, axis=0),
            "lower": np.min(sampled_levels, axis=0),
            "q75": np.percentile(sampled_levels, 75, axis=0),
            "q25": np.percentile(sampled_levels, 25, axis=0),
        }
        for bound_key in BOUND_KEYS:

            levels = IsotonicRegression(increasing=increasing).fit_transform(
                np.arange(len(summaries[bound_key])), summaries[bound_key]
            )
            specs[(feature, bound_key)] = BoundSpec(feature, bound_key, thresholds, levels)
        mass_maps[feature] = MassMap.fit(raw_fit[feature].to_numpy(dtype=float))
    return specs, mass_maps


def raw_design(raw: pd.DataFrame, specs, column_order):
    return np.column_stack(
        [specs[(feature, bound)].evaluate_step(raw[feature].to_numpy()) for feature, bound in column_order]
    )


def trapezoid_weights(n: int) -> np.ndarray:
    weights = np.ones(n, dtype=float) / (n - 1)
    weights[[0, -1]] *= 0.5
    return weights


def build_adaptive_problem(raw_val, specs, mass_maps, column_order, ref_weights, ref_intercept, dictionary):
    n_candidates = len(dictionary)
    n_val = len(raw_val)
    groups: list[TransitionGroup] = []
    columns: list[np.ndarray] = []
    derivative_columns: dict[str, list[tuple[int, np.ndarray]]] = {}
    z_constant = np.full(n_val, ref_intercept, dtype=float)
    grid_u = np.linspace(0.0, 1.0, 501)

    for column_idx, (feature, bound_key) in enumerate(column_order):
        spec = specs[(feature, bound_key)]
        weight = float(ref_weights[column_idx])
        z_constant += weight * spec.base
        if weight <= 1e-10:
            continue
        locations_u = mass_maps[feature](spec.thresholds)
        val_u = mass_maps[feature](raw_val[feature].to_numpy(dtype=float))
        for transition_idx, (location_u, delta) in enumerate(zip(locations_u, spec.deltas)):
            if abs(delta) <= 1e-10:
                continue
            start = len(columns)
            for kernel in dictionary:
                values, _ = kernel_value_derivative(val_u, float(location_u), kernel)
                _, derivative = kernel_value_derivative(grid_u, float(location_u), kernel)
                columns.append(weight * float(delta) * values)
                derivative_columns.setdefault(feature, []).append(
                    (len(columns) - 1, weight * float(delta) * derivative)
                )
            groups.append(
                TransitionGroup(
                    feature,
                    bound_key,
                    transition_idx,
                    float(spec.thresholds[transition_idx]),
                    float(location_u),
                    float(delta),
                    start,
                    start + n_candidates,
                )
            )

    A = np.column_stack(columns) if columns else np.zeros((n_val, 0))
    Q = np.zeros((A.shape[1], A.shape[1]), dtype=float)
    integration_weights = trapezoid_weights(len(grid_u))
    for feature, indexed_columns in derivative_columns.items():
        indices = np.asarray([item[0] for item in indexed_columns], dtype=int)
        D = np.column_stack([item[1] for item in indexed_columns])

        reference_curve = np.zeros_like(grid_u)
        for column_idx, (f, bound_key) in enumerate(column_order):
            if f != feature:
                continue
            spec = specs[(f, bound_key)]
            reference_curve += ref_weights[column_idx] * spec.evaluate_step(
                np.interp(grid_u, mass_maps[f].u_grid, mass_maps[f].x_grid)
            )
        amplitude = max(float(np.ptp(reference_curve)), 1e-3)
        block = D.T @ (D * integration_weights[:, None]) / (amplitude**2)
        Q[np.ix_(indices, indices)] += block
    Q = 0.5 * (Q + Q.T) + np.eye(len(Q)) * 1e-10
    return A, z_constant, Q, groups


def solve_transition_problem(A, z_constant, Q, groups, y_val, reference_loss, epsilon, dictionary):
    n_variables = A.shape[1]
    if n_variables == 0:
        return np.empty(0), {
            "minimum_smooth_loss": binary_log_loss(y_val, z_constant),
            "epsilon_min": binary_log_loss(y_val, z_constant) - reference_loss,
            "objective": 0.0,
            "iterations": 0,
        }
    E = np.zeros((len(groups), n_variables), dtype=float)
    initial = np.zeros(n_variables, dtype=float)
    sharp_local = min(
        range(len(dictionary)),
        key=lambda k: (dictionary[k].family != "beta33", dictionary[k].width),
    )
    for row, group in enumerate(groups):
        E[row, group.start : group.stop] = 1.0
        initial[group.start + sharp_local] = 1.0

    simplex = LinearConstraint(E, np.ones(len(groups)), np.ones(len(groups)))
    box = Bounds(np.zeros(n_variables), np.ones(n_variables))

    def logits(pi):
        return z_constant + A @ pi

    def risk(pi):
        return binary_log_loss(y_val, logits(pi))

    def risk_jac(pi):
        return A.T @ (expit(logits(pi)) - y_val) / len(y_val)

    risk_result = minimize(
        risk,
        initial,
        jac=risk_jac,
        method="SLSQP",
        bounds=box,
        constraints=[simplex],
        options={"maxiter": 1500, "ftol": 1e-11, "disp": False},
    )
    if not risk_result.success:
        raise RuntimeError(f"Minimum-risk transition fit failed: {risk_result.message}")
    minimum_smooth_loss = float(risk_result.fun)
    epsilon_min = minimum_smooth_loss - reference_loss
    effective_epsilon = max(float(epsilon), epsilon_min + 1e-8)
    risk_limit = reference_loss + effective_epsilon

    def objective(pi):
        return float(pi @ Q @ pi)

    def objective_jac(pi):
        return 2.0 * (Q @ pi)

    def risk_slack(pi):
        return risk_limit - risk(pi)

    def risk_slack_jac(pi):
        return -risk_jac(pi)

    smooth_result = minimize(
        objective,
        risk_result.x,
        jac=objective_jac,
        method="SLSQP",
        bounds=box,
        constraints=[
            simplex,
            NonlinearConstraint(risk_slack, 0.0, np.inf, jac=risk_slack_jac),
        ],
        options={"maxiter": 2500, "ftol": 1e-10, "disp": False},
    )
    if not smooth_result.success:
        raise RuntimeError(f"Smoothness optimization failed: {smooth_result.message}")
    final_loss = risk(smooth_result.x)
    if final_loss > risk_limit + 2e-6:
        raise RuntimeError(f"Risk constraint violated: {final_loss:.8f} > {risk_limit:.8f}")
    return smooth_result.x, {
        "minimum_smooth_loss": minimum_smooth_loss,
        "epsilon_min": epsilon_min,
        "effective_epsilon": effective_epsilon,
        "risk_limit": risk_limit,
        "objective": float(smooth_result.fun),
        "iterations": int(smooth_result.nit),
    }


def materialize_functions(specs, mass_maps, groups, pi, dictionary):
    mixtures: dict[tuple[str, str], np.ndarray] = {}
    for key, spec in specs.items():
        mixtures[key] = np.zeros((len(spec.thresholds), len(dictionary)), dtype=float)
        mixtures[key][:, 0] = 1.0
    for group in groups:
        mixtures[(group.feature, group.bound_key)][group.transition_index] = pi[group.start : group.stop]
    return {
        key: AdaptiveBoundFunction(
            mass_map=mass_maps[key[0]],
            base=spec.base,
            locations_u=mass_maps[key[0]](spec.thresholds),
            deltas=spec.deltas,
            dictionary=dictionary,
            mixtures=mixtures[key],
        )
        for key, spec in specs.items()
    }


def adaptive_logits(raw, functions, column_order, ref_weights, ref_intercept):
    logits = np.full(len(raw), ref_intercept, dtype=float)
    for idx, key in enumerate(column_order):
        logits += ref_weights[idx] * functions[key](raw[key[0]].to_numpy(dtype=float))
    return logits


def centered_feature_curves(raw_fit, functions, column_order, ref_weights):
    curves = {}
    features = sorted({feature for feature, _ in column_order})
    for feature in features:
        x_grid = np.linspace(raw_fit[feature].min(), raw_fit[feature].max(), 400)
        curve = np.zeros_like(x_grid)
        fit_values = np.zeros(len(raw_fit), dtype=float)
        for idx, (f, bound_key) in enumerate(column_order):
            if f != feature:
                continue
            curve += ref_weights[idx] * functions[(f, bound_key)](x_grid)
            fit_values += ref_weights[idx] * functions[(f, bound_key)](
                raw_fit[feature].to_numpy(dtype=float)
            )
        curve -= np.mean(fit_values)
        curves[feature] = (x_grid, curve)
    return curves


def centered_component_curves(raw_fit, functions, column_order, ref_weights):
    curves = {}
    for idx, (feature, bound_key) in enumerate(column_order):
        x_grid = np.linspace(raw_fit[feature].min(), raw_fit[feature].max(), 400)
        function = functions[(feature, bound_key)]
        train_x = raw_fit[feature].to_numpy(dtype=float)
        contribution = ref_weights[idx] * function(x_grid)
        contribution -= np.mean(ref_weights[idx] * function(train_x))
        curves[(feature, bound_key)] = (x_grid, contribution)
    return curves


def paper_feature_sobolev_norms(raw_fit, functions, column_order, ref_weights):
    norms = {}
    features = sorted({feature for feature, _ in column_order})
    for feature in features:
        terms = [
            (idx, bound_key)
            for idx, (candidate, bound_key) in enumerate(column_order)
            if candidate == feature
        ]

        def feature_function(x, terms=terms, feature=feature):
            x = np.asarray(x, dtype=float)
            result = np.zeros_like(x, dtype=float)
            for idx, bound_key in terms:
                result += ref_weights[idx] * functions[(feature, bound_key)](x)
            return result

        norms[feature] = compute_paper_sobolev_norm(
            feature_function,
            float(raw_fit[feature].min()),
            float(raw_fit[feature].max()),
        )
    return norms
