from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

if sys.version_info[:2] not in {(3, 9), (3, 10), (3, 11)}:
    raise RuntimeError("The bundled FastSparse extension requires Python 3.9, 3.10, or 3.11")

import rashomon_smoothing_core as poc


OptimizationMode = Literal["frozen", "joint", "unrestricted_joint"]
WeightMode = Literal["frozen", "post_refit", "joint", "unrestricted_joint"]
SOLVER_TOLERANCE = 2e-6
BasisEvaluator = Callable[["LiftedTransitionGroup", poc.KernelSpec, np.ndarray], tuple[np.ndarray, np.ndarray]]
TransitionBuilder = Callable[[poc.BoundSpec, poc.MassMap, float], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class JointAdaptiveConfig:
    families: tuple[str, ...] = ("logistic", "gaussian", "beta33", "beta24", "beta42")
    widths: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
    derivative_points: int = 501
    paper_sobolev_points: int = poc.PAPER_SOBOLEV_POINTS
    feasibility_tolerance: float = 1e-8
    constraint_tolerance: float = SOLVER_TOLERANCE
    maxiter_risk: int = 2000
    maxiter_smooth: int = 3000
    ftol_risk: float = 1e-11
    ftol_smooth: float = 1e-10
    zero_tolerance: float = 1e-10

    def dictionary(self) -> tuple[poc.KernelSpec, ...]:
        return tuple(poc.KernelSpec(family, width) for family in self.families for width in self.widths)


@dataclass(frozen=True)
class LiftedTransitionGroup:
    feature: str
    bound_key: str
    transition_index: int
    location_x: float
    location_u: float
    delta: float
    start: int
    stop: int
    weight_index: int


@dataclass(frozen=True)
class ComponentLayout:
    key: tuple[str, str]
    weight_index: int
    is_fixed: bool
    locations_x: np.ndarray
    locations_u: np.ndarray
    deltas: np.ndarray


@dataclass
class JointAdaptiveProblem:
    specs: dict[tuple[str, str], poc.BoundSpec]
    maps: dict[str, poc.MassMap]
    column_order: tuple[tuple[str, str], ...]
    dictionary: tuple[poc.KernelSpec, ...]
    reference_weights: np.ndarray
    reference_intercept: float
    reference_loss: float
    y_val: np.ndarray
    groups: tuple[LiftedTransitionGroup, ...]
    components: dict[tuple[str, str], ComponentLayout]
    feature_weight_totals: dict[str, float]
    feature_amplitudes: dict[str, float]
    fixed_keys: frozenset[tuple[str, str]]
    centered_design: np.ndarray
    theta_base_means: np.ndarray
    fixed_val_centered: np.ndarray
    fixed_base_mean: float
    roughness_matrix: np.ndarray
    equality_joint: LinearConstraint
    equality_unrestricted: LinearConstraint
    equality_frozen: LinearConstraint
    bounds_joint: Bounds
    bounds_unrestricted: Bounds
    bounds_frozen: Bounds
    reference_centered_intercept: float
    config: JointAdaptiveConfig
    basis_evaluator: BasisEvaluator | None = None
    transition_metadata: dict[tuple[str, str, int], dict[str, float | str]] = field(default_factory=dict)

    @property
    def n_weights(self) -> int:
        return len(self.column_order)

    @property
    def n_theta(self) -> int:
        return self.centered_design.shape[1]

    @property
    def n_variables(self) -> int:
        return 1 + self.n_weights + self.n_theta

    @property
    def weight_slice(self) -> slice:
        return slice(1, 1 + self.n_weights)

    @property
    def theta_slice(self) -> slice:
        return slice(1 + self.n_weights, self.n_variables)

    def unpack(self, params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=float)
        return float(params[0]), params[self.weight_slice], params[self.theta_slice]

    def logits(self, params: np.ndarray) -> np.ndarray:
        beta, _, theta = self.unpack(params)
        return beta + self.fixed_val_centered + self.centered_design @ theta

    def objective(self, params: np.ndarray) -> float:
        theta = np.asarray(params, dtype=float)[self.theta_slice]
        return float(theta @ self.roughness_matrix @ theta)

    def objective_gradient(self, params: np.ndarray) -> np.ndarray:
        gradient = np.zeros(self.n_variables, dtype=float)
        theta = np.asarray(params, dtype=float)[self.theta_slice]
        gradient[self.theta_slice] = 2.0 * (self.roughness_matrix @ theta)
        return gradient

    def raw_intercept(self, params: np.ndarray) -> float:
        beta, weights, theta = self.unpack(params)
        smooth_base = sum(weights[component.weight_index] * self.specs[key].base for key, component in self.components.items() if key not in self.fixed_keys)
        smooth_transition_mean = float(theta @ self.theta_base_means)
        return float(beta - self.fixed_base_mean - smooth_base - smooth_transition_mean)

    def evaluate_basis(
        self,
        group: LiftedTransitionGroup,
        kernel: poc.KernelSpec,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.basis_evaluator is not None:
            return self.basis_evaluator(group, kernel, np.asarray(u, dtype=float))
        return poc.kernel_value_derivative(u, group.location_u, kernel)


@dataclass
class RiskAnchor:
    mode: OptimizationMode
    params: np.ndarray
    minimum_loss: float
    epsilon_min: float
    success: bool
    message: str
    iterations: int
    equality_residual: float


@dataclass
class AdaptiveSolution:
    mode: WeightMode
    epsilon: float
    status: str
    params: np.ndarray | None
    minimum_loss: float
    epsilon_min: float
    final_loss: float
    risk_limit: float
    risk_slack: float
    objective: float
    iterations: int
    equality_residual: float
    risk_violation: float
    certified: bool
    message: str = ""
    solve_seconds: float = np.nan
    functions: dict[tuple[str, str], Callable] = field(default_factory=dict)
    weights: np.ndarray | None = None
    intercept: float = np.nan
    theta: np.ndarray | None = None
    mixtures: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)


@dataclass
class StepBoundFunction:
    spec: poc.BoundSpec

    def __call__(self, x):
        return self.spec.evaluate_step(x)


@dataclass
class TransitionAwareBoundFunction:
    mass_map: poc.MassMap
    base: float
    groups: tuple[LiftedTransitionGroup, ...]
    dictionary: tuple[poc.KernelSpec, ...]
    mixtures: np.ndarray
    evaluator: BasisEvaluator | None = None

    def __call__(self, x):
        x_array = np.asarray(x, dtype=float)
        scalar = x_array.ndim == 0
        u = np.atleast_1d(self.mass_map(x_array))
        values = np.full_like(u, self.base, dtype=float)
        for transition_index, group in enumerate(self.groups):
            for kernel_index, kernel in enumerate(self.dictionary):
                if self.evaluator is None:
                    basis, _ = poc.kernel_value_derivative(u, group.location_u, kernel)
                else:
                    basis, _ = self.evaluator(group, kernel, u)
                values += group.delta * self.mixtures[transition_index, kernel_index] * basis
        return values[0] if scalar else values


def uniform_raw_maps(raw_fit: pd.DataFrame, features: Sequence[str]) -> dict[str, poc.MassMap]:
    maps: dict[str, poc.MassMap] = {}
    for feature in features:
        x_min = float(raw_fit[feature].min())
        x_max = float(raw_fit[feature].max())
        if x_max <= x_min:
            maps[feature] = poc.MassMap(np.array([x_min]), np.array([0.0]))
        else:
            maps[feature] = poc.MassMap(np.array([x_min, x_max]), np.array([0.0, 1.0]))
    return maps


def _merged_transitions(spec: poc.BoundSpec, coordinate_map: poc.MassMap, tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records: list[list[float]] = []
    for location_x, delta in zip(spec.thresholds, spec.deltas):
        if abs(float(delta)) <= tolerance:
            continue
        location_u = float(coordinate_map(np.array([location_x]))[0])
        if records and abs(location_u - records[-1][1]) <= tolerance:
            records[-1][2] += float(delta)
        else:
            records.append([float(location_x), location_u, float(delta)])
    records = [record for record in records if abs(record[2]) > tolerance]
    if not records:
        return np.empty(0), np.empty(0), np.empty(0)
    array = np.asarray(records, dtype=float)
    return array[:, 0], array[:, 1], array[:, 2]


def _raw_reference_amplitude(raw_fit: pd.DataFrame, feature: str, specs, column_order, reference_weights) -> float:
    x_grid = np.linspace(float(raw_fit[feature].min()), float(raw_fit[feature].max()), 501)
    curve = np.zeros_like(x_grid)
    for index, (candidate, bound_key) in enumerate(column_order):
        if candidate == feature:
            curve += float(reference_weights[index]) * specs[(feature, bound_key)].evaluate_step(x_grid)
    return max(float(np.ptp(curve)), 1e-3)


def _constraint_residual(constraint: LinearConstraint, params: np.ndarray) -> float:
    values = np.asarray(constraint.A @ params, dtype=float)
    lower = np.broadcast_to(np.asarray(constraint.lb, dtype=float), values.shape)
    upper = np.broadcast_to(np.asarray(constraint.ub, dtype=float), values.shape)
    return float(max(np.max(np.maximum(lower - values, 0.0), initial=0.0), np.max(np.maximum(values - upper, 0.0), initial=0.0)))


def _linear_constraint(rows: list[np.ndarray], values: list[float], n_variables: int) -> LinearConstraint:
    matrix = np.vstack(rows) if rows else np.zeros((0, n_variables), dtype=float)
    rhs = np.asarray(values, dtype=float)
    return LinearConstraint(matrix, rhs, rhs)


def _initial_kernel_index(dictionary: Sequence[poc.KernelSpec]) -> int:
    def key(index: int) -> tuple[int, float]:
        kernel = dictionary[index]
        priority = 0 if kernel.family == "beta33" else 1 if kernel.family == "logistic" else 2
        width = float(kernel.width)
        return priority, width if np.isfinite(width) else np.inf

    return min(range(len(dictionary)), key=key)


def build_joint_problem(
    raw_fit: pd.DataFrame,
    raw_val: pd.DataFrame,
    specs: dict[tuple[str, str], poc.BoundSpec],
    column_order: Sequence[tuple[str, str]],
    reference_weights: np.ndarray,
    reference_intercept: float,
    reference_loss: float,
    *,
    target_column: str = "Outcome",
    config: JointAdaptiveConfig | None = None,
    dictionary: Sequence[poc.KernelSpec] | None = None,
    basis_evaluator: BasisEvaluator | None = None,
    transition_builder: TransitionBuilder | None = None,
    transition_metadata: dict[tuple[str, str, int], dict[str, float | str]] | None = None,
    allow_zero_total_activation: bool = False,
    coordinate_maps: dict[str, poc.MassMap] | None = None,
) -> JointAdaptiveProblem:

    config = config or JointAdaptiveConfig()
    dictionary = tuple(dictionary) if dictionary is not None else config.dictionary()
    if not dictionary:
        raise ValueError("The transition dictionary must contain at least one candidate")
    column_order = tuple(column_order)
    reference_weights = np.asarray(reference_weights, dtype=float)
    if len(reference_weights) != len(column_order):
        raise ValueError("reference_weights and column_order have different lengths")
    features = tuple(sorted({feature for feature, _ in column_order}))
    maps = dict(coordinate_maps) if coordinate_maps is not None else uniform_raw_maps(raw_fit, features)
    if set(maps) != set(features):
        raise ValueError("coordinate_maps must contain exactly the problem features")
    n_weights = len(column_order)
    feature_weight_totals = {
        feature: float(sum(reference_weights[index] for index, (candidate, _) in enumerate(column_order) if candidate == feature))
        for feature in features
    }

    transition_records: list[tuple[str, str, int, float, float, float, int]] = []
    components: dict[tuple[str, str], ComponentLayout] = {}
    fixed_keys: set[tuple[str, str]] = set()
    for weight_index, key in enumerate(column_order):
        feature, _ = key
        spec = specs[key]
        builder = transition_builder or _merged_transitions
        locations_x, locations_u, deltas = builder(spec, maps[feature], config.zero_tolerance)
        is_binary = raw_fit[feature].nunique(dropna=False) <= 2
        is_constant_feature = raw_fit[feature].nunique(dropna=False) <= 1
        is_fixed = (
            is_binary
            or is_constant_feature
            or len(deltas) == 0
            or (
                not allow_zero_total_activation
                and feature_weight_totals[feature] <= config.zero_tolerance
            )
        )
        if is_fixed:
            fixed_keys.add(key)
        components[key] = ComponentLayout(key, weight_index, is_fixed, locations_x, locations_u, deltas)
        if not is_fixed:
            for transition_index, (location_x, location_u, delta) in enumerate(zip(locations_x, locations_u, deltas)):
                transition_records.append((feature, key[1], transition_index, float(location_x), float(location_u), float(delta), weight_index))

    n_theta = len(transition_records) * len(dictionary)
    n_variables = 1 + n_weights + n_theta
    theta_offset = 1 + n_weights
    groups: list[LiftedTransitionGroup] = []
    for group_index, record in enumerate(transition_records):
        start = group_index * len(dictionary)
        groups.append(LiftedTransitionGroup(*record[:6], start, start + len(dictionary), record[6]))

    base_theta_columns: list[np.ndarray] = []
    val_theta_columns: list[np.ndarray] = []
    derivative_columns: dict[str, list[tuple[int, np.ndarray]]] = {}
    derivative_u = np.linspace(0.0, 1.0, config.derivative_points)
    for group in groups:
        base_u = maps[group.feature](raw_fit[group.feature].to_numpy(dtype=float))
        val_u = maps[group.feature](raw_val[group.feature].to_numpy(dtype=float))
        for kernel_index, kernel in enumerate(dictionary):
            if basis_evaluator is None:
                base_values, _ = poc.kernel_value_derivative(base_u, group.location_u, kernel)
                val_values, _ = poc.kernel_value_derivative(val_u, group.location_u, kernel)
                _, derivative = poc.kernel_value_derivative(derivative_u, group.location_u, kernel)
            else:
                base_values, _ = basis_evaluator(group, kernel, base_u)
                val_values, _ = basis_evaluator(group, kernel, val_u)
                _, derivative = basis_evaluator(group, kernel, derivative_u)
            base_theta_columns.append(group.delta * base_values)
            val_theta_columns.append(group.delta * val_values)
            derivative_columns.setdefault(group.feature, []).append((group.start + kernel_index, group.delta * derivative))

    base_theta = np.column_stack(base_theta_columns) if base_theta_columns else np.zeros((len(raw_fit), 0))
    val_theta = np.column_stack(val_theta_columns) if val_theta_columns else np.zeros((len(raw_val), 0))
    theta_base_means = base_theta.mean(axis=0) if n_theta else np.empty(0)
    centered_design = val_theta - theta_base_means

    fixed_base = np.zeros(len(raw_fit), dtype=float)
    fixed_val = np.zeros(len(raw_val), dtype=float)
    for index, key in enumerate(column_order):
        if key not in fixed_keys:
            continue
        weight = float(reference_weights[index])
        fixed_base += weight * specs[key].evaluate_step(raw_fit[key[0]].to_numpy(dtype=float))
        fixed_val += weight * specs[key].evaluate_step(raw_val[key[0]].to_numpy(dtype=float))
    fixed_base_mean = float(np.mean(fixed_base))
    fixed_val_centered = fixed_val - fixed_base_mean

    feature_amplitudes = {
        feature: _raw_reference_amplitude(raw_fit, feature, specs, column_order, reference_weights)
        for feature in features
    }
    roughness = np.zeros((n_theta, n_theta), dtype=float)
    integration_weights = poc.trapezoid_weights(config.derivative_points)
    for feature, indexed_columns in derivative_columns.items():
        indices = np.asarray([index for index, _ in indexed_columns], dtype=int)
        derivative_matrix = np.column_stack([column for _, column in indexed_columns])
        block = derivative_matrix.T @ (derivative_matrix * integration_weights[:, None])
        block /= feature_amplitudes[feature] ** 2
        roughness[np.ix_(indices, indices)] += block
    roughness = 0.5 * (roughness + roughness.T)

    joint_rows: list[np.ndarray] = []
    joint_values: list[float] = []
    unrestricted_rows: list[np.ndarray] = []
    unrestricted_values: list[float] = []
    frozen_rows: list[np.ndarray] = []
    frozen_values: list[float] = []
    for group in groups:
        row = np.zeros(n_variables, dtype=float)
        row[theta_offset + group.start : theta_offset + group.stop] = 1.0
        row[1 + group.weight_index] = -1.0
        joint_rows.append(row.copy()); joint_values.append(0.0)
        unrestricted_rows.append(row.copy()); unrestricted_values.append(0.0)
        frozen_rows.append(row.copy()); frozen_values.append(0.0)
    for feature in features:
        row = np.zeros(n_variables, dtype=float)
        for index, (candidate, _) in enumerate(column_order):
            if candidate == feature:
                row[1 + index] = 1.0
        joint_rows.append(row); joint_values.append(feature_weight_totals[feature])


    intercept_row = np.zeros(n_variables, dtype=float)
    intercept_row[0] = 1.0
    for index, key in enumerate(column_order):
        if key not in fixed_keys:
            intercept_row[1 + index] -= specs[key].base
    intercept_row[theta_offset:] -= theta_base_means
    frozen_rows.append(intercept_row)
    frozen_values.append(float(reference_intercept + fixed_base_mean))

    lower_joint = np.full(n_variables, -np.inf, dtype=float)
    upper_joint = np.full(n_variables, np.inf, dtype=float)
    lower_joint[1:] = 0.0
    for index, key in enumerate(column_order):
        feature = key[0]
        if key in fixed_keys or feature_weight_totals[feature] <= config.zero_tolerance:
            lower_joint[1 + index] = upper_joint[1 + index] = reference_weights[index]
    lower_frozen = lower_joint.copy(); upper_frozen = upper_joint.copy()
    lower_frozen[1 : 1 + n_weights] = reference_weights
    upper_frozen[1 : 1 + n_weights] = reference_weights
    lower_unrestricted = np.full(n_variables, -np.inf, dtype=float)
    upper_unrestricted = np.full(n_variables, np.inf, dtype=float)
    lower_unrestricted[1:] = 0.0
    for index, key in enumerate(column_order):
        if key in fixed_keys:
            lower_unrestricted[1 + index] = upper_unrestricted[1 + index] = reference_weights[index]

    smooth_reference_mean = 0.0
    for index, key in enumerate(column_order):
        if key not in fixed_keys:
            smooth_reference_mean += reference_weights[index] * specs[key].base
    reference_centered_intercept = float(reference_intercept + fixed_base_mean + smooth_reference_mean)

    return JointAdaptiveProblem(
        specs=specs,
        maps=maps,
        column_order=column_order,
        dictionary=dictionary,
        reference_weights=reference_weights,
        reference_intercept=float(reference_intercept),
        reference_loss=float(reference_loss),
        y_val=raw_val[target_column].to_numpy(dtype=int),
        groups=tuple(groups),
        components=components,
        feature_weight_totals=feature_weight_totals,
        feature_amplitudes=feature_amplitudes,
        fixed_keys=frozenset(fixed_keys),
        centered_design=centered_design,
        theta_base_means=theta_base_means,
        fixed_val_centered=fixed_val_centered,
        fixed_base_mean=fixed_base_mean,
        roughness_matrix=roughness,
        equality_joint=_linear_constraint(joint_rows, joint_values, n_variables),
        equality_unrestricted=_linear_constraint(unrestricted_rows, unrestricted_values, n_variables),
        equality_frozen=_linear_constraint(frozen_rows, frozen_values, n_variables),
        bounds_joint=Bounds(lower_joint, upper_joint),
        bounds_unrestricted=Bounds(lower_unrestricted, upper_unrestricted),
        bounds_frozen=Bounds(lower_frozen, upper_frozen),
        reference_centered_intercept=reference_centered_intercept,
        config=config,
        basis_evaluator=basis_evaluator,
        transition_metadata=dict(transition_metadata or {}),
    )


def _constraint_for(problem: JointAdaptiveProblem, mode: OptimizationMode) -> LinearConstraint:
    if mode == "frozen":
        return problem.equality_frozen
    if mode == "joint":
        return problem.equality_joint
    if mode == "unrestricted_joint":
        return problem.equality_unrestricted
    raise ValueError(f"Unsupported optimization mode: {mode}")


def _bounds_for(problem: JointAdaptiveProblem, mode: OptimizationMode) -> Bounds:
    if mode == "frozen":
        return problem.bounds_frozen
    if mode == "joint":
        return problem.bounds_joint
    if mode == "unrestricted_joint":
        return problem.bounds_unrestricted
    raise ValueError(f"Unsupported optimization mode: {mode}")


def initial_parameters(problem: JointAdaptiveProblem, mode: OptimizationMode, warm: np.ndarray | None = None) -> np.ndarray:
    if warm is not None:
        candidate = np.asarray(warm, dtype=float).copy()
        if candidate.shape == (problem.n_variables,):
            return candidate
    params = np.zeros(problem.n_variables, dtype=float)
    params[problem.weight_slice] = problem.reference_weights
    sharp_index = _initial_kernel_index(problem.dictionary)
    theta = params[problem.theta_slice]
    for group in problem.groups:
        theta[group.start + sharp_index] = problem.reference_weights[group.weight_index]
    params[0] = problem.reference_intercept + problem.fixed_base_mean
    for index, key in enumerate(problem.column_order):
        if key not in problem.fixed_keys:
            params[0] += problem.reference_weights[index] * problem.specs[key].base
    params[0] += float(theta @ problem.theta_base_means)
    return params


def _risk_and_gradient(problem: JointAdaptiveProblem, params: np.ndarray) -> tuple[float, np.ndarray]:
    logits = problem.logits(params)
    residual = expit(logits) - problem.y_val
    gradient = np.zeros(problem.n_variables, dtype=float)
    gradient[0] = float(np.mean(residual))
    gradient[problem.theta_slice] = problem.centered_design.T @ residual / len(residual)
    return poc.binary_log_loss(problem.y_val, logits), gradient


def _unrestricted_reduced_layout(problem: JointAdaptiveProblem):
    rows: list[np.ndarray] = []
    component_groups: dict[tuple[str, str], list[LiftedTransitionGroup]] = {}
    for group in problem.groups:
        component_groups.setdefault((group.feature, group.bound_key), []).append(group)
    for groups in component_groups.values():
        reference = groups[0]
        for group in groups[1:]:
            row = np.zeros(1 + problem.n_theta, dtype=float)
            row[1 + group.start : 1 + group.stop] = 1.0
            row[1 + reference.start : 1 + reference.stop] -= 1.0
            rows.append(row)
    equality = _linear_constraint(rows, [0.0] * len(rows), 1 + problem.n_theta)
    return component_groups, equality


def _full_to_unrestricted_reduced(problem: JointAdaptiveProblem, params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    return np.r_[params[0], params[problem.theta_slice]]


def _unrestricted_reduced_to_full(
    problem: JointAdaptiveProblem,
    reduced: np.ndarray,
    component_groups: dict[tuple[str, str], list[LiftedTransitionGroup]],
) -> np.ndarray:
    reduced = np.asarray(reduced, dtype=float)
    params = np.zeros(problem.n_variables, dtype=float)
    params[0] = reduced[0]
    theta = reduced[1:]
    params[problem.theta_slice] = theta
    weights = params[problem.weight_slice]
    for index, key in enumerate(problem.column_order):
        if key in problem.fixed_keys:
            weights[index] = problem.reference_weights[index]
            continue
        groups = component_groups[key]
        weights[index] = float(theta[groups[0].start : groups[0].stop].sum())
    return params


def _solve_unrestricted_smooth_budget(
    problem: JointAdaptiveProblem,
    epsilon: float,
    anchor: RiskAnchor,
    warm: np.ndarray | None,
) -> AdaptiveSolution:
    risk_limit = problem.reference_loss + float(epsilon)
    if not anchor.success:
        return AdaptiveSolution("unrestricted_joint", epsilon, "solver_failed", None, anchor.minimum_loss, anchor.epsilon_min, np.nan, risk_limit, np.nan, np.nan, anchor.iterations, anchor.equality_residual, np.nan, False, anchor.message)
    if risk_limit < anchor.minimum_loss - problem.config.feasibility_tolerance:
        return AdaptiveSolution("unrestricted_joint", epsilon, "infeasible", None, anchor.minimum_loss, anchor.epsilon_min, np.nan, risk_limit, risk_limit - anchor.minimum_loss, np.nan, 0, anchor.equality_residual, anchor.minimum_loss - risk_limit, False, "Requested risk budget is below the measured feasible minimum")

    component_groups, equality = _unrestricted_reduced_layout(problem)
    full_initial = anchor.params if warm is None else np.asarray(warm, dtype=float)
    reduced_initial = _full_to_unrestricted_reduced(problem, full_initial)
    if poc.binary_log_loss(problem.y_val, problem.logits(full_initial)) > risk_limit + problem.config.constraint_tolerance:
        reduced_initial = _full_to_unrestricted_reduced(problem, anchor.params)

    def logits(reduced):
        return reduced[0] + problem.fixed_val_centered + problem.centered_design @ reduced[1:]

    def risk(reduced):
        return poc.binary_log_loss(problem.y_val, logits(reduced))

    def risk_gradient(reduced):
        residual = expit(logits(reduced)) - problem.y_val
        return np.r_[np.mean(residual), problem.centered_design.T @ residual / len(residual)]

    def objective(reduced):
        theta = reduced[1:]
        return float(theta @ problem.roughness_matrix @ theta)

    def objective_gradient(reduced):
        return np.r_[0.0, 2.0 * (problem.roughness_matrix @ reduced[1:])]

    result = minimize(
        objective,
        reduced_initial,
        jac=objective_gradient,
        method="SLSQP",
        bounds=Bounds(np.r_[-np.inf, np.zeros(problem.n_theta)], np.full(1 + problem.n_theta, np.inf)),
        constraints=[
            equality,
            NonlinearConstraint(
                lambda reduced: risk_limit - risk(reduced),
                0.0,
                np.inf,
                jac=lambda reduced: -risk_gradient(reduced),
            ),
        ],
        options={"maxiter": problem.config.maxiter_smooth, "ftol": problem.config.ftol_smooth, "disp": False},
    )
    params = _unrestricted_reduced_to_full(problem, result.x, component_groups)
    final_loss = poc.binary_log_loss(problem.y_val, problem.logits(params))
    equality_residual = _constraint_residual(problem.equality_unrestricted, params)
    risk_violation = max(final_loss - risk_limit, 0.0)
    success = bool(
        result.success
        and equality_residual <= problem.config.constraint_tolerance
        and risk_violation <= problem.config.constraint_tolerance
    )
    solution = AdaptiveSolution(
        mode="unrestricted_joint",
        epsilon=float(epsilon),
        status="ok" if success else "solver_failed",
        params=params.copy() if success else None,
        minimum_loss=anchor.minimum_loss,
        epsilon_min=anchor.epsilon_min,
        final_loss=final_loss,
        risk_limit=risk_limit,
        risk_slack=risk_limit - final_loss,
        objective=problem.objective(params),
        iterations=int(result.nit),
        equality_residual=equality_residual,
        risk_violation=risk_violation,
        certified=success,
        message=str(result.message),
    )
    return materialize_solution(problem, solution) if success else solution


def solve_risk_anchor(
    problem: JointAdaptiveProblem,
    mode: OptimizationMode,
    *,
    warm: np.ndarray | None = None,
) -> RiskAnchor:
    equality = _constraint_for(problem, mode)
    bounds = _bounds_for(problem, mode)
    x0 = initial_parameters(problem, mode, warm)

    result = minimize(
        lambda params: _risk_and_gradient(problem, params)[0],
        x0,
        jac=lambda params: _risk_and_gradient(problem, params)[1],
        method="SLSQP",
        bounds=bounds,
        constraints=[equality],
        options={"maxiter": problem.config.maxiter_risk, "ftol": problem.config.ftol_risk, "disp": False},
    )
    minimum_loss = poc.binary_log_loss(problem.y_val, problem.logits(result.x))
    equality_residual = _constraint_residual(equality, result.x)
    success = bool(result.success and equality_residual <= problem.config.constraint_tolerance)
    return RiskAnchor(
        mode=mode,
        params=result.x.copy(),
        minimum_loss=minimum_loss,
        epsilon_min=minimum_loss - problem.reference_loss,
        success=success,
        message=str(result.message),
        iterations=int(result.nit),
        equality_residual=equality_residual,
    )


def solve_smooth_budget(
    problem: JointAdaptiveProblem,
    mode: OptimizationMode,
    epsilon: float,
    anchor: RiskAnchor,
    *,
    warm: np.ndarray | None = None,
) -> AdaptiveSolution:
    if mode == "unrestricted_joint":
        return _solve_unrestricted_smooth_budget(problem, epsilon, anchor, warm)
    risk_limit = problem.reference_loss + float(epsilon)
    if not anchor.success:
        return AdaptiveSolution(mode, epsilon, "solver_failed", None, anchor.minimum_loss, anchor.epsilon_min, np.nan, risk_limit, np.nan, np.nan, anchor.iterations, anchor.equality_residual, np.nan, False, anchor.message)
    if risk_limit < anchor.minimum_loss - problem.config.feasibility_tolerance:
        return AdaptiveSolution(mode, epsilon, "infeasible", None, anchor.minimum_loss, anchor.epsilon_min, np.nan, risk_limit, risk_limit - anchor.minimum_loss, np.nan, 0, anchor.equality_residual, anchor.minimum_loss - risk_limit, False, "Requested risk budget is below the measured feasible minimum")

    equality = _constraint_for(problem, mode)
    bounds = _bounds_for(problem, mode)
    x0 = anchor.params.copy() if warm is None else np.asarray(warm, dtype=float).copy()
    if poc.binary_log_loss(problem.y_val, problem.logits(x0)) > risk_limit + problem.config.constraint_tolerance:
        x0 = anchor.params.copy()

    def risk_slack(params):
        return risk_limit - _risk_and_gradient(problem, params)[0]

    def risk_slack_gradient(params):
        return -_risk_and_gradient(problem, params)[1]

    result = minimize(
        problem.objective,
        x0,
        jac=problem.objective_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[equality, NonlinearConstraint(risk_slack, 0.0, np.inf, jac=risk_slack_gradient)],
        options={"maxiter": problem.config.maxiter_smooth, "ftol": problem.config.ftol_smooth, "disp": False},
    )
    final_loss = poc.binary_log_loss(problem.y_val, problem.logits(result.x))
    equality_residual = _constraint_residual(equality, result.x)
    risk_violation = max(final_loss - risk_limit, 0.0)
    success = bool(result.success and equality_residual <= problem.config.constraint_tolerance and risk_violation <= problem.config.constraint_tolerance)
    status = "ok" if success else "solver_failed"
    solution = AdaptiveSolution(
        mode=mode,
        epsilon=float(epsilon),
        status=status,
        params=result.x.copy() if success else None,
        minimum_loss=anchor.minimum_loss,
        epsilon_min=anchor.epsilon_min,
        final_loss=final_loss,
        risk_limit=risk_limit,
        risk_slack=risk_limit - final_loss,
        objective=problem.objective(result.x),
        iterations=int(result.nit),
        equality_residual=equality_residual,
        risk_violation=risk_violation,
        certified=success,
        message=str(result.message),
    )
    if success:
        materialize_solution(problem, solution)
    return solution


def solution_from_risk_anchor(problem: JointAdaptiveProblem, anchor: RiskAnchor) -> AdaptiveSolution:
    if not anchor.success:
        return AdaptiveSolution(
            anchor.mode, anchor.epsilon_min, "solver_failed", None,
            anchor.minimum_loss, anchor.epsilon_min, np.nan, anchor.minimum_loss,
            np.nan, np.nan, anchor.iterations, anchor.equality_residual, np.nan,
            False, anchor.message,
        )
    solution = AdaptiveSolution(
        mode=anchor.mode,
        epsilon=anchor.epsilon_min,
        status="risk_anchor",
        params=anchor.params.copy(),
        minimum_loss=anchor.minimum_loss,
        epsilon_min=anchor.epsilon_min,
        final_loss=anchor.minimum_loss,
        risk_limit=anchor.minimum_loss,
        risk_slack=0.0,
        objective=problem.objective(anchor.params),
        iterations=anchor.iterations,
        equality_residual=anchor.equality_residual,
        risk_violation=0.0,
        certified=True,
        message="Minimum-risk feasibility anchor; roughness was not re-optimized at this endpoint",
    )
    return materialize_solution(problem, solution)


def _mixture_map(problem: JointAdaptiveProblem, weights: np.ndarray, theta: np.ndarray, fallback: dict[tuple[str, str], np.ndarray] | None = None) -> dict[tuple[str, str], np.ndarray]:
    mixtures = {
        key: np.zeros((len(component.deltas), len(problem.dictionary)), dtype=float)
        for key, component in problem.components.items()
        if not component.is_fixed
    }
    sharp_index = _initial_kernel_index(problem.dictionary)
    for key, values in mixtures.items():
        if fallback is not None and key in fallback and fallback[key].shape == values.shape:
            values[:] = fallback[key]
        else:
            values[:, sharp_index] = 1.0
    for group in problem.groups:
        key = (group.feature, group.bound_key)
        weight = float(weights[group.weight_index])
        if weight > problem.config.zero_tolerance:
            values = np.maximum(theta[group.start : group.stop], 0.0) / weight
            values /= max(float(values.sum()), problem.config.zero_tolerance)
            mixtures[key][group.transition_index] = values
    return mixtures


def materialize_solution(
    problem: JointAdaptiveProblem,
    solution: AdaptiveSolution,
    *,
    fallback_mixtures: dict[tuple[str, str], np.ndarray] | None = None,
) -> AdaptiveSolution:
    if solution.params is None:
        return solution
    _, weights, theta = problem.unpack(solution.params)
    mixtures = _mixture_map(problem, weights, theta, fallback=fallback_mixtures)
    functions: dict[tuple[str, str], Callable] = {}
    for key, component in problem.components.items():
        if component.is_fixed:
            functions[key] = StepBoundFunction(problem.specs[key])
        else:
            if problem.basis_evaluator is None:
                functions[key] = poc.AdaptiveBoundFunction(
                    mass_map=problem.maps[key[0]],
                    base=problem.specs[key].base,
                    locations_u=component.locations_u,
                    deltas=component.deltas,
                    dictionary=problem.dictionary,
                    mixtures=mixtures[key],
                )
            else:
                groups = tuple(
                    group for group in problem.groups
                    if (group.feature, group.bound_key) == key
                )
                functions[key] = TransitionAwareBoundFunction(
                    mass_map=problem.maps[key[0]],
                    base=problem.specs[key].base,
                    groups=groups,
                    dictionary=problem.dictionary,
                    mixtures=mixtures[key],
                    evaluator=problem.basis_evaluator,
                )
    solution.functions = functions
    solution.weights = weights.copy()
    solution.theta = theta.copy()
    solution.mixtures = mixtures
    solution.intercept = problem.raw_intercept(solution.params)
    return solution


def _select_original_style_c(X: np.ndarray, y: np.ndarray, c_grid: Sequence[float], seed: int) -> float:
    del seed
    splitter = StratifiedKFold(n_splits=5, shuffle=False)
    scores: dict[float, list[float]] = {float(C): [] for C in c_grid}
    for train_index, val_index in splitter.split(X, y):
        for C in c_grid:
            weights, intercept = poc.fit_nonnegative_logistic(X[train_index], y[train_index], float(C))
            probabilities = expit(X[val_index] @ weights + intercept)
            scores[float(C)].append(float(np.mean((probabilities >= 0.5) == y[val_index])))
    return max((float(C) for C in c_grid), key=lambda C: (float(np.mean(scores[C])), -list(map(float, c_grid)).index(C)))


def post_refit_solution(
    problem: JointAdaptiveProblem,
    frozen: AdaptiveSolution,
    raw_fit: pd.DataFrame,
    raw_val: pd.DataFrame,
    *,
    target_column: str = "Outcome",
    c_grid: Sequence[float] = poc.C_GRID,
    seed: int = 0,
) -> AdaptiveSolution:
    if frozen.status not in {"ok", "risk_anchor"}:
        return AdaptiveSolution("post_refit", frozen.epsilon, frozen.status, None, frozen.minimum_loss, frozen.epsilon_min, np.nan, frozen.risk_limit, np.nan, np.nan, 0, np.nan, np.nan, False, f"Frozen prerequisite failed: {frozen.message}")
    X_fit = np.column_stack([frozen.functions[key](raw_fit[key[0]].to_numpy(dtype=float)) for key in problem.column_order])
    X_val = np.column_stack([frozen.functions[key](raw_val[key[0]].to_numpy(dtype=float)) for key in problem.column_order])
    y_fit = raw_fit[target_column].to_numpy(dtype=int)
    y_val = raw_val[target_column].to_numpy(dtype=int)
    best_C = _select_original_style_c(X_fit, y_fit, c_grid, seed)
    weights, intercept = poc.fit_nonnegative_logistic(X_fit, y_fit, best_C)
    final_loss = poc.binary_log_loss(y_val, X_val @ weights + intercept)
    risk_violation = max(final_loss - frozen.risk_limit, 0.0)

    theta = np.zeros(problem.n_theta, dtype=float)
    for group in problem.groups:
        key = (group.feature, group.bound_key)
        theta[group.start : group.stop] = weights[group.weight_index] * frozen.mixtures[key][group.transition_index]
    params = np.zeros(problem.n_variables, dtype=float)
    params[problem.weight_slice] = weights
    params[problem.theta_slice] = theta
    smooth_mean = sum(weights[index] * problem.specs[key].base for index, key in enumerate(problem.column_order) if key not in problem.fixed_keys)
    params[0] = intercept + problem.fixed_base_mean + smooth_mean + float(theta @ problem.theta_base_means)
    status = "ok" if risk_violation <= problem.config.constraint_tolerance else "uncertified"
    solution = AdaptiveSolution(
        mode="post_refit",
        epsilon=frozen.epsilon,
        status=status,
        params=params,
        minimum_loss=frozen.minimum_loss,
        epsilon_min=frozen.epsilon_min,
        final_loss=final_loss,
        risk_limit=frozen.risk_limit,
        risk_slack=frozen.risk_limit - final_loss,
        objective=float(theta @ problem.roughness_matrix @ theta),
        iterations=0,
        equality_residual=np.nan,
        risk_violation=risk_violation,
        certified=risk_violation <= problem.config.constraint_tolerance,
        message=f"Original-style five-fold accuracy CV selected C={best_C:g}",
    )
    return materialize_solution(problem, solution, fallback_mixtures=frozen.mixtures)


def evaluate_solution(
    problem: JointAdaptiveProblem,
    solution: AdaptiveSolution,
    splits: dict[str, pd.DataFrame],
    *,
    target_column: str = "Outcome",
) -> tuple[dict[str, float | str | bool], dict[str, float], dict[str, tuple[np.ndarray, np.ndarray]]]:
    row: dict[str, float | str | bool] = {
        "mode": solution.mode,
        "epsilon": solution.epsilon,
        "status": solution.status,
        "certified": solution.certified,
        "minimum_val_loss": solution.minimum_loss,
        "epsilon_min": solution.epsilon_min,
        "risk_limit": solution.risk_limit,
        "risk_slack": solution.risk_slack,
        "risk_violation": solution.risk_violation,
        "smoothness_objective": solution.objective,
        "optimizer_iterations": solution.iterations,
        "equality_residual": solution.equality_residual,
        "message": solution.message,
        "solve_seconds": solution.solve_seconds,
    }
    if solution.params is None or solution.weights is None:
        return row, {}, {}
    for split, frame in splits.items():
        logits = poc.adaptive_logits(frame, solution.functions, problem.column_order, solution.weights, solution.intercept)
        for metric, value in poc.metric_row(frame[target_column], logits).items():
            row[f"{split}_{metric}"] = value
    raw_fit = splits["train"]
    sobolev = poc.paper_feature_sobolev_norms(raw_fit, solution.functions, problem.column_order, solution.weights)
    row["total_sobolev_norm"] = float(sum(sobolev.values()))
    curves = poc.centered_feature_curves(raw_fit, solution.functions, problem.column_order, solution.weights)
    return row, sobolev, curves


def component_weight_rows(problem: JointAdaptiveProblem, solution: AdaptiveSolution) -> list[dict[str, float | str | bool]]:
    if solution.weights is None:
        return []
    rows = []
    final_totals = {
        feature: float(sum(solution.weights[index] for index, (candidate, _) in enumerate(problem.column_order) if candidate == feature))
        for feature in problem.feature_weight_totals
    }
    for index, (feature, bound_key) in enumerate(problem.column_order):
        reference_total = problem.feature_weight_totals[feature]
        final_total = final_totals[feature]
        rows.append({
            "mode": solution.mode,
            "epsilon": solution.epsilon,
            "feature": feature,
            "bound": bound_key,
            "reference_weight": float(problem.reference_weights[index]),
            "final_weight": float(solution.weights[index]),
            "weight_change": float(solution.weights[index] - problem.reference_weights[index]),
            "reference_share": float(problem.reference_weights[index] / reference_total) if reference_total > problem.config.zero_tolerance else 0.0,
            "final_share": float(solution.weights[index] / final_total) if final_total > problem.config.zero_tolerance else 0.0,
            "feature_total_reference": reference_total,
            "feature_total_final": final_total,
            "fixed_component": (feature, bound_key) in problem.fixed_keys,
        })
    return rows


def transition_rows(problem: JointAdaptiveProblem, solution: AdaptiveSolution) -> list[dict[str, float | str | int]]:
    if solution.theta is None or solution.weights is None:
        return []
    rows = []
    for group in problem.groups:
        key = (group.feature, group.bound_key)
        mixture = solution.mixtures[key][group.transition_index]
        effective_count = float(1.0 / max(np.sum(mixture**2), problem.config.zero_tolerance))
        for kernel_index, kernel in enumerate(problem.dictionary):
            row = {
                "mode": solution.mode,
                "epsilon": solution.epsilon,
                "feature": group.feature,
                "bound": group.bound_key,
                "transition_index": group.transition_index,
                "threshold_value": group.location_x,
                "location_u": group.location_u,
                "delta": group.delta,
                "family": kernel.family,
                "width": kernel.width,
                "theta": float(solution.theta[group.start + kernel_index]),
                "mixture_weight": float(mixture[kernel_index]),
                "component_weight": float(solution.weights[group.weight_index]),
                "effective_kernel_count": effective_count,
            }
            row.update(problem.transition_metadata.get(key + (group.transition_index,), {}))
            rows.append(row)
    return rows


def direct_roughness(problem: JointAdaptiveProblem, theta: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, problem.config.derivative_points)
    integration_weights = poc.trapezoid_weights(len(grid))
    total = 0.0
    for feature in problem.feature_amplitudes:
        derivative = np.zeros_like(grid)
        for group in problem.groups:
            if group.feature != feature:
                continue
            for kernel_index, kernel in enumerate(problem.dictionary):
                _, values = problem.evaluate_basis(group, kernel, grid)
                derivative += theta[group.start + kernel_index] * group.delta * values
        total += float(np.sum(derivative**2 * integration_weights) / problem.feature_amplitudes[feature] ** 2)
    return total
