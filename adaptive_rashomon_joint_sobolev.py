from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import BFGS, Bounds, NonlinearConstraint, minimize

import rashomon_smoothing_core as poc
from adaptive_rashomon_joint import (
    AdaptiveSolution,
    JointAdaptiveProblem,
    OptimizationMode,
    RiskAnchor,
    _bounds_for,
    _constraint_for,
    _constraint_residual,
    _risk_and_gradient,
    materialize_solution,
)

RATIO_FLOOR = 1e-14


@dataclass(frozen=True)
class SobolevTerm:
    feature: str
    indices: np.ndarray
    d_hat: np.ndarray
    n_hat: np.ndarray


@dataclass(frozen=True)
class SobolevObjective:
    terms: tuple[SobolevTerm, ...]
    n_variables: int

    def value_and_gradient(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        params = np.asarray(params, dtype=float)
        total = 0.0
        gradient = np.zeros(self.n_variables, dtype=float)
        for term in self.terms:
            v = params[term.indices]
            dv = term.d_hat @ v
            nv = term.n_hat @ v
            q = float(v @ dv)
            n = float(v @ nv)
            if n <= RATIO_FLOOR:
                continue
            ratio = max(q / n, RATIO_FLOOR)
            norm = np.sqrt(ratio)
            total += norm
            gradient[term.indices] += norm * (dv / max(q, RATIO_FLOOR * n) - nv / n)
        return total, gradient

    def value(self, params: np.ndarray) -> float:
        return self.value_and_gradient(params)[0]

    def gradient(self, params: np.ndarray) -> np.ndarray:
        return self.value_and_gradient(params)[1]

    def per_feature(self, params: np.ndarray) -> dict[str, float]:
        params = np.asarray(params, dtype=float)
        norms = {}
        for term in self.terms:
            v = params[term.indices]
            q = float(v @ term.d_hat @ v)
            n = float(v @ term.n_hat @ v)
            norms[term.feature] = 0.0 if n <= RATIO_FLOOR else float(np.sqrt(max(q / n, 0.0)))
        return norms


def build_sobolev_objective(
    problem: JointAdaptiveProblem,
    raw_fit,
    n_points: int = poc.PAPER_SOBOLEV_POINTS,
) -> SobolevObjective:
    theta_offset = 1 + problem.n_weights
    h = 1.0 / (n_points - 1)
    trapezoid = np.full(n_points, h, dtype=float)
    trapezoid[0] = trapezoid[-1] = h / 2.0

    feature_groups: dict[str, list] = {}
    for group in problem.groups:
        feature_groups.setdefault(group.feature, []).append(group)

    terms: list[SobolevTerm] = []
    for feature, groups in sorted(feature_groups.items()):
        x_grid = np.linspace(float(raw_fit[feature].min()), float(raw_fit[feature].max()), n_points)
        u_grid = problem.maps[feature](x_grid)


        component_keys = sorted(
            {(group.feature, group.bound_key) for group in groups},
            key=lambda key: key[1],
        )
        weight_columns = {
            key: column for column, key in enumerate(component_keys)
        }
        indices = [1 + problem.components[key].weight_index for key in component_keys]
        columns: list[np.ndarray] = [
            np.full(n_points, problem.specs[key].base, dtype=float) for key in component_keys
        ]
        for group in groups:
            for kernel_index, kernel in enumerate(problem.dictionary):
                basis, _ = problem.evaluate_basis(group, kernel, u_grid)
                columns.append(group.delta * basis)
                indices.append(theta_offset + group.start + kernel_index)

        value_matrix = np.column_stack(columns)
        derivative_rows = (value_matrix[2:] - value_matrix[:-2]) / (2.0 * h)
        d_hat = derivative_rows.T @ derivative_rows * h
        n_hat = value_matrix.T @ (value_matrix * trapezoid[:, None])
        d_hat = 0.5 * (d_hat + d_hat.T)
        n_hat = 0.5 * (n_hat + n_hat.T)
        del weight_columns
        terms.append(
            SobolevTerm(
                feature=feature,
                indices=np.asarray(indices, dtype=int),
                d_hat=d_hat,
                n_hat=n_hat,
            )
        )
    return SobolevObjective(terms=tuple(terms), n_variables=problem.n_variables)


def solve_sobolev_budget(
    problem: JointAdaptiveProblem,
    objective: SobolevObjective,
    mode: OptimizationMode,
    epsilon: float,
    anchor: RiskAnchor,
    *,
    warm: np.ndarray | None = None,
) -> AdaptiveSolution:
    if mode == "unrestricted_joint":
        raise NotImplementedError("Sobolev-objective solves support frozen/joint modes")
    risk_limit = problem.reference_loss + float(epsilon)
    if not anchor.success:
        return AdaptiveSolution(mode, epsilon, "solver_failed", None, anchor.minimum_loss, anchor.epsilon_min, np.nan, risk_limit, np.nan, np.nan, anchor.iterations, anchor.equality_residual, np.nan, False, anchor.message)
    if risk_limit < anchor.minimum_loss - problem.config.feasibility_tolerance:
        return AdaptiveSolution(mode, epsilon, "infeasible", None, anchor.minimum_loss, anchor.epsilon_min, np.nan, risk_limit, risk_limit - anchor.minimum_loss, np.nan, 0, anchor.equality_residual, anchor.minimum_loss - risk_limit, False, "Requested risk budget is below the measured feasible minimum")

    equality = _constraint_for(problem, mode)
    bounds: Bounds = _bounds_for(problem, mode)
    x0 = anchor.params.copy() if warm is None else np.asarray(warm, dtype=float).copy()
    if poc.binary_log_loss(problem.y_val, problem.logits(x0)) > risk_limit + problem.config.constraint_tolerance:
        x0 = anchor.params.copy()

    def risk_slack(params):
        return risk_limit - _risk_and_gradient(problem, params)[0]

    def risk_slack_gradient(params):
        return -_risk_and_gradient(problem, params)[1]

    result = minimize(
        objective.value,
        x0,
        jac=objective.gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[equality, NonlinearConstraint(risk_slack, 0.0, np.inf, jac=risk_slack_gradient)],
        options={"maxiter": problem.config.maxiter_smooth, "ftol": problem.config.ftol_smooth, "disp": False},
    )
    final_loss = poc.binary_log_loss(problem.y_val, problem.logits(result.x))
    equality_residual = _constraint_residual(equality, result.x)
    risk_violation = max(final_loss - risk_limit, 0.0)
    feasible = bool(
        equality_residual <= problem.config.constraint_tolerance
        and risk_violation <= problem.config.constraint_tolerance
    )
    success = bool(result.success and feasible)


    if feasible:
        status = "ok" if success else "uncertified"
    else:
        status = "solver_failed"
    solution = AdaptiveSolution(
        mode=mode,
        epsilon=float(epsilon),
        status=status,
        params=result.x.copy() if feasible else None,
        minimum_loss=anchor.minimum_loss,
        epsilon_min=anchor.epsilon_min,
        final_loss=final_loss,
        risk_limit=risk_limit,
        risk_slack=risk_limit - final_loss,
        objective=objective.value(result.x),
        iterations=int(result.nit),
        equality_residual=equality_residual,
        risk_violation=risk_violation,
        certified=success,
        message=str(result.message),
    )
    if feasible:
        materialize_solution(problem, solution)
    return solution


def solve_sobolev_budget_trust(
    problem: JointAdaptiveProblem,
    objective: SobolevObjective,
    mode: OptimizationMode,
    epsilon: float,
    anchor: RiskAnchor,
    *,
    warm: np.ndarray | None = None,
    maxiter: int = 2000,
    gtol: float = 1e-7,
    xtol: float = 1e-10,
) -> AdaptiveSolution:
    if mode == "unrestricted_joint":
        raise NotImplementedError("Sobolev-objective solves support frozen/joint modes")
    risk_limit = problem.reference_loss + float(epsilon)
    if not anchor.success:
        return AdaptiveSolution(mode, epsilon, "solver_failed", None, anchor.minimum_loss, anchor.epsilon_min, np.nan, risk_limit, np.nan, np.nan, anchor.iterations, anchor.equality_residual, np.nan, False, anchor.message)
    if risk_limit < anchor.minimum_loss - problem.config.feasibility_tolerance:
        return AdaptiveSolution(mode, epsilon, "infeasible", None, anchor.minimum_loss, anchor.epsilon_min, np.nan, risk_limit, risk_limit - anchor.minimum_loss, np.nan, 0, anchor.equality_residual, anchor.minimum_loss - risk_limit, False, "Requested risk budget is below the measured feasible minimum")

    equality = _constraint_for(problem, mode)
    bounds: Bounds = _bounds_for(problem, mode)
    x0 = anchor.params.copy() if warm is None else np.asarray(warm, dtype=float).copy()
    if poc.binary_log_loss(problem.y_val, problem.logits(x0)) > risk_limit + problem.config.constraint_tolerance:
        x0 = anchor.params.copy()

    def risk(params):
        return _risk_and_gradient(problem, params)[0]

    def risk_gradient(params):
        return _risk_and_gradient(problem, params)[1]

    result = minimize(
        objective.value,
        x0,
        jac=objective.gradient,
        hess=BFGS(),
        method="trust-constr",
        bounds=bounds,
        constraints=[
            equality,
            NonlinearConstraint(risk, -np.inf, risk_limit, jac=risk_gradient, hess=BFGS()),
        ],
        options={"maxiter": maxiter, "gtol": gtol, "xtol": xtol, "verbose": 0},
    )
    final_loss = poc.binary_log_loss(problem.y_val, problem.logits(result.x))
    equality_residual = _constraint_residual(equality, result.x)
    risk_violation = max(final_loss - risk_limit, 0.0)
    feasible = bool(
        equality_residual <= problem.config.constraint_tolerance
        and risk_violation <= problem.config.constraint_tolerance
    )
    converged = result.status in (1, 2)
    success = bool(converged and feasible)
    if feasible:
        status = "ok" if success else "uncertified"
    else:
        status = "solver_failed"
    solution = AdaptiveSolution(
        mode=mode,
        epsilon=float(epsilon),
        status=status,
        params=result.x.copy() if feasible else None,
        minimum_loss=anchor.minimum_loss,
        epsilon_min=anchor.epsilon_min,
        final_loss=final_loss,
        risk_limit=risk_limit,
        risk_slack=risk_limit - final_loss,
        objective=objective.value(result.x),
        iterations=int(result.nit),
        equality_residual=equality_residual,
        risk_violation=risk_violation,
        certified=success,
        message=f"trust-constr status={result.status}: {result.message}",
    )
    if feasible:
        materialize_solution(problem, solution)
    return solution


def check_objective_matches_metric(
    problem: JointAdaptiveProblem,
    objective: SobolevObjective,
    solution: AdaptiveSolution,
    raw_fit,
) -> dict[str, tuple[float, float]]:
    if solution.params is None or solution.weights is None:
        return {}
    reported = poc.paper_feature_sobolev_norms(
        raw_fit, solution.functions, problem.column_order, solution.weights
    )
    predicted = objective.per_feature(solution.params)
    return {feature: (predicted[feature], reported[feature]) for feature in predicted}


def elapsed(solution: AdaptiveSolution, started: float) -> AdaptiveSolution:
    solution.solve_seconds = time.perf_counter() - started
    return solution
