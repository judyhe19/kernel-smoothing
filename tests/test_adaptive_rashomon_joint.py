import unittest

import numpy as np
import pandas as pd

from adaptive_rashomon_joint import (
    JointAdaptiveConfig,
    RiskAnchor,
    build_joint_problem,
    direct_roughness,
    initial_parameters,
    post_refit_solution,
    solve_risk_anchor,
    solve_smooth_budget,
)
import rashomon_smoothing_core as poc


def toy_problem(include_binary=True, duplicate=False, zero_feature=False):
    x_base = np.linspace(0.0, 1.0, 80)
    x_val = np.linspace(0.01, 0.99, 40)
    base = pd.DataFrame({"x": x_base, "Outcome": (x_base > 0.52).astype(int)})
    val = pd.DataFrame({"x": x_val, "Outcome": (x_val > 0.52).astype(int)})
    features = ["x"]
    if include_binary:
        base["binary"] = np.tile([0.0, 1.0], 40)
        val["binary"] = np.tile([0.0, 1.0], 20)
        features.append("binary")
    if zero_feature:
        base["zero"] = np.linspace(-1.0, 1.0, len(base))
        val["zero"] = np.linspace(-0.95, 0.95, len(val))
        features.append("zero")

    bounds = poc.BOUND_KEYS
    specs = {}
    column_order = []
    for feature in features:
        for bound_index, bound in enumerate(bounds):
            if feature == "binary":
                specs[(feature, bound)] = poc.BoundSpec(feature, bound, np.array([0.5]), np.array([-0.2, 0.3]))
            elif feature == "zero":
                specs[(feature, bound)] = poc.BoundSpec(feature, bound, np.array([-0.2, 0.4]), np.array([-0.3, 0.1, 0.4]))
            else:
                thresholds = np.array([0.35, 0.35, 0.72]) if duplicate else np.array([0.35, 0.72])
                levels = np.array([-0.8, -0.35, 0.15, 0.55]) if duplicate else np.array([-0.8, 0.15, 0.55])
                levels = levels + 0.04 * bound_index
                specs[(feature, bound)] = poc.BoundSpec(feature, bound, thresholds, levels)
            column_order.append((feature, bound))

    weights = []
    for feature in features:
        if feature == "x":
            weights.extend([0.7, 0.5, 0.4, 0.2])
        elif feature == "binary":
            weights.extend([0.1, 0.0, 0.0, 0.0])
        else:
            weights.extend([0.0, 0.0, 0.0, 0.0])
    weights = np.asarray(weights)
    column_order = tuple(column_order)
    intercept = -0.2
    raw_logits = poc.raw_design(val, specs, column_order) @ weights + intercept
    reference_loss = poc.binary_log_loss(val["Outcome"], raw_logits)
    config = JointAdaptiveConfig(
        families=("logistic", "beta33"), widths=(0.25, 0.75),
        derivative_points=101, maxiter_risk=1200, maxiter_smooth=1600,
    )
    problem = build_joint_problem(base, val, specs, column_order, weights, intercept, reference_loss, config=config)
    return base, val, problem


class JointProblemTests(unittest.TestCase):
    def test_layout_constraints_and_direct_objective(self):
        _, _, problem = toy_problem(include_binary=True, duplicate=True, zero_feature=True)
        self.assertEqual(len(problem.groups), 8)
        self.assertTrue(all(key[0] != "binary" for key in [(group.feature, group.bound_key) for group in problem.groups]))
        self.assertTrue(np.allclose(problem.roughness_matrix, problem.roughness_matrix.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(problem.roughness_matrix).min(), -1e-8)

        initial = initial_parameters(problem, "frozen")
        frozen_residual = np.max(np.abs(problem.equality_frozen.A @ initial - problem.equality_frozen.lb))
        joint_residual = np.max(np.abs(problem.equality_joint.A @ initial - problem.equality_joint.lb))
        unrestricted_residual = np.max(
            np.abs(problem.equality_unrestricted.A @ initial - problem.equality_unrestricted.lb)
        )
        self.assertLess(frozen_residual, 1e-10)
        self.assertLess(joint_residual, 1e-10)
        self.assertLess(unrestricted_residual, 1e-10)
        self.assertEqual(problem.equality_unrestricted.A.shape[0], len(problem.groups))
        self.assertEqual(problem.equality_joint.A.shape[0], len(problem.groups) + 3)
        theta = initial[problem.theta_slice]
        self.assertAlmostEqual(problem.objective(initial), direct_roughness(problem, theta), places=8)

        zero_indices = [index for index, key in enumerate(problem.column_order) if key[0] == "zero"]
        for index in zero_indices:
            self.assertEqual(problem.bounds_joint.lb[1 + index], 0.0)
            self.assertEqual(problem.bounds_joint.ub[1 + index], 0.0)

    def test_solvers_preserve_risk_totals_and_reconstruct_logits(self):
        base, val, problem = toy_problem(include_binary=True)
        frozen_anchor = solve_risk_anchor(problem, "frozen")
        self.assertTrue(frozen_anchor.success, frozen_anchor.message)
        frozen = solve_smooth_budget(problem, "frozen", 0.5, frozen_anchor)
        self.assertEqual(frozen.status, "ok", frozen.message)

        joint_anchor = solve_risk_anchor(problem, "joint", warm=frozen.params)
        self.assertTrue(joint_anchor.success, joint_anchor.message)
        joint = solve_smooth_budget(problem, "joint", 0.5, joint_anchor, warm=frozen.params)
        self.assertEqual(joint.status, "ok", joint.message)
        self.assertLessEqual(joint.final_loss, joint.risk_limit + 2e-6)
        self.assertLessEqual(joint.objective, frozen.objective + 1e-5)

        for feature, reference_total in problem.feature_weight_totals.items():
            final_total = sum(joint.weights[index] for index, key in enumerate(problem.column_order) if key[0] == feature)
            self.assertAlmostEqual(final_total, reference_total, places=6)
        self.assertGreaterEqual(joint.weights.min(), -1e-9)
        self.assertGreaterEqual(joint.theta.min(), -1e-9)
        for group in problem.groups:
            self.assertAlmostEqual(joint.theta[group.start : group.stop].sum(), joint.weights[group.weight_index], places=6)

        matrix_logits = problem.logits(joint.params)
        callable_logits = poc.adaptive_logits(val, joint.functions, problem.column_order, joint.weights, joint.intercept)
        np.testing.assert_allclose(matrix_logits, callable_logits, atol=2e-7, rtol=2e-7)

        repeated_anchor = solve_risk_anchor(problem, "joint", warm=frozen.params)
        repeated = solve_smooth_budget(problem, "joint", 0.5, repeated_anchor, warm=frozen.params)
        np.testing.assert_allclose(joint.params, repeated.params, atol=1e-7, rtol=1e-7)

        refit = post_refit_solution(problem, frozen, base, val, seed=9)
        self.assertIn(refit.status, {"ok", "uncertified"})
        self.assertEqual(refit.mode, "post_refit")

    def test_unrestricted_mode_does_not_preserve_feature_totals(self):
        _, val, problem = toy_problem(include_binary=False)
        initial = initial_parameters(problem, "unrestricted_joint")
        anchor = RiskAnchor(
            "unrestricted_joint",
            initial,
            problem.reference_loss,
            0.0,
            True,
            "reference anchor",
            0,
            0.0,
        )
        solution = solve_smooth_budget(
            problem, "unrestricted_joint", 0.5, anchor, warm=initial
        )
        self.assertEqual(solution.status, "ok", solution.message)
        self.assertGreaterEqual(solution.weights.min(), -1e-9)
        for group in problem.groups:
            self.assertAlmostEqual(
                solution.theta[group.start : group.stop].sum(),
                solution.weights[group.weight_index],
                places=6,
            )
        callable_logits = poc.adaptive_logits(
            val, solution.functions, problem.column_order, solution.weights, solution.intercept
        )
        np.testing.assert_allclose(problem.logits(solution.params), callable_logits, atol=2e-7)

    def test_infeasible_and_failed_anchor_are_explicit(self):
        _, _, problem = toy_problem(include_binary=False)
        anchor = solve_risk_anchor(problem, "joint")
        self.assertTrue(anchor.success, anchor.message)
        infeasible = solve_smooth_budget(problem, "joint", anchor.epsilon_min - 0.01, anchor)
        self.assertEqual(infeasible.status, "infeasible")
        self.assertIsNone(infeasible.params)
        self.assertFalse(infeasible.certified)

        failed_anchor = RiskAnchor("joint", anchor.params, anchor.minimum_loss, anchor.epsilon_min, False, "forced failure", 0, 1.0)
        failed = solve_smooth_budget(problem, "joint", 0.5, failed_anchor)
        self.assertEqual(failed.status, "solver_failed")
        self.assertIsNone(failed.params)


if __name__ == "__main__":
    unittest.main()
