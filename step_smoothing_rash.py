import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle


class StepFunctionPassthrough:

    is_binary = True

    def __init__(self, x_bounds, y_coefficients):
        self.x_bounds = np.asarray(x_bounds, dtype=float)
        self.y_coefficients = np.asarray(y_coefficients, dtype=float)

    def __call__(self, x):
        values = np.asarray(x, dtype=float)
        scalar = values.ndim == 0
        values = np.atleast_1d(values)
        indices = np.searchsorted(self.x_bounds, values, side="left")
        result = self.y_coefficients[np.clip(indices, 0, len(self.y_coefficients) - 1)]
        return float(result[0]) if scalar else result


RAMP_FAMILY = "logistic"
_BETA_PARAMETERS = {"beta33": (3.0, 3.0), "beta24": (2.0, 4.0), "beta42": (4.0, 2.0)}


def ramp_cdf(x, midpoint, k, family="logistic"):
    from scipy.special import expit, ndtr, ndtri, betainc, betaincinv
    x = np.asarray(x, dtype=float)
    if family == "logistic":
        return 1.0 / (1.0 + np.exp(-k * (x - midpoint)))
    width_10_90 = 2.0 * np.log(9.0) / k
    if family == "gaussian":
        sigma = width_10_90 / (2.0 * ndtri(0.9))
        return ndtr((x - midpoint) / sigma)
    if family in _BETA_PARAMETERS:
        a, b = _BETA_PARAMETERS[family]
        q10, median, q90 = (float(betaincinv(a, b, q)) for q in (0.1, 0.5, 0.9))
        span = width_10_90 / (q90 - q10)
        left = midpoint - median * span
        v = np.clip((x - left) / span, 0.0, 1.0)
        return betainc(a, b, v)
    raise ValueError(f"unknown ramp family {family!r}")


def detect_steps_and_flats_filtered(x_bounds, y_coefficients, tolerance=1e-6, min_flat_ratio=0.1):
    x_bounds = np.array(x_bounds)
    y_coefficients = np.array(y_coefficients)


    domain_width = x_bounds.max() - x_bounds.min()
    min_flat_width = domain_width * min_flat_ratio


    transitions = []
    flats = []

    i = 0
    while i < len(x_bounds) - 1:

        if abs(y_coefficients[i+1] - y_coefficients[i]) > tolerance:

            transitions.append((x_bounds[i], x_bounds[i+1],
                              y_coefficients[i], y_coefficients[i+1]))
            i += 1
        else:

            flat_start = i
            flat_y = y_coefficients[i]


            while (i < len(x_bounds) - 1 and
                   abs(y_coefficients[i+1] - flat_y) <= tolerance):
                i += 1


            flat_width = x_bounds[i] - x_bounds[flat_start]
            flats.append((x_bounds[flat_start], x_bounds[i], flat_y, flat_width))


    filtered_flats = []
    skipped_flats = []

    for idx, (flat_start, flat_end, flat_y, flat_width) in enumerate(flats):

        is_first = (idx == 0)
        is_last = (idx == len(flats) - 1)

        if flat_width >= min_flat_width or is_first or is_last:
            filtered_flats.append((flat_start, flat_end, flat_y))
        else:
            skipped_flats.append((flat_start, flat_end, flat_y, flat_width))


    filtered_transitions = []

    if len(skipped_flats) > 0:


        filtered_flats.sort(key=lambda x: x[0])


        for i in range(len(filtered_flats) - 1):
            current_flat = filtered_flats[i]
            next_flat = filtered_flats[i + 1]


            transition_start = current_flat[1]
            transition_end = next_flat[0]
            start_y = current_flat[2]
            end_y = next_flat[2]

            filtered_transitions.append((transition_start, transition_end, start_y, end_y))
    else:

        filtered_transitions = transitions

    return filtered_transitions, filtered_flats


class ImprovedAlignedLogisticStepFunction:

    def __init__(self, x_bounds, y_coefficients, steepness=1, min_flat_ratio=0.1, ramp=None):
        self.x_bounds = np.array(x_bounds)
        self.y_coefficients = np.array(y_coefficients)
        self.steepness = steepness
        self.min_flat_ratio = min_flat_ratio
        self.ramp_family = ramp if ramp is not None else RAMP_FAMILY


        self.domain_width = self.x_bounds.max() - self.x_bounds.min()
        self.domain_min = self.x_bounds.min()
        self.domain_max = self.x_bounds.max()


        self.transitions, self.flats = detect_steps_and_flats_filtered(
            x_bounds, y_coefficients, min_flat_ratio=min_flat_ratio
        )


        self._create_aligned_logistic()

    def _create_aligned_logistic(self):
        if len(self.transitions) == 0:

            self.constant_value = self.y_coefficients[0]
            self.is_constant = True
            return

        self.is_constant = False


        if len(self.flats) > 0:
            self.base_value = self.flats[0][2]
        else:
            self.base_value = self.y_coefficients[0]


        self.logistic_components = []

        for i, (start_x, end_x, start_y, end_y) in enumerate(self.transitions):

            amplitude = end_y - start_y


            step_location = (start_x + end_x) / 2


            step_width = end_x - start_x
            if step_width > 0:
                k = self.steepness / step_width
            else:
                k = self.steepness / 0.01

            self.logistic_components.append({
                'amplitude': amplitude,
                'midpoint': step_location,
                'steepness': k
            })

    def __call__(self, x):
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)

        if self.is_constant:
            y = np.full_like(x, self.constant_value, dtype=np.float64)
        else:

            y = np.full_like(x, self.base_value, dtype=np.float64)


            for component in self.logistic_components:
                amplitude = component['amplitude']
                midpoint = component['midpoint']
                k = component['steepness']


                logistic_term = amplitude * ramp_cdf(x, midpoint, k, self.ramp_family)
                y += logistic_term

        return y[0] if scalar_input else y


class StepSmoothingTrainer:

    def __init__(self, aligned_logistic_data, intercept, features):
        self.aligned_logistic_data = aligned_logistic_data
        self.intercept = intercept
        self.features = features


        self.best_min_flat_ratio = None
        self.last_min_flat_ratio = None
        self.best_results = None
        self.cv_result = None


        self.steepness_multipliers = {feature: 1.0 for feature in features}
        self.best_steepness_multipliers = None

    def _build_functions(self, min_flat_ratio, steepness_multipliers=None):
        if steepness_multipliers is None:
            steepness_multipliers = self.steepness_multipliers

        funcs = {}
        for feature in self.features:
            data = self.aligned_logistic_data[feature]
            bounds = data['bounds']
            coefficients = data['coefficients']


            base_steepness = 1 / (max(bounds) - min(bounds)) * 100
            multiplier = steepness_multipliers.get(feature, 1.0)
            steepness = base_steepness * multiplier

            func = ImprovedAlignedLogisticStepFunction(
                bounds, coefficients,
                steepness=steepness,
                min_flat_ratio=min_flat_ratio
            )
            funcs[feature] = func
        return funcs

    def _predict_sample(self, sample_row, funcs):
        outcome = self.intercept
        for feature in self.features:
            value = float(sample_row[feature])
            outcome += funcs[feature](value)
        return outcome

    def _predict_proba(self, logit):
        return np.exp(logit) / (1 + np.exp(logit))

    def _evaluate_on_data(self, X, y, funcs, threshold=0.5):
        y_preds = []
        y_probs = []

        for idx, row in X.iterrows():
            logit = self._predict_sample(row, funcs)
            prob = self._predict_proba(logit)
            y_preds.append(1 if prob > threshold else 0)
            y_probs.append(prob)

        y_true = np.array(y.values if hasattr(y, 'values') else y)
        y_preds = np.array(y_preds)
        y_probs = np.array(y_probs)

        accuracy = np.mean(y_preds == y_true)

        try:
            auc = roc_auc_score(y_true, y_probs)
        except ValueError:
            auc = 0.5

        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_preds,
            'probabilities': y_probs
        }

    def _evaluate_across_folds(self, min_flat_ratio, fold_data):
        funcs = self._build_functions(min_flat_ratio)

        val_accuracies = []
        val_aucs = []
        test_accuracies = []
        test_aucs = []

        for fold in fold_data:

            val_result = self._evaluate_on_data(
                fold['X_val'], fold['y_val'], funcs
            )
            val_accuracies.append(val_result['accuracy'])
            val_aucs.append(val_result['auc'])


            test_result = self._evaluate_on_data(
                fold['X_test'], fold['y_test'], funcs
            )
            test_accuracies.append(test_result['accuracy'])
            test_aucs.append(test_result['auc'])

        return {
            'val_acc_mean': np.mean(val_accuracies),
            'val_acc_std': np.std(val_accuracies),
            'val_auc_mean': np.mean(val_aucs),
            'val_auc_std': np.std(val_aucs),
            'test_acc_mean': np.mean(test_accuracies),
            'test_acc_std': np.std(test_accuracies),
            'test_auc_mean': np.mean(test_aucs),
            'test_auc_std': np.std(test_aucs),
            'val_accuracies': val_accuracies,
            'val_aucs': val_aucs,
            'test_accuracies': test_accuracies,
            'test_aucs': test_aucs
        }

    def _tune_per_feature_steepness(self, fold_data, min_flat_ratio, metric='accuracy',
                                      multiplier_candidates=None, max_iterations=3, verbose=True):
        if multiplier_candidates is None:
            multiplier_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]


        current_multipliers = self.steepness_multipliers.copy()

        if verbose:
            print("\n" + "=" * 80)
            print("PER-FEATURE STEEPNESS TUNING")
            print("=" * 80)
            print(f"Multiplier candidates: {multiplier_candidates}")
            print(f"Iterations: {max_iterations}")


        baseline_results = self._evaluate_across_folds(min_flat_ratio, fold_data)
        baseline_metric = baseline_results['val_acc_mean'] if metric == 'accuracy' else baseline_results['val_auc_mean']

        if verbose:
            print(f"\nBaseline (all multipliers = 1.0): {metric} = {baseline_metric:.4f}")

        best_overall_metric = baseline_metric
        best_overall_multipliers = current_multipliers.copy()

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            for feature in self.features:
                if verbose:
                    print(f"\n  Tuning '{feature}':")

                best_multiplier = current_multipliers[feature]
                best_metric_value = best_overall_metric

                for mult in multiplier_candidates:

                    test_multipliers = current_multipliers.copy()
                    test_multipliers[feature] = mult


                    funcs = self._build_functions(min_flat_ratio, test_multipliers)


                    results = self._evaluate_across_folds_with_funcs(funcs, fold_data)
                    current_metric = results['val_acc_mean'] if metric == 'accuracy' else results['val_auc_mean']

                    if verbose:
                        marker = " *" if current_metric > best_metric_value else ""
                        print(f"    multiplier={mult:.2f}: {metric}={current_metric:.4f}{marker}")

                    if current_metric > best_metric_value:
                        best_metric_value = current_metric
                        best_multiplier = mult


                current_multipliers[feature] = best_multiplier

                if verbose:
                    print(f"    → Best for '{feature}': {best_multiplier:.2f}")


                if best_metric_value > best_overall_metric:
                    best_overall_metric = best_metric_value
                    best_overall_multipliers = current_multipliers.copy()

        if verbose:
            print("\n" + "=" * 80)
            print("STEEPNESS TUNING RESULTS")
            print("=" * 80)
            print(f"Final {metric}: {best_overall_metric:.4f} (baseline: {baseline_metric:.4f})")
            print(f"Improvement: {best_overall_metric - baseline_metric:+.4f}")
            print("\nOptimal multipliers per feature:")
            for feature, mult in best_overall_multipliers.items():
                changed = " (changed)" if mult != 1.0 else ""
                print(f"  {feature}: {mult:.2f}{changed}")

        return best_overall_multipliers

    def _evaluate_across_folds_with_funcs(self, funcs, fold_data):
        val_accuracies = []
        val_aucs = []
        test_accuracies = []
        test_aucs = []

        for fold in fold_data:

            val_result = self._evaluate_on_data(
                fold['X_val'], fold['y_val'], funcs
            )
            val_accuracies.append(val_result['accuracy'])
            val_aucs.append(val_result['auc'])


            test_result = self._evaluate_on_data(
                fold['X_test'], fold['y_test'], funcs
            )
            test_accuracies.append(test_result['accuracy'])
            test_aucs.append(test_result['auc'])

        return {
            'val_acc_mean': np.mean(val_accuracies),
            'val_acc_std': np.std(val_accuracies),
            'val_auc_mean': np.mean(val_aucs),
            'val_auc_std': np.std(val_aucs),
            'test_acc_mean': np.mean(test_accuracies),
            'test_acc_std': np.std(test_accuracies),
            'test_auc_mean': np.mean(test_aucs),
            'test_auc_std': np.std(test_aucs),
            'val_accuracies': val_accuracies,
            'val_aucs': val_aucs,
            'test_accuracies': test_accuracies,
            'test_aucs': test_aucs
        }

    def _collect_all_steps(self):
        all_steps = []

        for feature, data in self.aligned_logistic_data.items():
            if feature not in self.features:
                continue

            bounds = data['bounds']
            coefficients = data['coefficients']

            domain_width = max(bounds) - min(bounds)


            raw_bounds = [bounds[0]]
            raw_coefficients = [coefficients[0]]

            for i in range(1, len(bounds)):
                if bounds[i] - bounds[i-1] > 0.1:
                    raw_bounds.append(bounds[i])
                    raw_coefficients.append(coefficients[i])


            for i in range(1, len(raw_bounds)):
                all_steps.append({
                    'feature': feature,
                    'width': (raw_bounds[i] - raw_bounds[i-1]) / domain_width,
                    'start_x': raw_bounds[i-1],
                    'end_x': raw_bounds[i],
                    'start_y': raw_coefficients[i-1],
                    'end_y': raw_coefficients[i],
                    'amplitude': abs(raw_coefficients[i] - raw_coefficients[i-1])
                })

        return all_steps

    def train(self, fold_data, accuracy_threshold=0.02, metric='accuracy', verbose=True,
              tune_steepness=False, steepness_multiplier_candidates=None, steepness_max_iterations=3):
        if verbose:
            print("=" * 80)
            print("ADAPTIVE STEP ELIMINATION TRAINING (K-Fold Cross-Validation)")
            print("=" * 80)
            print(f"\nUsing {len(fold_data)} folds")


        all_steps = self._collect_all_steps()
        all_steps_sorted = sorted(all_steps, key=lambda x: x['width'])

        if verbose:
            print(f"\nFound {len(all_steps_sorted)} total steps across all features:")
            for i, step in enumerate(all_steps_sorted):
                print(f"   {i+1}. {step['feature']:25s} | width={step['width']:.4f} | "
                      f"amplitude={step['amplitude']:.4f}")


        unique_widths = sorted(set(step['width'] for step in all_steps_sorted))


        min_flat_ratio_candidates = [0.0] + [w + 0.001 for w in unique_widths]

        if verbose:
            print(f"\nTesting {len(min_flat_ratio_candidates)} min_flat_ratio thresholds...")


        baseline_results = self._evaluate_across_folds(0.0, fold_data)

        baseline_metric = baseline_results['val_acc_mean'] if metric == 'accuracy' else baseline_results['val_auc_mean']

        if verbose:
            print(f"\nBaseline (all steps, min_flat_ratio=0.0):")
            print(f"   Avg Validation Accuracy: {baseline_results['val_acc_mean']:.4f} ± {baseline_results['val_acc_std']:.4f}")
            print(f"   Avg Validation AUC:      {baseline_results['val_auc_mean']:.4f} ± {baseline_results['val_auc_std']:.4f}")
            print(f"   Avg Test Accuracy:       {baseline_results['test_acc_mean']:.4f} ± {baseline_results['test_acc_std']:.4f}")
            print(f"   Avg Test AUC:            {baseline_results['test_auc_mean']:.4f} ± {baseline_results['test_auc_std']:.4f}")


        history = []
        best_min_flat_ratio = 0.0
        best_results = baseline_results
        steps_eliminated = 0

        if verbose:
            print(f"\nIterating through min_flat_ratio thresholds (stopping if avg {metric} drops > {accuracy_threshold:.2%}):")
            print("-" * 80)
            print(f"{'min_flat_ratio':>14} | {'eliminated':>10} | {'val_acc':>12} | {'val_auc':>12} | {'test_acc':>12} | {'test_auc':>12}")
            print("-" * 80)

        for min_flat_ratio in min_flat_ratio_candidates:

            eliminated = sum(1 for s in all_steps_sorted if s['width'] < min_flat_ratio)


            results = self._evaluate_across_folds(min_flat_ratio, fold_data)

            history.append({
                'min_flat_ratio': min_flat_ratio,
                'steps_eliminated': eliminated,
                **results
            })

            current_metric = results['val_acc_mean'] if metric == 'accuracy' else results['val_auc_mean']

            if verbose:
                print(f"{min_flat_ratio:14.4f} | {eliminated:10d} | "
                      f"{results['val_acc_mean']:.4f}±{results['val_acc_std']:.2f} | "
                      f"{results['val_auc_mean']:.4f}±{results['val_auc_std']:.2f} | "
                      f"{results['test_acc_mean']:.4f}±{results['test_acc_std']:.2f} | "
                      f"{results['test_auc_mean']:.4f}±{results['test_auc_std']:.2f}")


            if baseline_metric - current_metric > accuracy_threshold:
                if verbose:
                    print("-" * 80)
                    print(f"\nWarning: Avg {metric} dropped by {(baseline_metric - current_metric):.4f} "
                          f"(threshold: {accuracy_threshold:.4f})")
                    print(f"   Stopping at previous min_flat_ratio: {best_min_flat_ratio:.4f}")
                break

            self.last_min_flat_ratio = min_flat_ratio


            best_metric = best_results['val_acc_mean'] if metric == 'accuracy' else best_results['val_auc_mean']

            if current_metric >= best_metric:
                best_min_flat_ratio = min_flat_ratio
                best_results = results
                steps_eliminated = eliminated

        if verbose:
            print("\n" + "=" * 80)
            print("FINAL RESULTS")
            print("=" * 80)
            print(f"   Best min_flat_ratio: {best_min_flat_ratio:.4f}")
            print(f"   Steps eliminated: {steps_eliminated} / {len(all_steps_sorted)}")
            print(f"\n   Validation Performance:")
            print(f"      Accuracy: {best_results['val_acc_mean']:.4f} ± {best_results['val_acc_std']:.4f} "
                  f"(baseline: {baseline_results['val_acc_mean']:.4f})")
            print(f"      AUC:      {best_results['val_auc_mean']:.4f} ± {best_results['val_auc_std']:.4f} "
                  f"(baseline: {baseline_results['val_auc_mean']:.4f})")
            print(f"\n   Test Performance:")
            print(f"      Accuracy: {best_results['test_acc_mean']:.4f} ± {best_results['test_acc_std']:.4f}")
            print(f"      AUC:      {best_results['test_auc_mean']:.4f} ± {best_results['test_auc_std']:.4f}")


            print(f"\n   Per-Fold Test Results at best min_flat_ratio:")
            for i, (acc, auc) in enumerate(zip(best_results['test_accuracies'], best_results['test_aucs'])):
                print(f"      Fold {i+1}: Accuracy={acc:.4f}, AUC={auc:.4f}")


        self.best_min_flat_ratio = best_min_flat_ratio
        self.best_results = best_results
        final_results = best_results


        if tune_steepness:
            self.best_steepness_multipliers = self._tune_per_feature_steepness(
                fold_data=fold_data,
                min_flat_ratio=best_min_flat_ratio,
                metric=metric,
                multiplier_candidates=steepness_multiplier_candidates,
                max_iterations=steepness_max_iterations,
                verbose=verbose
            )
            self.steepness_multipliers = self.best_steepness_multipliers.copy()


            final_funcs = self._build_functions(best_min_flat_ratio, self.best_steepness_multipliers)
            final_results = self._evaluate_across_folds_with_funcs(final_funcs, fold_data)

            if verbose:
                print("\n" + "=" * 80)
                print("FINAL PERFORMANCE (with tuned steepness)")
                print("=" * 80)
                print(f"   Validation Accuracy: {final_results['val_acc_mean']:.4f} ± {final_results['val_acc_std']:.4f}")
                print(f"   Validation AUC:      {final_results['val_auc_mean']:.4f} ± {final_results['val_auc_std']:.4f}")
                print(f"   Test Accuracy:       {final_results['test_acc_mean']:.4f} ± {final_results['test_acc_std']:.4f}")
                print(f"   Test AUC:            {final_results['test_auc_mean']:.4f} ± {final_results['test_auc_std']:.4f}")

        else:
            self.best_steepness_multipliers = self.steepness_multipliers.copy()

        self.cv_result = {
            'best_min_flat_ratio': best_min_flat_ratio,
            'last_min_flat_ratio': self.last_min_flat_ratio,
            'baseline_results': baseline_results,
            'optimal_min_flat_ratio_results': best_results,
            'steepness_tuned_results': final_results,
            'steps_eliminated': steps_eliminated,
            'total_steps': len(all_steps_sorted),
            'all_steps': all_steps_sorted,
            'history': history,
            'steepness_multipliers': self.best_steepness_multipliers,
            'steepness_tuned': tune_steepness
        }

        return self.cv_result

    def get_optimal_functions(self, min_flat_ratio=None, steepness_multipliers=None):
        if min_flat_ratio is None:
            if self.best_min_flat_ratio is None:
                raise ValueError("No best_min_flat_ratio found. Run train() first or provide min_flat_ratio.")
            min_flat_ratio = self.best_min_flat_ratio

        if steepness_multipliers is None:
            steepness_multipliers = self.best_steepness_multipliers

        return self._build_functions(min_flat_ratio, steepness_multipliers)

    def predict(self, X, min_flat_ratio=None, return_proba=False):
        funcs = self.get_optimal_functions(min_flat_ratio)

        predictions = []
        probabilities = []

        for idx, row in X.iterrows():
            logit = self._predict_sample(row, funcs)
            prob = self._predict_proba(logit)
            predictions.append(1 if prob > 0.5 else 0)
            probabilities.append(prob)

        if return_proba:
            return np.array(probabilities)
        return np.array(predictions)

    def export(self, filename):
        if self.cv_result is None:
            raise ValueError("No results to export. Run train() first.")

        optimal_funcs = self.get_optimal_functions()

        export_data = {
            'functions': optimal_funcs,
            'min_flat_ratio': self.best_min_flat_ratio,
            'steepness_multipliers': self.best_steepness_multipliers,
            'intercept': self.intercept,
            'features': self.features,
            'cv_result': self.cv_result,
            'aligned_logistic_data': self.aligned_logistic_data
        }

        with open(filename, 'wb') as f:
            pickle.dump(export_data, f)

        print(f"Exported model to '{filename}'")


def create_fold_data_from_dataframe(data, target_column, n_folds=5, val_ratio=0.2, random_state=42):
    from sklearn.model_selection import KFold

    X = data.drop(columns=[target_column])
    y = data[target_column]

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_data = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train_full = X.iloc[train_idx]
        y_train_full = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]


        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_ratio, random_state=random_state
        )

        fold_data.append({
            'fold': fold_idx + 1,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        })

    return fold_data


def create_fold_data_from_splits(train_test_pairs, val_ratio=0.2, random_state=42):
    fold_data = []

    for fold_idx, (train_df, test_df, target_column) in enumerate(train_test_pairs):
        X_train_full = train_df.drop(columns=[target_column])
        y_train_full = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]


        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_ratio, random_state=random_state
        )

        fold_data.append({
            'fold': fold_idx + 1,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        })

    return fold_data


def compute_sobolev_semi_norm(func, x_min, x_max, n_points=200):
    domain_length = x_max - x_min

    if domain_length == 0:
        return np.nan


    x_vals = np.linspace(x_min, x_max, n_points)


    f_vals = func(x_vals)


    f_range = np.max(f_vals) - np.min(f_vals)
    if f_range < 1e-10:
        return 0.0


    h_norm = 1.0 / (n_points - 1)


    f_squared = f_vals ** 2

    l2_norm_sq = h_norm * (f_squared[0]/2 + np.sum(f_squared[1:-1]) + f_squared[-1]/2)
    l2_norm = np.sqrt(l2_norm_sq)

    if l2_norm < 1e-10:
        return 0.0


    f_vals_normalized = f_vals / l2_norm


    f_first_deriv = np.zeros(n_points - 2)
    for j in range(1, n_points - 1):
        f_first_deriv[j - 1] = (f_vals_normalized[j + 1] - f_vals_normalized[j - 1]) / (2 * h_norm)


    integrand = f_first_deriv ** 2
    sobolev_seminorm_sq = np.sum(integrand) * h_norm

    return np.sqrt(sobolev_seminorm_sq)
