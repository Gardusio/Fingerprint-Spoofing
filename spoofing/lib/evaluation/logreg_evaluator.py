from lib.evaluation.bayes_evaluator import BinaryBayesEvaluator


class LogRegEvaluator:

    @staticmethod
    def evaluate_logregs(
        applications,
        logreg_experiments,
        weighted_experiments,
        quadratic_experiments,
        verbose=False,
    ):
        if verbose:
            print("-" * 40)
            print("\nLinear logreg results")

        app_to_logreg_results = BinaryBayesEvaluator.evaluate_models_on_applications(
            logreg_experiments, applications, verbose=verbose
        )

        if verbose:
            print("-" * 40)
            print("\nPrior Weighted logreg results")

        app_to_w_logreg_results = BinaryBayesEvaluator.evaluate_models_on_applications(
            weighted_experiments, applications, verbose=verbose
        )
        if verbose:
            print("-" * 40)
            print("\nQuadratic logreg results")

        app_to_q_logreg_results = BinaryBayesEvaluator.evaluate_models_on_applications(
            quadratic_experiments, applications, verbose=verbose
        )
        print("-" * 40)
        return (
            app_to_logreg_results,
            app_to_w_logreg_results,
            app_to_q_logreg_results,
        )
