from evaluation.bayes_evaluator import BinaryBayesEvaluator
from models.mvg_binary_classifiers import *


class MVGEvaluator:

    

    @staticmethod
    def evaluate_mvgs(
        applications,
        mvg_experiments,
        nb_experiments,
        tied_experiments,
        verbose=False,
    ):
        if verbose:
            print("-" * 40)
            print("\nMVG results")

        app_to_mvg_results = BinaryBayesEvaluator.evaluate_models_on_applications(
            mvg_experiments, applications, verbose=verbose
        )

        if verbose:
            print("-" * 40)
            print("\nNB results")

        app_to_nb_results = BinaryBayesEvaluator.evaluate_models_on_applications(
            nb_experiments, applications, verbose=verbose
        )
        if verbose:
            print("-" * 40)
            print("\nTIED results")

        app_to_tied_results = BinaryBayesEvaluator.evaluate_models_on_applications(
            tied_experiments, applications, verbose=verbose
        )
        print("-" * 40)
        return (
            app_to_mvg_results,
            app_to_nb_results,
            app_to_tied_results,
        )
