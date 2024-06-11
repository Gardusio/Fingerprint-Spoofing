from evaluation.bayes_evaluator import BinaryBayesEvaluator
from models.mvg_binary_classifiers import *


class MVGEvaluator:

    def evaluate_mvgs(
        self,
        applications,
        mvg_experiments,
        nb_experiments,
        tied_experiments,
        verbose=False,
    ):
        if verbose:
            print("-" * 40)
            print("\nMVG results")

        app_to_mvg_results = self.evaluate_models_on_applications(
            mvg_experiments, applications, verbose=verbose
        )

        if verbose:
            print("-" * 40)
            print("\nNB results")

        app_to_nb_results = self.evaluate_models_on_applications(
            nb_experiments, applications, verbose=verbose
        )
        if verbose:
            print("-" * 40)
            print("\nTIED results")

        app_to_tied_results = self.evaluate_models_on_applications(
            tied_experiments, applications, verbose=verbose
        )
        print("-" * 40)
        return (
            app_to_mvg_results,
            app_to_nb_results,
            app_to_tied_results,
        )

    def evaluate_models_on_applications(self, experiments, applications, verbose=False):
        application_to_models_results = {
            application.get_name(): [] for application in applications
        }

        application_to_best_results = {
            application.get_name(): [] for application in applications
        }

        for application in applications:
            app_name = application.get_name()
            for experiment in experiments:
                model, x_val, y_val = experiment

                model_result = self.evaluate_on_application(
                    classifier=model,
                    application=application,
                    val_samples=x_val,
                    val_labels=y_val,
                    verbose=verbose,
                )

                application_to_models_results[app_name].append(model_result)

            best_app_values_w_info = self.get_best_app_result(
                application_to_models_results[app_name]
            )
            application_to_best_results[app_name] = best_app_values_w_info

        return application_to_best_results, application_to_models_results

    def evaluate_on_application(
        self, classifier, application, val_samples, val_labels, verbose=False
    ):
        classifier.with_application(application)
        llrs, predictions, _ = classifier.classify(
            val_samples, val_labels, verbose=verbose
        )

        evaluator = BinaryBayesEvaluator(
            evaluation_labels=val_labels,
            opt_predictions=predictions,
            llrs=llrs,
            application=application,
            model_name=classifier.get_name(),
            verbose=verbose,
        )

        results = evaluator.evaluate()
        results["model"] = classifier

        return results

    def get_best_app_result(self, models_results):
        best_values_w_info = {
            "udcf": (float("inf"), None),
            "norm_dcf": (float("inf"), None),
            "mindcf": (float("inf"), None),
            "calibration_loss": (float("inf"), None),
        }

        for model_results in models_results:
            model_obj = model_results["model"]
            for metric in ["udcf", "norm_dcf", "mindcf", "calibration_loss"]:
                value = model_results[metric]

                if value < best_values_w_info[metric][0]:
                    best_values_w_info[metric] = value, model_obj

        return best_values_w_info
