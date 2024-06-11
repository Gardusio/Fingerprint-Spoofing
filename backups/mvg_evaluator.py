from evaluation.bayes_evaluator import BinaryBayesEvaluator
from preprocessing import pca


class MVGEvaluator:

    def compute_bests(self, results):
        best_udcf = min(
            (results["MVG"]["udcf"], "MVG"),
            (results["Naive bayes"]["udcf"], "NB"),
            (results["Tied MVG"]["udcf"], "Tied"),
        )
        best_dcf = min(
            (results["MVG"]["norm_dcf"], "MVG"),
            (results["Naive bayes"]["norm_dcf"], "NB"),
            (results["Tied MVG"]["norm_dcf"], "Tied"),
        )
        best_mindcf = min(
            (results["MVG"]["mindcf"], "MVG"),
            (results["Naive bayes"]["mindcf"], "NB"),
            (results["Tied MVG"]["mindcf"], "Tied"),
        )
        best_calibration_loss = min(
            (results["MVG"]["calibration_loss"], "MVG"),
            (results["Naive bayes"]["calibration_loss"], "NB"),
            (results["Tied MVG"]["calibration_loss"], "Tied"),
        )

        return best_udcf, best_dcf, best_mindcf, best_calibration_loss

    def get_bests_from_results(self, results, verbose=False):
        best_udcf, best_dcf, best_mindcf, best_calibration_loss = self.compute_bests(
            results
        )

        best_results = {
            "udcf": best_udcf,
            "dcf": best_dcf,
            "mindcf": best_mindcf,
            "calibration_loss": best_calibration_loss,
        }

        if verbose:
            print("Best values:")
            for metric, (value, name) in best_results.items():
                print(f"{metric}: {name} with value {value}")

        return tuple(
            best_results[metric]
            for metric in ["udcf", "dcf", "mindcf", "calibration_loss"]
        )

    def evaluate_mvg_on_application(
        self, mvg_classifier, application, val_samples, val_labels, verbose=False
    ):
        def get_evaluator(name, with_naive_bayes=False, with_tied=False):
            llrs, predictions, _ = mvg_classifier.classify(
                val_samples,
                val_labels,
                with_naive_bayes=with_naive_bayes,
                with_tied=with_tied,
            )
            return BinaryBayesEvaluator(
                evaluation_labels=val_labels,
                opt_predictions=predictions,
                llrs=llrs,
                application=application,
                model_name=name,
                verbose=verbose,
            )

        mvg_classifier.with_application(application)

        evaluators = [
            get_evaluator("MVG"),
            get_evaluator("Naive bayes", with_naive_bayes=True),
            get_evaluator("Tied MVG", with_tied=True),
        ]

        results = {
            evaluator.model_name: evaluator.evaluate() for evaluator in evaluators
        }

        return results

    def evaluate_mvg(
        self, mvg_classifier, applications, val_samples, val_labels, verbose=False
    ):
        """
        params:
        - mvg_classifier: a fitted model to evaluate
        - applications: a list of Application
        - val_samples: validation set on which to perform evaluation
        - val_labels: relative val_labels
        - verbose: turn on printing intermediate logs

        return:
        - a list containing tuples: (application_name, best_metrics)

        where best_metrics is a {"metric": (model_name, metric_value)} object that represent
        which model achieved the best metric_value for that metric and application
        """
        bests = []
        for application in applications:
            results = self.evaluate_mvg_on_application(
                mvg_classifier, application, val_samples, val_labels, verbose=verbose
            )
            (
                (best_udcf_name, best_udcf),
                (best_dcf_name, best_dcf),
                (best_mindcf_name, best_mindcf),
                (best_calibration_loss_name, best_calibration_loss),
            ) = self.get_bests_from_results(results)

            bests.append(
                {
                    "app_name": application.get_name(),
                    "best_metrics": {
                        "udcf": (best_udcf_name, best_udcf),
                        "dcf": (best_dcf_name, best_dcf),
                        "mindcf": (best_mindcf_name, best_mindcf),
                        "calibration_loss": (
                            best_calibration_loss_name,
                            best_calibration_loss,
                        ),
                    },
                }
            )

        return results, bests

    def compare_results(self, best_list):

        best_values = {
            "udcf": (None, float("inf")),
            "dcf": (None, float("inf")),
            "mindcf": (None, float("inf")),
            "calibration_loss": (None, float("inf")),
        }

        for entry, info in best_list:
            for metric in ["udcf", "dcf", "mindcf", "calibration_loss"]:
                value, name = entry[metric]
                if value < best_values[metric][1]:
                    best_values[metric] = (name, value)

        # Print the results
        for metric, (entry, value) in best_values.items():
            print(
                f"\nThe best {metric} is from: {entry} - {info} -  with a value of {value}\n"
            )
