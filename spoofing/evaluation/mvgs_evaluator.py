from evaluation.bayes_evaluator import BinaryBayesEvaluator


class MVGEvaluator:

    def evaluate_models_on_applications(self, experiments, applications):

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
                )

                application_to_models_results[app_name].append(
                    (model.get_name(), model_result)
                )

            best_app_values_w_info = self.get_best_app_result(
                application_to_models_results[app_name]
            )

            application_to_best_results[app_name].append(best_app_values_w_info)

        return application_to_best_results, application_to_models_results

    def evaluate_on_application(
        self, classifier, application, val_samples, val_labels, verbose=False
    ):
        classifier.with_application(application)
        llrs, predictions, _ = classifier.classify(
            val_samples,
            val_labels,
        )
        evaluator = BinaryBayesEvaluator(
            evaluation_labels=val_labels,
            opt_predictions=predictions,
            llrs=llrs,
            application=application,
            model_name=classifier.__class__.__name__,
            verbose=verbose,
        )

        return evaluator.evaluate()

    def evaluate_on_applications(
        self, classifier, applications, val_samples, val_labels, verbose=False
    ):
        """
        params:
        - classifier: a fitted model to evaluate
        - applications: a list of Application
        - val_samples: validation set on which to perform evaluation
        - val_labels: relative val_labels
        - verbose: turn on printing intermediate logs

        return:
        - a list containing tuples: (application_name, best_metrics)

        where best_metrics is a {"metric": (model_name, metric_value)} object that represent
        which model achieved the best metric_value for that metric and application
        """
        app_to_results = []
        for application in applications:
            app_results = (
                self.evaluate_mvg_on_application(
                    classifier,
                    application,
                    val_samples,
                    val_labels,
                    verbose=verbose,
                ),
            )
            app_to_results.append(
                {"app_name": application.get_name(), "results": app_results}
            )

        return app_to_results

    def get_best_app_result(self, results):
        best_values_w_info = {
            "udcf": (float("inf"), ""),
            "norm_dcf": (float("inf"), ""),
            "mindcf": (float("inf"), ""),
            "calibration_loss": (float("inf"), ""),
        }

        for info, result in results:
            for metric in ["udcf", "norm_dcf", "mindcf", "calibration_loss"]:
                value = result[metric]
                if value < best_values_w_info[metric][0]:
                    best_values_w_info[metric] = value, info

        return best_values_w_info
