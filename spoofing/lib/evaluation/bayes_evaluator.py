import numpy as np
from lib.util.math_utils import get_confusion_matrix


class BinaryBayesEvaluator:
    def __init__(
        self,
        evaluation_labels,
        opt_predictions,
        llrs,
        application,
        model_name,
        verbose=False,
    ) -> None:
        self.application = application
        self.opt_predictions = opt_predictions
        # TODO: check that this work with any "scores" (e.g svms, logreg..)
        self.llrs = llrs
        self.evaluation_labels = evaluation_labels
        self.model_name = model_name
        self.verbose = verbose

    def evaluate(self):
        """
        Return:
        - an object containing the udcf, dcf, mindcf and calibration_loss values computed on this evaluator llrs and predictions
        """
        udcf = self.compute_udcf()
        norm_dcf = self.compute_dcf()
        mindcf = self.compute_mindcf()
        cal_loss = (norm_dcf - mindcf) * 100

        return {
            "udcf": udcf,
            "norm_dcf": norm_dcf,
            "mindcf": mindcf,
            "calibration_loss": cal_loss,
        }

    def compute_rates(self):
        cm = get_confusion_matrix(
            self.opt_predictions, self.evaluation_labels, num_classes=2
        )
        p_fn = cm[0, 1] / (cm[0, 1] + cm[1, 1])
        p_fp = cm[1, 0] / (cm[0, 0] + cm[1, 0])
        return p_fn, p_fp

    """
    def get_dcf(self):
        cm = self.cm
        p_fn = cm[0, 1] / (cm[0, 1] + cm[1, 1])
        p_fp = cm[1, 0] / (cm[0, 0] + cm[1, 0])
        return self.t_prior * self.c_fn * p_fn + self.n_prior * self.c_fp * p_fp
    """

    def compute_udcf(self):
        p_fn, p_fp = self.compute_rates()
        ep = self.application.get_effective_prior()
        return ep * p_fn + (1 - ep) * p_fp

    def compute_dcf(self):
        u_dcf = self.compute_udcf()
        norm = self.application.get_norm()
        return u_dcf / norm

    def get_an_effective_prior(self, p):
        return 1 / (1 + np.exp(-p))

    # If we sort the scores, then, as we sweep the scores, we can have that at most one prediction changes everytime.
    # We can then keep a running confusion matrix (or simply the number of false positives and false negatives)
    # that is updated everytime we move the threshold
    # Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
    # We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
    def compute_all_Pfn_Pfp(self):
        llrSorter = np.argsort(self.llrs)
        llrSorted = self.llrs[llrSorter]
        sorted_val_labels = self.evaluation_labels[llrSorter]

        Pfp = []
        Pfn = []

        nTrue = (sorted_val_labels == 1).sum()
        nFalse = (sorted_val_labels == 0).sum()
        # With the left-most theshold all samples are assigned to class 1
        nFalseNegative = 0
        nFalsePositive = nFalse

        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

        for idx in range(len(llrSorted)):
            if sorted_val_labels[idx] == 1:
                # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
                nFalseNegative += 1
            if sorted_val_labels[idx] == 0:
                # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
                nFalsePositive -= 1
            Pfn.append(nFalseNegative / nTrue)
            Pfp.append(nFalsePositive / nFalse)

        # The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
        # Pfn.append(1.0) # Corresponds to the np.inf threshold, all samples are assigned to class 0
        # Pfp.append(0.0) # Corresponds to the np.inf threshold, all samples are assigned to class 0
        llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

        # In case of repeated scores,
        # we need to "compact" the Pfn and Pfp arrays
        # (i.e., we need to keep only the value that corresponds to an actual change of the threshold
        PfnOut = []
        PfpOut = []
        thresholdsOut = []
        for idx in range(len(llrSorted)):
            if (
                idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[idx]
            ):  # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
                PfnOut.append(Pfn[idx])
                PfpOut.append(Pfp[idx])
                thresholdsOut.append(llrSorted[idx])

        return (
            np.array(PfnOut),
            np.array(PfpOut),
            np.array(thresholdsOut),
        )  # we return also the corresponding thresholds

    def compute_mindcf(self, returnThreshold=False):
        prior = self.application.get_effective_prior()
        Cfn = self.application.c_fn
        Cfp = self.application.c_fp

        Pfn, Pfp, th = self.compute_all_Pfn_Pfp()

        minDCF = (
            (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp)
        ) / self.application.get_dummy()
        idx = np.argmin(minDCF)
        if returnThreshold:
            return minDCF[idx], th[idx]
        else:
            return minDCF[idx]

    @staticmethod
    def evaluate_models_on_applications(experiments, applications, verbose=False):
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

                model_result = BinaryBayesEvaluator.evaluate_on_application(
                    classifier=model,
                    application=application,
                    val_samples=x_val,
                    val_labels=y_val,
                    verbose=verbose,
                )

                application_to_models_results[app_name].append(model_result)

            best_app_values_w_info = BinaryBayesEvaluator.group_best_app_result(
                application_to_models_results[app_name]
            )
            application_to_best_results[app_name] = best_app_values_w_info

        return application_to_best_results, application_to_models_results

    @staticmethod
    def evaluate_on_application(
        classifier, application, val_samples, val_labels, verbose=False
    ):
        classifier.with_application(application)
        llrs, predictions = classifier.classify(
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

        if verbose:
            print(
                f"\nEvaluating {classifier.get_name()} with application: {application.info()}"
            )

        results = evaluator.evaluate()

        if verbose:
            print("Evaluation results: ", results)
            print("-" * 40)

        results["model"] = classifier

        return results

    @staticmethod
    def group_best_app_result(models_results):
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
