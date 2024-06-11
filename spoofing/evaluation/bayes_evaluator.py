import numpy as np
from util.math_utils import get_confusion_matrix


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
        # TODO: this should be any "scores"
        self.llrs = llrs
        self.evaluation_labels = evaluation_labels
        self.model_name = model_name
        self.verbose = verbose

    def evaluate(self):
        """
        Return:
        - an object containing the udcf, dcf, mindcf and calibration_loss values computed on this evaluator llrs and predictions
        """
        if self.verbose:
            print(
                f"\nEvaluating {self.model_name} with application: {self.application.info()}"
            )

        udcf = self.get_udcf()
        norm_dcf = self.get_normalized_dcf()
        mindcf = self.get_mindcf()
        cal_loss = (norm_dcf - mindcf) * 100
        if self.verbose:
            print("DCF: ", udcf)
            print("normalized DCF: ", norm_dcf)
            print("mindcf: ", mindcf)
            print("calibration loss: ", cal_loss)

        print("-" * 40)

        return {
            "udcf": udcf,
            "norm_dcf": norm_dcf,
            "mindcf": mindcf,
            "calibration_loss": cal_loss,
        }

    def get_rates(self):
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

    def get_udcf(self):
        p_fn, p_fp = self.get_rates()
        return self.get_dcf(self.application.get_effective_prior(), p_fn, p_fp)

    def get_dcf(self, ep, p_fn, p_fp):
        return ep * p_fn + (1 - ep) * p_fp

    def get_normalized_dcf(self):
        u_dcf = self.get_udcf()
        norm = self.application.get_norm()
        return u_dcf / norm

    def get_an_effective_prior(self, p):
        return 1 / (1 + np.exp(-p))

    # Compute minDCF (fast version)
    # If we sort the scores, then, as we sweep the scores, we can have that at most one prediction changes everytime. We can then keep a running confusion matrix (or simply the number of false positives and false negatives) that is updated everytime we move the threshold
    # Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
    # We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
    def get_all_Pfn_Pfp(self):
        llrSorter = np.argsort(self.llrs)
        llrSorted = self.llrs[llrSorter]  # We sort the llrs
        sorted_val_labels = self.evaluation_labels[
            llrSorter
        ]  # we sort the labels so that they are aligned to the llrs

        Pfp = []
        Pfn = []

        nTrue = (sorted_val_labels == 1).sum()
        nFalse = (sorted_val_labels == 0).sum()
        nFalseNegative = (
            0  # With the left-most theshold all samples are assigned to class 1
        )
        nFalsePositive = nFalse

        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

        for idx in range(len(llrSorted)):
            if sorted_val_labels[idx] == 1:
                nFalseNegative += 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
            if sorted_val_labels[idx] == 0:
                nFalsePositive -= 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
            Pfn.append(nFalseNegative / nTrue)
            Pfp.append(nFalsePositive / nFalse)

        # The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
        # Pfn.append(1.0) # Corresponds to the np.inf threshold, all samples are assigned to class 0
        # Pfp.append(0.0) # Corresponds to the np.inf threshold, all samples are assigned to class 0
        llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

        # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
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

    # Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
    # We can therefore directly pass the logistic regression scores, or the SVM scores
    def get_mindcf(self, returnThreshold=False):
        prior = self.application.get_effective_prior()
        Cfn = self.application.c_fn
        Cfp = self.application.c_fp

        Pfn, Pfp, th = self.get_all_Pfn_Pfp()

        minDCF = (
            (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp)
        ) / self.application.get_dummy()  # We exploit broadcasting to compute all DCFs for all thresholds
        idx = np.argmin(minDCF)
        if returnThreshold:
            return minDCF[idx], th[idx]
        else:
            return minDCF[idx]
