import numpy as np
from math_utils import *


class MVGClassifier:
    def __init__(
        self, t_samples, t_labels, v_samples, v_labels, c1_label, c2_label
    ) -> None:
        self.t_samples = t_samples
        self.t_labels = t_labels
        self.v_samples = v_samples
        self.v_labels = v_labels
        self.c1_label = c1_label
        self.c2_label = c2_label

        self.parameters = {
            "c1_mle": self.get_class_mle(self.c1_label),
            "c2_mle": self.get_class_mle(self.c2_label),
        }

    def get_class_mle(self, class_label):
        class_samples = self.t_samples[:, self.t_labels == class_label]
        c1_mean = vcol(class_samples.mean(axis=1))
        c1_cov_m = get_covariance_matrix(class_samples)
        return c1_mean, c1_cov_m

    def get_llrs_predictions(self, c1_mle, c2_mle, v_samples):

        fxc_1 = log_gaussian_density_set(v_samples, c2_mle[0], c2_mle[1])

        fxc_2 = log_gaussian_density_set(v_samples, c1_mle[0], c1_mle[1])

        llr_set = fxc_2 - fxc_1
        llr_predictions = llr_set
        llr_predictions[llr_predictions > 0] = self.c1_label
        llr_predictions[llr_predictions < 0] = self.c2_label
        return llr_predictions.ravel()

    def get_error_rate(self, predictions, v_labels):
        return len(predictions[predictions != v_labels]) / len(v_labels)

    def classify(self, with_naive_bayes=False, with_tied=False):

        c1_mle = self.parameters["c1_mle"]
        c2_mle = self.parameters["c2_mle"]

        if with_naive_bayes:
            c1_mle = (c1_mle[0], c1_mle[1] * np.identity(c1_mle[1].shape[0]))
            c2_mle = (c2_mle[0], c2_mle[1] * np.identity(c2_mle[1].shape[0]))
        if with_tied:
            cov_m = get_within_class_covariance_matrix(self.t_samples, self.t_labels)
            c1_mle = (c1_mle[0], cov_m)
            c2_mle = (c2_mle[0], cov_m)

        predictions = self.get_llrs_predictions(c1_mle, c2_mle, self.v_samples)
        err_rate = self.get_error_rate(predictions, self.v_labels)

        if with_naive_bayes:
            print("NAIVE BAYES MVG error rate percentage: ", err_rate * 100)
        elif with_tied:
            print("TIED MVG error rate percentage: ", err_rate * 100)
        else:
            print("MVG error rate percentage: ", err_rate * 100)
