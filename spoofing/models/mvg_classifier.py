import numpy as np
from math_utils import *


class MVGClassifier:
    def __init__(self, dataset) -> None:
        self.ds = dataset
        self.parameters = {
            "g_mle": self.get_genuines_mle(),
            "c_mle": self.get_counterfeits_mle(),
        }

    def get_genuines_mle(self):
        t_genuines = self.ds.training_genuines
        g_mean = vcol(t_genuines.mean(axis=1))
        g_cov_m = get_covariance_matrix(t_genuines)
        return g_mean, g_cov_m

    def get_counterfeits_mle(self):
        t_counterfeits = self.ds.training_counterfeits
        c_mean = vcol(t_counterfeits.mean(axis=1))
        c_cov_m = get_covariance_matrix(t_counterfeits)
        return c_mean, c_cov_m

    def get_llrs_predictions(self, g_mle, c_mle, v_samples):

        fxc_1 = log_gaussian_density_set(v_samples, c_mle[0], c_mle[1])

        fxc_2 = log_gaussian_density_set(v_samples, g_mle[0], g_mle[1])

        llr_set = fxc_2 - fxc_1
        llr_predictions = llr_set
        llr_predictions[llr_predictions > 0] = 1
        llr_predictions[llr_predictions < 0] = 0
        return llr_predictions.ravel()

    def get_error_rate(self, predictions, v_labels):
        return len(predictions[predictions != v_labels]) / len(v_labels)

    def classify(self, with_naive_bayes=False, with_tied=False):
        v_samples = self.ds.validation_samples
        v_labels = self.ds.validation_labels

        g_mle = self.parameters["g_mle"]
        c_mle = self.parameters["c_mle"]

        if with_naive_bayes:
            g_mle = (g_mle[0], g_mle[1] * np.identity(g_mle[1].shape[0]))
            c_mle = (c_mle[0], c_mle[1] * np.identity(c_mle[1].shape[0]))
        if with_tied:
            cov_m = get_within_class_covariance_matrix(
                self.ds.training_samples, self.ds.training_labels
            )
            g_mle = (g_mle[0], cov_m)
            c_mle = (c_mle[0], cov_m)

        predictions = self.get_llrs_predictions(g_mle, c_mle, v_samples)
        err_rate = self.get_error_rate(predictions, v_labels)

        if with_naive_bayes:
            print("NAIVE BAYES MVG error rate percentage: ", err_rate * 100)
        elif with_tied:
            print("TIED MVG error rate percentage: ", err_rate * 100)
        else:
            print("MVG error rate percentage: ", err_rate * 100)
