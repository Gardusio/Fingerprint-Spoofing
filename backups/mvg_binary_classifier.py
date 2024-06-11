import numpy as np
from spoofing.utils.math_utils import *


class MVGClassifierOriginal:
    def __init__(
        self,
        c1_label,
        c2_label,
        use_application=False,
        application=(),
    ) -> None:
        self.c1_label = c1_label
        self.c2_label = c2_label

        self.use_application = use_application
        self.application = application

        self.parameters = {"c1_mle": (), "c2_mle": (), "wc_cov_matrix": ()}

    def set_use_application(self, v):
        self.use_application = v

    def set_application(self, application):
        self.application = application

    def with_application(self, application):
        self.use_application = True
        self.set_application(application)

    def fit(self, t_samples, t_labels):
        c1_mle = self.get_class_mle(t_samples, t_labels, self.c1_label)
        c2_mle = self.get_class_mle(t_samples, t_labels, self.c2_label)
        wc_cov_matrix = get_within_class_covariance_matrix(t_samples, t_labels)

        self.parameters = {
            "c1_mle": c1_mle,
            "c2_mle": c2_mle,
            "wc_cov_matrix": wc_cov_matrix,
        }

        return self

    def get_class_mle(self, t_samples, t_labels, class_label):
        class_samples = t_samples[:, t_labels == class_label]
        c_mean = vcol(class_samples.mean(axis=1))
        c_cov_m = get_covariance_matrix(class_samples)
        return c_mean, c_cov_m

    def get_llrs(self, v_samples, with_naive_bayes=False, with_tied=False):
        c1_mle = self.parameters["c1_mle"]
        c2_mle = self.parameters["c2_mle"]

        if with_naive_bayes:
            c1_mle = (c1_mle[0], c1_mle[1] * np.identity(c1_mle[1].shape[0]))
            c2_mle = (c2_mle[0], c2_mle[1] * np.identity(c2_mle[1].shape[0]))
        if with_tied:
            cov_m = self.parameters["wc_cov_matrix"]
            c1_mle = (c1_mle[0], cov_m)
            c2_mle = (c2_mle[0], cov_m)

        fxc_2 = log_gaussian_density_set(v_samples, c2_mle[0], c2_mle[1])
        fxc_1 = log_gaussian_density_set(v_samples, c1_mle[0], c1_mle[1])

        return (fxc_1 - fxc_2).ravel()

    def get_predictions(self, llr_set):
        if self.use_application:
            trh = self.application.get_treshold()
        else:
            trh = 0

        llr_predictions = np.where(llr_set > trh, self.c1_label, self.c2_label)

        return llr_predictions

    def get_error_rate(self, predictions, y_val):
        return len(predictions[predictions != y_val]) / len(y_val)

    def classify(
        self, x_val, y_val, with_naive_bayes=False, with_tied=False, verbose=False
    ):

        llrs = self.get_llrs(x_val, with_naive_bayes, with_tied)
        predictions = self.get_predictions(llrs)
        err_rate = self.get_error_rate(predictions, y_val)

        if verbose:
            if self.use_application:
                print("USING APPLICATION: ", self.application.info())
            if with_naive_bayes:
                print("NAIVE BAYES MVG error rate percentage: ", err_rate * 100)
            elif with_tied:
                print("TIED MVG error rate percentage: ", err_rate * 100)
            else:
                print("MVG error rate percentage: ", err_rate * 100)

        return llrs, predictions, err_rate