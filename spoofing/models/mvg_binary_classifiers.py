import numpy as np
from util.math_utils import *


class BaseMVGClassifier:
    def __init__(
        self,
        c1_label,
        c2_label,
        use_application=False,
        application=(),
        name="",
    ):
        self.name = name
        self.c1_label = c1_label
        self.c2_label = c2_label
        self.use_application = use_application
        self.application = application
        self.parameters = {"c1_mle": (), "c2_mle": ()}

    def fit(self, t_samples, t_labels):
        print(f"Fitting {self.name}...")

        c1_mle = self.get_class_mle(t_samples, t_labels, self.c1_label)
        c2_mle = self.get_class_mle(t_samples, t_labels, self.c2_label)
        self.parameters = {"c1_mle": c1_mle, "c2_mle": c2_mle}
        return self

    def get_llrs(self, v_samples):
        c1_mle = self.parameters["c1_mle"]
        c2_mle = self.parameters["c2_mle"]
        fxc_2 = log_gaussian_density_set(v_samples, c2_mle[0], c2_mle[1])
        fxc_1 = log_gaussian_density_set(v_samples, c1_mle[0], c1_mle[1])
        return (fxc_1 - fxc_2).ravel()

    def get_predictions(self, llr_set):
        trh = self.application.get_treshold() if self.use_application else 0
        return np.where(llr_set > trh, self.c1_label, self.c2_label)

    def get_error_rate(self, predictions, y_val):
        return len(predictions[predictions != y_val]) / len(y_val)

    def classify(self, x_val, y_val, verbose=False):
        llrs = self.get_llrs(x_val)
        predictions = self.get_predictions(llrs)
        err_rate = self.get_error_rate(predictions, y_val)

        if verbose:
            if self.use_application:
                print("\nClassifying using Application: ", self.application.info())
            else:
                print(
                    f"{self.__class__.__name__} error rate percentage: ",
                    err_rate * 100,
                )

        return llrs, predictions

    def get_name(self):
        return self.name

    def set_use_application(self, v):
        self.use_application = v

    def set_application(self, application):
        self.application = application

    def with_application(self, application):
        self.use_application = True
        self.set_application(application)
        return self

    def get_class_samples(self, t_samples, t_labels, class_label):
        return t_samples[:, t_labels == class_label]


class MVGClassifier(BaseMVGClassifier):
    def with_name(self, name, new):
        self.name = name
        if new:
            return MVGClassifier(
                c1_label=self.c1_label,
                c2_label=self.c2_label,
                application=self.application,
                use_application=self.use_application,
                name=name,
            )
        return self

    def get_class_mle(self, t_samples, t_labels, class_label):
        class_samples = self.get_class_samples(t_samples, t_labels, class_label)
        c_mean = vcol(class_samples.mean(axis=1))
        c_cov_m = get_covariance_matrix(class_samples)
        return c_mean, c_cov_m


# TODO: add optional mvg parameters ti avoid fitting if an mvg on the same ds as already been trained
class NBClassifier(BaseMVGClassifier):
    def with_name(self, name, new=False):
        self.name = name
        if new:
            return NBClassifier(
                c1_label=self.c1_label,
                c2_label=self.c2_label,
                application=self.application,
                use_application=self.use_application,
                name=name,
            )
        return self

    def get_class_mle(self, t_samples, t_labels, class_label):
        class_samples = self.get_class_samples(t_samples, t_labels, class_label)
        c_mean = vcol(class_samples.mean(axis=1))
        c_cov_m = get_covariance_matrix(class_samples)
        return c_mean, c_cov_m * np.identity(c_cov_m.shape[0])


# TODO: add optional mvg parameters to re-use and avoid fitting if an mvg on the same ds as already been trained
class TIEDClassifier(BaseMVGClassifier):
    def with_name(self, name, new=False):
        self.name = name
        if new:
            return TIEDClassifier(
                c1_label=self.c1_label,
                c2_label=self.c2_label,
                application=self.application,
                use_application=self.use_application,
                name=name,
            )
        return self

    def get_class_mle(self, t_samples, t_labels, class_label):
        class_samples = self.get_class_samples(t_samples, t_labels, class_label)
        c_mean = vcol(class_samples.mean(axis=1))
        c_cov_m = get_within_class_covariance_matrix(t_samples, t_labels)
        return c_mean, c_cov_m
