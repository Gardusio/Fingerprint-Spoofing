import numpy as np
from evaluation.application import Application
from scipy.optimize import fmin_l_bfgs_b
from util.math_utils import (
    prior_weighted_logreg_objective,
    logreg_objective,
)


class LogisticRegressionBinaryClassifier:
    def __init__(
        self,
        c1_label,
        c2_label,
        empirical_prior,
        use_application=False,
        application=None,
        name="",
        weighted=False,
    ):
        self.name = name
        self.c1_label = c1_label
        self.c2_label = c2_label
        self.use_application = use_application
        self.application = application
        self.weighted = weighted
        self.empirical_prior = empirical_prior
        self.application = application

        if weighted and application is None:
            print(
                """
                Weighted logistic regression requires application specific prior log-odds.
                Not specifying an application results in using the empirical prior with uniform costs application.
                """
            )

    def with_name(self, name, new=False):
        self.name = name
        if new:
            return LogisticRegressionBinaryClassifier(
                c1_label=self.c1_label,
                c2_label=self.c2_label,
                empirical_prior=self.empirical_prior,
                application=self.application,
                weighted=self.weighted,
                use_application=self.use_application,
                name=name,
            )

        return self

    def get_name(self):
        return self.name

    def with_weighted(self):
        if not self.use_application:
            print(
                """
                Weighted logistic regression requires application specific prior log-odds
                Not specifying an application for this model results in  using the empirical prior with uniform costs application.
                """
            )
            self.application = Application(self.empirical_prior, 1, 1)
        self.weighted = True
        return self

    def with_application(self, application):
        self.use_application = True
        self.application = application
        return self

    def get_weights(self, z_train):
        prior = self.application.get_effective_prior()
        t_weight = prior / (z_train > 0).sum()
        f_weight = (1 - prior) / (z_train < 0).sum()

        return t_weight, f_weight

    # TODO : store parameters in a class obj "parameters"
    def fit(self, x_train, y_train, l):
        n = x_train.shape[0]
        x0 = np.zeros(n + 1)
        z_train = (2 * y_train) - 1

        if self.weighted:
            t_weight, f_weight = self.get_weights(z_train)
            objective = prior_weighted_logreg_objective
            obj_args = (x_train, z_train, l, t_weight, f_weight)
        else:
            objective = logreg_objective
            obj_args = (x_train, z_train, l)

        v_est, vf, _ = fmin_l_bfgs_b(
            func=objective,
            x0=x0,
            args=obj_args,
            approx_grad=False,
        )

        self.w = v_est[:-1]
        self.b = v_est[-1]

        return self

    def get_predictions(self, llr_set):
        trh = self.application.get_treshold() if self.use_application else 0

        return np.where(llr_set > trh, self.c1_label, self.c2_label)

    def get_scores(self, x_val):
        return (self.w.T @ x_val) + self.b

    def get_llrs(self, x_val):
        scores = self.get_scores(x_val)

        if self.weighted:
            prior = self.application.get_effective_prior()
            print("Effective prior: ", prior)
        else:
            prior = self.empirical_prior
            print("Empirical prior: ", prior)

        return scores - np.log(prior / (1 - prior))

    def classify(self, x_val, y_val, verbose=False):
        llrs = self.get_llrs(x_val)
        predictions = self.get_predictions(llrs)

        if verbose:
            if self.use_application:
                print("\nClassifying using Application: ", self.application.info())

        return llrs, predictions
