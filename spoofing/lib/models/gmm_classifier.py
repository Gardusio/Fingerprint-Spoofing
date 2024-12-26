import numpy as np
from util.GMM_load import *
from util.math_utils import (
    vcol,
    vrow,
    get_mean_vector,
    get_covariance_matrix,
    log_gmm_density,
    smooth_covariance_matrix,
    get_gmm_density_mean,
)


class GMMClassifier:
    def __init__(
        self,
        c1_label,
        c2_label,
        use_application=False,
        application=None,
        cov_type="Full",
        name="GMM",
    ):
        self.c1_label = c1_label
        self.c2_label = c2_label
        self.use_application = use_application
        self.application = application
        self.name = name
        self.cov_type = cov_type

    def em_iteration(self, x_train, gmm, psi_eig=None):

        log_joint_densities, log_marginal_densities = log_gmm_density(x_train, gmm)

        # Compute posterior probabilities
        gamma_all_components = np.exp(log_joint_densities - log_marginal_densities)

        # M-step
        updated_gmm = []

        for g_idx in range(len(gmm)):
            # Responsibilities for component g_idx
            gamma = gamma_all_components[g_idx]
            Z = gamma.sum()
            F = vcol((vrow(gamma) * x_train).sum(1))
            S = (vrow(gamma) * x_train) @ x_train.T

            mu_upd = F / Z
            covariance_upd = S / Z - mu_upd @ mu_upd.T
            weight_upd = Z / x_train.shape[1]

            if self.cov_type.lower() == "diagonal":
                covariance_upd = np.diag(np.diag(covariance_upd))

            updated_gmm.append((weight_upd, mu_upd, covariance_upd))

        # Handle tied covariance type
        if self.cov_type.lower() == "tied":
            tied_covariance = sum(w * C for w, mu, C in updated_gmm)
            updated_gmm = [(w, mu, tied_covariance) for w, mu, C in updated_gmm]

        # Smooth covariance if psi_eig is provided
        if psi_eig is not None:
            updated_gmm = [
                (w, mu, smooth_covariance_matrix(C, psi_eig))
                for w, mu, C in updated_gmm
            ]

        return updated_gmm

    # Train a GMM until the average dela log-likelihood becomes <= epsLLAverage
    def run_em(
        self,
        x_train,
        initial_gmm,
        psi_eig=None,
        eps_ll_average=1e-6,
        verbose=True,
    ):

        # Initialize variables
        gmm = initial_gmm
        ll_old = get_gmm_density_mean(x_train, gmm)
        ll_delta = None

        if verbose:
            print("GMM - Iteration %3d - Average Log-Likelihood %.8e" % (0, ll_old))

        # EM algorithm iterations
        iteration = 1
        while ll_delta is None or ll_delta > eps_ll_average:
            updated_gmm = self.em_iteration(x_train, gmm, psi_eig)
            ll_updated = get_gmm_density_mean(x_train, updated_gmm)
            ll_delta = ll_updated - ll_old

            if verbose:
                print(
                    "GMM - Iteration %3d - Average Log-Likelihood %.8e"
                    % (iteration, ll_updated)
                )

            gmm = updated_gmm
            ll_old = ll_updated
            iteration += 1

        if verbose:
            print(
                "GMM - Iteration %3d - Final Average Log-Likelihood %.8e (eps = %e)"
                % (iteration, ll_updated, eps_ll_average)
            )

        return gmm

    def split_gmm_with_lbg(self, gmm, alpha=0.1, verbose=True):
        gmmOut = []
        if verbose:
            print("LBG - going from %d to %d components" % (len(gmm), len(gmm) * 2))
        for w, mu, C in gmm:
            U, s, Vh = np.linalg.svd(C)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            gmmOut.append((0.5 * w, mu - d, C))
            gmmOut.append((0.5 * w, mu + d, C))
        return gmmOut

    # Train a full model using LBG + EM,
    # starting from a single Gaussian model, until we have num_components components.
    # lbgAlpha is the value 'alpha' used for LBG, the otehr parameters are the same as in the EM functions above
    def fit(
        self,
        x_train,
        num_components,
        psi_eig=None,
        eps_ll_average=1e-6,
        lbg_alpha=0.1,
        verbose=True,
    ):

        # Initialize the mean and covariance matrix
        mean_vector = get_mean_vector(x_train)
        covariance_matrix = get_covariance_matrix(x_train)

        if self.cov_type.lower() == "diagonal":
            covariance_matrix = covariance_matrix * np.eye(x_train.shape[0])

        # Initialize the Gaussian Mixture Model with one component
        if psi_eig is not None:
            gmm = [
                (1.0, mean_vector, smooth_covariance_matrix(covariance_matrix, psi_eig))
            ]
        else:
            gmm = [(1.0, mean_vector, covariance_matrix)]  # One-component model

        # Iterate until the desired number of components is reached
        while len(gmm) < num_components:

            if verbose:
                print(
                    "Average log-likelihood before LBG: %.8e"
                    % get_gmm_density_mean(x_train, gmm)
                )

            gmm = self.split_gmm_with_lbg(gmm, lbg_alpha, verbose=verbose)

            if verbose:
                print(
                    "Average log-likelihood after LBG: %.8e"
                    % get_gmm_density_mean(x_train, gmm)
                )

            gmm = self.run_em(
                x_train,
                gmm,
                psi_eig=psi_eig,
                verbose=verbose,
                eps_ll_average=eps_ll_average,
            )

        return gmm

    def get_predictions(self, llrs):
        trh = self.application.get_treshold() if self.use_application else 0
        return np.where(llrs > trh, self.c1_label, self.c2_label)

    def classify(self, x_val, y_val, gmm1, gmm0):
        llrs = log_gmm_density(y_val, gmm1) - log_gmm_density(y_val, gmm0)
        predictions = self.get_predictions(llrs)

        if self.use_application:
            print("\nClassifying using Application: ", self.application.info())

        return llrs, predictions
