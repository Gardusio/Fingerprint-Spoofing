import numpy as np
from util.math_utils import *
from scipy.optimize import fmin_l_bfgs_b


class SVMClassifier:
    def __init__(
        self,
        c1_label,
        c2_label,
        name="",
        use_application=False,
        application=None,
    ):
        self.name = name
        self.c1_label = c1_label
        self.c2_label = (c2_label,)
        self.use_application = use_application
        self.application = application
        self.w = None
        self.b = None
        self.kernel_svm = None

    def with_application(self, application):
        self.use_application = True
        self.application = application
        return self

    def get_name(self):
        return self.name

    def fit(self, x_train, y_train, C=1, K=1, verbose=False):
        print(f"Fitting {self.name}...")

        z_train = y_train * 2.0 - 1.0

        x_train_EXT = np.vstack([x_train, np.ones((1, x_train.shape[1])) * K])
        H = np.dot(x_train_EXT.T, x_train_EXT) * vcol(z_train) * vrow(z_train)

        # Dual objective with gradient
        def dual_objective_with_gradient(alpha):
            Ha = H @ vcol(alpha)
            loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
            grad = Ha.ravel() - np.ones(alpha.size)
            return loss, grad

        alphaStar, _, _ = fmin_l_bfgs_b(
            dual_objective_with_gradient,
            np.zeros(x_train_EXT.shape[1]),
            bounds=[(0, C) for i in y_train],
            factr=1.0,
        )

        # Primal loss
        def primalLoss(w_hat):
            S = (vrow(w_hat) @ x_train_EXT).ravel()
            return (
                0.5 * np.linalg.norm(w_hat) ** 2
                + C * np.maximum(0, 1 - z_train * S).sum()
            )

        # Compute primal solution for extended data matrix
        w_hat = (vrow(alphaStar) * vrow(z_train) * x_train_EXT).sum(1)

        # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
        w, b = (
            w_hat[0 : x_train.shape[0]],
            w_hat[-1] * K,
        )  # b must be rescaled in case K != 1, since we want to compute w'x + b * K

        if verbose:
            primalLoss, dualLoss = (
                primalLoss(w_hat),
                -dual_objective_with_gradient(alphaStar)[0],
            )
            print(
                "SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e"
                % (C, K, primalLoss, dualLoss, primalLoss - dualLoss)
            )

        self.w = w
        self.b = b

        return self

    def fit_kernel(self, x_train, y_train, kernel=None, C=1, eps=1.0, verbose=False):
        z_train = y_train * 2.0 - 1.0  # Convert labels to +1/-1
        K = kernel(x_train, x_train) + eps
        H = vcol(z_train) * vrow(z_train) * K

        # Dual objective with gradient
        def fOpt(alpha):
            Ha = H @ vcol(alpha)
            loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
            grad = Ha.ravel() - np.ones(alpha.size)
            return loss, grad

        alphaStar, _, _ = fmin_l_bfgs_b(
            fOpt,
            np.zeros(x_train.shape[1]),
            bounds=[(0, C) for i in y_train],
            factr=1.0,
        )

        if verbose:
            print("SVM (kernel) - C %e - dual loss %e" % (C, -fOpt(alphaStar)[0]))

        # Function to compute the scores for samples in DTE
        def kernel_svm_scoring(DTE):
            K = kernel(x_train, DTE) + eps
            H = vcol(alphaStar) * vcol(z_train) * K
            return H.sum(0)

        self.kernel_svm = kernel_svm_scoring
        return self

    # We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
    # The inner function will be able to access the arguments of the outer function
    @staticmethod
    def polyKernel(degree, c):
        def polyKernelFunc(D1, D2):
            return (np.dot(D1.T, D2) + c) ** degree

        return polyKernelFunc

    @staticmethod
    def rbfKernel(gamma):
        def rbfKernelFunc(D1, D2):
            # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
            D1Norms = (D1**2).sum(0)
            D2Norms = (D2**2).sum(0)
            Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
            return np.exp(-gamma * Z)

        return rbfKernelFunc

    def get_predictions(self, scores):
        trh = self.application.get_treshold() if self.use_application else 0
        return np.where(scores > trh, self.c1_label, self.c2_label)

    def classify(self, x_val, y_val, use_kernel=False):
        if use_kernel:
            scores = self.kernel_svm(x_val)
        else:
            scores = (vrow(self.w) @ x_val + self.b).ravel()

        predictions = self.get_predictions(scores)

        return scores, predictions
