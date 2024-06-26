from lib.util.math_utils import *
from scipy.linalg import eigh


def get_lda_directions(Sb, Sw, m=2):
    # Solves generalized eigenvalues
    s, U = eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    return W


def get_lda_matrix(samples, labels, nc=[0, 1], m=1):
    Sb = get_between_class_covariance_matrix(samples, labels, nc)
    Sw = get_within_class_covariance_matrix(samples, labels, nc)
    W = get_lda_directions(Sb, Sw, m)
    return W


def apply_lda(samples, labels, nc=[0, 1], m=2):
    return get_lda_matrix(samples, labels, nc, m).T @ samples
