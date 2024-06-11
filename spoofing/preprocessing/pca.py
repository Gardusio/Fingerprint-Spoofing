import numpy as np

from util.math_utils import get_covariance_matrix


def get_pca_matrix(samples, m):

    covariance_m = get_covariance_matrix(samples)

    # eig values are already sorted by svd
    s_eigh_vectors, s_values, vh = np.linalg.svd(covariance_m)

    # hence we can directly access first m columns of eigh_vectors
    P = s_eigh_vectors[:, 0:m]

    return P


def pca(samples, m):
    P = get_pca_matrix(samples, m)
    return P.T @ samples


def pca_fit(x_train, x_val, m):
    P = get_pca_matrix(x_train, m)
    pcad_x_train = P.T @ x_train
    pcad_x_val = P.T @ x_val
    return pcad_x_train, pcad_x_val
