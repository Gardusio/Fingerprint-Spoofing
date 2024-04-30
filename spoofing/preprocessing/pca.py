import numpy as np

from mathutils.math_utils import get_covariance_matrix


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
