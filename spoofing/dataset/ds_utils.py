import numpy as np
from mathutils.math_utils import *


def get_class_samples(samples, labels, c):
    return samples[:, labels == c]


def get_genuines_samples(samples, labels):
    return get_class_samples(samples, labels, 1)


def get_counterfeits_samples(samples, labels):
    return get_class_samples(samples, labels, 0)


def get_class_mean(samples, labels, c):
    return get_class_samples(samples, labels, c).mean()


# GAUSSIAN DENSITY
def fit_gaussian_to_feature(feature_samples, f_idx):

    f_mean = get_mean_vector(feature_samples)
    f_cov = get_covariance_matrix(feature_samples)

    f_row = vrow(feature_samples[0, :])

    f_min = f_row.min()
    f_max = f_row.max()

    plot = np.linspace(f_min, f_max, feature_samples.shape[1])
    gaussian_pdf = gaussian_density(vrow(plot), f_mean, f_cov)

    return plot, gaussian_pdf
