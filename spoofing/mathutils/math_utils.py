import numpy as np
import math


def vrow(v):
    return v.reshape(1, v.shape[0])


def vcol(v):
    return v.reshape(1, v.shape[0])


def get_mean_vector(samples):
    return samples.mean(axis=1).reshape(samples.shape[0], 1)


def get_covariance_matrix(features_matrix):

    ds_mean = features_matrix.mean(axis=1).reshape(features_matrix.shape[0], 1)

    centered_features = features_matrix - ds_mean

    return (centered_features @ centered_features.T) / float(features_matrix.shape[1])


def get_scatter(class_samples, ds_mean):
    class_mean = get_mean_vector(class_samples)
    return (class_mean - ds_mean) @ (class_mean - ds_mean).T


def get_within_class_covariance_matrix(samples, labels, nc=[0, 1]):
    num_samples = float(samples.shape[1])

    Sw_acc = 0
    for i in nc:
        class_samples = samples[:, labels == i]
        num_class_samples = float(class_samples.shape[1])
        Sw_acc = Sw_acc + (num_class_samples * get_covariance_matrix(class_samples))

    Sw = Sw_acc / num_samples
    return Sw


def get_between_class_covariance_matrix(samples, labels, nc=[0, 1]):
    num_samples = float(samples.shape[1])
    ds_mean = samples.mean(axis=1).reshape(samples.shape[0], 1)

    Sb_acc = 0
    for i in nc:
        class_samples = samples[:, labels == i]
        num_class_samples = float(class_samples.shape[1])
        Sb_acc = Sb_acc + (num_class_samples * get_scatter(class_samples, ds_mean))

    Sb = Sb_acc / num_samples
    return Sb


def log_gaussian_density_set(sample_set, ds_mean_vec, cov_matrix):

    ll = [
        log_gaussian_density(sample_set[:, i : i + 1], ds_mean_vec, cov_matrix)
        for i in range(sample_set.shape[1])
    ]

    return np.array(ll)


def log_gaussian_density(sample, ds_mean_vec, cov_matrix):
    n_features = ds_mean_vec.shape[0]
    _, cov_matrix_log_det = np.linalg.slogdet(cov_matrix)
    inverse_cov_matrix = np.linalg.inv(cov_matrix)
    centered_sample = sample - ds_mean_vec

    log_2pi = math.log(2 * math.pi)

    result = (
        (-(n_features / 2) * log_2pi)
        - (cov_matrix_log_det / 2)
        - (((centered_sample.T) @ inverse_cov_matrix @ centered_sample) / 2)
    )

    return result.ravel()


def gaussian_density(sample_set, ds_mean_vec, cov_matrix):
    return np.exp(log_gaussian_density_set(sample_set, ds_mean_vec, cov_matrix).ravel())


def get_gaussian_to_feature_plotline(feature_samples, f_idx):

    f_mean = get_mean_vector(feature_samples)
    f_cov = get_covariance_matrix(feature_samples)

    f_row = vrow(feature_samples[0, :])

    f_min = f_row.min()
    f_max = f_row.max()

    plot = np.linspace(f_min, f_max, feature_samples.shape[1])
    gaussian_pdf = gaussian_density(vrow(plot), f_mean, f_cov)

    return plot, gaussian_pdf
