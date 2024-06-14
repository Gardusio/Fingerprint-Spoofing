import numpy as np
import math


def vrow(v):
    return v.reshape(1, v.shape[0])


def vcol(v):
    return v.reshape(v.shape[0], 1)


def l2_norm(w):
    return np.linalg.norm(w)


# TODO: GET ACCURACY AND ERROR RATE


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


def log_gaussian_density_set(samples, mean_vec, cov_matrix):

    ll = [
        log_gaussian_density(samples[:, i : i + 1], mean_vec, cov_matrix)
        for i in range(samples.shape[1])
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


def get_pearson_correlation_matrix(cov_m):
    return cov_m / (vcol(cov_m.diagonal() ** 0.5) * vrow(cov_m.diagonal() ** 0.5))


def get_confusion_matrix(predictions, labels, num_classes):
    confusion_matrix = np.zeros(shape=(num_classes, num_classes))
    for p, l in zip(predictions, labels):
        confusion_matrix[p, l] += 1
    return confusion_matrix


def logreg_grad(v, x_train, z_train, l, scores):
    w = v[0:-1]

    G = -z_train / (1.0 + np.exp(z_train * scores))
    dJ_w = (vrow(G) * x_train).mean(1) + l * w.ravel()
    dJ_b = G.mean()

    return np.hstack([dJ_w, np.array(dJ_b)])


def logreg_objective(v, x_train, z_train, l):
    w, b = v[:-1], v[-1]
    reg = l2_norm(w) ** 2 * (l / 2)

    # Slow, use broadcasting instead
    """    
    S = 0
    for i, x_i in enumerate(x_train):
        z_i = (2 * y_train[i]) - 1
        S = S + np.logaddexp(0, -z_i*(w.T*x_i + b))
    J = reg + 1/n * S
    """

    scores = (vcol(w).T @ x_train).ravel() + b
    f = reg + np.logaddexp(0, -z_train * scores).mean()

    grad = logreg_grad(v, x_train, z_train, l, scores)

    return (f, grad)


def prior_weighted_logreg_grad(v, x_train, z_train, l, scores, t_weight, f_weight):
    w = v[0:-1]

    G = -z_train / (1.0 + np.exp(z_train * scores))
    G[z_train > 0] *= t_weight
    G[z_train < 0] *= f_weight

    dJ_w = (vrow(G) * x_train).sum(1) + l * w.ravel()
    dJ_b = G.sum()

    return np.hstack([dJ_w, np.array(dJ_b)])


def prior_weighted_logreg_objective(v, x_train, z_train, l, t_weight, f_weight):
    w, b = v[:-1], v[-1]

    reg = (l2_norm(w) ** 2) * (l / 2)

    scores = (vcol(w).T @ x_train).ravel() + b

    loss = np.logaddexp(0, -z_train * scores)
    loss[z_train > 0] *= t_weight
    loss[z_train < 0] *= f_weight
    f = loss.sum() + reg

    grad = prior_weighted_logreg_grad(
        v, x_train, z_train, l, scores, t_weight, f_weight
    )

    return (f, grad)


def get_err_rate(predictions, labels):
    return (predictions != labels).sum() / float(labels.size)


def expand_features(X):
    num_features, num_samples = X.shape

    xxT = X[:, np.newaxis, :] * X[np.newaxis, :, :]

    vec_xxT = xxT.reshape(num_features * num_features, num_samples)

    phi_X = np.vstack([vec_xxT, X])

    return phi_X
