import numpy as np
from scipy.linalg import eigh


def get_class_samples(samples, labels, c):
    return samples[:, labels == c]


def get_genuines_samples(samples, labels):
    return get_class_samples(samples, labels, 1)


def get_counterfeits_samples(samples, labels):
    return get_class_samples(samples, labels, 0)


# TODO clean this
def geat_feature_mean(ds, feature_idx):
    return ds.mean(axis=1).reshape(ds.shape[0], 1)[feature_idx]


def geat_feature_var(ds, feature_idx):
    return ds.var(axis=1).reshape(ds.shape[0], 1)[feature_idx]


def get_mean(samples, labels, c):
    return get_class_samples(samples, labels, c).mean()


### COMPUTING PCA
def get_mean_vector(samples):
    return samples.mean(axis=1).reshape(samples.shape[0], 1)


def get_covariance_matrix(features_matrix):
    ds_mean = features_matrix.mean(axis=1).reshape(features_matrix.shape[0], 1)

    centered_features = features_matrix - ds_mean

    return (centered_features @ centered_features.T) / float(features_matrix.shape[1])


def get_pca_matrix(features_matrix, m):

    covariance_m = get_covariance_matrix(features_matrix)

    # eig values are already sorted by svd
    s_eigh_vectors, s_values, vh = np.linalg.svd(covariance_m)

    # hence we can directly access first m columns of eigh_vectors
    P = s_eigh_vectors[:, 0:m]

    return P


def pca(samples, m):
    P = get_pca_matrix(samples, m)
    return P.T @ samples


### Computing LDA
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


# Returns the lda matrix oriented in a way that genuines have greather projected mean
def orient_lda_matrix(Ui, samples, labels):
    if (
        get_genuines_samples(samples, labels).mean()
        < get_counterfeits_samples(samples, labels).mean()
    ):
        return -Ui
    return Ui


def lda(samples, labels, nc=[0, 1], m=2):
    return get_lda_matrix(samples, labels, nc, m).T @ samples


def get_lda_mean_dist_treshold(samples, labels):
    return (get_mean(samples, labels, 0) + get_mean(samples, labels, 1)) / 2


def get_accuracy(validation_samples, threshold, validation_labels):
    PVAL = np.zeros(shape=validation_labels.shape, dtype=np.int32)
    PVAL[validation_samples[0] >= threshold] = 1
    PVAL[validation_samples[0] < threshold] = 0
    missed = np.sum(PVAL != validation_labels)
    return missed


def get_best_threshold(samples, labels, num_samples, range):
    rates = []
    for i in range:
        curr_threshold = i
        curr_rate = get_accuracy(samples, curr_threshold, labels) / num_samples
        rates.append((curr_rate, curr_threshold))

    return min(rates, key=lambda t: t[0])


def get_error_rate(samples, n_samples, t, labels):
    missed = get_accuracy(samples, t, labels)
    return missed / n_samples


def get_accuracy_as_pca_dims_function(
    training_samples,
    training_labels,
    validation_samples,
    num_validation_samples,
    validation_labels,
    pca_dims=[2, 3, 4, 5],
):
    results = []
    for m in pca_dims:
        Pi = get_pca_matrix(training_samples, m)
        training_pcad_m = Pi.T @ training_samples

        Ui = get_lda_matrix(training_pcad_m, training_labels)

        training_pcad_m_lda = Ui.T @ training_pcad_m

        Ui = orient_lda_matrix(Ui, training_pcad_m_lda, training_labels)

        validation_pcad_m = Pi.T @ validation_samples
        validation_pcad_lda_m = Ui.T @ validation_pcad_m
        pmean_diff_threshold_m = get_lda_mean_dist_treshold(
            training_pcad_m_lda, training_labels
        )
        error_rate_m = get_error_rate(
            validation_pcad_lda_m,
            num_validation_samples,
            pmean_diff_threshold_m,
            validation_labels,
        )
        """plt.plot_hist(
            genuines=get_genuines_samples(training_pcad_m_lda, training_labels),
            counterfeits=get_counterfeits_samples(training_pcad_m_lda, training_labels),
            range_v=1,
        )"""
        results.append((m, pmean_diff_threshold_m, error_rate_m))
    return results
