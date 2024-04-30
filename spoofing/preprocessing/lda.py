from dataset.ds_utils import *
from mathutils.math_utils import *
from scipy.linalg import eigh
from validation.test_utils import *
from preprocessing.pca import get_pca_matrix


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


def lda(samples, labels, nc=[0, 1], m=2):
    return get_lda_matrix(samples, labels, nc, m).T @ samples


def get_lda_mean_dist_treshold(samples, labels):
    return (get_class_mean(samples, labels, 0) + get_class_mean(samples, labels, 1)) / 2


# Returns the lda matrix oriented in a way that genuines have greather projected mean
def orient_lda_matrix(Ui, samples, labels):
    if (
        get_genuines_samples(samples, labels).mean()
        < get_counterfeits_samples(samples, labels).mean()
    ):
        return -Ui
    return Ui


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

        results.append((m, pmean_diff_threshold_m, error_rate_m))
    return results
