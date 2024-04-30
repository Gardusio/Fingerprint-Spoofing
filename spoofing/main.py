from dataset.loader import DatasetLoader
from plotter import Plotter
from dataset.ds_utils import *
from preprocessing.pca import *
from preprocessing.lda import *


def main():

    # TODO: specify ds file path via args
    loader = DatasetLoader()
    ds = loader.load()
    samples, labels = ds

    plt = Plotter()

    # Q2: if ds is big, how to do this in a single pass while leveraging np?
    genuines = get_genuines_samples(samples, labels)
    counterfeits = get_counterfeits_samples(samples, labels)

    """
    ############################################## LAB 2 ##########################################

    plt.plot_features(genuines, counterfeits)
    plt.plot_scatters(genuines, counterfeits)
    """

    """
    ############################################## LAB 3 ##########################################
    ############################################## LAB 3 - PCA ####################################

    pcad = pca(samples, 6)
    pcad_genuines = get_genuines_samples(pcad, labels)
    pcad_counterfeits = get_counterfeits_samples(pcad, labels)
    plt.plot_features(pcad_genuines, pcad_counterfeits, x_label="Component")

    ############################################## LAB 3 - LDA #####################################

    U_lda = -get_lda_matrix(samples, labels)
    LDA_samples = U_lda.T @ samples

    plt.plot_features(
        get_genuines_samples(LDA_samples, labels),
        get_counterfeits_samples(LDA_samples, labels),
        x_label="LDA direction",
        range_v=1,
    )

    ############################################## LAB 3 - LDA classification ######################

    training_samples, training_labels, validation_samples, validation_labels = (
        loader.split_ds_2to1(samples, labels)
    )

    Ut_lda = get_lda_matrix(training_samples, training_labels)
    training_samples_lda = Ut_lda.T @ training_samples
    plt.plot_features(
        get_genuines_samples(training_samples_lda, training_labels),
        get_counterfeits_samples(training_samples_lda, training_labels),
        x_label="LDA direction",
        range_v=1,
    )
    validation_samples_lda = Ut_lda.T @ validation_samples
    num_validation_samples = float(validation_samples_lda.shape[1])
    threshold = get_lda_mean_dist_treshold(training_samples_lda, training_labels)

    error_rate = get_error_rate(
        validation_samples_lda, num_validation_samples, threshold, validation_labels
    )
    print(
        f"LDA classification error rate with threshold {threshold}: ",
        error_rate,
    )

    # We get the range by observing the training LDA histogram
    # The hope is within this range there's a better treshold than the projected mean diff
    error_rate, best_threshold = get_best_threshold(
        validation_samples_lda,
        validation_labels,
        num_validation_samples,
        np.arange(-1, 1, 0.001),
    )

    print("Best LDA threshold results in : ", error_rate, best_threshold)

    ########################################### LAB 3 - PCA + LDA classification ######################
    # We now employ the projected mean diff as threshold, but we could employ the same observations as before on each m

    results = get_accuracy_as_pca_dims_function(
        training_samples,
        training_labels,
        validation_samples,
        num_validation_samples,
        validation_labels,
    )

    for m, t, error_rate in results:
        print(f"Error rate with {m} pca dimensions and threshold {t} is: ", error_rate)
    """

    ############################################## LAB 4 - Fitting Gaussians #######################

    for f_idx in range(0, 6):
        genuines_feature_samples = genuines[f_idx, :].reshape(1, genuines.shape[1])
        counterfeits_feature_samples = counterfeits[f_idx, :].reshape(
            1, counterfeits.shape[1]
        )

        g_plot, g_gaussian_pdf = fit_gaussian_to_feature(
            genuines_feature_samples, f_idx
        )
        c_plot, c_gaussian_pdf = fit_gaussian_to_feature(
            counterfeits_feature_samples, f_idx
        )

        plt.plot_1d_gau(
            c1_sample_set=genuines_feature_samples.ravel(),
            c1_plot=g_plot,
            c1_pdf=g_gaussian_pdf,
            c2_sample_set=counterfeits_feature_samples.ravel(),
            c2_plot=c_plot,
            c2_pdf=c_gaussian_pdf,
            f_idx=f_idx,
            save=True,
        )

    ############################################## LAB 5 - MVG, NB-MVG, Tied MVG #######################


if __name__ == "__main__":
    main()
