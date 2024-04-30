from dataset.loader import DatasetLoader
from plotter import Plotter
from preprocessing.pca import *
from preprocessing.lda import *
from models.lda_classifier import LDAClassifier
from models.mvg_classifier import MVGClassifier


def main():

    # TODO: specify ds file path via args
    loader = DatasetLoader()
    ds = loader.load()

    samples = ds.samples
    labels = ds.labels

    genuines = ds.genuines
    counterfeits = ds.counterfeits

    plt = Plotter()

    """PRELIMINARY PLOTS
    plt.plot_features(genuines, counterfeits)
    plt.plot_scatters(genuines, counterfeits)
    """

    """PCA
    pcad = pca(samples, 6)
    pcad_genuines = ds.get_genuines_from(pcad)
    pcad_counterfeits = ds.get_counterfeits_from(pcad)
    plt.plot_features(pcad_genuines, pcad_counterfeits, x_label="Component")
    """

    """LDA
    U_lda = -get_lda_matrix(samples, labels)
    LDA_samples = U_lda.T @ samples
    
    plt.plot_features(
        ds.get_genuines_from(LDA_samples),
        ds.get_counterfeits_from(LDA_samples),
        x_label="LDA direction",
        range_v=1,
    )
    """

    """LDA CLASSIFIER
    # lda_classifier = LDAClassifier(ds)
    # lda_classifier.classify_with_mean_dist_treshold()
    # lda_classifier.classify_with_mean_dist_treshold(with_pca=True)
    # lda_classifier.classify_with_best_threshold()
    """

    """PLOT GAUSSIANS TO FEATURES
    # plt.plot_all_1d_gau(genuines, counterfeits)
    """

    # """MVG CLASSIFIER
    mvg_classifier = MVGClassifier(ds)
    mvg_classifier.classify()
    mvg_classifier.classify(with_naive_bayes=True)
    mvg_classifier.classify(with_tied=True)

    # g_corr_matrix = get_pearson_matrix(mvg_classifier.parameters["g_mle"][1])
    # c_corr_matrix = get_pearson_matrix(mvg_classifier.parameters["c_mle"][1])
    # plt.plot_correlation_matrixes(g_corr_matrix, c_corr_matrix)
    # """

    """FEATURE SELECTION (dropping features)
    #ds.drop_features([4, 5])
    #ds.drop_features([2, 3, 4, 5])
    print("FEATURE SELECTED")
    fs_mvg_classifier = MVGClassifier(ds)
    fs_mvg_classifier.classify()
    fs_mvg_classifier.classify(with_naive_bayes=True)
    fs_mvg_classifier.classify(with_tied=True)
    """

    """MVG WITH PCA
    """



if __name__ == "__main__":
    main()
