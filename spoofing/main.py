from dataset.loader import DatasetLoader
from plotter import Plotter
from preprocessing.pca import *
from preprocessing.lda import *
from models.lda_binary_classifier import LDABinaryClassifier

# from models.mvg_classifier import MVGClassifier
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
    plt.plot_features_onefig(genuines, counterfeits, x_label="Feature", save=True, name="features_hists")
    plt.plot_scatters(genuines, counterfeits)
    """

    """PCA
    pcad = pca(samples, 6)
    pcad_genuines = ds.get_genuines_from(pcad)
    pcad_counterfeits = ds.get_counterfeits_from(pcad)
    plt.plot_features_onefig(
        pcad_genuines,
        pcad_counterfeits,
        x_label="Component",
        name="principal_components",
    )
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
    lda_classifier = LDABinaryClassifier(
        t_samples=ds.training_samples,
        t_labels=ds.training_labels,
        v_samples=ds.validation_samples,
        v_labels=ds.validation_labels,
        c1_label=1,
        c2_label=0,
    )
    lda_classifier.classify_with_mean_dist_treshold()
    lda_classifier.classify_with_mean_dist_treshold(with_pca=True)
    lda_classifier.classify_with_best_threshold()
    """

    """PLOT GAUSSIANS TO FEATURES
    # plt.plot_all_1d_gau(genuines, counterfeits)
    """

    """MVG CLASSIFIER
    mvg_classifier = MVGClassifier(
        t_samples=ds.training_samples,
        t_labels=ds.training_labels,
        v_samples=ds.validation_samples,
        v_labels=ds.validation_labels,
        c1_label=1,
        c2_label=0,
    )
    mvg_classifier.classify()
    mvg_classifier.classify(with_naive_bayes=True)
    mvg_classifier.classify(with_tied=True)

    # g_corr_matrix = get_pearson_matrix(mvg_classifier.parameters["g_mle"][1])
    # c_corr_matrix = get_pearson_matrix(mvg_classifier.parameters["c_mle"][1])
    # plt.plot_correlation_matrixes(g_corr_matrix, c_corr_matrix)
    """

    """FEATURE SELECTION (dropping features)
    #ds.drop_features([4, 5])
    #ds.drop_features([2, 3, 4, 5])
    print("FEATURE SELECTED")
    fs_mvg_classifier = MVGClassifier(
        t_samples=ds.training_samples,
        t_labels=ds.training_labels,
        v_samples=ds.validation_samples,
        v_labels=ds.validation_labels,
        c1_label=1,
        c2_label=0,
    )
    fs_mvg_classifier.classify()
    fs_mvg_classifier.classify(with_naive_bayes=True)
    fs_mvg_classifier.classify(with_tied=True)
    """

    # """MVG WITH PCA

    # """


"""
def function_one(plt, ds):
    plt.plot_features(ds.genuines, ds.counterfeits)
    print("Nice informing analysis about this plots")
    print("Do you want to procede with scatters? (Y/N)")
    choice = input("Enter your choice: ")

    if choice == "Y":
        plt.plot_scatters(ds.genuines, ds.counterfeits)
    
    return 

def function_two():
    print("Function Two was triggered.")

def function_three():
    print("Function Three was triggered.")

def display_menu():
    print("Welcome to this Fingerprint spoofing journey!")
    print("Choose what to do with this spoofing dataset:")
    print("1. Show preliminary plots")
    print("2. Apply PCA to the dataset")
    print("3. Apply LDA to the dataset")
    print("0. Exit")

def main():
    loader = DatasetLoader()
    ds = loader.load()

    samples = ds.samples
    labels = ds.labels

    genuines = ds.genuines
    counterfeits = ds.counterfeits

    plt = Plotter()

    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            function_one(plt, ds)
        elif choice == "2":
            function_two()
        elif choice == "3":
            function_three()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
"""


if __name__ == "__main__":
    main()
