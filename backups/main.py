import sys
from dataset.loader import DatasetLoader
from util.plotter import Plotter
from preprocessing.pca import *
from preprocessing.lda import *

from models.lda_binary_classifier import LDABinaryClassifier
from models.mvg_binary_classifiers import MVGClassifier, NBClassifier, TIEDClassifier
from evaluation.application import Application
from evaluation.mvgs_evaluator import MVGEvaluator


def main():

    # TODO: specify ds file path via args
    file_path = sys.argv[1] if len(sys.argv) == 2 else "./dataset/trainData.txt"
    loader = DatasetLoader(file_path=file_path)

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

    """LDA CLASSIFIER TODO: REFACTOR THIS with "fit"
    lda_classifier = LDABinaryClassifier(
        t_samples=x_train,
        t_labels=y_train,
        v_samples=x_val,
        v_labels=y_val,
        c1_label=1,
        c2_label=0,
    )
    lda_classifier.classify_with_mean_dist_treshold()
    lda_classifier.classify_with_best_threshold()
    lda_classifier.classify_with_mean_dist_treshold(with_pca=True)
    """

    """PLOT GAUSSIANS TO FEATURES
    # plt.plot_all_1d_gau(genuines, counterfeits)
    """

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    """MVG CLASSIFICATION WITH NB AND TIED
    print("-" * 80)
    print("\n MVG CLASSIFICATION WITH NB AND TIED MODELS\n")
    
    mvg_classifier = MVGClassifier(c1_label=1, c2_label=0, name="MVG")
    nb_classifier = NBClassifier(c1_label=1, c2_label=0, name="Naive bayes")
    tied_classifier = TIEDClassifier(c1_label=1, c2_label=0, name="Tied MVG")

    mvg_classifier.fit(x_train, y_train)
    nb_classifier.fit(x_train, y_train)
    tied_classifier.fit(x_train, y_train)

    mvg_classifier.classify(x_val, y_val, verbose=True)
    nb_classifier.classify(x_val, y_val, verbose=True)
    tied_classifier.classify(x_val, y_val, verbose=True)

    # TODO: WHAT'S THIS
    # g_corr_matrix = get_pearson_matrix(mvg_classifier.parameters["g_mle"][1])
    # c_corr_matrix = get_pearson_matrix(mvg_classifier.parameters["c_mle"][1])
    # plt.plot_correlation_matrixes(g_corr_matrix, c_corr_matrix)
    """

    """FEATURE SELECTION (dropping features)
    x_train_dropped, x_val_dropped = ds.drop_features([4, 5])
    # x_train_dropped, x_val_dropped = ds.drop_features([2, 3, 4, 5])
    print("-" * 80)
    print("\nFEATURE SELECTED\n")
    mvg_classifier.fit(x_train_dropped, y_train)
    nb_classifier.fit(x_train_dropped, y_train)
    tied_classifier.fit(x_train_dropped, y_train)
    mvg_classifier.classify(x_val_dropped, y_val, verbose=True)
    nb_classifier.classify(x_val_dropped, y_val, verbose=True)
    tied_classifier.classify(x_val_dropped, y_val, verbose=True)
    """

    """MVG WITH PCA
    print("-" * 80)
    print("APPLYING PCA ON THE MVG AND ITS VARIANTS")
    for m in [1, 2, 3, 4, 5, 6]:
        pcad_x_train, pcad_x_val = pca_fit(x_train, x_val, m)
        print("\nMVG Classification after PCA with M =", m)
        mvg_classifier.fit(pcad_x_train, y_train)
        nb_classifier.fit(pcad_x_train, y_train)
        tied_classifier.fit(pcad_x_train, y_train)

        mvg_classifier.classify(pcad_x_val, y_val, verbose=True)
        nb_classifier.classify(pcad_x_val, y_val, verbose=True)
        tied_classifier.classify(pcad_x_val, y_val, verbose=True)
    """

    # """ MVG BAYES EVALUATION
    print("-" * 80)
    print("MVG BAYES EVALUATION ")
    print("\nEvaluating MVG with and without PCA...")
    evaluator = MVGEvaluator()

    # uniform_app = Application(0.5, 1.0, 1.0, "Uniform")
    # higher_gen_app = Application(0.9, 1.0, 1.0, "Higher genuine prior")
    higher_fake_app = Application(0.1, 1.0, 1.0, "Higher counterfeits prior")
    # applications = [uniform, higher_fake_app, higher_gen_app]
    applications = [higher_fake_app]

    evaluator.evaluate_pca_on_mvg(applications, x_train, y_train, x_val, y_val)
    evaluator.evaluate_pca_on_nb(applications, x_train, y_train, x_val, y_val)
    evaluator.evaluate_pca_on_tied(applications, x_train, y_train, x_val, y_val)

    # prior_log_odds = np.linspace(-4, 4, 21)
    # mindcf = evaluator.get_log_odds_min_dcf(prior_log_odds)
    # plt.plot_bayes_errors(dcf, prior_log_odds, mindcf, (-1, 1))
    # """


if __name__ == "__main__":
    main()
"""
def function_one(plt, ds):
    plt.plot_features(ds.genuines, ds.counterfeits)
    print("Nice informing analysis about this plots OR BETTER")
    print("Do you want to procede with scatters? (Y/N)")
    choice = input("Enter your choice: ")

    if choice == "Y":
        plt.plot_scatters(ds.genuines, ds.counterfeits)
    
    return 

def display_menu():
    print("Welcome to this Fingerprint spoofing journey!")
    print("Choose what to do with this spoofing dataset:")
    print("1. Show preliminary plots")
    print("2. Apply PCA to the dataset")
    print("3. Apply LDA to the dataset")
    print("4. Apply an LDA classifier")
    print("5. Apply a MVG classifier")
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
