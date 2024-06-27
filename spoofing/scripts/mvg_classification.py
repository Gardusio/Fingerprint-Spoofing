from lib.preprocessing.pca import pca_fit
from lib.models.mvg_binary_classifiers import *


# MVG CLASSIFICATION WITH NB AND TIED
def run_mvg_classification(ds):
    print("-" * 80)
    print("\nRunning MVG and variants classification on spoofing dataset...\n")

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    mvg_classifier = MVGClassifier(c1_label=1, c2_label=0, name="MVG")
    nb_classifier = NBClassifier(c1_label=1, c2_label=0, name="Naive bayes")
    tied_classifier = TIEDClassifier(c1_label=1, c2_label=0, name="Tied MVG")

    mvg_classifier.fit(x_train, y_train)
    nb_classifier.fit(x_train, y_train)
    tied_classifier.fit(x_train, y_train)

    print()

    mvg_classifier.classify(x_val, y_val, verbose=True)
    nb_classifier.classify(x_val, y_val, verbose=True)
    tied_classifier.classify(x_val, y_val, verbose=True)

    # TODO: WHAT'S THIS
    # g_corr_matrix = get_pearson_matrix(mvg_classifier.parameters["g_mle"][1])
    # c_corr_matrix = get_pearson_matrix(mvg_classifier.parameters["c_mle"][1])
    # plt.plot_correlation_matrixes(g_corr_matrix, c_corr_matrix)
    print("\n")

    print("-" * 80)


def run_mvg_classification_with_pca(ds):
    print("-" * 80)
    print("\nRunning MVG and its variant classification after applying PCA...\n")

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    mvg_classifier = MVGClassifier(c1_label=1, c2_label=0, name="MVG")
    nb_classifier = NBClassifier(c1_label=1, c2_label=0, name="Naive bayes")
    tied_classifier = TIEDClassifier(c1_label=1, c2_label=0, name="Tied MVG")

    for m in [1, 2, 3, 4, 5, 6]:
        print("-" * 40)
        print("\nClassification after PCA with M =", m)

        pcad_x_train, pcad_x_val = pca_fit(x_train, x_val, m)

        mvg_classifier.fit(pcad_x_train, y_train)
        nb_classifier.fit(pcad_x_train, y_train)
        tied_classifier.fit(pcad_x_train, y_train)

        print()

        mvg_classifier.classify(pcad_x_val, y_val, verbose=True)
        nb_classifier.classify(pcad_x_val, y_val, verbose=True)
        tied_classifier.classify(pcad_x_val, y_val, verbose=True)
        print("-" * 40)

    print("-" * 80)
