from lib.models.mvg_binary_classifiers import *
from lib.preprocessing.pca import pca_fit


def run_feature_selection_on_mvgs(ds, to_drop=[4, 5]):
    print("-" * 80)
    print(
        "\nRunning MVG and variants classification after feature selection (DROPPING features)...\n"
    )

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()
    x_train_dropped, x_val_dropped = ds.drop_features(to_drop)

    print(f"Dropped features: {to_drop}\n")

    mvg_classifier = MVGClassifier(c1_label=1, c2_label=0, name="MVG")
    nb_classifier = NBClassifier(c1_label=1, c2_label=0, name="Naive bayes")
    tied_classifier = TIEDClassifier(c1_label=1, c2_label=0, name="Tied MVG")

    mvg_classifier.fit(x_train_dropped, y_train)
    nb_classifier.fit(x_train_dropped, y_train)
    tied_classifier.fit(x_train_dropped, y_train)

    print()

    mvg_classifier.classify(x_val_dropped, y_val, verbose=True)
    nb_classifier.classify(x_val_dropped, y_val, verbose=True)
    tied_classifier.classify(x_val_dropped, y_val, verbose=True)

    print()
    print("-" * 80)


def run_feature_selection_on_mvgs_pca(ds, to_drop=[4, 5]):
    print("-" * 80)
    print(
        "\nRunning MVG and variants classification after feature selection and PCA...\n"
    )

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()
    x_train_dropped, x_val_dropped = ds.drop_features(to_drop)

    print(f"Dropped features: {to_drop}\n")

    mvg_classifier = MVGClassifier(c1_label=1, c2_label=0, name="MVG")
    nb_classifier = NBClassifier(c1_label=1, c2_label=0, name="Naive bayes")
    tied_classifier = TIEDClassifier(c1_label=1, c2_label=0, name="Tied MVG")

    for m in [1, 2, 3, 4, 5, 6]:
        print("-" * 40)
        print("\nClassification after PCA with M =", m)

        pcad_x_train, pcad_x_val = pca_fit(x_train_dropped, x_val_dropped, m)

        mvg_classifier.fit(pcad_x_train, y_train)
        nb_classifier.fit(pcad_x_train, y_train)
        tied_classifier.fit(pcad_x_train, y_train)

        print()

        mvg_classifier.classify(pcad_x_val, y_val, verbose=True)
        nb_classifier.classify(pcad_x_val, y_val, verbose=True)
        tied_classifier.classify(pcad_x_val, y_val, verbose=True)
        print("-" * 40)

    print("-" * 80)
