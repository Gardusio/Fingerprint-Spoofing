from lib.models.mvg_binary_classifiers import *


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
