from lib.models.lda_binary_classifier import LDABinaryClassifier


def run_lda_classification(ds):
    print("-" * 80)
    print("\nRunning LDA classification on spoofing dataset...\n")

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    lda_classifier = LDABinaryClassifier(
        t_samples=x_train,
        t_labels=y_train,
        v_samples=x_val,
        v_labels=y_val,
        c1_label=1,
        c2_label=0,
    )

    print("-" * 40)
    print("\nClassification best treshold selection with PCA...")
    lda_classifier.classify_with_best_threshold()
    lda_classifier.classify_with_best_threshold_pca()
    print("-" * 40)
    print("\nRunning min_dist treshold and  PCA...")
    lda_classifier.classify_with_mean_dist_treshold(with_pca=True)

    print()
    print("-" * 80)
