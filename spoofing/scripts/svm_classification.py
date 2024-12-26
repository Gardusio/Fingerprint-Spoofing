import numpy as np
from lib.models.svm_classifier import SVMClassifier
from lib.evaluation.bayes_evaluator import BinaryBayesEvaluator
from lib.evaluation.application import Application
from lib.util.plotter import Plotter
from math import exp


def fit_and_classify_linear(x_train, y_train, x_val, y_val, application, C):
    svm = SVMClassifier(
        c1_label=1,
        c2_label=0,
        name=f"LinearSVM-C-{C}",
        use_application=True,
        application=application,
    )
    svm = svm.fit(x_train, y_train, C, K=1)
    scores, predictions = svm.classify(x_val, y_val)

    return scores, predictions


def fit_and_classify_kernel(
    x_train, y_train, x_val, y_val, application, C, eps, kernel=None, name="SVM"
):
    svm = SVMClassifier(
        c1_label=1,
        c2_label=0,
        name=f"{name}-C-{C}",
        use_application=True,
        application=application,
    )
    svm = svm.fit_kernel(x_train, y_train, kernel=kernel, C=C, eps=eps)

    scores, predictions = svm.classify(x_val, y_val, use_kernel=True)

    return scores, predictions


def svm_classification(x_train, y_train, x_val, y_val, use_quadratic=False):
    main_app = Application(0.1, 1, 1)

    c_to_dcfs = []
    for C in np.logspace(-5, 0, 11):
        print("-" * 40)

        if use_quadratic:
            model_name = "QuadraticSVM"
            # NOTE: Using application to compute bayes decisions with svm in a context in which emp prior is different from app prior
            scores, predictions = fit_and_classify_kernel(
                x_train,
                y_train,
                x_val,
                y_val,
                main_app,
                C,
                eps=0,
                kernel=SVMClassifier.polyKernel(2, 1),
                name=model_name,
            )
        else:
            model_name = "LinearSVM"
            # NOTE: Using application to compute bayes decisions with svm in a context in which emp prior is different from app prior
            scores, predictions = fit_and_classify_linear(
                x_train, y_train, x_val, y_val, main_app, C
            )

        err = (predictions != y_val).sum() / float(y_val.size)

        evaluator = BinaryBayesEvaluator(
            llrs=scores,
            opt_predictions=predictions,
            application=main_app,
            evaluation_labels=y_val,
            verbose=False,
            model_name=model_name,
        )

        mindcf = evaluator.compute_mindcf()
        dcf = evaluator.compute_dcf()

        c_to_dcfs.append((C, dcf, mindcf))

        print(f"{model_name}-C-{C}")
        print("Error rate: %.1f" % (err * 100))
        print("minDCF - pT = 0.1: %.4f" % mindcf)
        print("actDCF - pT = 0.1: %.4f" % dcf)
        print("-" * 40)

    cs, dcfs, min_dcfs = zip(*c_to_dcfs)
    Plotter().plot_dcf_vs_reg(cs, dcfs, min_dcfs, model_name=model_name)


def run_linear_svm_classification(ds):
    print("-" * 80)
    print("\nRunning SVM classification on spoofing dataset...\n")

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    svm_classification(x_train, y_train, x_val, y_val)
    print("-" * 80)


def run_linear_svm_classification_after_centering(ds):
    print("-" * 80)
    print("\nRunning SVM classification on CENTERED spoofing dataset...\n")

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()
    mean = x_train.mean(axis=1).reshape(x_train.shape[0], 1)
    x_train = x_train - mean
    x_val = x_val - mean

    svm_classification(x_train, y_train, x_val, y_val)
    print("-" * 80)


def run_quadratic_svm_classification(ds):
    print("-" * 80)
    print("\nRunning Quadratic SVM classification on spoofing dataset...\n")

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    svm_classification(x_train, y_train, x_val, y_val, use_quadratic=True)
    print("-" * 80)


def run_rbf_svm_classification(ds, verbose=True):
    main_app = Application(0.1, 1, 1)

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    gamma_to_dcfs = []
    c_to_gamma = []
    for C in np.logspace(-5, 0, 11):
        print("-" * 40)
        for gamma in [exp(-4), exp(-3), exp(-2), exp(-1)]:
            scores, predictions = fit_and_classify_kernel(
                x_train,
                y_train,
                x_val,
                y_val,
                main_app,
                C,
                eps=1.0,
                kernel=SVMClassifier.rbfKernel(gamma),
                name=f"RbfSVM-gamma-{gamma}",
            )

            err = (predictions != y_val).sum() / float(y_val.size)

            evaluator = BinaryBayesEvaluator(
                llrs=scores,
                opt_predictions=predictions,
                application=main_app,
                evaluation_labels=y_val,
                verbose=False,
                model_name="RbfSVM-gamma-{gamma}",
            )

            mindcf = evaluator.compute_mindcf()
            dcf = evaluator.compute_dcf()

            gamma_to_dcfs.append((gamma, dcf, mindcf))

            print(f"RbfSVM-gamma-{gamma}")
            print("Error rate: %.1f" % (err * 100))
            print("minDCF - pT = 0.1: %.4f" % mindcf)
            print("actDCF - pT = 0.1: %.4f" % dcf)
            print("-" * 40)
        c_to_gamma.append((C, gamma_to_dcfs))

    # TODO: Print a line for each gamma