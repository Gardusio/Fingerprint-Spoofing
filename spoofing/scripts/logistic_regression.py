import numpy as np
from evaluation.application import Application
from evaluation.bayes_evaluator import BinaryBayesEvaluator
from models.logistic_regression_classifier import LogisticRegressionBinaryClassifier
from util.plotter import Plotter
from util.math_utils import get_err_rate, expand_features


def logistic_regression_classification(x_train, y_train, x_val, y_val, weighted=False):
    primary_app = Application(0.1, 1, 1, name="Higher fake prior")

    empirical_prior = (y_train == 1).sum() / y_train.size
    logreg_classifier = LogisticRegressionBinaryClassifier(
        name="LogRegClassifier",
        c1_label=1,
        c2_label=0,
        empirical_prior=empirical_prior,
        application=primary_app,
        use_application=True,
    )

    if weighted:
        logreg_classifier = logreg_classifier.with_weighted()

    l_to_dcfs = []
    for l in np.logspace(-4, 2, 13):
        print("-" * 40)

        logreg_classifier.fit(x_train, y_train, l)
        llrs, predictions = logreg_classifier.classify(x_val, y_val, verbose=True)
        err_rate = get_err_rate(predictions, y_val)
        print(f"Lambda = {l} -> error rate = {err_rate*100}")

        evaluator = BinaryBayesEvaluator(
            application=primary_app,
            llrs=llrs,
            evaluation_labels=y_val,
            opt_predictions=predictions,
            model_name=f"LogReg_reg_of_{l}",
        )
        print(f"Evaluating with {evaluator.model_name}")

        # TODO: use evaluator.evaluate instead
        mindcf = evaluator.compute_mindcf()
        dcf = evaluator.compute_dcf()
        l_to_dcfs.append((l, dcf, mindcf))
        print(f"minDCF = {mindcf}")
        print(f"actDCF = {dcf}")

        print("-" * 40)

    lambdas, dcfs, min_dcfs = zip(*l_to_dcfs)
    Plotter().plot_dcf_vs_reg(lambdas, dcfs, min_dcfs)


def run_logistic_reg(ds):
    print("-" * 80)
    print(
        "\nTraining and evaluating logistic regression with multiple regularizations\n"
    )

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()
    logistic_regression_classification(x_train, y_train, x_val, y_val)

    print("-" * 80)


def run_logistic_reg_with_few_samples(ds, few_samples=50):
    print("-" * 80)
    print("\nTraining on few samples ({few_samples})\n")

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()
    x_train = x_train[:, ::few_samples]
    y_train = y_train[::few_samples]

    logistic_regression_classification(x_train, y_train, x_val, y_val)

    print("-" * 80)


def run_logistic_prior_weighted_logreg(ds):
    x_train, y_train, x_val, y_val = ds.split_ds_2to1()
    logistic_regression_classification(x_train, y_train, x_val, y_val, weighted=True)


def run_logreg_after_centering(ds):
    x_train, y_train, x_val, y_val = ds.split_ds_2to1()
    mean = x_train.mean(axis=1).reshape(x_train.shape[0], 1)
    x_train = x_train - mean
    x_val = x_val - mean

    logistic_regression_classification(x_train, y_train, x_val, y_val)


def run_quadratic_logreg(ds):
    x_train, y_train, x_val, y_val = ds.split_ds_2to1()
    x_train, x_val = expand_features(x_train), expand_features(x_val)
    logistic_regression_classification(x_train, y_train, x_val, y_val)
