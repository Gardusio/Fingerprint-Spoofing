import numpy as np
from models.svm_classifier import SVMClassifier
from evaluation.bayes_evaluator import BinaryBayesEvaluator
from evaluation.application import Application
from util.plotter import Plotter
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
                name="QuadraticSVM",
            )
        else:
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
        )

        mindcf = evaluator.compute_mindcf()
        dcf = evaluator.compute_dcf()

        c_to_dcfs.append((C, dcf, mindcf))

        print(f"{'Quadratic' if use_quadratic else 'Linear'}SVM-C-{C}")
        print("Error rate: %.1f" % (err * 100))
        print("minDCF - pT = 0.1: %.4f" % mindcf)
        print("actDCF - pT = 0.1: %.4f" % dcf)
        print("-" * 40)

    cs, dcfs, min_dcfs = zip(*c_to_dcfs)
    Plotter().plot_dcf_vs_reg(cs, dcfs, min_dcfs, model_name="LinearSVM")


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

    svm_classification(x_train, y_train, x_val, y_val, use_kernel=True)
    print("-" * 80)


def run_rbf_svm_classification(ds):
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

    cs, gammas = zip(*c_to_gamma)
    Plotter().plot_multiple_dcf_vs_reg(cs, gammas[0], model_name="RBF")


"""
if __name__ == "__main__":

    for kernelFunc in [
        polyKernel(2, 0),
        polyKernel(2, 1),
        rbfKernel(1.0),
        rbfKernel(10.0),
    ]:
        for eps in [0.0, 1.0]:
            fScore = train_dual_SVM_kernel(x_train, y_train, 1.0, kernelFunc, eps)
            SVAL = fScore(DVAL)
            PVAL = (SVAL > 0) * 1
            err = (PVAL != y_val).sum() / float(y_val.size)
            print("Error rate: %.1f" % (err * 100))
            print(
                "minDCF - pT = 0.5: %.4f"
                % bayesRisk.compute_minDCF_binary_fast(SVAL, y_val, 0.5, 1.0, 1.0)
            )
            print(
                "actDCF - pT = 0.5: %.4f"
                % bayesRisk.compute_actDCF_binary_fast(SVAL, y_val, 0.5, 1.0, 1.0)
            )
            print()
"""
