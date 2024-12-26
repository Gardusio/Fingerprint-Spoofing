from lib.evaluation.bayes_evaluator import BinaryBayesEvaluator
from lib.evaluation.application import Application
from lib.preprocessing.pca import pca_fit
from lib.models.mvg_binary_classifiers import *
from lib.util.plotter import Plotter


# TODO: refactor this to get a suitable object instead of checking PCA- is in the name
def run_bayes_plots(ds, models):

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    for model in models:
        if "PCA-" in model.get_name():
            pca_components = model.get_name().split("PCA-")[1]
            x_train, x_val = pca_fit(x_train, x_val, int(pca_components))

        print(f"Plotting bayes errors for: {model.get_name()}...")

        run_bayes_plot(model, x_val, y_val, title=model.get_name())


def run_bayes_plot(model, x_val, y_val, title):
    plt = Plotter()

    prior_log_odds = np.linspace(-4, 4, 21)
    e_priors = 1.0 / (1.0 + np.exp(-prior_log_odds))

    actdcfs = []
    mindcfs = []

    for e_prior in e_priors:
        current_application = Application(e_prior, 1, 1)

        model.with_application(current_application)

        llrs, opt_predictions = model.classify(x_val, y_val)

        c_evaluator = BinaryBayesEvaluator(
            evaluation_labels=y_val,
            opt_predictions=opt_predictions,
            application=current_application,
            llrs=llrs,
            model_name=model.get_name(),
        )

        mindcfs.append(c_evaluator.compute_mindcf())

        actdcfs.append(c_evaluator.compute_dcf())

    plt.plot_bayes_errors(prior_log_odds, actdcfs, mindcfs, title=title)
