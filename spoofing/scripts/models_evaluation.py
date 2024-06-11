from evaluation.mvgs_evaluator import MVGEvaluator
from evaluation.bayes_evaluator import BinaryBayesEvaluator
from evaluation.application import Application
from preprocessing.pca import pca_fit
from models.mvg_binary_classifiers import *
from util.load_store import store_models


def run_mvgs_pca_evaluations_on_main_app(
    ds,
    verbose=False,
    metrics=["mindcf", "norm_dcf"],
    store=False,
    store_paths=["./models/best_models/mindcf", "./models/best_models/dcf"],
):
    (
        (mvg_best_mindcf, mvg_best_dcf),
        (nb_best_mindcf, nb_best_dcf),
        (tied_best_mindcf, tied_best_dcf),
    ) = evaluate_mvgs_on_main_app(ds, verbose=verbose, metrics=metrics)

    if store:
        store_models(
            store_paths[0],
            [mvg_best_mindcf[1], nb_best_mindcf[1], tied_best_mindcf[1]],
        )

        store_models(
            store_paths[1],
            [mvg_best_dcf[1], nb_best_dcf[1], tied_best_dcf[1]],
        )

    return (
        (mvg_best_mindcf, mvg_best_dcf),
        (nb_best_mindcf, nb_best_dcf),
        (tied_best_mindcf, tied_best_dcf),
    )


def evaluate_mvgs_on_main_app(ds, verbose=False, metrics=["mindcf", "norm_dcf"]):
    """
    Evaluate on project main application and return only best model for given metrics
    """

    print("-" * 80)
    print("\nEvaluating MVG with and without PCA on MAIN APPLICATION...")

    mvg_results, nb_results, tied_results = get_main_app_results(ds, verbose=verbose)

    mvg_models = (mvg_results[metric] for metric in metrics)
    nb_models = (nb_results[metric] for metric in metrics)
    tied_models = (tied_results[metric] for metric in metrics)

    print("-" * 80)
    return mvg_models, nb_models, tied_models


def get_main_app_results(ds, verbose=False):
    higher_fake_app = Application(0.1, 1.0, 1.0, "Higher counterfeits prior")

    app_to_mvg_results, app_to_nb_results, app_to_tied_results = (
        run_mvgs_pca_evaluations(
            ds,
            [higher_fake_app],
            return_best_only=True,
            verbose=verbose,
        )
    )

    mvg_results = app_to_mvg_results[higher_fake_app.get_name()]
    nb_results = app_to_nb_results[higher_fake_app.get_name()]
    tied_results = app_to_tied_results[higher_fake_app.get_name()]

    return mvg_results, nb_results, tied_results


def run_mvgs_pca_evaluations(
    ds,
    applications,
    return_best_only,
    verbose=False,
):
    """
    Use this to evaluate on multiple applications,

    if return best only return, for each application and for each metric, only the best performing model and stats.

    Otherwise return all models and stats
    """
    print("-" * 40)
    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    mvg_results, nb_results, tied_results = run_pca_evaluations(
        applications, x_train, y_train, x_val, y_val, verbose=verbose
    )

    if return_best_only:
        print_best_report(applications, mvg_results[0], nb_results[0], tied_results[0])
        return mvg_results[0], nb_results[0], tied_results[0]

    print("-" * 40)
    return mvg_results[1], nb_results[1], tied_results[1]


def get_pca_experiments(x_train, y_train, x_val, y_val):
    mvg_experiments = []
    nb_experiments = []
    tied_experiments = []

    mvg_classifier = MVGClassifier(1, 0, name=f"MVG").fit(x_train, y_train)
    nb_classifier = NBClassifier(1, 0, name=f"NB").fit(x_train, y_train)
    tied_classifier = TIEDClassifier(1, 0, name=f"TIED").fit(x_train, y_train)

    mvg_experiments.append((mvg_classifier, x_val, y_val))
    nb_experiments.append((nb_classifier, x_val, y_val))
    tied_experiments.append((tied_classifier, x_val, y_val))

    for m in [1, 2, 3, 4, 5]:
        pcad_x_train, pcad_x_val = pca_fit(x_train, x_val, m)

        mvg_classifier = MVGClassifier(1, 0, name=f"MVG-PCA-{m}").fit(
            pcad_x_train, y_train
        )
        nb_classifier = NBClassifier(1, 0, name=f"NB-PCA-{m}").fit(
            pcad_x_train, y_train
        )
        tied_classifier = TIEDClassifier(1, 0, name=f"TIED-PCA-{m}").fit(
            pcad_x_train, y_train
        )

        mvg_experiments.append((mvg_classifier, pcad_x_val, y_val))
        nb_experiments.append((nb_classifier, pcad_x_val, y_val))
        tied_experiments.append((tied_classifier, pcad_x_val, y_val))

    return mvg_experiments, nb_experiments, tied_experiments


def run_pca_evaluations(applications, x_train, y_train, x_val, y_val, verbose=False):

    mvg_experiments, nb_experiments, tied_experiments = get_pca_experiments(
        x_train, y_train, x_val, y_val
    )

    print("-" * 40)
    print("\nEvaluating models...\n")
    evaluator = MVGEvaluator()

    app_to_mvg_results, app_to_nb_results, app_to_tied_results = (
        evaluator.evaluate_mvgs(
            applications,
            mvg_experiments,
            nb_experiments,
            tied_experiments,
            verbose=verbose,
        )
    )

    return app_to_mvg_results, app_to_nb_results, app_to_tied_results



###################### PRINT MVG EVALUATION REPORTS #######################
def print_best_report(applications, app_to_mvg_best, app_to_nb_best, app_to_tied_best):
    print("-" * 40)
    print("\nModel evaluations Best only Report")
    print_apps_report(applications, [app_to_mvg_best, app_to_nb_best, app_to_tied_best])


def print_apps_report(applications, models_results):
    for model_result in models_results:
        for app in applications:
            app_name = app.get_name()
            best = model_result[app_name]

            print(f"\nBest models for {app.info()}: \n")
            for metric, result in best.items():
                value, model = result
                print(
                    f"{model.get_name()} is the best model considering: {metric} - {value}"
                )
