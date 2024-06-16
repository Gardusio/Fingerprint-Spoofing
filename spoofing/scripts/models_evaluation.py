from evaluation.mvgs_evaluator import MVGEvaluator
from evaluation.application import Application
from preprocessing.pca import pca_fit
from models.mvg_binary_classifiers import *
from util.load_store import store_models
from models.logistic_regression_classifier import *
from evaluation.logreg_evaluator import LogRegEvaluator


############################## LOG REG ###########################################

def run_logregs_pca_evaluations_on_main_app(
    ds,
    verbose=False,
    metrics=["mindcf", "norm_dcf"],
    store=False,
    store_paths=[
        "./models/best_models/mindcf/logreg",
        "./models/best_models/dcf/logreg",
    ],
):
    (
        (lr_best_mindcf, lr_best_dcf),
        (pw_best_mindcf, pw_best_dcf),
        (q_best_mindcf, q_best_dcf),
    ) = evaluate_logregs_on_main_app(ds, verbose=verbose, metrics=metrics)
    if store:
        store_models(
            store_paths[0],
            [lr_best_mindcf[1], pw_best_mindcf[1], q_best_mindcf[1]],
        )
        store_models(
            store_paths[1],
            [lr_best_dcf[1], pw_best_dcf[1], q_best_dcf[1]],
        )
    return (
        (lr_best_mindcf, lr_best_dcf),
        (pw_best_mindcf, pw_best_dcf),
        (q_best_mindcf, q_best_dcf),
    )


def evaluate_logregs_on_main_app(ds, verbose=False, metrics=["mindcf", "norm_dcf"]):
    """
    Evaluate on project main application and return only best model for given metrics
    """
    print("-" * 80)
    print("\nEvaluating LOG-REGS with and without PCA on MAIN APPLICATION...")
    higher_fake_app = Application(0.1, 1.0, 1.0, "Higher counterfeits prior")
    app_to_mvg_results, app_to_nb_results, app_to_tied_results = (
        run_logregs_pca_evaluations(
            ds,
            application=higher_fake_app,
            return_best_only=True,
            verbose=verbose,
        )
    )
    l_results = app_to_mvg_results[higher_fake_app.get_name()]
    pw_results = app_to_nb_results[higher_fake_app.get_name()]
    q_results = app_to_tied_results[higher_fake_app.get_name()]
    l_models = (l_results[metric] for metric in metrics)
    pw_models = (pw_results[metric] for metric in metrics)
    q_models = (q_results[metric] for metric in metrics)
    print("-" * 80)
    return l_models, pw_models, q_models


def run_logregs_pca_evaluations(
    ds,
    application,
    return_best_only,
    verbose=False,
):
    """
    Use this to evaluate on a single application,
    if return best only return, for each metric, only the best performing model and stats.
    Otherwise return all models and stats
    """
    print("-" * 40)
    x_train, y_train, x_val, y_val = ds.split_ds_2to1()
    l_experiments, pw_experiments, q_experiments = (
        get_logreg_regularization_pca_experiments(
            x_train, y_train, x_val, y_val, application
        )
    )
    print("-" * 40)
    print("\nEvaluating models...\n")
    l_results, pw_results, quad_results = LogRegEvaluator.evaluate_logregs(
        [application],
        l_experiments,
        pw_experiments,
        q_experiments,
        verbose=verbose,
    )
    if return_best_only:
        print_best_report([application], l_results[0], pw_results[0], quad_results[0])
        return l_results[0], pw_results[0], quad_results[0]
    print("-" * 40)
    return l_results[1], pw_results[1], quad_results[1]


def get_logreg_regularization_pca_experiments(
    x_train, y_train, x_val, y_val, application
):
    logreg_experiments = []
    w_logreg_experiments = []
    q_logreg_experiments = []

    empirical_prior = (y_train == 1).sum() / y_train.size

    logreg = LogisticRegressionBinaryClassifier(
        1, 0, name="LogReg", empirical_prior=empirical_prior, application=application
    )
    w_logreg = LogisticRegressionBinaryClassifier(
        1,
        0,
        name="WeightedLogReg",
        weighted=True,
        empirical_prior=empirical_prior,
        application=application,
    )
    q_logreg = LogisticRegressionBinaryClassifier(
        1,
        0,
        name="QuadraticLogReg",
        empirical_prior=empirical_prior,
        application=application,
    )
    x_train_expanded = expand_features(x_train)
    x_val_expanded = expand_features(x_val)

    for l in np.logspace(-4, 2, 13):
        logreg = logreg.with_name(f"LogReg-LAMBDA-{l}", new=True)
        logreg_experiments.extend(
            get_model_pca_experiments(x_train, y_train, x_val, y_val, logreg, l)
        )
        w_logreg = w_logreg.with_name(f"WeightedLogReg-LAMBDA-{l}", new=True)
        w_logreg_experiments.extend(
            get_model_pca_experiments(x_train, y_train, x_val, y_val, w_logreg, l)
        )
        q_logreg = q_logreg.with_name(f"QuadraticLogReg-LAMBDA-{l}", new=True)
        q_logreg_experiments.extend(
            get_model_pca_experiments(
                x_train_expanded, y_train, x_val_expanded, y_val, q_logreg, l
            )
        )
    return logreg_experiments, w_logreg_experiments, q_logreg_experiments



############################## MVGs ###########################################
def run_mvgs_pca_evaluations_on_main_app(
    ds,
    verbose=False,
    metrics=["mindcf", "norm_dcf"],
    store=False,
    store_paths=["./models/best_models/mindcf/mvg", "./models/best_models/dcf/mvg"],
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

    mvg_models = (mvg_results[metric] for metric in metrics)
    nb_models = (nb_results[metric] for metric in metrics)
    tied_models = (tied_results[metric] for metric in metrics)

    print("-" * 80)
    return mvg_models, nb_models, tied_models


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

    mvg_experiments, nb_experiments, tied_experiments = get_pca_experiments(
        x_train, y_train, x_val, y_val
    )

    for mvg_exp in mvg_experiments:
        print("Name: ", mvg_exp[0].get_name())

    print("-" * 40)
    print("\nEvaluating models...\n")
    evaluator = MVGEvaluator()

    mvg_results, nb_results, tied_results = evaluator.evaluate_mvgs(
        applications,
        mvg_experiments,
        nb_experiments,
        tied_experiments,
        verbose=verbose,
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

    mvg_classifier = MVGClassifier(1, 0, name=f"MVG")
    nb_classifier = NBClassifier(1, 0, name=f"NB")
    tied_classifier = TIEDClassifier(1, 0, name=f"TIED")

    mvg_experiments.extend(
        get_model_pca_experiments(x_train, y_train, x_val, y_val, mvg_classifier)
    )
    nb_experiments.extend(
        get_model_pca_experiments(x_train, y_train, x_val, y_val, nb_classifier)
    )
    tied_experiments.extend(
        get_model_pca_experiments(x_train, y_train, x_val, y_val, tied_classifier)
    )

    return mvg_experiments, nb_experiments, tied_experiments



def get_model_pca_experiments(x_train, y_train, x_val, y_val, classifier, *fit_args):
    experiments = []

    # Fit the initial classifier without PCA and append to experiments
    initial_classifier = classifier.with_name(f"{classifier.get_name()}", new=True)
    experiments.append(
        (initial_classifier.fit(x_train, y_train, *fit_args), x_val, y_val)
    )

    name = classifier.get_name()

    for m in [1, 2, 3, 4, 5]:
        pcad_x_train, pcad_x_val = pca_fit(x_train, x_val, m)
        pca_classifier = classifier.with_name(f"{name}-PCA-{m}", new=True).fit(
            pcad_x_train, y_train, *fit_args
        )

        experiments.append((pca_classifier, pcad_x_val, y_val))

    return experiments


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
