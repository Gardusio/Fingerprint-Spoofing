import pprint
from preprocessing.pca import pca_fit
from models.mvg_binary_classifiers import *
from evaluation.mvgs_evaluator import MVGEvaluator


def print_mvg_pca_evaluation(
    model, applications, x_train, y_train, x_val, y_val, print_all=False
):
    evaluator = MVGEvaluator()
    experiments = []

    for m in [1, 2, 3, 4, 5, 6]:
        pcad_x_train, pcad_x_val = pca_fit(x_train, x_val, m)

        if model == "MVG":
            classifier = MVGClassifier(1, 0, name=f"MVG - with PCA: {m}").fit(
                pcad_x_train, y_train
            )
        elif model == "NB":
            classifier = NBClassifier(1, 0, name=f"NB - with PCA: {m}").fit(
                pcad_x_train, y_train
            )
        elif model == "TIED":
            classifier = TIEDClassifier(1, 0, name=f"TIED MVG - with PCA: {m}").fit(
                pcad_x_train, y_train
            )

        experiments.append((classifier, pcad_x_val, y_val))

    app_to_best_results, app_to_models_results = (
        evaluator.evaluate_models_on_applications(experiments, applications)
    )

    if print_all:
        print("\n Application Evaluation Report:")
        pprint.pprint(app_to_models_results)

    print("\n Best models:")
    for app in applications:
        print("\n For app: ", app.get_name())
        pprint.pprint(app_to_best_results[app.get_name()])
