import sys
from lib.dataset.loader import DatasetLoader
from lib.preprocessing.pca import *
from lib.preprocessing.lda import *

from scripts.preliminary import *
from scripts.dim_red import *
from scripts.feature_selection import *
from scripts.lda_classification import *
from scripts.models_evaluation import *
from scripts.mvg_classification import *
from scripts.bayes_plots import *
from scripts.logistic_regression import *
from scripts.svm_classification import *


from lib.util.load_store import *


def main():

    file_path = sys.argv[1] if len(sys.argv) == 2 else "./dataset/trainData.txt"
    loader = DatasetLoader(file_path=file_path)
    ds = loader.load()
    """
    # run_preliminary_plots(ds, save_plots=False)
    # run_dim_red_on_ds(ds, save_plots=False)
    # run_lda_classification(ds)
    # run_gaussians_to_features_plot(ds, save_plots=False)

    # Classify with MVG, NB, TIED on the unmodified dataset
    run_mvg_classification(ds)
    # Classify with MVG, NB, TIED after applying PCA
    run_mvg_classification_with_pca(ds)
    # Classify with MVG, NB, TIED after dropping 5,6
    run_feature_selection_on_mvgs(ds, to_drop=[4, 5])
    # Classify with MVG, NB, TIED with only 1,2
    run_feature_selection_on_mvgs(ds, to_drop=[2, 3, 4, 5])
    # Classify with MVG, NB, TIED with only 3,4
    run_feature_selection_on_mvgs(ds, to_drop=[0, 1, 4, 5])

    
    # Run MVG and Variants models evaluation on 3 different applications, use verbose=True to see the results
    mvg_results, nb_results, tied_results = run_mvgs_pca_evaluations(
        ds,
        [Application(0.1, 1, 1), Application(0.5, 1, 1), Application(0.9, 1, 1)],
        return_best_only=True,
        verbose=True,
    )
    
    """
    # Run evaluation on main app, store the best models in ./models/best_models/.
    # Use verbose to see results and store to store best models
    run_mvgs_pca_evaluations_on_main_app(
        ds=ds,
        verbose=False,
        store=False,
        metrics=["mindcf", "norm_dcf"],
    )

    # models = read_models("./best-models/mindcf/mvg/")
    # run_bayes_plots(ds, models)

    """
    run_logistic_reg(ds)
    run_logistic_reg_with_few_samples(ds)
    run_logistic_prior_weighted_logreg(ds)
    run_quadratic_logreg(ds)
    run_logreg_after_centering(ds)

    run_logregs_pca_evaluations_on_main_app(
        ds=ds,
        verbose=False,
        store=False,
        metrics=["mindcf", "norm_dcf"],
    )

    run_linear_svm_classification(ds)
    run_linear_svm_classification_after_centering(ds)
    run_quadratic_svm_classification(ds)
    run_rbf_svm_classification(ds)
    """


if __name__ == "__main__":
    main()
