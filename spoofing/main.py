import sys
from dataset.loader import DatasetLoader
from preprocessing.pca import *
from preprocessing.lda import *

from scripts.preliminary import *
from scripts.dim_red import *
from scripts.feature_selection import *
from scripts.lda_classification import *
from scripts.models_evaluation import *
from scripts.mvg_classification import *
from scripts.bayes_plots import *
from scripts.logistic_regression import *
from scripts.svm_classification import *


from util.load_store import *


def main():

    file_path = sys.argv[1] if len(sys.argv) == 2 else "./dataset/trainData.txt"
    loader = DatasetLoader(file_path=file_path)
    ds = loader.load()

    # run_preliminary_plots(ds, save_plots=True)
    # run_dim_red_on_ds(ds, save_plots=True)
    # run_lda_classification(ds)
    # run_gaussians_to_features_plot(ds, save_plots=True)
    # run_mvg_classification(ds)
    # run_feature_selection_on_mvgs(ds, to_drop=[4, 5])
    # run_mvg_classification_with_pca(ds)

    # Run MVG and Variants models evaluation on 3 different applications, use verbose=True to see the results
    """mvg_results, nb_results, tied_results = run_mvgs_pca_evaluations(
        ds,
        [Application(0.1, 1, 1), Application(0.5, 1, 1), Application(0.9, 1, 1)],
        return_best_only=True,
        verbose=True,
    )"""

    # Run evaluation on main app, store the best models in ./models/best_models/.
    # Use verbose to see results and store to store best models
    """
    run_mvgs_pca_evaluations_on_main_app(
        ds=ds,
        verbose=True,
        store=False,
        metrics=["mindcf", "norm_dcf"],
    )
    """

    # models = read_models("./models/best_models/mindcf")
    # run_bayes_plots(ds, models)

    # run_logistic_reg(ds)
    # run_logistic_reg_with_few_samples(ds)
    # run_logistic_prior_weighted_logreg(ds)
    # run_quadratic_logreg(ds)
    # run_logreg_after_centering(ds)
    """
    run_logregs_pca_evaluations_on_main_app(
        ds=ds,
        verbose=True,
        store=True,
        metrics=["mindcf", "norm_dcf"],
    )
    """

    # run_linear_svm_classification(ds)
    # run_linear_svm_classification_after_centering(ds)
    # run_quadratic_svm_classification(ds)
    run_rbf_svm_classification(ds)


if __name__ == "__main__":
    main()

"""
def function_one(plt, ds):
    plt.plot_features(ds.genuines, ds.counterfeits)
    print("Nice informing analysis about this plots OR BETTER")
    print("Do you want to procede with scatters? (Y/N)")
    choice = input("Enter your choice: ")

    if choice == "Y":
        plt.plot_scatters(ds.genuines, ds.counterfeits)
    
    return 

def display_menu():
    print("Welcome to this Fingerprint spoofing journey!")
    print("Choose what to do with this spoofing dataset:")
    print("1. Show preliminary plots")
    print("2. Apply PCA to the dataset")
    print("3. Apply LDA to the dataset")
    print("4. Apply an LDA classifier")
    print("5. Apply a MVG classifier")
    print("0. Exit")

def main():
    loader = DatasetLoader()
    ds = loader.load()

    samples = ds.samples
    labels = ds.labels

    genuines = ds.genuines
    counterfeits = ds.counterfeits

    plt = Plotter()

    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            function_one(plt, ds)
        elif choice == "2":
            function_two()
        elif choice == "3":
            function_three()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
"""
