from lib.util.plotter import Plotter
from lib.util.math_utils import get_pearson_correlation_matrix, get_covariance_matrix


def run_preliminary_plots(ds, save_plots=False):
    print("-" * 80)
    print("\nPreliminary plots...\n")

    plt = Plotter()

    genuines = ds.genuines
    counterfeits = ds.counterfeits
    
    print("\nPlotting feature distributions...\n")
    plt.plot_features(
        genuines,
        counterfeits,
        x_label="Feature",
        save=save_plots,
        name="features_hists",
    )

    print("\nPlotting features scatters...\n")
    plt.plot_scatters(genuines, counterfeits, save_plots=save_plots)

    print("\nPlotting correlation matrices...\n")
    g_pearson = get_pearson_correlation_matrix(get_covariance_matrix(genuines))
    c_pearson = get_pearson_correlation_matrix(get_covariance_matrix(counterfeits))
    plt.plot_correlation_matrixes(g_pearson, c_pearson)

    print("-" * 80)


def run_gaussians_to_features_plot(ds, save_plots=False):
    print("-" * 80)
    print("\nPlotting gaussians to class features...\n")

    plt = Plotter()

    genuines = ds.genuines
    counterfeits = ds.counterfeits

    plt.plot_all_1d_gau(genuines, counterfeits, save_plots=save_plots)

    print("\n")
    print("-" * 80)
