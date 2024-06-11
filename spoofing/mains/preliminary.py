from util.plotter import Plotter


def run_preliminary_plots(ds, save_plots=False):
    print("-" * 80)
    print("\nPreliminary plots...\n")

    plt = Plotter()

    genuines = ds.genuines
    counterfeits = ds.counterfeits

    plt.plot_features_onefig(
        genuines,
        counterfeits,
        x_label="Feature",
        save=save_plots,
        name="features_hists",
    )
    plt.plot_scatters(genuines, counterfeits)

    print("-" * 80)


def run_gaussians_to_features_plot(ds, save_plots=False):
    print("-" * 80)
    print("\nPlotting gaussians to class features...\n")

    plt = Plotter()

    genuines = ds.genuines
    counterfeits = ds.counterfeits

    plt.plot_all_1d_gau(genuines, counterfeits)

    print("\n")
    print("-" * 80)
