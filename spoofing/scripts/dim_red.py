from preprocessing.pca import *
from preprocessing.lda import *

from util.plotter import Plotter


def run_dim_red_on_ds(ds, save_plots=False):
    print("-" * 80)
    print("\nRunning dimensionality reduction on spoofing dataset...\n")

    plt = Plotter()

    samples = ds.samples
    labels = ds.labels

    # PCA
    pcad = pca(samples, 6)
    pcad_genuines = ds.get_genuines_from(pcad)
    pcad_counterfeits = ds.get_counterfeits_from(pcad)

    plt.plot_features_onefig(
        pcad_genuines,
        pcad_counterfeits,
        x_label="Component",
        name="principal_components",
        save=save_plots,
    )
    
    # LDA
    U_lda = -get_lda_matrix(samples, labels)
    LDA_samples = U_lda.T @ samples

    plt.plot_features(
        ds.get_genuines_from(LDA_samples),
        ds.get_counterfeits_from(LDA_samples),
        x_label="LDA direction",
        range_v=1,
        save=True,
        name="LDA hist",
    )

    print()
    print("-" * 80)
