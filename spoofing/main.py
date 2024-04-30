from dataset.loader import DatasetLoader
from plotter import Plotter
from dataset.ds_utils import *
from preprocessing.pca import *
from preprocessing.lda import *
from models.lda_classifier import LDAClassifier


def main():

    # TODO: specify ds file path via args
    loader = DatasetLoader()
    ds = loader.load()

    samples = ds.samples
    labels = ds.labels

    plt = Plotter(ds)

    """
    ############################################## LAB 2 ##########################################
    plt.plot_features(ds.genuines, ds.counterfeits)
    plt.plot_scatters(ds.genuines, ds.counterfeits)
    """

     
    """
    ############################################## LAB 3 - PCA ####################################

    pcad = pca(samples, 6)
    pcad_genuines = ds.get_genuines_from(pcad)
    pcad_counterfeits = ds.get_counterfeits_from(pcad)
    plt.plot_features(pcad_genuines, pcad_counterfeits, x_label="Component")

    
    
    
    ############################################## LAB 3 - LDA #####################################

    U_lda = -get_lda_matrix(samples, labels)
    LDA_samples = U_lda.T @ samples
    
    plt.plot_features(
        ds.get_genuines_from(LDA_samples),
        ds.get_counterfeits_from(LDA_samples),
        x_label="LDA direction",
        range_v=1,
    )
    """
    
    
    """
    ############################################## LAB 3 - LDA classification ######################
    lda_classifier = LDAClassifier(ds)
    lda_classifier.classify_with_mean_dist_treshold()
    lda_classifier.classify_with_mean_dist_treshold(with_pca=True)
    lda_classifier.classify_with_best_threshold()
    """


    """
    ############################################## LAB 4 - Fitting Gaussians #######################
    plt.plot_all_1d_gau(genuines, counterfeits)
    """
    
    
    ############################################## LAB 5 - MVG, NB-MVG, Tied MVG #######################
    


if __name__ == "__main__":
    main()
