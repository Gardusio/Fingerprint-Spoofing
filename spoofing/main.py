from loader import DatasetLoader
from plotter import Plotter
from analytics import *


def main():

    # TODO: specify ds file path via args
    ds = DatasetLoader().load()

    features_matrix, labels = ds

    # select all rows (:), match column_idx with mask_idx
    # Q1: Why don't just include the labels in the last column in the matrix?
    # Q2: if ds is big, how to do this in a single pass while leveraging np?
    genuines = features_matrix[:, labels == 1]
    counterfeits = features_matrix[:, labels == 0]

    plt = Plotter()
    plt.plot_features(genuines, counterfeits)
    # plt.plot_scatters(genuines, counterfeits)
    # plt.print_features_stats(genuines, counterfeits)



if __name__ == "__main__":
    main()
