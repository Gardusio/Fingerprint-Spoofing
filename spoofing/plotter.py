import matplotlib
import matplotlib.pyplot as plt
from math_utils import *


class Plotter:
    def __init__(self) -> None:
        self.features_idx = {
            0: "Feature 1",
            1: "Feature 2",
            2: "Feature 3",
            3: "Feature 4",
            4: "Feature 5",
            5: "Feature 6",
        }

    def plot_features_onefig(
        self, genuines, counterfeits, save=False, range_v=6, x_label="", name=""
    ):
        fig, axs = plt.subplots(2, range_v // 2, figsize=(20, 8))
        for row in range(2):
            for col in range(range_v // 2):
                f_idx = row * (range_v // 2) + col
                axs[row, col].set_xlabel(x_label + f" {f_idx}")
                self.plot_hist_x(
                    axs[row, col], genuines[f_idx], bins=100, alpha=0.4, label="Genuine"
                )
                self.plot_hist_x(
                    axs[row, col],
                    counterfeits[f_idx],
                    bins=100,
                    alpha=0.4,
                    label="Counterfeit",
                )
                axs[row, col].legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"./analysis/plots/features_hists/{name}.pdf")
        plt.show()

    def plot_hist_x(self, ax, data, **kwargs):
        ax.hist(data, **kwargs)

    def plot_hist(self, samples, bins, alpha, label):
        plt.hist(samples, bins=bins, alpha=alpha, label=label)

    def plot_features(
        self, genuines, counterfeits, save=False, range_v=6, x_label="", name=""
    ):
        for f_idx in range(0, range_v):
            plt.figure()
            plt.xlabel(x_label + f" {f_idx}")

            self.plot_hist(genuines[f_idx], bins=100, alpha=0.4, label="Genuine")

            self.plot_hist(
                counterfeits[f_idx],
                bins=100,
                alpha=0.4,
                label="Counterfeit",
            )
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(
                    f"./analysis/plots/features_hists/{name}_%d_hist.pdf" % (f_idx + 1)
                )
        plt.show()

    def plot_scatter(self, genuines, counterfeits, i, j, save=False):
        plt.figure()
        plt.xlabel(self.features_idx[i])
        plt.ylabel(self.features_idx[j])
        plt.scatter(x=genuines[i], y=genuines[j], label="Genuines")
        plt.scatter(x=counterfeits[i], y=counterfeits[j], label="Counterfeits")
        plt.legend()
        if save:
            plt.savefig(f"./analysis/plots/scatter_plots/scatter_{i}-{j}_hist.pdf")

    def plot_scatters(self, genuines, counterfeits):
        self.plot_scatter(genuines, counterfeits, 0, 1)
        self.plot_scatter(genuines, counterfeits, 2, 3)
        self.plot_scatter(genuines, counterfeits, 4, 5)
        plt.show()

    def plot_1d_gau(
        self,
        c1_sample_set,
        c1_plot,
        c1_pdf,
        c2_sample_set,
        c2_plot,
        c2_pdf,
        f_idx="",
        save=False,
    ):
        plt.figure()
        plt.xlabel(f"Feature {f_idx}")
        plt.hist(c1_sample_set, bins=50, density=True, alpha=0.4, label="Genuines")
        plt.plot(c1_plot, c1_pdf, label="Genuines Estimated PDF")
        plt.hist(c2_sample_set, bins=50, density=True, alpha=0.4, label="Counterfeits")
        plt.plot(c2_plot, c2_pdf, label="Counterfeits Estimated PDF")

        if save:
            plt.savefig(
                f"./analysis/plots/features_hists/f_%d_hist_with_gaussian.pdf"
                % (f_idx + 1)
            )
        plt.legend()

    def plot_all_1d_gau(self, genuines, counterfeits):
        for f_idx in range(0, 6):
            genuines_feature_samples = genuines[f_idx, :].reshape(1, genuines.shape[1])
            counterfeits_feature_samples = counterfeits[f_idx, :].reshape(
                1, counterfeits.shape[1]
            )

            g_plot, g_gaussian_pdf = get_gaussian_to_feature_plotline(
                genuines_feature_samples, f_idx
            )
            c_plot, c_gaussian_pdf = get_gaussian_to_feature_plotline(
                counterfeits_feature_samples, f_idx
            )

            self.plot_1d_gau(
                c1_sample_set=genuines_feature_samples.ravel(),
                c1_plot=g_plot,
                c1_pdf=g_gaussian_pdf,
                c2_sample_set=counterfeits_feature_samples.ravel(),
                c2_plot=c_plot,
                c2_pdf=c_gaussian_pdf,
                f_idx=f_idx,
                save=True,
            )
        plt.show()

    def plot_corr_matrix(self, corr_matrix, title):
        plt.imshow(corr_matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.show()

    def plot_correlation_matrixes(self, g_corr_matrix, c_corr_matrix):
        plt.plot_corr_matrix(g_corr_matrix, "Correlation matrix for genuines")
        plt.plot_corr_matrix(c_corr_matrix, "Correlation matrix for counterfeits")

    def plot_bayes_errors(self, prior_log_odds, dcf, mindcf, range):
        plt.plot(prior_log_odds, dcf, label="DCF", color="r")
        plt.plot(prior_log_odds, mindcf, label="min DCF", color="b")
        plt.ylim([0, 1.1])
        plt.xlim([range[0], range[1]])
        plt.show()
