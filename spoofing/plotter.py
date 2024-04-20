import matplotlib.pyplot as plt


class Plotter:
    def __init__(self) -> None:
        self.num_features = 6
        self.features_idx = {
            0: "Feature 1",
            1: "Feature 2",
            2: "Feature 3",
            3: "Feature 4",
            4: "Feature 5",
            5: "Feature 6",
        }

    def plot_hist(self, samples, bins, alpha, label):
        plt.hist(samples, bins=bins, alpha=alpha, label=label)

    def plot_features(
        self, genuines, counterfeits, save=False, range_v=6, x_label="", name=""
    ):
        for dIdx in range(0, range_v):
            plt.figure()
            plt.xlabel(x_label + f" {dIdx}")
            self.plot_hist(genuines[dIdx], bins=100, alpha=0.4, label="Genuine")

            self.plot_hist(
                counterfeits[dIdx],
                bins=100,
                alpha=0.4,
                label="Counterfeit",
            )
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(f"./plots/features_hists/{name}_%d_hist.pdf" % (dIdx + 1))
        plt.show()

    def plot_scatter(self, genuines, counterfeits, i, j, save=False):
        plt.figure()
        plt.xlabel(self.features_idx[i])
        plt.ylabel(self.features_idx[j])
        plt.scatter(x=genuines[i], y=genuines[j], label="Genuines")
        plt.scatter(x=counterfeits[i], y=counterfeits[j], label="Counterfeits")
        plt.legend()
        plt.savefig(f"./plots/scatter_plots/scatter_{i}-{j}_hist.pdf")

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
                f"./plots/features_hists/f_%d_hist_with_gaussian.pdf" % (f_idx + 1)
            )
        plt.legend()
        plt.show()
