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

    def plot_features(
        self, genuines, counterfeits, save=False, range_v=6, x_label="", name=""
    ):
        for dIdx in range(0, range_v):
            plt.figure()
            plt.xlabel(x_label + f" {dIdx}")
            plt.hist(genuines[dIdx], bins=100, alpha=0.4, label="Genuine")
            plt.hist(
                counterfeits[dIdx],
                bins=100,
                alpha=0.4,
                label="Counterfeit",
            )
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(f"./plots/{name}_%d_hist.pdf" % (dIdx + 1))
        plt.show()

    def plot_scatter(self, genuines, counterfeits, i, j):
        plt.figure()
        plt.xlabel(self.features_idx[i])
        plt.ylabel(self.features_idx[j])
        plt.scatter(x=genuines[i], y=genuines[j], label="Genuines")
        plt.scatter(x=counterfeits[i], y=counterfeits[j], label="Counterfeits")
        plt.legend()
        plt.savefig(f"./plots/scatter_{i}-{j}_hist.pdf")

    def plot_scatters(self, genuines, counterfeits):
        self.plot_scatter(genuines, counterfeits, 0, 1)
        self.plot_scatter(genuines, counterfeits, 2, 3)
        self.plot_scatter(genuines, counterfeits, 4, 5)
        plt.show()

    def print_feature_stats(self, genuines, counterfeits, f_idx):
        print(
            f"Genuines mean for feature {f_idx+1}: ", geat_feature_mean(genuines, f_idx)
        )
        print(
            f"Counterfeits mean for feature {f_idx+1}: ",
            geat_feature_mean(counterfeits, f_idx),
        )
        print(
            f"Genuines variance for feature {f_idx+1}: ",
            geat_feature_var(genuines, f_idx),
        )
        print(
            f"Counterfeits variance for feature {f_idx+1}: ",
            geat_feature_var(counterfeits, f_idx),
        )

    def print_features_stats(self, genuines, counterfeits):
        self.print_feature_stats(genuines, counterfeits, 0)
        self.print_feature_stats(genuines, counterfeits, 1)
        self.print_feature_stats(genuines, counterfeits, 2)
        self.print_feature_stats(genuines, counterfeits, 3)
        self.print_feature_stats(genuines, counterfeits, 4)
        self.print_feature_stats(genuines, counterfeits, 5)
