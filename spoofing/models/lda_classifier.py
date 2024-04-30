import numpy as np
from preprocessing import lda
from preprocessing.pca import get_pca_matrix


class LDAClassifier:
    def __init__(self, dataset) -> None:
        self.ds = dataset

    def classify_with_mean_dist_treshold(self, with_pca=False):
        training_samples = self.ds.training_samples
        training_labels = self.ds.training_labels
        validation_samples = self.ds.validation_samples

        Ut_lda = lda.get_lda_matrix(training_samples, training_labels)
        validation_samples_lda = Ut_lda.T @ validation_samples
        training_samples_lda = Ut_lda.T @ training_samples

        threshold = self.get_lda_mean_dist_treshold(
            training_samples_lda, training_labels
        )

        error_rate = self.get_error_rate(validation_samples_lda, threshold)
        print(
            f"LDA classification error rate with threshold {threshold}: ",
            error_rate,
        )

        if with_pca:
            # We now employ the projected mean diff as threshold, but we could employ the same observations as before on each m
            results = self.get_accuracy_as_pca_dims_function(
                training_samples,
                training_labels,
                validation_samples,
            )

            for m, t, error_rate in results:
                print(
                    f"Error rate with {m} pca dimensions and threshold {t} is: ",
                    error_rate,
                )

    def classify_with_best_threshold(self):
        Ut_lda = lda.get_lda_matrix(self.ds.training_samples, self.ds.training_labels)
        validation_samples_lda = Ut_lda.T @ self.ds.validation_samples
        # We get the range by observing the training LDA histogram
        # The hope is within this range there's a better treshold than the projected mean diff
        error_rate, best_threshold = self.get_best_threshold(
            validation_samples_lda,
            np.arange(-1, 1, 0.001),
        )

        print("Best LDA threshold results in : ", error_rate, best_threshold)

    def get_best_threshold(self, samples, range):
        num_samples = float(samples.shape[1])
        rates = []
        for i in range:
            curr_threshold = i
            curr_rate = self.get_accuracy(samples, curr_threshold) / num_samples
            rates.append((curr_rate, curr_threshold))

        return min(rates, key=lambda t: t[0])

    def get_accuracy(self, samples, threshold):
        PVAL = np.zeros(shape=self.ds.validation_labels.shape, dtype=np.int32)
        PVAL[samples[0] >= threshold] = 1
        PVAL[samples[0] < threshold] = 0
        missed = np.sum(PVAL != self.ds.validation_labels)
        return missed

    def get_error_rate(self, samples, t):
        n_samples = float(samples.shape[1])
        missed = self.get_accuracy(samples, t)
        return missed / n_samples

    def get_accuracy_as_pca_dims_function(
        self,
        training_samples,
        training_labels,
        validation_samples,
        pca_dims=[2, 3, 4, 5],
    ):
        results = []
        for m in pca_dims:
            Pi = get_pca_matrix(training_samples, m)
            training_pcad_m = Pi.T @ training_samples

            Ui = lda.get_lda_matrix(training_pcad_m, training_labels)

            training_pcad_m_lda = Ui.T @ training_pcad_m

            Ui = self.orient_lda_matrix(Ui, training_pcad_m_lda, training_labels)

            validation_pcad_m = Pi.T @ validation_samples
            validation_pcad_lda_m = Ui.T @ validation_pcad_m
            pmean_diff_threshold_m = self.get_lda_mean_dist_treshold(
                training_pcad_m_lda, training_labels
            )
            error_rate_m = self.get_error_rate(
                validation_pcad_lda_m,
                pmean_diff_threshold_m,
            )

            results.append((m, pmean_diff_threshold_m, error_rate_m))
        return results

    def get_lda_mean_dist_treshold(self, samples, labels):
        return 0.5 * (
            self.ds.get_counterfeits_from(samples, labels).mean()
            + self.ds.get_genuines_from(samples, labels).mean()
        )

    # Returns the lda matrix oriented in a way that genuines have greather projected mean
    def orient_lda_matrix(self, Ui, samples, labels):
        if (
            self.ds.get_genuines_from(samples, labels).mean()
            < self.ds.get_counterfeits_from(samples, labels).mean()
        ):
            return -Ui
        return Ui
