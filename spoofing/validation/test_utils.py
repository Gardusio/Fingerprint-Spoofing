import numpy as np

def get_accuracy(validation_samples, threshold, validation_labels):
    PVAL = np.zeros(shape=validation_labels.shape, dtype=np.int32)
    PVAL[validation_samples[0] >= threshold] = 1
    PVAL[validation_samples[0] < threshold] = 0
    missed = np.sum(PVAL != validation_labels)
    return missed


def get_best_threshold(samples, labels, num_samples, range):
    rates = []
    for i in range:
        curr_threshold = i
        curr_rate = get_accuracy(samples, curr_threshold, labels) / num_samples
        rates.append((curr_rate, curr_threshold))

    return min(rates, key=lambda t: t[0])


def get_error_rate(samples, n_samples, t, labels):
    missed = get_accuracy(samples, t, labels)
    return missed / n_samples
