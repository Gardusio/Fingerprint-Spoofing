import numpy as np


class Dataset:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.num_classes = 2
        self.genuines = samples[:, labels == 1]
        self.counterfeits = samples[:, labels == 0]
        t_s, t_l, v_s, v_l = self.split_ds_2to1()
        self.training_samples = t_s
        self.training_labels = t_l
        self.validation_samples = v_s
        self.validation_labels = v_l

    def genuines_mean(self):
        return self.genuines.mean()

    def counterfeits_mean(self):
        return self.genuines.mean()

    def ds_mean_vcol(self):
        self.samples.mean(axis=1).reshape(self.samples.shape[0], 1)

    def get_genuines_from(self, samples, from_labels=None):
        if from_labels is not None:
            return samples[:, from_labels == 1]

        return samples[:, self.labels == 1]

    def get_counterfeits_from(self, samples, from_labels=None):
        if from_labels is not None:
            return samples[:, from_labels == 0]
        return samples[:, self.labels == 0]

    def split_ds_2to1(self, seed=0):
        nTrain = int(self.samples.shape[1] * 2.0 / 3.0)
        np.random.seed(seed)
        idx = np.random.permutation(self.samples.shape[1])
        idxTrain = idx[0:nTrain]
        idxTest = idx[nTrain:]
        DTR = self.samples[:, idxTrain]
        DVAL = self.samples[:, idxTest]
        LTR = self.labels[idxTrain]
        LVAL = self.labels[idxTest]

        return DTR, LTR, DVAL, LVAL