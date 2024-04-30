import numpy as np


class Dataset:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.genuines = samples[labels == 1]
        self.counterfeits = samples[labels == 0]

    def split_ds_2to1(self, D, L, seed=0):
        nTrain = int(D.shape[1] * 2.0 / 3.0)
        np.random.seed(seed)
        idx = np.random.permutation(D.shape[1])
        idxTrain = idx[0:nTrain]
        idxTest = idx[nTrain:]
        DTR = D[:, idxTrain]
        DVAL = D[:, idxTest]
        LTR = L[idxTrain]
        LVAL = L[idxTest]

        return DTR, LTR, DVAL, LVAL

    def genuines_mean(self):
        return self.genuines.mean()

    def counterfeits_mean(self):
        return self.genuines.mean()

    def ds_mean_vcol(self):
        self.samples.mean(axis=1).reshape(self.samples.shape[0], 1)
