import numpy as np
from preprocessing.pca import pca


class Dataset:
    """
    Spoofing dataset util class

    Init sets a default training/test split of 2 to 1

    - samples: contains the whole dataset samples as a np array
    - labels: contains the samples relative labels
    samples and labels must be ordered accordingly

    - genuines: contains samples of class 1 (genuines fingerprints)
    - counterfeits: contains samples of class 0 (fake fingerprint)

    - training_samples: 2/3 of samples by default
    - training_labels: 2/3 of labels by default, ordered according to training_samples
    - validation_samples: 1/3 of samples by default
    - validation_labels: 1/3 of labels by default, ordered according to validation_samples

    """

    def __init__(self, samples, labels):
        self.num_classes = 2
        self.samples = samples
        self.labels = labels
        self.genuines = samples[:, labels == 1]
        self.counterfeits = samples[:, labels == 0]

        t_s, t_l, v_s, v_l = self.split_ds_2to1()
        self.training_samples = t_s
        self.training_labels = t_l
        self.validation_samples = v_s
        self.validation_labels = v_l

    def get_genuines_from(self, samples, from_labels=None):
        """
        Get the samples belonging to class 1 from any set of given samples

        params:
            - samples: any set of samples from which to extract samples belonging to class 1
            - from_labels: if not None, extract samples of class 1 using this set of labels assignments
        """
        if from_labels is not None:
            return samples[:, from_labels == 1]

        return samples[:, self.labels == 1]

    def get_counterfeits_from(self, samples, from_labels=None):
        """
        Get the samples belonging to class 0 from any set of given samples

        params:
            - samples: any set of samples from which to extract samples belonging to class 0
            - from_labels: if not None, extract samples of class 0 using this set of labels assignments
        """
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

    def drop_features(self, to_drop=[]):
        to_keep_mask = np.ones(self.training_samples.shape[0], dtype=bool)
        to_keep_mask[to_drop] = False
        self.training_samples = self.training_samples[to_keep_mask, :]
        self.validation_samples = self.validation_samples[to_keep_mask, :]
        return self.training_samples, self.validation_samples
