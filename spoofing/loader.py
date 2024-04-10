import numpy as np


class DatasetLoader:

    def __init__(self, file_path="trainData.txt", separator=" , "):
        self.file_path = file_path
        self.separator = separator

    def read_samples(self):
        with open(self.file_path) as file:
            return [self.build_sample(line) for line in file]

    def build_sample(self, raw_line):
        line = raw_line.strip().split(self.separator)

        label = int(line[-1])

        features = np.array(line[:-1], dtype=float)

        return features, label

    def load(self):
        raw_samples_w_label = self.read_samples()
        raw_features, raw_labels = zip(*raw_samples_w_label)

        # row: feature, column: sample
        features_matrix = np.column_stack(raw_features)

        labels = np.array(raw_labels)

        return (features_matrix, labels)

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
