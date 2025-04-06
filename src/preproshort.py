import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessing():
    def __init__(self, X):
        self.X = X

    # normalization -> Give each feature a mean of 0 and a variance of 1
    def normalizingNOprepo(self, X):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)
        return self.X

    # todo: think about if really 1/3 of data, that have just no labels should be removed?
    # removes LABELS that are unclassified 'NA'
    def labelDeleter(self, X_id, y, y_id):
        rm = np.where(y == 2)
        self.X = np.delete(self.X, rm[0], axis=0)
        X_id = np.delete(X_id, rm[0])
        y = np.delete(y, rm[0])
        y_id = np.delete(y_id, rm[0])
        return self.X, X_id, y, y_id