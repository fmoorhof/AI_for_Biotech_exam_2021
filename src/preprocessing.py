import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def featureDeleter(X, X_head):
    #remove features that have more than 50% 0-values
    rm = []
    for j in range(X.shape[1]):
        lst = np.where(X[:, j] == 0)[0]
        freq = lst.shape[0] / X.shape[0] * 100
        if freq > 50:
            rm.append(j)
        # print('%d\t%d\t\t\t%.1f' % (j, lst.shape[0], freq))
    X = np.delete(X, rm, axis=1)
    X_head = np.delete(X_head, rm)  # also remove headders of deleted features
    return X, X_head

#todo: think about if really 1/3 of data, that have just no labels should be removed?
def labelDeleter(X, X_id, y, y_id):
    #remove labels that are unclassified 'NA'
    rm = np.where(y==2)
    X = np.delete(X, rm[0], axis=0)
    X_id = np.delete(X_id, rm[0])
    y = np.delete(y, rm[0])
    y_id = np.delete(y_id, rm[0])
    return X, X_id, y, y_id

def imputing(X_train, X_test):
    imputer = SimpleImputer(missing_values=0, strategy='mean')# ,strategy='most_frequent'
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)  # Apply same transformation on test data
    return X_train, X_test

def normalizing(X_train, X_test):
    # normalization of X -> n_score normlaization needed for PCA -> Give each feature a mean of 0 and a variance of 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test