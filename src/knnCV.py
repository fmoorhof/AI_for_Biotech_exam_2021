import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def knnCV(X, y):
    # initialize Cross-Validation
    kf = StratifiedKFold(n_splits=5)

    best_acc_val = []
    best_acc_test = []
    best_neighbors = []
    best_pre_test = []
    best_mcc_test = []
    # Perform cross-validation
    for train_index, test_index in kf.split(X, y):
        # Split data in each iteration
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # Create train-test split für hyperparameter optimization (line-seach)
        X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.2)
        neighbors = np.arange(1, 15)

        acc_train = []
        acc_val = []

        # Perform line search to find best k
        for k in neighbors:
            knn = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)
            knn.fit(X_subtrain, y_subtrain)
            predictions_training = knn.predict(X_subtrain)
            predictions_testing = knn.predict(X_val)

            # Store Accuracy
            acc_train.append(metrics.accuracy_score(y_subtrain, predictions_training))
            acc_val.append(metrics.accuracy_score(y_val, predictions_testing))

        # Store information to best model in line-search
        best_k = neighbors[np.argmax(acc_val)]
        best_neighbors.append(best_k)
        best_acc_val.append(acc_val[np.argmax(acc_val)])

        # retrain Model with best k and predict on test data
        knn = KNeighborsClassifier(n_neighbors=best_k, metric="minkowski", p=2)
        knn.fit(X_train, y_train)
        y_prediction = knn.predict(X_test)
        test_accuracy = metrics.accuracy_score(y_test, y_prediction)
        test_precition = metrics.precision_score(y_test, y_prediction)
        test_mcc = metrics.matthews_corrcoef(y_test, y_prediction)
        best_acc_test.append(test_accuracy)
        best_pre_test.append(test_precition)
        best_mcc_test.append(test_mcc)

    # transform python list into numpy array
    best_neighbors = np.array(best_neighbors)
    best_acc_val = np.array(best_acc_val)
    best_acc_test = np.array(best_acc_test)
    best_pre_test = np.array(best_pre_test)
    best_mcc_test = np.array(best_mcc_test)
    # print metrics
    print('knnVC:')
    print("Average k: %.2f (+- %.2f)" % (best_neighbors.mean(), best_neighbors.std()))
    print("Average Accuracy (Val): %.2f (+- %.2f)" % (best_acc_val.mean(), best_acc_val.std()))
    print("Average Accuracy (Test): %.2f (+- %.2f)" % (best_acc_test.mean(), best_acc_test.std()))
    print("Average precision (Test): %.2f (+- %.2f)" % (best_pre_test.mean(), best_pre_test.std()))
    print("Average MCC (Test): %.2f (+- %.2f)" % (best_mcc_test.mean(), best_mcc_test.std()))