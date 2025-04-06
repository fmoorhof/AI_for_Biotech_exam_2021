import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def knnl(X_train, y_train, y_test):
    X_subtrain, X_test, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.2)
    neighbors = np.arange(1, 15)
    acc_train = []
    acc_test = []
    # Perform line-search to find best k
    for k in neighbors:
        kn = KNeighborsClassifier(n_neighbors=k, algorithm="brute", metric="minkowski", p=2)
        kn.fit(X_subtrain, y_subtrain)
        predictions_training = kn.predict(X_subtrain)
        predictions_testing = kn.predict(X_test)

        # Store Accuracy
        acc_train.append(metrics.accuracy_score(y_subtrain, predictions_training))
        acc_test.append(metrics.accuracy_score(y_val, predictions_testing))

    # Best k?
    best_k = neighbors[np.argmax(acc_test)]
    print("Best k=\t" + str(best_k))
    print("Best Accuracy (test):\t%.2f" % acc_test[np.argmax(acc_test)])

    # now retrain model with best k on full training set and predict on test
    kn = KNeighborsClassifier(n_neighbors=best_k, algorithm="brute", metric="minkowski", p=2)
    kn.fit(X_train, y_train)
    y_prediction = kn.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_prediction)
    test_accuracy = metrics.accuracy_score(y_test, y_prediction)
    test_mcc = metrics.matthews_corrcoef(y_test, y_prediction)
    test_f1 = metrics.f1_score(y_test, y_prediction)
    test_rcl = metrics.recall_score(y_test, y_prediction)
    print("Accuracy on Test:\t%.2f" % test_accuracy)
    print("mcc on Test:\t%.2f" % test_mcc)
    print("f1 score on Test:\t%.2f" % test_f1)
    print("recall on Test:\t%.2f" % test_rcl)
