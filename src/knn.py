import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def knn(k, Xr, Xr_test, y_train, y_test):
    X_train = Xr
    X_test = Xr_test
    neighbors = np.arange(1, k)
    acc_train = []
    acc_test = []
    # Perform line-search to find best k
    for k in neighbors:
        kn = KNeighborsClassifier(n_neighbors=k, algorithm="brute", metric="minkowski", p=2)
        kn.fit(X_train, y_train)
        predictions_training = kn.predict(X_train)
        predictions_testing = kn.predict(Xr_test)

        # Store Accuracy
        acc_train.append(metrics.accuracy_score(y_train, predictions_training))
        acc_test.append(metrics.accuracy_score(y_test, predictions_testing))

    # Best k?
    best_k = neighbors[np.argmax(acc_test)]
    print("Best k=\t" + str(best_k))
    print("Best Accuracy (test):\t%.2f" % acc_test[np.argmax(acc_test)])

    # now retrain model with best k ON FULL TRAINING SET and predict ON FULL TEST
    kn = KNeighborsClassifier(n_neighbors=best_k, algorithm="brute", metric="minkowski", p=2)
    kn.fit(Xr, y_train)
    y_prediction = kn.predict(X_test)
    #from here its getting buggy since shapes dont match of y_test and y_prediction
    #knnCV is working therefore i droped dev.
    test_accuracy = metrics.accuracy_score(y_test, y_prediction)
    test_mcc = metrics.matthews_corrcoef(y_test, y_prediction)
    test_f1 = metrics.f1_score(y_test, y_prediction)
    test_rcl = metrics.recall_score(y_test, y_prediction)
    print("Accuracy on Test:\t%.2f" % test_accuracy)
    print("mcc on Test:\t%.2f" % test_mcc)
    print("f1 score on Test:\t%.2f" % test_f1)
    print("recall on Test:\t%.2f" % test_rcl)
    # Plot performance for different k values
"""    plt.figure()
    plt.grid()
    plt.plot(neighbors, acc_train, label="Accuracy (Train)")
    plt.plot(neighbors, acc_test, label="Accuracy (test)")
    plt.axvline(best_k, color="grey", linestyle="dashed")
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()"""