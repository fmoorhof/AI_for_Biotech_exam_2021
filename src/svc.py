import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import src.pca as pca


def svc(X_train_outer, X_test_outer, y_train_outer, y_test_outer, auc_outer, accuracy_outer, recall_outer, precision_outer, mcc_outer):
    X = X_train_outer
    y = y_train_outer
    pca = PCA()
    svc = SVC(kernel="rbf")
    pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])

    auc_values_svm_rbf = []
    precision_values = []
    recall_values = []
    accuracy_values = []
    mcc_values = []
    best_logi_C = []
    best_kpca_gamma = []
    best_kpca_kernel = []


    # Parameters of pipelines can be set using ‘__’ separated parameter names:#todo: adapt this to programms output to reduce hyperspace and calc. time
    param_grid = {"svc__C": np.logspace(-2, 1, 10),
                "svc__gamma": np.logspace(-4, 0, 10)
                  #"svc__kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed']
                  }
    search = GridSearchCV(pipe, param_grid, cv=2, n_jobs=-1)#search.get_params().keys()    to see valid param_grid definitions
    search.fit(X_train_outer, y_train_outer)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    # compute metrics for ROC Curve
    scores = search.decision_function(X_test_outer)

    y_pred = search.predict(X_test_outer)
    auc_values_svm_rbf.append(metrics.roc_auc_score(y_test_outer, y_pred))
    accuracy_values.append(metrics.accuracy_score(y_test_outer, y_pred))
    recall_values.append(metrics.recall_score(y_test_outer, y_pred))
    precision_values.append(metrics.precision_score(y_test_outer, y_pred))
    mcc_values.append(metrics.matthews_corrcoef(y_test_outer, y_pred))
    best_logi_C.append(search.best_params_['svc__C'])
    best_kpca_gamma.append(search.best_params_['svc__gamma'])
    #best_kpca_kernel.append(search.best_params_["svc__kernel"])

    # print results
    print("Accuracy:\t%.2f (+-%.2f)" % (np.mean(accuracy_values), np.std(accuracy_values)))
    print("Recall:\t\t%.2f (+-%.2f)" % (np.mean(recall_values), np.std(recall_values)))
    print("Precision:\t%.2f (+-%.2f)" % (np.mean(precision_values), np.std(precision_values)))
    print("MCC:\t\t%.2f (+-%.2f)" % (np.mean(mcc_values), np.std(mcc_values)))

    # compute metrics for ROC Curve
    scores = search.decision_function(X_test_outer)
    y_pred_outer = search.predict(X_test_outer)
    auc_outer.append(metrics.roc_auc_score(y_test_outer, y_pred_outer))
    accuracy_outer.append(metrics.accuracy_score(y_test_outer, y_pred_outer))
    recall_outer.append(metrics.recall_score(y_test_outer, y_pred_outer))
    precision_outer.append(metrics.precision_score(y_test_outer, y_pred_outer))
    mcc_outer.append(metrics.matthews_corrcoef(y_test_outer, y_pred_outer))

def svcM(y, X_train, X_test, X_head):

    Xr, Xt, Xr_test, Xt_test, i, ratios_variance_explained = pca.pca(X_train, X_test, X_head)
    Xrt = Xr[0:3500, 0:20]
    yrt = y[0:3500]

    params = {"C": np.logspace(-2, 1, 10),
              "gamma": np.logspace(-4, 0, 10)}

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    auc_values_svm_rbf = []
    precision_values = []
    recall_values = []
    accuracy_values = []
    mcc_values = []
    best_C = []
    best_y = []
    # train simple logistic regression without any penalty
    for train_index, val_index in cv.split(Xrt, yrt):
        X_train = Xrt[train_index]
        X_val = Xrt[val_index]
        y_train = yrt[train_index]
        y_val = yrt[val_index]

        # Train Model
        svc = SVC(kernel="rbf")
        gridsearch = GridSearchCV(svc, params, cv=2, scoring='f1')
        gridsearch.fit(X_train, y_train)

        # compute metrics for ROC Curve
        scores = gridsearch.decision_function(X_val)

        y_pred = gridsearch.predict(X_val)
        auc_values_svm_rbf.append(metrics.roc_auc_score(y_val, y_pred))
        accuracy_values.append(metrics.accuracy_score(y_val, y_pred))
        recall_values.append(metrics.recall_score(y_val, y_pred))
        precision_values.append(metrics.precision_score(y_val, y_pred))
        mcc_values.append(metrics.matthews_corrcoef(y_val, y_pred))
        best_C.append(gridsearch.best_params_['C'])
        best_y.append(gridsearch.best_params_['gamma'])

    # print results
    print("Accuracy:\t%.2f (+-%.2f)" % (np.mean(accuracy_values), np.std(accuracy_values)))
    print("Recall:\t\t%.2f (+-%.2f)" % (np.mean(recall_values), np.std(recall_values)))
    print("Precision:\t%.2f (+-%.2f)" % (np.mean(precision_values), np.std(precision_values)))
    print("MCC:\t\t%.2f (+-%.2f)" % (np.mean(mcc_values), np.std(mcc_values)))

    print('best_C ', best_C)
    print('best_y ', best_y)