import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#best linear kernel, gamma=0.03, logistic__C=0.45059440644047144 (MCC=0.48)
def pipeKPCAlogireg(X_train_outer, X_test_outer, y_train_outer, y_test_outer, auc_outer, accuracy_outer, recall_outer, precision_outer, mcc_outer):
    X = X_train_outer
    y = y_train_outer
    kpca = KernelPCA(fit_inverse_transform=True, kernel='linear', gamma=0.03)#remove gamma and kernel for hyperparameter optimization
    logistic = LogisticRegression(max_iter=10000, tol=0.1, C=0.03162277660168379)
    pipe = Pipeline(steps=[('kpca', kpca), ('logistic', logistic)])

    cv = StratifiedKFold(n_splits=2, shuffle=True)
    auc_values_svm_rbf = []
    precision_values = []
    recall_values = []
    accuracy_values = []
    mcc_values = []
    best_logi_C = []
    best_kpca_gamma = []
    best_kpca_kernel = []

    # nested cross validation
    for train_index, val_index in cv.split(X, y):
        X_train = X[train_index]
        X_val = X[val_index]
        y_train = y[train_index]
        y_val = y[val_index]

        # Parameters of pipelines can be set using ‘__’ separated parameter names:#todo: adapt this to programms output to reduce hyperspace and calc. time
        param_grid = {#"kpca__gamma": np.linspace(0.03, 0.05, 5),
                      #"kpca__kernel_params": ['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'],
                      #"logistic__C": np.logspace(0.04, 0.52, 10)
        }
        search = GridSearchCV(pipe, param_grid, n_jobs=-1)#search.get_params().keys()    to see valid param_grid definitions
        search.fit(X_train, y_train)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

        # compute metrics for ROC Curve
        scores = search.decision_function(X_val)

        y_pred = search.predict(X_val)
        auc_values_svm_rbf.append(metrics.roc_auc_score(y_val, y_pred))
        accuracy_values.append(metrics.accuracy_score(y_val, y_pred))
        recall_values.append(metrics.recall_score(y_val, y_pred))
        precision_values.append(metrics.precision_score(y_val, y_pred))
        mcc_values.append(metrics.matthews_corrcoef(y_val, y_pred))
        #best_logi_C.append(search.best_params_['logistic__C'])
        #best_kpca_gamma.append(search.best_params_['kpca__gamma'])
        #best_kpca_kernel.append(search.best_params_["kpca__kernel_params"])

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


# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
def tryy(X_digits, y_digits):
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    logistic = LogisticRegression(max_iter=10000, tol=0.1)
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    #X_digits, y_digits = datasets.load_digits(return_X_y=True)

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'pca__n_components': [5, 15, 30, 45, 64],
        'logistic__C': np.logspace(-4, 4, 4),
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(X_digits, y_digits)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    # Plot the PCA spectrum
    pca.fit(X_digits)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(np.arange(1, pca.n_components_ + 1),
             pca.explained_variance_ratio_, '+', linewidth=2)
    ax0.set_ylabel('PCA explained variance ratio')

    ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = 'param_pca__n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                   legend=False, ax=ax1)
    ax1.set_ylabel('Classification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.xlim(-1, 70)

    plt.tight_layout()
    plt.show()