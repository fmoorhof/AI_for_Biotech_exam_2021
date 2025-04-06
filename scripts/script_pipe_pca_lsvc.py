"""Script is executable w/o main, just like that. However prediction is rather bad with best MCC of 0.3;
but execution and hyperparameter finding is rather fast
lsvc = linear svc = much faster than a SVC with linear kernel"""

import numpy as np
import time
start_time = time.time()

#data sources
txpath =  '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/tox_head.csv'
data = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/data_head.csv'
txpath =  '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/toxicity_labels.csv'
data = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/data.csv'

y_id = []
y = []
file = open(txpath, "r")
for line in file:
    ys = line.strip().split(",")
    y_id.append(ys[0])
    y.append(
        ys[1].replace('NA', '2'))
file.close()
X_id = []
X = []
file = open(data, "r")
for line in file:
    ys = line.strip().split(",")
    X_id.append(ys[0])
    X.append(ys[1:])

y_id = np.array(y_id[1:])
X_id = np.array(X_id[1:])
y_head = np.array(y[0])
X_head = np.array(X[0])
X = np.array(X[1:], dtype='float64')
y = np.array(y[1:], dtype='i1')  # int8

print('Number of DATA samples: ', X.shape[0])
print('Number of DATA features: ' + str(X.shape[1]))
print('Number of LABEL samples in total:\t' + str(y.shape[0]))
lab, freq = np.unique(y, return_counts=True)
print('Number of values with missing label NA:\t%d' % freq[2])
print('Percentage of values of label ' + str(lab[0]) + ':\t%.2f' % (freq[0] / y.size * 100))
print('Percentage of values in label %s:\t%.2f' % (lab[1], (freq[1] / y.size * 100)))
print('Percentage of values with missing labels:\t%.2f' % ((freq[2] / y.size * 100)))

print("time elapsed: {:.2f}s".format(time.time() - start_time))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

rm = np.where(y == 2)
X = np.delete(X, rm[0], axis=0)
X_id = np.delete(X_id, rm[0])
y = np.delete(y, rm[0])
y_id = np.delete(y_id, rm[0])

print("time elapsed: {:.2f}s".format(time.time() - start_time))

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

cv = StratifiedKFold(n_splits=2, shuffle=True)
auc_values_svm_rbf = []
precision_values = []
recall_values = []
accuracy_values = []
mcc_values = []

pca = PCA()
#svm = SVC()
lsvc = LinearSVC(max_iter=10000)
# pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
pipe = Pipeline(steps=[('pca', pca), ('lsvc', lsvc)])
# Check the training time for the SVC
n_components = [50]#[20, 40, 50]

params_grid = {#old block for the svc, that takes too long for calculation
    'svm__C': [1],  # , 10, 100, 1000],
    'svm__kernel': ['linear'],  # , 'rbf'],
    'svm__gamma': [0.001],  # , 0.0001],
    'pca__n_components': n_components,
}
params_grid = {
    'lsvc__C': [0.03162277660168379],#[1, 10, 100, 1000],
    'pca__n_components': n_components,
}

# nested cross validation
for train_index, val_index in cv.split(X, y):
    X_train = X[train_index]
    X_val = X[val_index]
    y_train = y[train_index]
    y_val = y[val_index]

    estimator = GridSearchCV(pipe, params_grid, cv=2, n_jobs=-1)
    estimator.fit(X_train, y_train)

    print(estimator.best_params_, estimator.best_score_)
    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    # compute metrics for ROC Curve
    scores = estimator.decision_function(X_val)

    y_pred = estimator.predict(X_val)
    auc_values_svm_rbf.append(metrics.roc_auc_score(y_val, y_pred))
    accuracy_values.append(metrics.accuracy_score(y_val, y_pred))
    recall_values.append(metrics.recall_score(y_val, y_pred))
    precision_values.append(metrics.precision_score(y_val, y_pred))
    mcc_values.append(metrics.matthews_corrcoef(y_val, y_pred))
print(auc_values_svm_rbf, accuracy_values, recall_values, precision_values, mcc_values)