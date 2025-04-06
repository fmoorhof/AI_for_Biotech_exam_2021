import numpy as np
import time
start_time = time.time()

"""txpath = 'toxicity_labels.csv'
data = 'data.csv'
un = 'unknown_data.csv'
txpath = 'label_500.csv'
data = 'data_500.csv'"""
txpath = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/tox_head.csv'
data = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/data_head.csv'
un = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/unknown_head.csv'
txpath = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/toxicity_labels.csv'
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
file.close()
Xu_id = []
Xu = []
file = open(un, "r")
for line in file:
    ys = line.strip().split(",")
    Xu_id.append(ys[0])
    Xu.append(ys[1:])
file.close()
y_id = np.array(y_id[1:])
X_id = np.array(X_id[1:])
y_head = np.array(y[0])
X_head = np.array(X[0])
X = np.array(X[1:], dtype='float64')
Xu = np.array(Xu[1:], dtype='float64')
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
print('Number of unknown DATA samples: ', Xu.shape[0])
print('Number of unknown DATA features: ' + str(Xu.shape[1]))


from sklearn.preprocessing import StandardScaler

rm = np.where(y == 2)
X = np.delete(X, rm[0], axis=0)
X_id = np.delete(X_id, rm[0])
y = np.delete(y, rm[0])
y_id = np.delete(y_id, rm[0])


scaler = StandardScaler()
X = scaler.fit_transform(X)
Xu = scaler.transform(Xu)
print("time elapsed: {:.2f}s".format(time.time() - start_time))

from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import make_scorer

cv = StratifiedKFold(n_splits=3, shuffle=True)
auc_values_svm_rbf = []
precision_values = []
recall_values = []
accuracy_values = []
mcc_values = []
un_pred_lst = []

pca = PCA()
svm = SVC()  # max_iter=10000)
pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
n_components = [45, 55]

param_dist = {"max_depth": [3, None], "min_samples_split": np.random.randint(2, 11, 2)}
params_grid = {
    'svm__C': [10, 1000, 10000],
    'svm__kernel': ['rbf'],  # 'linear', 'precomputed', 'sigmoid', 'poly']
    'svm__gamma': [0.1, 0.01, 0.001],
    'pca__n_components': n_components,
}
# nested cross validation
for train_index, val_index in cv.split(X, y):
    X_train = X[train_index]
    X_val = X[val_index]
    y_train = y[train_index]
    y_val = y[val_index]

    #estimator = HalvingRandomSearchCV(estimator=pipe, param_distributions=params_grid, n_candidates='exhaust', factor=2, random_state=0)
    estimator = HalvingGridSearchCV(estimator=pipe, param_grid=params_grid, factor=2, cv=2)#, random_state=0)

    estimator.fit(X_train, y_train)
    print(estimator.best_params_, estimator.best_score_)

    # compute metrics for ROC Curve
    scores = estimator.decision_function(X_val)

    y_pred = estimator.predict(X_val)
    un_pred = estimator.predict(Xu)
    un_pred_lst.append(un_pred)
    auc_values_svm_rbf.append(metrics.roc_auc_score(y_val, y_pred))
    accuracy_values.append(metrics.accuracy_score(y_val, y_pred))
    recall_values.append(metrics.recall_score(y_val, y_pred))
    precision_values.append(metrics.precision_score(y_val, y_pred))
    mcc_values.append(metrics.matthews_corrcoef(y_val, y_pred))
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    # print(auc_values_svm_rbf, accuracy_values, recall_values, precision_values, mcc_values)

print("Accuracy:\t%.2f (+-%.2f)" % (np.mean(auc_values_svm_rbf), np.std(auc_values_svm_rbf)))
print("Recall:\t\t%.2f (+-%.2f)" % (np.mean(recall_values), np.std(recall_values)))
print("Precision:\t%.2f (+-%.2f)" % (np.mean(precision_values), np.std(precision_values)))
print("MCC:\t\t%.2f (+-%.2f)" % (np.mean(mcc_values), np.std(mcc_values)))
#todo: not working yet this part of predicting the unknown data
print(auc_values_svm_rbf, accuracy_values, recall_values, precision_values, mcc_values)
print(un_pred_lst)
print("unknown:\t\t%.2f (+-%.2f)" % (np.mean(un_pred_lst), np.std(un_pred_lst)))