# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
import threading
from sklearn.model_selection import StratifiedKFold, train_test_split

import src.kpca
import src.parsing as parsing#import has to be same as file name
import src.preprocessing
import src.pca as pca
import src.logisticRegression
import src.knn as knn
import src.knnCV
import src.knnLoop
import src.preproshort as preproshort
import src.pipelining as pipelining
import src.svc as svc

start_time = time.time()

#data sources
txpath = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/tox_head.csv'
data = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/data_head.csv'
txpath = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/toxicity_labels.csv'
data = '/home/lnx/TUM/SS21/Artifical_intelligence_for_biotech/ExamProject/Data/data.csv'

# Press the green button in the gutter to run the script.
def main():
    # function name with import.def
    X, y, X_id, y_id, X_head, y_head = parsing.parsing(txpath, data)
    parsing.stats(X, y)
    #new class fashion
    prepro = preproshort.Preprocessing(X)
    X = prepro.normalizingNOprepo(X)
    X, X_id, y, y_id = prepro.labelDeleter(X_id, y, y_id)

    """"#for visualization issues:
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    Xr, Xt, Xr_test, Xt_test, i, ratios_variance_explained = pca.pca(X_train, X_test, X_head, y_train)
    pca.featurePC(i, X_head, ratios_variance_explained)
    pca.pcaVis(5, Xt, y_train)
    kpca.simpleKPCA(X, y)
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    kpca.kpca_try(X, y)
    #pipelining.tryy(X_train, y_train)"""


    svc.svcM(y, X_train, X_test, X_head)#not running!!!!!!!!!!!!!!!!!!!
    #old fashion
    #X = preprocessing.normalizingNOprepo(X)
    #X, X_id, y, y_id = preprocessing.labelDeleter(X, X_id, y, y_id)


    #all values were computationally generated; hence there is no further preprocessing necessary
    #X, X_head = preprocessing.featureDeleter(X, X_head)

    # No imputing and normalization necessary. instead data were normalized one time before
    #X_train, X_test, y_train, y_test = preprocessing.imputing(X, y)
    #X_train, X_test = preprocessing.normalizing(X_train, X_test)


    #outer nested CV
    cv_out = StratifiedKFold(n_splits=2, shuffle=True)
    auc_outer = []
    accuracy_outer = []
    recall_outer = []
    precision_outer = []
    mcc_outer = []
    for train_index, val_index in cv_out.split(X, y):
        X_train_outer = X[train_index]
        X_test_outer = X[val_index]
        y_train_outer = y[train_index]
        y_test_outer = y[val_index]
        #pipelining.pipeKPCAlogireg(X_train_outer, X_test_outer, y_train_outer, y_test_outer, auc_outer, accuracy_outer, recall_outer, precision_outer, mcc_outer)
        svc.svc(X_train_outer, X_test_outer, y_train_outer, y_test_outer, auc_outer, accuracy_outer, recall_outer, precision_outer, mcc_outer)
        print("time elapsed: {:.2f}s".format(time.time() - start_time))
        print(auc_outer, accuracy_outer, recall_outer, precision_outer, mcc_outer)
    # , auc_outer, accuracy_outer, recall_outer, precision_outer, mcc_outer




    #todo: before run any methods and CV, data have to be separabl somehow -> apply kernel trick!
    from sklearn.decomposition import PCA
    pc = PCA()
    out = pc.fit_transform(X_train)
    X_test = pc.transform(X_test)
    print('hello')


    Xr, Xt, Xr_test, Xt_test, i, ratios_variance_explained = pca.pca(X_train, X_test, X_head, y_train)
    #visualization of the PCA stuff:
#    pca.featurePC(i, X_head, ratios_variance_explained)
#    pca.pcaVis(number_of_PCs=2, Xt=Xt, y_train=y_train)#change number of plots to visualize in number_of_PCs=



    X_train = Xr; X_test = Xr_test; del Xr, Xr_test
    y_train[y_train == 2] = 0;    y_test[y_test == 2] = 0#if preprocessing.labelDeleter is not called: ValueError: multiclass format is not supported in logireg function
    #declaring multithreading
    t1 = threading.Thread(target=logisticRegression.logireg, args=(X_train, X_test, y_train, y_test))
    t2 = threading.Thread(target=knn.knn, args=(10, X_train, X_test, y_train, y_test))
    #function call using multithreading
    t1.start()
    t2.start()
    t1.join()
#    logisticRegression.logireg(X_train, X_test, y_train, y_test)
#    knn.knn(10, X_train, X_test, y_train, y_test)
    print("time elapsed: {:.2f}s".format(time.time() - start_time))


    #knnCV.knnCV(X_train, y_train)#Xr_test, y_test
    #knnLoop.knnl(Xr, Xr_test, y_train, y_test)
"""
Information/Interpetation:
MCC: +1 is a perfect classifier, 0 random and -1 indicates negative correlation (total disagreement)
ROC: 0.5 (random) < AUC < 1.0 (perfect)
%Precision and Recall are good to evaluate the performance of models, if the positive class is much smaller than the negative class
"""

if __name__ == '__main__':
    main()
    print("time elapsed: {:.2f}s".format(time.time() - start_time))