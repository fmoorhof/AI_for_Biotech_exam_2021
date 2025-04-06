import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

def logireg(X_train,X_test, y_train, y_test):#X=Xr
    #copy paste form exercise 7.7 for exeplarily see logistic regression
    model = LogisticRegression(penalty="none")
    model.fit(X_train,y_train)
    #compute metrics for ROC Curve
    scores = model.decision_function(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, drop_intermediate=False)#Note: this implementation is restricted to the binary classification task.
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print("Logistic Regression for training data")
    print("ROC-AUC:\t%.2f " % metrics.roc_auc_score(y_train, y_pred_train))
    print("Accuracy:\t%.2f" % metrics.accuracy_score(y_train, y_pred_train))
    print("Recall:\t\t%.2f" % metrics.recall_score(y_train, y_pred_train))
    print("Precision:\t%.2f" % metrics.precision_score(y_train, y_pred_train))
    print("MCC:\t\t%.2f" % metrics.matthews_corrcoef(y_train, y_pred_train))
    print("Logistic Regression for test data")
    print("ROC-AUC:\t%.2f " % metrics.roc_auc_score(y_test, y_pred))
    print("Accuracy:\t%.2f" % metrics.accuracy_score(y_test, y_pred))
    print("Recall:\t\t%.2f" % metrics.recall_score(y_test, y_pred))
    print("Precision:\t%.2f" % metrics.precision_score(y_test, y_pred))
    print("MCC:\t\t%.2f" % metrics.matthews_corrcoef(y_test, y_pred))
    #generate figure
"""    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    #plot roc curve for all 10 foldes
    plt.plot(fpr, tpr, label="Logistic Regression (AUC=%.2f)" % metrics.roc_auc_score(y_test, y_pred))
    ax.plot([0,1], [0,1], color="grey",label="Random Classifier",linestyle="--")
    #Set axis labels
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    #Set axis limits
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    #show grid in grey and set top and right axis to invisible
    ax.grid(color="#CCCCCC")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.tight_layout()
    plt.show()"""