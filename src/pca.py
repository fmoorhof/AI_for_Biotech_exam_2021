import numpy as np
import matplotlib.pyplot as plt

def pca(X_train, X_test, X_head, y_train):
    #todo: think about wheather y_train labels should be included or better not because they are anyways just some

    # X = X_train#this was done do to lazyness to copy paste code; X should not have been overwritten with X_train!
    # n = X_train.shape[0]; C = 1/(n-1) * np.dot(X_train.T,X_train)
    C = np.cov(X_train.T)
    if C.shape[1] == X_train.shape[1]:
        print('Dimension of C correct with: %d x %d' % (C.shape[0], C.shape[1]))
    d, V = np.linalg.eig(C)
    #complex data handling -> extracting the real part
    if d.dtype == complex:
        d = d.real
    if V.dtype == complex:
        V = V.real
    ind = np.argsort(d)[::-1]  # largest to smallest sorting
    d = d[ind]  # d= eigenvalues define the magnitude of the principal components
    V = V[:, ind]  # V = eigenvectors of the covariance matrix represent the principal components,which are the directions of maximum variance
    X_head[ind]
    # mhh sorting results in just some little changes?????

    required_explanation = 0.90
    for i in range(2, C.shape[1]):
        ratios_variance_explained = d / d.sum()
        va = ratios_variance_explained[0:i].sum()
        if va > required_explanation:
            print("r for explanation of %.1f %% is %d" % (required_explanation * 100, i))
            break
    # Xr = lower dimensional representation;; Xt original matrix
    Xr = np.dot(X_train, V[:, 0:i])
    #complex data handling -> extracting the real part
    if Xr.dtype == complex:
        Xr = Xr.real
    Xr_test = np.dot(X_test, V[:, 0:i])
    if Xr_test.dtype == complex:
        Xr_test.real
    Xt_test = np.dot(X_test, V)
    Xt = np.dot(X_train, V)  # original matrix in old lecture this was V.T, whats correct now?

    #stats printing:
    print('PC 1 is accounting for %.2f %% of the data' % (ratios_variance_explained[0] * 100))
    print('PC 2 is accounting for %.2f %% of the data' % (ratios_variance_explained[1] * 100))
    # Euklidean norm same as euclidean distance for r=2 -> results seem to be very similar?:
    se = (X_train - Xt) ** 2
    r = np.sqrt(se.sum())
    print('Reconstruction error r on normalized data is :' + str(r))
    r = np.linalg.norm(X_train - Xt)
    print('Reconstruction error r on normalized data is :' + str(r))  # 3643 is quite high. what is that telling us?

    return Xr, Xt, Xr_test, Xt_test, i, ratios_variance_explained


def featurePC(i, X_head, ratios_variance_explained):
    # Visualization: Features and their abundancy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(X_head[0:i], ratios_variance_explained[0:i])
    ax.set_xticks(np.arange(0, len(X_head[0:i])))
    ax.set_xticklabels(X_head[0:i], rotation=90)
    plt.show()

def pcaVis(number_of_PCs, Xt, y_train):# PCA visualization for single features; 2D:
    # change number here to see more Principal components 2D plotted
    ind = np.where(y_train == 1)[0]
    indu = np.where(y_train == 2)[0]#when 'NA' samples in y get removed this line will invoke itself. dont have to be deleted
    tox = Xt[ind]
    unkn = Xt[indu]
    good = np.delete(Xt, np.append(ind, indu), axis=0)
    fig = plt.figure()
    for i in range(1, number_of_PCs):
        ax = fig.add_subplot(number_of_PCs, 2, i)
        ax.scatter(good[:, i - 1], good[:, i], alpha=0.25, label='good', zorder=30, color='green')
        ax.scatter(tox[:, i - 1], tox[:, i], alpha=0.25, label='tox', zorder=30, color='red')
        ax.scatter(unkn[:, i - 1], unkn[:, i], alpha=0.25, label='unknown', zorder=30, color='blue')
        #ax.set_xlabel("PC - Feature x" + str(i))
        #ax.set_ylabel("PC - Feature x" + str(i + 1))
    #plt.xlabel('PC - feature x')
    #plt.ylabel('PC - feature x+1')
    plt.legend(loc='lower center')#, bbox_to_anchor=(2.5, 0))
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    fig.tight_layout(h_pad=50)
    plt.show()