import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA

def simpleKPCA(X, y):
    kpca = KernelPCA(kernel="linear", fit_inverse_transform=True, gamma=0.03)#["rbf", "sigmoid", "linear", "poly"]
    X_kpca = kpca.fit_transform(X)

    # Plot results
    good = y == 0
    toxic = y == 1
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_kpca[good, 0], X_kpca[good, 1], X_kpca[good, 2], c="green",
                s=20, edgecolor='k', alpha=0.25)
    ax.scatter(X_kpca[toxic, 0], X_kpca[toxic, 1], X_kpca[toxic, 2], c="red",
                s=20, edgecolor='k', alpha=0.25)
    plt.title("Projection by KPCA")
    plt.xlabel(r"1st component")
    plt.ylabel("2nd component")
    plt.show()

def kpca_try(X, y):
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(X)
    X_back = kpca.inverse_transform(X_kpca)
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Plot results
    plt.figure()
    plt.subplot(2, 2, 1, aspect='equal')
    plt.title("Original space")
    greens = y == 0 #greens = good
    reds = y == 1   #reds = toxic

    plt.scatter(X[greens, 0], X[greens, 1], c="red",
                s=20, edgecolor='k')
    plt.scatter(X[reds, 0], X[reds, 1], c="blue",
                s=20, edgecolor='k')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    #X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
    #X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
    # projection on the first principal component (in the phi space)
    #Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
    #plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

    plt.subplot(2, 2, 2, aspect='equal')
    plt.scatter(X_pca[greens, 0], X_pca[greens, 1], c="greens",
                s=20, edgecolor='k')
    plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="reds",
                s=20, edgecolor='k')
    plt.title("Projection by PCA")
    plt.xlabel("1st principal component")
    plt.ylabel("2nd component")

    plt.subplot(2, 2, 3, aspect='equal')
    plt.scatter(X_kpca[greens, 0], X_kpca[greens, 1], c="red",
                s=20, edgecolor='k')
    plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="blue",
                s=20, edgecolor='k')
    plt.title("Projection by KPCA")
    plt.xlabel(r"1st principal component in space induced by $\phi$")
    plt.ylabel("2nd component")

    plt.subplot(2, 2, 4, aspect='equal')
    plt.scatter(X_back[greens, 0], X_back[greens, 1], c="red",
                s=20, edgecolor='k')
    plt.scatter(X_back[reds, 0], X_back[reds, 1], c="blue",
                s=20, edgecolor='k')
    plt.title("Original space after inverse transform")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.tight_layout()
    plt.show()
    print('hello world')




