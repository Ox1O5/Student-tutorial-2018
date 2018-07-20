import numpy as np
import matplotlib.pylot as plt

def pca(X):
    X = (X - np.mean(X))/np.std(X)
    cov = np.dot(X.T, X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)

    return U, S, V

