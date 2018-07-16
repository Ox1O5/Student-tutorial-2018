import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def find_closest_centroids(X, centroids):
    num_sample = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(num_sample)

    for i in range(num_sample):
        min_dist = np.inf
        for j in range(K):
            dist = np.sum(np.square(X[i, :] - centroids[j, :]))
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx  #(num_sample,)

def compute_centroids(X, idx, K):
    D = X.shape[1]
    centroids = np.zeros((K, D))
    for i in range(K):
        indices = np.where(idx == i)[0] #result of np.where is a tuple of ndarray
        centroids[i, :] = np.sum(X[indices, :], axis=0, keepdims=False) /indices.shape[0]
    return centroids

def init_centroids(data_set, K):
    num_sample, D = X.shape
    centroids = np.zeros((K, D))
    idx = np.random.choice(num_sample, K, replace=False)
    centroids = X[idx, :]
    return centroids

def run_Kmeans(X, K, iters):
    num_sample, D = X.shape
    idx = np.zeros(num_sample)
    centroids = init_centroids(X, K)

    for i in range(iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    
    return idx, centroids

data = loadmat('ex7data2.mat')
X = data['X']
K = 3
idx, centroids = run_Kmeans(X, K, 100)

clusters={}
fig, ax = plt.subplots(figsize=(12,8))  
for i in range(K):
    clusters['cluster'+str(i+1)] = X[np.where(idx == i)[0], :]
    ax.scatter(clusters['cluster'+str(i+1)][:,0], clusters['cluster'+str(i+1)][:,1], s=30, label='Cluster'+str(i+1))
ax.legend()
plt.show()