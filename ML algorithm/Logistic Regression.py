import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


X, y = make_classification(n_samples= 300, n_features=2, n_redundant=0, n_informative=2)

def sigmoid(x):
    return 1.0 / (1+np.exp(-x))

class LogisiticReg(object):
    def __init__(self):
        pass
    def train(self, X, y, learning_rate = 1e-4, iterations = 10000):
        """
        :param X: A Numpy array of shape (num_train, D) containing the train data consisting of
            num_train samples and dimension D.

        :param y: A Numpy array of shape (num_train,) containing the train labels, where y[i] is
            the label for X[i]
        :return:
        """
        y_hat = y.reshape(-1,1)
        W = np.zeros((X.shape[1], 1))
        for i in range(iterations):
            dW = 1.0 / X.shape[0] * np.dot(X.T, (sigmoid(np.dot(X, W))- y_hat))
            W = W - learning_rate * dW
        return W

    def predict(self, X, weight):
        A = sigmoid(np.dot(X, weight))
        y_pred = np.where(A<0.5, 0, 1).squeeze()
        return y_pred

classifier = LogisiticReg()
weights = classifier.train(X, y, learning_rate= 1e-3, iterations=5000)

def plot_decision_boundary(pred_func):

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k', s=25)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X, y)

plt.subplot(211)
plot_decision_boundary(lambda x: classifier.predict(x,weights))
plt.title("Logistic Regression by 261")
plt.subplot(212)
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.show()