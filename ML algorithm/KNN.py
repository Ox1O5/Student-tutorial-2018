import numpy as np
import utils

mnist_dir = 'MNIST-data'
X_train, y_train, X_test, y_test = utils.read_mnist(mnist_dir, flatten=True)

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

class KNearestNeighbor(object):
    def _init_(self):
        pass

    def train(self, X, y):
        """
        :param X: A Numpy array of shape (num_train, D) containing the train data consisting of
            num_train samples and dimension D.

        :param y: A Numpy array of shape (num_train,) containing the train labels, where y[i] is
            the label for X[i]

        """
        self.X_train = X
        self.y_train = y

    def compute_distance(self, X):
        """
        :return: dists: A numpy array of shape (num_test, num_train) where dists[i, j]
            is the Euclidean distance between the ith test point and the jth training
            point.
            The specific formula of dists is written in the learning report
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.sqrt(-2*np.dot(X, self.X_train.T) + np.sum(np.square(X), axis=1, keepdims=True)
                        + np.sum(np.square(self.X_train), axis=1, keepdims=True).T)
        dists=np.reshape(dists,(num_test, num_train))

        return dists

    def predict_label(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return  y_pred

    def predict(self, X, k=1):
        """

        :param X: A Numpy array of shape (num_train, D) containing the train data consisting of
            num_train samples and dimension D.

        :param k: The number of nearest neighbors that vote for the predicted labels.

        dists: A numpy array of shape (num_test, num_train) where dists[i, j]
            is the Euclidean distance between the ith test point and the jth training
            point.
        """
        dists = self.compute_distance(X)

        return self.predict_label(dists, k = k)


classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
dists = classifier.compute_distance(X_test)

K = 7

y_test_pred = classifier.predict(X_test, k=K)

num_correct = np.sum(y_test_pred == y_test)
accuracy = 1.0 * num_correct / X_test.shape[0]

print("k = %d, accuracy = %f" %(K, accuracy))


