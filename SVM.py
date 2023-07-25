import os
import numpy as np
import re
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils_svm

from sklearn.model_selection import KFold


# Example1
# Load from ex6data1
data = loadmat(os.path.join('SVM','Data','ex6data1.mat'))
X,y = data['X'], data['y'][:, 0]

# Plot training data
# utils_svm.plotData(X, y)

C = 1

model = utils_svm.svmTrain(X, y, C, utils_svm.linearKernel, 1e-3, 20)
utils_svm.visualizeBoundaryLinear(X, y, model)
pyplot.show()


# Example2
def gaussianKernel(x1, x2, sigma):

    # Initialize the target value
    sim = 0

    # Compute the RBF
    # K(x1, x2) = exp(-sum( (x1 - x2) ^ 2 ) / (2 * sigma ^ 2))
    sim = np.exp(-np.sum(np.square(x1 - x2)) / (2 * np.square(sigma)))

    return sim

# Test
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussianKernel(x1, x2, sigma)
print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:\n\t%f'
      '\n(for sigma = 2, this value should be about 0.324652)\n'% (sigma, sim))


# Load from ex6data2
# You will have X, y as keys in the dict data
data = loadmat(os.path.join('SVM', 'Data', 'ex6data2.mat'))
X, y = data['X'], data['y'][:, 0]

# Plot training data
utils_svm.plotData(X, y)

C = 1
sigma = 0.1
model = utils_svm.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils_svm.visualizeBoundary(X, y, model)
pyplot.show()


# Example3
# Load from ex6data3
# You will have X, y, Xval, yval as keys in the dice=t data
data = loadmat(os.path.join('SVM', 'Data', 'ex6data3.mat'))
X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]

# Plot training data
utils_svm.plotData(X, y)
pyplot.show()


# Cross-Validation
# Setting the times of split
def dataset3Params(X, y, Xval, yval):
    ts = 10
    kf = KFold(n_splits=ts)

    # Set the parameter
    # C参数一般越大非线性拟合能力越强
    # sigma一般用来调整函数的平滑程度
    parameter_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    best_c = 0
    best_sig = 0
    mean_accuracies = 0
    for c in parameter_set:
        for sig in parameter_set:
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = utils_svm.svmTrain(X_train, y_train, c, gaussianKernel, args=(sig,))
                prediction = utils_svm.svmPredict(model, X_test)

                accuracies = []
                accuracy = np.mean(prediction == y_test)
                accuracies.append(accuracy)

            mean_acc = np.mean(accuracies)

            if mean_acc > mean_accuracies:
                mean_accuracies = mean_acc
                best_c = c
                best_sig = sig

    return best_c, best_sig

# Train for the parameters
C, sigma = dataset3Params(X, y, Xval, yval)

# Train the SVM
model = utils_svm.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils_svm.visualizeBoundary(X, y, model)
pyplot.show()
print(C, sigma)