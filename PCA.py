import os
import numpy as np
import re
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from IPython.display import HTML, display, clear_output

import utils_pca

try:
    pyplot.rcParams["animation.html"] = "jshtml"
except ValueError:
    pyplot.rcParams["animation.html"] = "html5"

from scipy import optimize
from scipy.io import loadmat


# Load the data set
data = loadmat(os.path.join('PCA', 'Data', 'ex7data1.mat'))
X= data['X']

# Visualize
pyplot.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=1)
pyplot.axis([0.5, 6.5, 2, 8])
pyplot.gca().set_aspect('equal')
pyplot.grid(False)

def pca(X):
    """ Run principle component analysis

        Parameters
        ----------
        X : array like
            The dataset to be used for computing PCA. It has dimensions (m x n) where m is the number of examples (observations) and n is the number of features.

        Returns
        -------
        U : array like
            The eigenvectors, representing the computed principal components of X. U has dimensions (n x n) where each column is a principal component.
        S : array like
            A vector of size n, containing the singular values for each principal component. Note this is the diagonal of the mentioned matrix in class.

        Instructions
        ------------
        You should first compute the covariance matrix. Then, you should use the "svd" function to compute the eigenvectors and eigenvalues of the covariance matrix.

        Notes
        -----
        When computing the covariance matrix, remember to divide by m (the number of examples).
        """
    m,n = X.shape
    U = np.zeros(n)
    S = np.zeros(n)

    # Compute the covariance matrix
    covmatrix = (1 / m) * np.dot(X.T, X)

    #
    U, S, V = np.linalg.svd(covmatrix)

    return U, S


# It's important to first normalize(centralize) X before running PCA
X_norm, mu, sigma = utils_pca.featureNormalize(X)

# Run PCA
U, S = pca(X_norm)

# Draw the eigenvectors centered at mean of data.
# These line show the directions of maximum variations in the dataset
fig, ax = pyplot.subplots()
ax.plot(X[:, 0], X[:, 1], 'bo', ms = 10, mec = 'k', mew = 0.25)

for i in range(2):
    ax.arrow(mu[0], mu[1], 1.5 * S[i] * U[0, i], 1.5 * S[i] * U[1,i],
             head_width = 0.25, head_length = 0.2, fc = 'k', ec = 'k', lw = 2,
             zorder = 1000)

ax.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
ax.grid(False)

print('Top eigenvector: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
print(' (you should expect to see [-0.707107 -0.707107])')


def projectData(X, U, K):
    """
    Computes the reduced data representation when projecting only on to the top K eigenvectors.

    Parameters
    ----------
    X : array like
        The input dataset of shape (m x n). The dataset is assumed to be normalized.
    U : array like
        The computed eigenvectors using PCA. This is a matrix of shape (n x n). Each column in the matrix represents a single
        eigenvector (or a single principal component).
    K : int
        Number of dimensions to project onto. Must be smaller than n.

    Returns
    -------
    Z : array like
        The projects of the dataset onto the top K eigenvectors. This will be a matrix of shape (m x k).

    Instructions
    ------------
    Compute the projection of the data using only the top K eigenvectors in U (first K columns).
    For the i-th example X[i, :], the projection on to the k-th eigenvector is given as follows:
    x = X[i, :]
    projection_k = np.dot(x, U[:, k])
    """

    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # ====================== YOUR CODE HERE ======================
    # Compute the projection of the data using the top K eigenvectors in U (first K columns)
    Z = np.dot(X, U[:, :K])

    # =============================================================
    return Z

# Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: {:.6f}'.format(Z[0, 0]))
print('(this value should be about    : 1.481274)')


def recoverData(Z, U, K):
    """
    Recovers an approximation of the original data when using the projected data.

    Parameters
    ----------
    Z : array like
        The reduced data after applying PCA. This is a matrix of shape (m x K).
    U : array like
        The eigenvectors (principal components) computed by PCA. This is a matrix of shape (n x n) where each column represents a single eigenvector.
    K : int
        The number of principal components retained (should be less than n).

    Returns
    -------
    X_rec : array like
        The recovered data after transformation back to the original dataset space. This is a matrix of shape (m x n), where m is the number of examples and n is the dimensions (number of features) of the original dataset.

    Instructions
    ------------
    Compute the approximation of the data by projecting back onto the original space using the top K eigenvectors in U.
    For the i-th example Z[i, :], the (approximate) recovered data for dimension j is given as follows:
    v = Z[i, :]
    recovered_j = np.dot(v, U[j, :K])
    Notice that U[j, :K] is a vector of size K.
    """

    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # ====================== YOUR CODE HERE ======================
    # Compute the approximation of the data by projecting back onto the original space using the top K eigenvectors in U
    X_rec = np.dot(Z, U[:, :K].T)

    # =============================================================
    return X_rec


X_rec = recoverData(Z, U, K)
print('Approximation of the first example: [{:.6f} {:.6f}]'.format(X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about [-1.047419 -1.047419])')

# Plot the normalized dataset (returned from featureNormalize)
fig, ax = pyplot.subplots(figsize=(5, 5))
ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo', ms=8, mec='b', mew=0.5)
ax.set_aspect('equal')
ax.grid(False)
pyplot.axis([-3, 2.75, -3, 2.75])

# Draw lines connecting the projected points to the original points
ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', mec='r', mew=2, mfc='none')

for xnorm, xrec in zip(X_norm, X_rec):
    ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)

