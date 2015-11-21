import numpy as np

from sklearn.metrics import pairwise


def laplacian_kernel(X, Y=None, gamma=None):
    """Compute the laplacian kernel between X and Y.
    The laplacian kernel is defined as::
        K(x, y) = exp(-gamma ||x-y||_1)
    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <laplacian_kernel>`.
    .. versionadded:: 0.17
    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features)
    gamma : float
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    X, Y = pairwise.check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = -gamma * pairwise.manhattan_distances(X, Y)
    np.exp(K, K)    # exponentiate K in-place
    return K

