import numpy as np


def gaussian_1d(x: np.ndarray, A: float, mu: float, var: float) -> np.ndarray:
    """Compute a gaussian at the specified points.

    Parameters
    ----------
    x : np.ndarray
        Points where to compute the gaussian curve values
    A : float
        Multiplicative factor of the Gaussian
    mu : float
        Gaussian mean
    var : float
        Gaussian variance

    Returns
    -------
    np.ndarray
        Gaussian curve values at the specified points
    """
    return A * 1 / np.sqrt(2 * np.pi * var) * np.exp(-((x - mu) ** 2) / (2 * var))


def multidimensional_linspace(X: np.array, n_points: int) -> np.array:
    """Compute linearly separated points from the minimum to the maximum
    of the input array with n_points."""
    elements = np.linspace(np.min(X, axis=0), np.max(X, axis=0), n_points)
    return elements


def variance_for_n_std_for_all_range(X: np.ndarray, n_std: float=4) -> np.ndarray:
    """Compute a variance such that `n_std` standard deviations 
    will occupy the full range of the data."""
    return ((X.max() - X.min())/n_std)**2
