import numpy as np
from scipy import stats


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


def variance_for_n_std_for_all_range(X: np.ndarray, n_std: float = 4) -> np.ndarray:
    """Compute a variance such that `n_std` standard deviations
    will occupy the full range of the data."""
    return ((X.max() - X.min()) / n_std) ** 2


def ndim_normal_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Sample a normal distribution with mean `mu` and covariance `cov`.

    Parameters
    ----------
    x : np.ndarray (, n_dims)
        Points where to compute the gaussian curve
    mu : np.ndarray (n_dims, )
        Mean of the gaussian
    cov : np.ndarray (n_dims, n_dims)
        Covariance matrix of the gaussian

    Returns
    -------
    np.ndarray
        Normal pdf value at the specified points
    """
    assert x.shape[-1] == len(mu), "last x dimension and mu must match"
    assert (
        x.shape[-1] == cov.shape[0]
    ), "last x dimension must match with cov dimensions"

    rv = stats.multivariate_normal(mean=mu, cov=cov)
    return rv.pdf(x)


def ndim_isotropic_normal_pdf(x: np.ndarray, mu: np.ndarray, var: float) -> np.ndarray:
    """Sample a isotropic normal distribution with mean `mu` and variance `var`.

    Parameters
    ----------
    x : np.ndarray (, n_dims)
        Points where to compute the gaussian curve
    mu : np.ndarray (n_dims, )
        Mean of the gaussian
    var : float
        Variance of the gaussian

    Returns
    -------
    np.ndarray
        Normal pdf value at the specified points
    """
    cov = np.eye(len(mu)) * var
    return ndim_normal_pdf(x, mu, cov)


def isotropic_normal_pdf_on_the_2d_grid(
    min_x: np.ndarray, max_x: np.ndarray, mu: np.ndarray, var: float
):
    """Sample a isotropic normal distribution with mean `mu` and variance `var` on a 2D grid
    defined by `min_x` and `max_x`.

    Parameters
    ----------
    min_x : np.ndarray (2,)
        Diagonal vertex of the grid
    max_x : np.ndarray (2,)
        Other diagonal vertex of the grid
    mu : np.ndarray
        Mean of the gaussian
    var : float
        Variance of the isotrpoic gaussian

    Returns
    -------
    (np.ndarray, np.ndarray)
        Coordinates of each matrix element and normal pdf values at the specified points
    """
    x, y = np.mgrid[min_x[0] : max_x[0] : 0.1, min_x[1] : max_x[1] : 0.1]
    pos = np.dstack((x, y))
    probs = ndim_isotropic_normal_pdf(pos, mu, var)
    return (pos, probs)
