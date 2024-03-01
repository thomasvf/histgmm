import numpy as np
import matplotlib.pyplot as plt

from .core import HistogramGMM
from .utils import gaussian_1d


def compute_gaussian_amplitudes_on_histogram(
    gmm: HistogramGMM, X: np.ndarray, h: np.ndarray
) -> np.array:
    """Compute the amplitudes of the Gaussians on the histogram.

    For each Normal component, the amplitude is computed as how much it
    needs to be stretched vertically so that its top matches the top histogram
    at the closest point to the mean of the Gaussian.

    Parameters
    ----------
    gmm : HistogramGMM
        Fitted HistogramGMM object
    X : np.ndarray
        X used to fit the HistogramGMM
    h : np.ndarray
        Histogram values used to fit the HistogramGMM

    Returns
    -------
    amplitudes : np.array(n_components,)
        Amplitudes of the Gaussians on the histogram
    """
    amplitudes = np.zeros(gmm.weights_.shape)

    for cluster in range(gmm.n_components):
        index_closest = np.argmin(np.abs(X - gmm.means_[cluster]))

        inverse_gaussian_normalization_factor = np.sqrt(
            (2 * np.pi) ** gmm.n_dims * np.linalg.det(gmm.covariances_[cluster])
        )
        amplitudes[cluster] = h[index_closest] * inverse_gaussian_normalization_factor

    return amplitudes


def plot_1d_gaussian_fit(
    gmm: HistogramGMM,
    X: np.ndarray,
    h: np.ndarray,
    ax=None,
    component_std_extension=5,
    n_points_for_gaussian=None,
):
    """Plot the fitted gaussian curves scaled to match the histogram.

    Parameters
    ----------
    gmm : HistogramGMM
        Fitted HistogramGMM object
    X : np.ndarray (n_bins, n_dims)
        Bin positions of the histogram
    h : np.ndarray (n_bins, )
        Histogram
    ax : plt.Axes, optional
        Axis where to plot, by default None and creates a new figure
    component_std_extension : int, optional
        Extension, in standard deviations from the mean, of the gaussian 
        curve to be plotted. By default 5.
    n_points_for_gaussian : int, optional
        Number of points for the gaussian curve to be plotted. If None, 
        uses given X. Increase its value to increase the resolution of the
        gaussian curve. By default None.

    Returns
    -------
    plt.Axes
        Axis where the plot was made
    """    
    stds = component_std_extension
    amplitudes = compute_gaussian_amplitudes_on_histogram(gmm=gmm, X=X, h=h)

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(X.squeeze(), h, color="lightgray")

    for k in range(gmm.n_components):
        color = plt.cm.viridis(k / gmm.n_components)
        if n_points_for_gaussian is not None:
            x_gaussian = np.linspace(
                gmm.means_[k] - stds * np.sqrt(gmm.covariances_[k]),
                gmm.means_[k] + stds * np.sqrt(gmm.covariances_[k]),
                n_points_for_gaussian,
            )
        else:
            x_gaussian = X.squeeze()

        y_gaussian = gaussian_1d(
            x_gaussian.squeeze(),
            amplitudes[k].squeeze(),
            gmm.means_[k].squeeze(),
            gmm.covariances_[k].squeeze(),
        )
        ax.plot(x_gaussian.squeeze(), y_gaussian, color=color)

    return ax


def plot_1d_assignments_and_pdfs(
    gmm: HistogramGMM,
    X: np.ndarray,
    h: np.ndarray,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    
    y_pred = gmm.predict(X)

    for k in range(gmm.n_components):
        color = plt.cm.viridis(k / gmm.n_components)
        ax.plot(X[y_pred == k], h[y_pred == k], "o", color=color)

    return ax
