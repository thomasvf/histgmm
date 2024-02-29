import numpy as np

from .core import HistogramGMM


def compute_gaussian_amplitudes_on_histogram(
    gmm: HistogramGMM,
    X: np.ndarray, 
    h: np.ndarray
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
            (2*np.pi)**gmm.n_dims * np.linalg.det(gmm.covariances_[cluster])
        )
        amplitudes[cluster] = h[index_closest] * inverse_gaussian_normalization_factor

    return amplitudes