#!/usr/bin/env python

import numpy as np

from histgmm import HistogramGMM
from histgmm.utils import gaussian_1d


def test_histgmm():
    x = np.arange(-5, 5, 0.1).reshape((100, 1))
    true_amplitudes = np.array([2, 0.4, 2])
    true_means = np.array([[2], [3], [0]])
    true_variances = np.array([0.1, 0.05, 0.01])

    h = (
        gaussian_1d(
            x.squeeze(), A=true_amplitudes[0], mu=true_means[0], var=true_variances[0]
        )
        + gaussian_1d(
            x.squeeze(), A=true_amplitudes[1], mu=true_means[1], var=true_variances[1]
        )
        + gaussian_1d(
            x.squeeze(), A=true_amplitudes[2], mu=true_means[2], var=true_variances[2]
        )
    )

    weights_0 = np.array([0.5, 0.5, 0.5])
    means_0 = np.array([[0.1], [5], [3]])
    covs_0 = np.array([[[0.5]], [[0.5]], [[0.5]]])

    histgmm = HistogramGMM(
        n_components=3,
        init_params=(means_0, covs_0, weights_0),
    )
    histgmm.fit(x, h)

    np.testing.assert_allclose(
        np.sort(histgmm.means_, axis=0), np.sort(means_0, axis=0), rtol=1e-5, atol=1e-8
    )


def test_auto_init():
    x = np.arange(-5, 5, 0.1).reshape((100, 1))
    true_amplitudes = np.array([2, 0.4, 2])
    true_means = np.array([[2], [3], [0]])
    true_variances = np.array([0.1, 0.05, 0.01])

    h = (
        gaussian_1d(
            x.squeeze(), A=true_amplitudes[0], mu=true_means[0], var=true_variances[0]
        )
        + gaussian_1d(
            x.squeeze(), A=true_amplitudes[1], mu=true_means[1], var=true_variances[1]
        )
        + gaussian_1d(
            x.squeeze(), A=true_amplitudes[2], mu=true_means[2], var=true_variances[2]
        )
    )

    histgmm = HistogramGMM(
        init_params="auto",
        n_components=3,
        max_iter=1000
    )
    histgmm.fit(x, h)

    np.testing.assert_allclose(
        np.sort(histgmm.means_, axis=0), np.sort(true_means, axis=0), rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        np.sort(histgmm.covariances_.squeeze(), axis=0), np.sort(true_variances, axis=0), rtol=1e-5, atol=1e-8
    )


if __name__ == "__main__":
    test_histgmm()
