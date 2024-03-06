#!/usr/bin/env python

import numpy as np

from histgmm import HistogramGMM
from histgmm.utils import gaussian_1d


def test_histgmm(gauss_mix_1d_3component):
    x = gauss_mix_1d_3component["x"]
    h = gauss_mix_1d_3component["h"]
    true_means = gauss_mix_1d_3component["true_means"]

    weights_0 = np.array([0.5, 0.5, 0.5])
    means_0 = np.array([[0.1], [5], [3]])
    covs_0 = np.array([[[0.5]], [[0.5]], [[0.5]]])

    histgmm = HistogramGMM(
        n_components=3,
        init_params=(means_0, covs_0, weights_0),
    )
    histgmm.fit(x, h)

    np.testing.assert_allclose(
        np.sort(histgmm.means_, axis=0),
        np.sort(true_means, axis=0),
        rtol=1e-3,
        atol=1e-3,
    )


def test_auto_init(gauss_mix_1d_3component):
    x = gauss_mix_1d_3component["x"]
    h = gauss_mix_1d_3component["h"]
    true_means = gauss_mix_1d_3component["true_means"]
    true_variances = gauss_mix_1d_3component["true_variances"]

    histgmm = HistogramGMM(init_params="auto", n_components=3, max_iter=1000)
    histgmm.fit(x, h)

    np.testing.assert_allclose(
        np.sort(histgmm.means_, axis=0),
        np.sort(true_means, axis=0),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.sort(histgmm.covariances_.squeeze(), axis=0),
        np.sort(true_variances, axis=0),
        rtol=1e-3,
        atol=1e-3,
    )


def test_predict_proba(gauss_mix_1d_2component):
    x = gauss_mix_1d_2component["x"]
    h = gauss_mix_1d_2component["h"]

    histgmm = HistogramGMM(init_params="auto", n_components=2, max_iter=1000)
    histgmm.fit(x, h)

    scores = histgmm.predict_proba(x)
    n_zeros = np.sum(scores.argmax(axis=1) == 0)

    assert n_zeros == 51
    np.testing.assert_allclose(scores[50], np.array([0.5, 0.5]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scores[75], np.array([0.0, 1.0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scores[25], np.array([1.0, 0.0]), rtol=1e-5, atol=1e-5)


def test_predict(gauss_mix_1d_2component):
    x = gauss_mix_1d_2component["x"]
    h = gauss_mix_1d_2component["h"]

    histgmm = HistogramGMM(init_params="auto", n_components=2, max_iter=1000)
    histgmm.fit(x, h)

    predictions = histgmm.predict(x)
    n_zeros = np.sum(predictions == 0)
    n_ones = np.sum(predictions == 1)

    assert n_zeros == 51
    assert n_ones == 49


def test_aic(gauss_mix_1d_2component):
    x = gauss_mix_1d_2component["x"]
    h = gauss_mix_1d_2component["h"]

    aic_list = []
    for k in range(1, 10):
        histgmm = HistogramGMM(n_components=k, init_params=None, max_iter=200)
        histgmm.fit(x, h)
        aic = histgmm.aic(x, h)
        aic_list.append(aic)

    best_k = np.argmin(aic_list) + 1
    assert best_k == 2


def test_histgmm_2d(gauss_2d_2component):
    x = gauss_2d_2component["x"]
    h = gauss_2d_2component["h"]
    true_means = gauss_2d_2component["true_means"]
    true_covariances = gauss_2d_2component["true_covariances"]

    histgmm = HistogramGMM(n_components=2, n_dimensions=2, max_iter=100)
    histgmm.fit(x, h)

    np.testing.assert_allclose(histgmm.means_, true_means, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(histgmm.covariances_, true_covariances, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_predict()
