import numpy as np
from histgmm.core import HistogramGMM
from histgmm.utils import gaussian_1d
from histgmm.visualization import (
    compute_gaussian_amplitudes_on_histogram,
    plot_1d_assignments_and_pdfs,
    plot_1d_gaussian_fit,
)


def test_amplitude_computation(gauss_mix_1d_3component):
    x = gauss_mix_1d_3component["x"]
    h = gauss_mix_1d_3component["h"]
    true_amplitudes = gauss_mix_1d_3component["true_amplitudes"]

    histgmm = HistogramGMM(init_params="auto", n_components=3, max_iter=100)
    histgmm.fit(x, h)

    amplitudes = compute_gaussian_amplitudes_on_histogram(histgmm, x, h)
    np.testing.assert_allclose(
        np.sort(true_amplitudes, axis=0),
        np.sort(amplitudes, axis=0),
        rtol=1e-2,
        atol=1e-2,
    )


def test_plot_histogram_and_gaussians(gauss_mix_1d_3component):
    x = gauss_mix_1d_3component["x"]
    h = gauss_mix_1d_3component["h"]

    histgmm = HistogramGMM(init_params="auto", n_components=3, max_iter=100)
    histgmm.fit(x, h)

    plot_1d_gaussian_fit(
        histgmm, x, h, component_std_extension=5, n_points_for_gaussian=len(x)
    )


def test_plot_1d_assignments_and_pdfs(gauss_mix_1d_3component):
    x = gauss_mix_1d_3component["x"]
    h = gauss_mix_1d_3component["h"]

    histgmm = HistogramGMM(init_params="auto", n_components=3, max_iter=100)
    histgmm.fit(x, h)

    plot_1d_assignments_and_pdfs(
        histgmm, x, h
    )


if __name__ == "__main__":
    test_plot_1d_assignments_and_pdfs()
