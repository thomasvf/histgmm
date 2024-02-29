import numpy as np
from histgmm.core import HistogramGMM
from histgmm.utils import gaussian_1d
from histgmm.visualization import compute_gaussian_amplitudes_on_histogram


def test_amplitude_computation():
    x = np.arange(-5, 5, 0.1).reshape((100, 1))
    true_amplitudes = np.array([2, 0.4, 2])
    true_means = np.array([[-2], [3], [0]])
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
        max_iter=100
    )
    histgmm.fit(x, h)

    print(histgmm.weights_)
    
    amplitudes = compute_gaussian_amplitudes_on_histogram(histgmm, x, h)
    np.testing.assert_allclose(
        np.sort(true_amplitudes, axis=0), np.sort(amplitudes, axis=0), rtol=1e-5, atol=1e-8
    )


if __name__ == "__main__":
    test_amplitude_computation()