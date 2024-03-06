import numpy as np
import pytest

from histgmm.utils import gaussian_1d, isotropic_normal_pdf_on_the_2d_grid


@pytest.fixture
def gauss_mix_1d_3component():
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

    distribution_data = {
        "x": x,
        "h": h,
        "true_amplitudes": true_amplitudes,
        "true_means": true_means,
        "true_variances": true_variances,
    }

    return distribution_data


@pytest.fixture
def gauss_mix_1d_2component():
    x = np.arange(-5, 5, 0.1).reshape((100, 1))
    true_amplitudes = np.array([1, 1])
    true_means = np.array([[-2.5], [2.5]])
    true_variances = np.array([0.1, 0.1])

    h = (
        gaussian_1d(
            x.squeeze(), A=true_amplitudes[0], mu=true_means[0], var=true_variances[0]
        )
        + gaussian_1d(
            x.squeeze(), A=true_amplitudes[1], mu=true_means[1], var=true_variances[1]
        )
    )

    distribution_data = {
        "x": x,
        "h": h,
        "true_amplitudes": true_amplitudes,
        "true_means": true_means,
        "true_variances": true_variances,
    }

    return distribution_data


@pytest.fixture
def gauss_2d_2component():
    x1, h1 = isotropic_normal_pdf_on_the_2d_grid((-5, -5), (5, 5), [-2.5, -2.5], 0.1)
    x2, h2 = isotropic_normal_pdf_on_the_2d_grid((-5, -5), (5, 5), [2.5, 2.5], 0.3)
    assert np.allclose(x1, x2)
    x = x1.reshape(-1, 2)

    distribution_data = {
        "x": x,
        "h": (h1 + h2).reshape(-1),
        "true_means": np.array([[-2.5, -2.5], [2.5, 2.5]]),
        "true_covariances": np.array([
            [[0.1, 0], [0, 0.1]],
            [[0.3, 0], [0, 0.3]],
        ])
    }

    return distribution_data
