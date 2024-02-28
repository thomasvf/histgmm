import numpy as np
import pytest

from histgmm.utils import multidimensional_linspace


@pytest.mark.parametrize("X, n_points, expected", [
    (np.array([[0, -1], [1, 1]]), 3, np.array([[0, -1], [0.5, 0.0], [1, 1]])),
    (np.array([[0, 0], [1, 1]]), 2, np.array([[0, 0], [1, 1]])),
    (np.array([[0], [1]]), 4, np.array([[0], [1/3], [2/3], [1]])),    
])
def test_multidimensional_linspace(X, n_points, expected):
    computed = multidimensional_linspace(X, n_points)
    np.testing.assert_allclose(computed, expected, rtol=1e-5, atol=1e-8)
