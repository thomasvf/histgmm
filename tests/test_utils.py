import numpy as np
import pytest
from scipy import stats

from histgmm.utils import isotropic_normal_pdf_on_the_2d_grid, multidimensional_linspace


@pytest.mark.parametrize("X, n_points, expected", [
    (np.array([[0, -1], [1, 1]]), 3, np.array([[0, -1], [0.5, 0.0], [1, 1]])),
    (np.array([[0, 0], [1, 1]]), 2, np.array([[0, 0], [1, 1]])),
    (np.array([[0], [1]]), 4, np.array([[0], [1/3], [2/3], [1]])),    
])
def test_multidimensional_linspace(X, n_points, expected):
    computed = multidimensional_linspace(X, n_points)
    np.testing.assert_allclose(computed, expected, rtol=1e-5, atol=1e-8)


def test_isotropic_normal_pdf_on_the_2d_grid():
    rv1 = stats.multivariate_normal(mean=[-2.5, -2.5], cov=[[0.1, 0], [0, 0.1]])

    x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
    pos = np.dstack((x, y))  
    probs = rv1.pdf(pos)

    pos_t, probs_t = isotropic_normal_pdf_on_the_2d_grid((-5, -5), (5, 5), [-2.5, -2.5], 0.1)
    
    np.testing.assert_allclose(pos, pos_t)
    np.testing.assert_allclose(probs, probs_t)