from time import sleep
from typing import Tuple, Union
import numpy as np
from scipy import stats
import logging

from histgmm.utils import multidimensional_linspace, variance_for_n_std_for_all_range


logger = logging.getLogger("histgmm")


class HistogramGMM:
    def __init__(
        self,
        n_components: Union[int, str] = 2,
        n_dimensions: int = 1,
        init_params: Union[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = "auto",
        max_iter: int = 100,
        tol: float = 1e-3,
    ):
        """Fit a Gaussian Mixture Model over a histogram.

        Parameters
        ----------
        n_components : int, optional
            Number of gaussian components. If auto, uses AIC to try multiple values, by default 2.
        n_dimensions : int, optional
            Number of dimensions of the data, by default 1
        init_params : str, optional
            Tuple (means, covariances, weights), by default 'auto'
        max_iter : int, optional
            Maximum number of EM steps, by default 100
        tol : float, optional
            Tolerance when comparing floats, by default 1e-3
        """
        self.n_components = n_components
        self.init_params = init_params
        self.max_iter = max_iter
        self.tol = tol
        self.n_dims = n_dimensions

    def fit(self, X: np.ndarray, h: np.ndarray):
        """Fit the gaussians to the data

        Parameters
        ----------
        X : np.ndarray (n_bins, n_dims)
            N-dimensional position of each bin.
        h : np.ndarray (n_bins, )
            Number of counts in each bin.

        Returns
        -------
        HistogrammGMM
            Self
        """
        self._initialize_temporary_variables(X=X, h=h)
        self._initialize_parameter_containers()

        try:
            self._run_em_loop()
        except Exception as e:
            logger.error(f"Error while fitting the model: {e}")
            raise e
        finally:
            logger.info(f"EM algorithm finished after {self._n_steps_passed} steps.")
            self._delete_temporary_variables()

        return self

    def _run_em_loop(self):
        while not self._has_converged():
            self._e_step()
            self._m_step()

            self._n_steps_passed += 1
            if self._reached_maximum_iterations():
                logger.warning(
                    "Maximum number of iterations reached. Stopping without convergence"
                )
                break

    def _e_step(self):
        """Update the responsibilities (r) of each cluster."""
        for cluster in range(self.n_components):
            self._compute_unnormalized_responsabilities(cluster)
        self._normalize_responsabilities()

    def _m_step(self):
        """Update the parameters of the gaussians."""
        n_points_in_each_cluster = self.effective_number_of_points_in_each_cluster()

        self._update_weights(n_points_in_each_cluster)
        for cluster in range(self.n_components):
            self._update_means(n_points_in_each_cluster, cluster)
            self._update_covariances(n_points_in_each_cluster, cluster)

    def _update_means(self, n_points_in_cluster: np.ndarray, cluster: int):
        """Update the means of each Gaussian component

        Parameters
        ----------
        n_points_in_cluster : np.ndarray
            Effective number of points in each cluster
        cluster : int
            Which cluster to update
        """
        rk = self._responsabilities[:, cluster]
        self.means_[cluster] = (
            np.einsum("bm,b->m", self._X, rk * self._h) / n_points_in_cluster[cluster]
        )

    def _update_covariances(self, n_points_in_cluster: np.ndarray, cluster: int):
        """Update the covariance matrices of each Gaussian compoenent

        Parameters
        ----------
        n_points_in_cluster : np.ndarray
            Effective number of points in each cluster
        cluster : int
            Which cluster to update
        """
        rk = self._responsabilities[:, cluster]
        p = np.einsum(
            "bm,bn->bmn", self._X - self.means_[cluster], self._X - self.means_[cluster]
        )
        self.covariances_[cluster] = (
            np.einsum("b,bmn->mn", rk * self._h, p) / n_points_in_cluster[cluster]
        )

    def _update_weights(self, n_points_in_cluster):
        self.weights_ = n_points_in_cluster / np.sum(self._h)

    def _normalize_responsabilities(self):
        self._responsabilities = self._responsabilities / np.sum(
            self._responsabilities, axis=1, keepdims=True
        )
        return self._responsabilities

    def effective_number_of_points_in_each_cluster(self) -> np.ndarray:
        """Compute the effective number of points in each cluster.

        Returns
        -------
        np.ndarray (n_clusters,)
            Effective number of points in each cluster
        """
        return np.einsum("bk,b->k", self._responsabilities, self._h)

    def _initialize_parameter_containers(self):
        if self._initial_parameters_are_given():
            self.means_, self.covariances_, self.weights_ = (
                self.init_params[0].copy(),
                self.init_params[1].copy(),
                self.init_params[2].copy(),
            )
        else:
            self._automatic_initialization_of_parameters()

        self._check_parameters_shape()

    def _initial_parameters_are_given(self):
        if self.init_params == "auto" or self.init_params is None:
            return False
        return True

    def _automatic_initialization_of_parameters(self):
        self.means_ = multidimensional_linspace(self._X, self.n_components)
        cov = variance_for_n_std_for_all_range(self._X)
        self.covariances_ = np.tile(
            np.eye(self.n_dims) * cov, (self.n_components, self.n_dims, self.n_dims)
        )
        self.weights_ = np.ones(self.n_components) / self.n_components

    def _check_parameters_shape(self):
        assert self.means_.shape == (self.n_components, self.n_dims)
        assert self.covariances_.shape == (self.n_components, self.n_dims, self.n_dims)
        assert self.weights_.shape == (self.n_components,)

    def _initialize_temporary_variables(self, X, h):
        self._X = X.copy()
        self._h = h.copy()
        self._responsabilities = np.zeros((self._X.shape[0], self.n_components))
        self._n_steps_passed = 0

    def _delete_temporary_variables(self):
        del self._X
        del self._h
        del self._responsabilities
        del self._n_steps_passed

    def _reached_maximum_iterations(self):
        return self._n_steps_passed >= self.max_iter

    def _has_converged(self):
        return False

    def _compute_unnormalized_responsabilities(self, cluster):
        p_x_given_k = stats.multivariate_normal(
            mean=self.means_[cluster], cov=self.covariances_[cluster]
        ).pdf(self._X)
        rj = self.weights_[cluster] * p_x_given_k
        self._responsabilities[:, cluster] = rj

    def predict(self, X):
        raise NotImplementedError()

    def score(self, X):
        raise NotImplementedError()

    def sample(self, n_samples=1):
        raise NotImplementedError()

    def score_samples(self, X):
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError()

    def aic(self, X):
        raise NotImplementedError()

    def bic(self, X):
        raise NotImplementedError()
