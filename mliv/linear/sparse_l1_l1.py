"""
This module provides implementations of sparse linear NPIV estimators.

Classes:
    _SparseLinearAdversarialGMM: Base class for sparse linear adversarial GMM.
    sparse_l1vsl1: Sparse Linear NPIV estimator using $\ell_1-\ell_1$ optimization.
    sparse_ridge_l1vsl1: Sparse Ridge NPIV estimator using $\ell_1-\ell_1$ optimization.
"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.base import clone
from mliv.linear.utilities import cross_product

class _SparseLinearAdversarialGMM:
    """
    Base class for sparse linear adversarial GMM.

    This class implements common functionality for sparse linear models using adversarial GMM.

    Attributes:
        lambda_theta (float): Regularization parameter.
        B (int): Budget parameter.
        eta_theta (str or float): Learning rate for theta.
        eta_w (str or float): Learning rate for w.
        n_iter (int): Number of iterations.
        tol (float): Tolerance for duality gap.
        sparsity (int or None): Sparsity level for the model.
        fit_intercept (bool): Whether to fit an intercept.

    Methods:
        fit(Z, X, Y): Fit the model.
        predict(X): Predict using the fitted model.
    """

    def __init__(self, lambda_theta=0.01, B=100, eta_theta='auto', eta_w='auto',
                 n_iter=2000, tol=1e-2, sparsity=None, fit_intercept=True):
        """
        Initialize the sparse linear adversarial GMM model.

        Args:
            lambda_theta (float, optional): Regularization parameter. Defaults to 0.01.
            B (int, optional): Budget parameter. Defaults to 100.
            eta_theta (str or float, optional): Learning rate for theta. Defaults to 'auto'.
            eta_w (str or float, optional): Learning rate for w. Defaults to 'auto'.
            n_iter (int, optional): Number of iterations. Defaults to 2000.
            tol (float, optional): Tolerance for duality gap. Defaults to 1e-2.
            sparsity (int or None, optional): Sparsity level for the model. Defaults to None.
            fit_intercept (bool, optional): Whether to fit an intercept. Defaults to True.
        """
        self.B = B
        self.lambda_theta = lambda_theta
        self.eta_theta = eta_theta
        self.eta_w = eta_w
        self.n_iter = n_iter
        self.tol = tol
        self.sparsity = sparsity
        self.fit_intercept = fit_intercept

    def _check_input(self, Z, X, Y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            Z = np.hstack([np.ones((X.shape[0], 1)), Z])
        return Z, X, Y.flatten()

    def predict(self, X):
        """
        Predict using the fitted model.

        Args:
            X (array-like): Covariates.

        Returns:
            array: Predicted values.
        """
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.coef_)

    @property
    def coef(self):
        return self.coef_[1:] if self.fit_intercept else self.coef_

    @property
    def intercept(self):
        return self.coef_[0] if self.fit_intercept else 0


class sparse_l1vsl1(_SparseLinearAdversarialGMM):
    """
    Sparse Linear NPIV estimator using $\ell_1-\ell_1$ optimization.

    This class solves the high-dimensional sparse linear problem using $\ell_1$ relaxations for the minimax optimization problem.

    Attributes:
        Same as `_SparseLinearAdversarialGMM`.

    Methods:
        fit(Z, X, Y): Fit the model.
        predict(X): Predict using the fitted model.
    """

    def fit(self, Z, X, Y):
        """
        Fit the model.

        Args:
            Z (array-like): Instrumental variables.
            X (array-like): Covariates.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        n = X.shape[0]
        B = self.B
        eta_theta = .5 if self.eta_theta == 'auto' else self.eta_theta
        eta_w = .5 if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        yz = np.mean(Y.reshape(-1, 1) * Z, axis=0)
        if d_x * d_z < n**2:
            xz = np.mean(cross_product(X, Z),
                         axis=0).reshape(d_z, d_x).T

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.ones(2 * d_x) * B / (2 * d_x)
                theta_acc = np.ones(2 * d_x) * B / (2 * d_x)
                w = np.ones(2 * d_z) / (2 * d_z)
                w_acc = np.ones(2 * d_z) / (2 * d_z)
                res = np.zeros(2 * d_z)
                res_pre = np.zeros(2 * d_z)
                cors = 0

            # quantities for updating theta
            if d_x * d_z < n**2:
                cors_t = xz @ (w[:d_z] - w[d_z:])
            else:
                test_fn = np.dot(Z, w[:d_z] -
                                 w[d_z:]).reshape(-1, 1)
                cors_t = np.mean(test_fn * X, axis=0)
            cors += cors_t

            # quantities for updating w
            if d_x * d_z < n**2:
                res[:d_z] = (theta[:d_x] -
                             theta[d_x:]).T @ xz - yz
            else:
                pred_fn = np.dot(X, theta[:d_x] -
                                 theta[d_x:]).reshape(-1, 1)
                res[:d_z] = np.mean(Z * pred_fn, axis=0) - yz
            res[d_z:] = - res[:d_z]

            # update theta
            theta[:d_x] = np.exp(-1 - eta_theta *
                                 (cors + cors_t + (t + 1) * lambda_theta))
            theta[d_x:] = np.exp(-1 - eta_theta *
                                 (- cors - cors_t + (t + 1) * lambda_theta))
            normalization = np.sum(theta)
            if normalization > B:
                theta[:] = theta * B / normalization

            # update w
            w[:] = w * \
                np.exp(2 * eta_w * res - eta_w * res_pre)
            w[:] = w / np.sum(w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res

            if t % 50 == 0:
                self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
                self.w_ = w_acc[:d_z] - w_acc[d_z:]
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.rho_ = theta_acc
        self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
        self.w_ = w_acc[:d_z] - w_acc[d_z:]

        self._post_process(Z, X, Y)

        return self


class sparse_ridge_l1vsl1(_SparseLinearAdversarialGMM):
    """
    Sparse Ridge NPIV estimator using $\ell_1-\ell_1$ optimization.

    This class solves the high-dimensional sparse ridge problem using $\ell_1$ relaxations for the minimax optimization problem.

    Attributes:
        Same as `_SparseLinearAdversarialGMM`.

    Methods:
        fit(Z, X, Y): Fit the model.
        predict(X): Predict using the fitted model.
    """

    def fit(self, Z, X, Y):
        """
        Fit the model.

        Args:
            Z (array-like): Instrumental variables.
            X (array-like): Covariates.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        n = X.shape[0]
        B = self.B
        eta_theta = .5 if self.eta_theta == 'auto' else self.eta_theta
        eta_w = .5 if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        yz = np.mean(Y.reshape(-1, 1) * Z, axis=0)
        xx = np.mean(cross_product(X, X),
                     axis=0).reshape(d_x, d_x).T
        self.xx = xx
        # Perform SVD on E_n[xx^T]
        Sigma = np.linalg.svd(xx, compute_uv=False)
        # Find the minimum non-zero singular value
        sigma_min = np.min(Sigma[Sigma > 1e-10])
        # Compute the maximum singular value of the pseudoinverse
        self.msvp = 1 / sigma_min

        if d_x * d_z < n**2:
            xz = np.mean(cross_product(X, Z),
                         axis=0).reshape(d_z, d_x).T

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.ones(2 * d_x) * B / (2 * d_x)
                theta_acc = np.ones(2 * d_x) * B / (2 * d_x)
                w = np.ones(2 * d_z) / (2 * d_z)
                w_acc = np.ones(2 * d_z) / (2 * d_z)
                res = np.zeros(2 * d_z)
                res_pre = np.zeros(2 * d_z)
                cors = 0

            # quantities for updating theta
            xx_theta = xx @ (theta[:d_x] - theta[d_x:])
            if d_x * d_z < n**2:
                cors_t = xz @ (w[:d_z] - w[d_z:]) + lambda_theta * xx_theta
            else:
                test_fn = np.dot(Z, w[:d_z] -
                                 w[d_z:]).reshape(-1, 1)
                cors_t = np.mean(test_fn * X, axis=0) + lambda_theta * xx_theta
            cors += cors_t

            # quantities for updating w
            if d_x * d_z < n**2:
                res[:d_z] = (theta[:d_x] -
                             theta[d_x:]).T @ xz - yz
            else:
                pred_fn = np.dot(X, theta[:d_x] -
                                 theta[d_x:]).reshape(-1, 1)
                res[:d_z] = np.mean(Z * pred_fn, axis=0) - yz
            res[d_z:] = - res[:d_z]

            # update theta
            theta[:d_x] = np.exp(-1 - eta_theta * (cors + cors_t))
            theta[d_x:] = np.exp(-1 - eta_theta * (- cors - cors_t))
            normalization = np.sum(theta)
            if normalization > B:
                theta[:] = theta * B / normalization

            # update w
            w[:] = w * \
                np.exp(2 * eta_w * res - eta_w * res_pre)
            w[:] = w / np.sum(w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res

            if t % 50 == 0:
                self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
                self.w_ = w_acc[:d_z] - w_acc[d_z:]
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.rho_ = theta_acc
        self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
        self.w_ = w_acc[:d_z] - w_acc[d_z:]

        self._post_process(Z, X, Y)

        return self
