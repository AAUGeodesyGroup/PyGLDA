import numpy as np

class CovRegu:

    def __init__(self):
        self._cov = None
        self._reguCOV = None
        pass

    def set_COV(self, cov):
        self._cov = cov
        return self

    def get_COV(self):
        return self._reguCOV

    def method_add_unity_empirical(self, lamb=None):
        """
        Suggest lambda within [1e-6, 1e-3], depending on scale
        """

        n = self._cov.shape[0]

        if lamb is None:
            lamb = 1e-4 * np.trace(self._cov) / n

        assert lamb > 0

        self._reguCOV = self._cov + lamb * np.eye(n)
        pass

    def method_add_unity_simplest(self):
        n = self._cov.shape[0]

        lamb = 1e-6

        self._reguCOV = self._cov + lamb * np.eye(n)
        pass

    def method_shrinkage(self, alpha):
        n = self._cov.shape[0]
        self._reguCOV = (1 - alpha) * self._cov + alpha * np.diag(np.diag(self._cov))
        pass

    def method_eigenvalue_clipping(self, min_eig=None):
        """
        Stabilize a covariance matrix using eigenvalue clipping.

        Parameters
        ----------
        Sigma : ndarray (n x n)
            Symmetric covariance matrix (can be nearly singular or slightly indefinite)
        min_eig : float
            Minimum allowed eigenvalue (default = 1e-8)

        Returns
        -------
        Sigma_stable : ndarray
            Positive definite stabilized covariance matrix
        """

        # Ensure symmetry (important for numerical stability)
        Sigma = self._cov
        Sigma = 0.5 * (Sigma + Sigma.T)

        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(Sigma)

        if min_eig is None:
            min_eig = 1e-10 * max(eigvals)

        # Clip eigenvalues
        eigvals_clipped = np.clip(a=eigvals, a_min=min_eig)

        # Reconstruct matrix
        self._reguCOV = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

        pass

    def method_nearest_positive_definite(self):
        self._reguCOV = self._nearest_positive_definite(A=self._cov)
        pass

    def _nearest_positive_definite(self, A):
        """
        Higham (1988) nearest positive definite matrix.

        This computes the closest positive definite matrix in Frobenius norm.

        This is more mathematically rigorous than simple clipping.
        """

        B = 0.5 * (A + A.T)

        # SVD
        U, s, Vt = np.linalg.svd(B)
        H = Vt.T @ np.diag(s) @ Vt

        A2 = 0.5 * (B + H)
        A3 = 0.5 * (A2 + A2.T)

        # Check if PD
        def is_pd(X):
            try:
                np.linalg.cholesky(X)
                return True
            except np.linalg.LinAlgError:
                return False

        if is_pd(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1

        while not is_pd(A3):
            min_eig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-min_eig * k ** 2 + spacing)
            k += 1

        return A3

    def method_threshold_correlation_covariance(self, tau=0.05):
        """
        Threshold weak correlations and project back to nearest PD.
        Step 1 — Threshold Correlation (Thresholding correlations does NOT preserve positive definiteness)
        Step 2 — Project back to nearest positive definite matrix

        That combination is stable.
        """
        Sigma = self._cov

        # Ensure symmetry
        Sigma = 0.5 * (Sigma + Sigma.T)

        # Convert to correlation
        d = np.sqrt(np.diag(Sigma))
        D_inv = np.diag(1 / d)
        R = D_inv @ Sigma @ D_inv

        # Threshold correlations
        R[np.abs(R) < tau] = 0.0

        # Force symmetry & unit diagonal
        R = 0.5 * (R + R.T)
        np.fill_diagonal(R, 1.0)

        # Project to nearest PD
        R_pd = self._nearest_positive_definite(A=R)

        # Convert back to covariance
        D = np.diag(d)
        Sigma_new = D @ R_pd @ D

        return Sigma_new

    def method_domain_localization(self, radius):

        """
        cov set to 0 outside the given domain
        """

        pass
