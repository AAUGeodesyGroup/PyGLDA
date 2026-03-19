import numpy as np
import geopandas as gpd


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
        """be careful, not choosing a big alpha. best within [0.05, 0.15]"""
        # n = self._cov.shape[0]
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
        # R_pd = R

        # Convert back to covariance
        D = np.diag(d)
        Sigma_new = D @ R_pd @ D

        self._reguCOV = Sigma_new

        pass

    def method_domain_localization(self, radius):

        """
        cov set to 0 outside the given domain
        """

        pass


class DomainLocalization:
    """
    Be aware that this method is primarily developed for the covariance localization in the observation space. Therefore,
    the localization will be designed according to the shapefile that defines the network of the observation. Currently,
    the distance between the geocenter of two basins will be used as the basis for domain localization.

    if this is for a covariance of a small dimension, call the Pmatrix which was pre-saved for accelerating computation.
    if the target covariance is huge, please call the Poperator instead which whill instantly localize the covariance each time.
    """

    def __init__(self, shapefile, radius):
        self._sh = gpd.read_file(filename=shapefile)
        self._radius = radius
        self._Pmatrix = None

        self._center_coordinate()
        pass

    def Pmatrix(self):
        if self._Pmatrix is not None:
            '''load the already-existing P matrix'''
            return self._Pmatrix

        '''calculate the Pmatrix and save it for next use'''
        basin_num = len(self._sh.ID)
        P = np.zeros(shape=(basin_num, basin_num))
        distance = np.zeros_like(P)

        for i in range(basin_num):
            point1 = np.array([self._coordinate['lat'][i], self._coordinate['lon'][i]])
            point2 = np.array([self._coordinate['lat'][i:], self._coordinate['lon'][i:]])
            distance[i, i:] = self._distance_metric_1(point1=point1, point2=point2)

        distance = distance + distance.T

        P[distance < self._radius] = 1

        self._Pmatrix = P

        return self._Pmatrix

    def Poperator(self):
        pass

    def _center_coordinate(self):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            gdf = self._sh.to_crs(4326).centroid

        coordinate_lat = gdf.y.values
        coordinate_lon = gdf.x.values

        self._coordinate = {
            'lat': coordinate_lat,
            'lon': coordinate_lon
        }
        pass

    def _distance_metric_1(self, point1, point2):
        dd = np.abs(np.array(point1[:,None] - point2))
        return np.sqrt(dd[0] ** 2 + dd[1] ** 2)


def demo1():
    dl = DomainLocalization(shapefile='/media/user/My Book/Fan/ESA_SING/shapefiles/EuropeContinent/Europe_valid'
                                      '/Europe_subbasins.shp', radius=4)

    P = dl.Pmatrix()
    Q = dl.Pmatrix()
    pass


if __name__ == '__main__':
    demo1()