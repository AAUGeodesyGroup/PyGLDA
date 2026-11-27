"""Spherical harmonic synthesis of ESM2.0 (ICGEM, fully normalized Stokes coefficients) to 0.5 deg EWH.

Used by scripts/09_ice_solid_earth.py (I and S components of ESM2.0 HIS).
Grid: latitude -89.75..89.75 (south first), longitude -179.75..179.75, as Global_EWH_05_HIS.hdf5.
EWH (m) = R rho_e / (3 rho_w) sum_n (2n+1)/(1+k'_n) sum_m Pnm(sin phi) (Cnm cos m lam + Snm sin m lam)
The load Love numbers k'_n, densities and radius are arguments (validated against the published truth grid).
"""
from datetime import datetime, timedelta

import numpy as np


def read_icgem(fn, lmax=180):
    """C, S (lmax+1, lmax+1) of an ICGEM 'gfc' file (header up to end_of_head)"""
    C = np.zeros((lmax + 1, lmax + 1))
    S = np.zeros((lmax + 1, lmax + 1))
    with open(fn) as f:
        for line in f:
            if line.startswith('end_of_head'):
                break
        rows = [ln.split() for ln in f if ln.startswith('gfc')]
    a = np.array([r[1:5] for r in rows], dtype=float)
    n, m = a[:, 0].astype(int), a[:, 1].astype(int)
    ok = n <= lmax
    C[n[ok], m[ok]] = a[ok, 2]
    S[n[ok], m[ok]] = a[ok, 3]
    return C, S


def write_icgem(fn, C, S, modelname, gm=3.986005e14, radius=6378137.0):
    lmax = C.shape[0] - 1
    with open(fn, 'w') as f:
        f.write('begin_of_head ' + '=' * 60 + '\n')
        f.write('product_type                  gravity_field\n')
        f.write('modelname                     %s\n' % modelname)
        f.write('earth_gravity_constant        %.11E\n' % gm)
        f.write('radius                        %.11E\n' % radius)
        f.write('max_degree                    %d\n' % lmax)
        f.write('norm                          fully_normalized\n')
        f.write('errors                        no\n\n')
        f.write('key   L   M       C                   S                   \n')
        f.write('end_of_head ' + '=' * 60 + '\n')
        for n in range(lmax + 1):
            for m in range(n + 1):
                f.write('gfc %5d%5d %21.14E %21.14E\n' % (n, m, C[n, m], S[n, m]))


def legendre(lmax, lat_deg):
    """fully normalized associated Legendre functions P[nlat, n, m] at the given latitudes"""
    x = np.sin(np.deg2rad(lat_deg))
    u = np.cos(np.deg2rad(lat_deg))
    P = np.zeros((len(x), lmax + 1, lmax + 1))
    P[:, 0, 0] = 1.0
    if lmax >= 1:
        P[:, 1, 1] = np.sqrt(3.0) * u
    for m in range(2, lmax + 1):
        P[:, m, m] = u * np.sqrt((2.0 * m + 1) / (2.0 * m)) * P[:, m - 1, m - 1]
    for m in range(0, lmax):
        P[:, m + 1, m] = np.sqrt(2.0 * m + 3) * x * P[:, m, m]
        for n in range(m + 2, lmax + 1):
            a = np.sqrt((2.0 * n - 1) * (2.0 * n + 1) / ((n - m) * (n + m)))
            b = np.sqrt((2.0 * n + 1) * (n + m - 1) * (n - m - 1) / ((n - m) * (n + m) * (2.0 * n - 3)))
            P[:, n, m] = a * x * P[:, n - 1, m] - b * P[:, n - 2, m]
    return P


class Synthesis:
    """synthesis of (C, S) to the 0.5 deg grid (lat south->north) after scaling by degree factors"""

    def __init__(self, lmax=180, res=0.5):
        self.lmax = lmax
        self.lat = np.arange(-90 + res / 2, 90, res)
        self.lon = np.arange(-180 + res / 2, 180, res)
        self.P = legendre(lmax, self.lat)                      # (nlat, n, m)
        m = np.arange(lmax + 1)
        ml = np.deg2rad(np.outer(m, self.lon))
        self.cos, self.sin = np.cos(ml), np.sin(ml)            # (m, nlon)

    def grid(self, C, S, factor):
        """factor (lmax+1,) per degree; returns (nlat, nlon)"""
        Cf = C[:self.lmax + 1, :self.lmax + 1] * factor[:, None]
        Sf = S[:self.lmax + 1, :self.lmax + 1] * factor[:, None]
        A = np.einsum('inm,nm->im', self.P, Cf)
        B = np.einsum('inm,nm->im', self.P, Sf)
        return A @ self.cos + B @ self.sin


def ewh_factor(love, lmax=180, radius=6378136.3, rho_e=5517.0, rho_w=1000.0, nmin=0):
    """degree factors Stokes -> EWH (m); love: k'_n array (index = degree); degrees < nmin set to 0"""
    n = np.arange(lmax + 1)
    f = radius * rho_e / (3.0 * rho_w) * (2 * n + 1) / (1.0 + np.asarray(love[:lmax + 1]))
    f[:nmin] = 0.0
    return f


# load Love numbers k'_n, n = 0..70, of Wang et al. (2012) as in SGPPToolbox data/Auxiliary/LoveNumber.mat
# (column 4, degree 1 = 0.02616765; k'_0 = 0)
WANG_K = [0.0, 0.02616765, -0.30516104, -0.19585733, -0.13352284, -0.10456531, -0.090184841, -0.081906787,
          -0.076379141, -0.072250183, -0.068934145, -0.066147992, -0.063736253, -0.06160287, -0.059683159,
          -0.05793118, -0.056313294, -0.054804452, -0.053385807, -0.052043088, -0.050765423, -0.049544487,
          -0.048373866, -0.047248575, -0.046164708, -0.04511916, -0.044109431, -0.043133464, -0.04218954,
          -0.041276184, -0.040392105, -0.039536148, -0.03870726, -0.037904463, -0.037126837, -0.03637351,
          -0.035643644, -0.034936432, -0.034251094, -0.033586873, -0.032943035, -0.032318866, -0.031713675,
          -0.031126789, -0.030557557, -0.030005347, -0.029469547, -0.028949568, -0.028444838, -0.027954806,
          -0.02747894, -0.027016729, -0.026567679, -0.026131317, -0.025707185, -0.025294846, -0.024893877,
          -0.024503874, -0.024124449, -0.023755228, -0.023395852, -0.02304598, -0.02270528, -0.022373437,
          -0.022050147, -0.021735119, -0.021428073, -0.021128742, -0.020836869, -0.020552206, -0.020274516]
PUBLISHED_LMAX = 70


def published_factor(lmax_grid=180):
    """degree factors that reproduce Global_EWH_05_HIS.hdf5 (EWHA + EWH_background) to ~1e-8 m:
    degrees 0..70 (degree 0 and 1 included, no smoothing), Wang load Love numbers, R = 6378136.3 m,
    rho_e = 5517, rho_w = 1000 kg/m3; degrees > 70 dropped"""
    f = np.zeros(lmax_grid + 1)
    f[:PUBLISHED_LMAX + 1] = ewh_factor(np.array(WANG_K), PUBLISHED_LMAX, 6378136.3, 5517.0, 1000.0, 0)
    return f


def six_hourly_window(epoch_start):
    """the 21 six-hourly time stamps 00:00 of epoch_start .. 00:00 of epoch_start + 5 d (inclusive).
    This reproduces the ESM2.0 5-day HIS files used for the published truth (file label = epoch_start + 1 d;
    verified to 1e-25 on the coefficients for 2004 and 2006 epochs)."""
    t0 = datetime(epoch_start.year, epoch_start.month, epoch_start.day)
    return [t0 + timedelta(hours=6 * k) for k in range(21)]
