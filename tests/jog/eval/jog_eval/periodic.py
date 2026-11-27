"""Periodic (10/15/30-day) errors of the 5-day synthetic observations (script 08).

Least squares per unit on the residual r = obs - truth (both de-meaned) with
    nuisance: constant, trend, annual, semi-annual (cos/sin)
    periodic: cos/sin at P days for every P in `periods` (default 15, 30) + one term (-1)^k for 10 d
              (10 d is the Nyquist period of 5-day sampling; its sine term is zero at the epochs).
All harmonic terms use the epoch index k (global ESM2.0 index, 5-day steps), not dates: t = 5 k days.
"""
import numpy as np
from scipy import signal

STEP = 5.0
YEAR = 365.25


def design(k, periods=(15.0, 30.0), nyquist=True):
    """design matrix (n, m) and column names; k = global epoch indices"""
    k = np.asarray(k, float)
    t = STEP * k
    kc = (k - k.mean()) / max(np.ptp(k), 1.0)
    cols, names = [np.ones_like(t), kc], ['const', 'trend']
    for P, nm in ((YEAR, 'annual'), (YEAR / 2, 'semiannual')):
        cols += [np.cos(2 * np.pi * t / P), np.sin(2 * np.pi * t / P)]
        names += [nm + '_c', nm + '_s']
    for P in periods:
        cols += [np.cos(2 * np.pi * t / P), np.sin(2 * np.pi * t / P)]
        names += ['p%g_c' % P, 'p%g_s' % P]
    if nyquist:
        cols.append(np.where(np.asarray(k, int) % 2 == 0, 1.0, -1.0))
        names.append('p10_nyq')
    return np.stack(cols, 1), names


def fit(r, k, periods=(15.0, 30.0), nyquist=True):
    """r (n, units) -> dict: coef (m, units), p (periodic part, n x units), amp/phase per period (units,)"""
    A, names = design(k, periods, nyquist)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    per = np.array([n.startswith('p') for n in names])
    p = A[:, per] @ coef[per]
    amp, pha = {}, {}
    for P in periods:
        c, s = coef[names.index('p%g_c' % P)], coef[names.index('p%g_s' % P)]
        amp[P] = np.hypot(c, s)
        pha[P] = np.degrees(np.arctan2(s, c))          # r ~ amp cos(2 pi t/P - phase)
    if nyquist:
        c = coef[names.index('p10_nyq')]
        amp[10.0] = np.abs(c)
        pha[10.0] = np.where(c >= 0, 0.0, 180.0)        # (-1)^k: phase 0 or 180 deg
    return {'coef': coef, 'names': names, 'p': p, 'amp': amp, 'phase': pha}


def spectrum(x, pad=8):
    """Hann-window periodogram along axis 0 of a linearly detrended series, zero-padded to pad*n (even).
    returns periods (days, Nyquist included), power (freq, ...) in mm^2 day"""
    n = x.shape[0]
    nfft = pad * n + (pad * n) % 2
    f, P = signal.periodogram(x, fs=1 / STEP, window='hann', nfft=nfft, detrend='linear', axis=0)
    P[-1] *= 2      # scipy's one-sided spectrum does not double the Nyquist bin; make it comparable to its neighbours
    return f, P


def peaks(f, P, pmin=10.0, pmax=60.0, n_top=6, bg_halfwidth_days=None):
    """local maxima of a 1-D spectrum between pmin and pmax days (Nyquist edge included by mirroring).
    returns list of (period, power, ratio to median background of the band)"""
    sel = (f > 0) & (1 / np.maximum(f, 1e-12) >= pmin - 1e-9) & (1 / np.maximum(f, 1e-12) <= pmax)
    ff, pp = f[sel], P[sel]
    mirrored = np.concatenate([pp, pp[-2::-1]]) if np.isclose(1 / ff[-1], 10.0) else pp
    idx, prop = signal.find_peaks(mirrored, prominence=0)
    idx = idx[idx < len(pp)]
    bg = np.median(pp)
    order = idx[np.argsort(pp[idx])[::-1]][:n_top]
    return [(float(1 / ff[i]), float(pp[i]), float(pp[i] / bg)) for i in order]


def power_at(f, P, period, halfwidth=12):
    """power at the bin nearest 1/period and its ratio to the median of the +-halfwidth*4 neighbouring bins
    excluding +-halfwidth bins (local background)"""
    i = int(np.argmin(np.abs(f - 1 / period)))
    lo, hi = max(i - 4 * halfwidth, 1), min(i + 4 * halfwidth + 1, len(f))
    nb = np.r_[np.arange(lo, max(i - halfwidth, lo)), np.arange(min(i + halfwidth + 1, hi), hi)]
    return float(P[i]), float(P[i] / np.median(P[nb]))
