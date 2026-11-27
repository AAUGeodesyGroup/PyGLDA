"""Metrics of the JoG evaluation.

Two definitions are provided on purpose:

* ``definition='paper'`` follows Appendix B of the manuscript: reference = truth (Q0), evaluated = model or DA (Qm),
  NSE = 1 - sum((Qm-Q0)^2) / sum(Q0^2)   (anomalies, so the mean of Q0 is 0),
  KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2), alpha = sigma_m / sigma_0  (beta = 1 after de-meaning),
  RMSE = std(Qm - Q0)  (centred RMSE, as stated in Appendix B).
* ``definition='published_code'`` reproduces ESAsing/paper_RSE/plot_fig_2.py exactly: NSE_2D(a, b) and KGE_2D(a, b)
  are called with a = truth and b = model/DA, so NSE is normalised by sum(b^2) (the evaluated data set) and the KGE
  variability ratio is sigma_truth / sigma_eval. RMSE is identical in both.

All arrays are (time, ...) and are de-meaned along time first (as in the paper). NaN cells give NaN.
Spatial means follow the paper (Section 5.3): NSE and KGE values below -1 are set to -1 before averaging.
"""
import numpy as np

DEFINITIONS = ('paper', 'published_code')


def demean(x):
    return x - np.nanmean(x, axis=0)


def corr(a, b):
    a = demean(a)
    b = demean(b)
    return np.sum(a * b, axis=0) / np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))


def rmse(truth, sim):
    return np.std(demean(sim) - demean(truth), axis=0)


def nse(truth, sim, definition='paper'):
    t, s = demean(truth), demean(sim)
    if definition == 'paper':
        return 1 - np.sum((s - t) ** 2, axis=0) / np.sum(t ** 2, axis=0)
    if definition == 'published_code':
        return 1 - np.sum((t - s) ** 2, axis=0) / np.sum(s ** 2, axis=0)
    raise ValueError(definition)


def kge(truth, sim, definition='paper'):
    t, s = demean(truth), demean(sim)
    r = corr(t, s)
    if definition == 'paper':
        a = np.std(s, axis=0) / np.std(t, axis=0)
    elif definition == 'published_code':
        a = np.std(t, axis=0) / np.std(s, axis=0)
    else:
        raise ValueError(definition)
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2)


def all_metrics(truth, sim, definition='paper'):
    return {'nse': nse(truth, sim, definition), 'kge': kge(truth, sim, definition), 'rmse': rmse(truth, sim),
            'r': corr(truth, sim)}


def spatial_mean(values, name, clip=-1.0):
    """spatial mean of a metric map as in the paper (NSE/KGE clipped at -1; RMSE and r not clipped)"""
    v = np.asarray(values, dtype=float)
    if name in ('nse', 'kge') and clip is not None:
        v = np.where(v < clip, clip, v)
    return float(np.nanmean(v))


def improvement(model, grace_c, other):
    """Table 1 convention: (other - GRACE-C) and its ratio to (GRACE-C - model), in %"""
    d = other - grace_c
    return d, 100.0 * d / (grace_c - model)
