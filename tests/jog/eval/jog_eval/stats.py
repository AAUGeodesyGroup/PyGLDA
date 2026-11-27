"""Uncertainty and robustness tools for the revision (R1.2, R1.3, R1.4/R4.8, R3.5).

* moving-block bootstrap over 5-day epochs, paired across products (same resampled epochs for model, GRACE-C, NGGM,
  MAGIC), giving percentile CIs of spatial-mean metrics and of their differences (R1.3);
* split-sample selection: choose the best configuration on one period, evaluate on the other (R1.2);
* band-pass decomposition of errors (high/mid/low frequency) for model, DA and observation errors (R1.4, R4.8);
* record-length sensitivity: metrics over growing windows (R3.5).
"""
import numpy as np
from scipy import signal

from . import metrics as M


def block_bootstrap_indices(n, block, n_boot, seed=20021):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return idx


def bootstrap_table(truth, sims, metric_names=('nse', 'kge', 'rmse'), definition='paper', block=18, n_boot=1000,
                    seed=20021, pairs=()):
    """truth (T, ...); sims {name: (T, ...)}; pairs [(a, b), ...] for differences a - b.
    Returns {'point': {name: {metric: v}}, 'ci': {name or 'a-b': {metric: (lo, hi)}}, 'block': block, 'n_boot': n}"""
    T = truth.shape[0]
    idx = block_bootstrap_indices(T, block, n_boot, seed)

    def table(sel):
        out = {}
        for k, s in sims.items():
            m = M.all_metrics(truth[sel], s[sel], definition)
            out[k] = {n: M.spatial_mean(m[n], n) for n in metric_names}
        return out

    point = table(np.arange(T))
    boot = [table(i) for i in idx]
    ci = {}
    for k in sims:
        ci[k] = {n: tuple(np.percentile([b[k][n] for b in boot], [2.5, 97.5])) for n in metric_names}
    for a, b in pairs:
        ci['%s-%s' % (a, b)] = {n: tuple(np.percentile([x[a][n] - x[b][n] for x in boot], [2.5, 97.5]))
                               for n in metric_names}
        point['%s-%s' % (a, b)] = {n: point[a][n] - point[b][n] for n in metric_names}
    return {'point': point, 'ci': ci, 'block': block, 'n_boot': n_boot, 'definition': definition}


def split_sample(scores_a, scores_b, key='kge'):
    """scores_x: {config: {metric: value}} on period A (selection) and B (evaluation).
    Returns the config chosen on A, its score on B, the best config on B and the regret."""
    pick = max(scores_a, key=lambda c: scores_a[c][key])
    best_b = max(scores_b, key=lambda c: scores_b[c][key])
    return {'chosen_on_A': pick, 'score_on_B': scores_b[pick][key], 'best_on_B': best_b,
            'best_score_on_B': scores_b[best_b][key], 'regret': scores_b[best_b][key] - scores_b[pick][key]}


def butter_band(x, low_days=None, high_days=None, step_days=5, order=3):
    """zero-phase Butterworth filter along time (5-day steps).
    low_days = longest period kept (high-pass cut-off), high_days = shortest period kept (low-pass cut-off):
    (30, None) keeps periods < 30 d; (90, 30) keeps 30-90 d; (None, 365) keeps periods > 365 d."""
    fs = 1.0 / step_days
    nyq = fs / 2
    if low_days and high_days:
        b, a = signal.butter(order, [1 / low_days / nyq, 1 / high_days / nyq], btype='band')
    elif low_days:
        b, a = signal.butter(order, 1 / low_days / nyq, btype='high')
    elif high_days:
        b, a = signal.butter(order, 1 / high_days / nyq, btype='low')
    else:
        return x
    y = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x).all(axis=0)
    y[:, ok] = signal.filtfilt(b, a, x[:, ok], axis=0)
    return y


BANDS = {'<60 d': (60, None), '<30 d': (30, None), '30-90 d': (90, 30), '90-365 d': (365, 90), '>365 d': (None, 365)}


def band_errors(truth, sim, bands=BANDS):
    """RMS error per band (spatial mean of the per-cell std of the filtered error), mm"""
    e = M.demean(sim) - M.demean(truth)
    out = {}
    for name, (low, high) in bands.items():
        f = butter_band(e, low_days=low, high_days=high)
        out[name] = float(np.nanmean(np.nanstd(f, axis=0)))
    return out


def record_length(truth, sims, lengths, metric='kge', definition='paper'):
    """spatial-mean metric over windows of growing length (number of epochs from the end of the record)"""
    T = truth.shape[0]
    res = {}
    for L in lengths:
        sel = np.arange(T - L, T)
        res[L] = {k: M.spatial_mean(M.all_metrics(truth[sel], s[sel], definition)[metric], metric) for k, s in sims.items()}
    return res
