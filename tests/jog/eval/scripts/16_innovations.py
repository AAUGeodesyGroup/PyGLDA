"""Step 16 (R4.7) - normalized innovation statistics of DA runs made with output.save_innovations (PyGLDA v1).

Reads cases/<case>/DA/innovations.h5 (one group per 5-day window: y = unperturbed obs, obs_pert_mean, hx_mean, hx_var
(forecast H X over the members, ddof 1), R_diag, hxa_mean, inc_<compartment>; all mm). Per unit and window:
    d = y - hx_mean,  nis = d^2 / (hx_var + R_diag)                   (consistent filter: E[nis] = 1)
and, for comparison, the innovation the update used (perturbed obs): d_u = obs_pert_mean - hx_mean.
Per run: mean nis over units x windows (and its 95 % block-bootstrap CI over windows), per-unit mean nis (median, range),
window-wise chi2 test: sum_units nis ~ chi2(m) (m = number of units, diagonal R and an approximately diagonal HPH^T assumed),
share of windows outside the central 95 % chi2 interval; mean d (bias) and RMS d, mean spread sqrt(hx_var), sqrt(R).
Usage: python 16_innovations.py --cases jog_MAGIC_Brahmaputra_s1 [...] [--data-root /work/data_for_w3] [--out results]
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

from _common import write_csv, write_json, C


def load(fn):
    with h5py.File(fn, 'r') as f:
        keys = sorted(f.keys())
        out = {k: np.array([f[w][k][()] for w in keys]) for k in ('y', 'obs_pert_mean', 'hx_mean', 'hx_var', 'R_diag',
                                                                    'hxa_mean')}
        out['epoch'] = [f[w].attrs['epoch'] for w in keys]
    return out


def analyse(x, n_boot=1000, block=18, seed=20021):
    d = x['y'] - x['hx_mean']
    du = x['obs_pert_mean'] - x['hx_mean']
    var = x['hx_var'] + x['R_diag']
    nis = d ** 2 / var
    T, m = nis.shape
    win_sum = nis.sum(axis=1)
    lo, hi = stats.chi2.ppf([0.025, 0.975], m)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, max(T - block, 0) + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T]
    boot = nis[idx].mean(axis=(1, 2))
    unit_mean = nis.mean(axis=0)
    return dict(windows=T, units=m, nis_mean=float(nis.mean()), nis_ci95=[float(v) for v in np.percentile(boot, [2.5, 97.5])],
                nis_unit_median=float(np.median(unit_mean)), nis_unit_min=float(unit_mean.min()),
                nis_unit_max=float(unit_mean.max()), chi2_window_low=float(lo), chi2_window_high=float(hi),
                share_windows_below=float(np.mean(win_sum < lo)), share_windows_above=float(np.mean(win_sum > hi)),
                bias_d=float(d.mean()), rms_d=float(np.sqrt((d ** 2).mean())), rms_d_update=float(np.sqrt((du ** 2).mean())),
                spread_hx=float(np.sqrt(x['hx_var']).mean()), sigma_R=float(np.sqrt(x['R_diag']).mean()),
                rms_analysis_residual=float(np.sqrt(((x['y'] - x['hxa_mean']) ** 2).mean()))), nis, unit_mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--data-root', default=str(C.V1_DATA_ROOT))
    p.add_argument('--out', default=str(C.OUT_ROOT))
    a = p.parse_args()
    out = Path(a.out) / '16_innovations'
    out.mkdir(parents=True, exist_ok=True)
    rows, full = [], {}
    for case in a.cases:
        fn = Path(a.data_root) / 'w3ra' / 'cases' / case / 'DA' / 'innovations.h5'
        r, nis, unit_mean = analyse(load(fn))
        full[case] = r
        rows.append(dict(case=case, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                       if not isinstance(v, list)}, nis_ci_low=round(r['nis_ci95'][0], 4),
                         nis_ci_high=round(r['nis_ci95'][1], 4)))
        write_csv(out / ('%s_unit_nis.csv' % case), [dict(unit=k + 1, mean_nis=round(float(v), 4)) for k, v in
                                                        enumerate(unit_mean)])
    write_csv(out / 'innovations_summary.csv', rows)
    write_json(out / 'innovations_summary.json', full)
    print('written', out)


if __name__ == '__main__':
    main()
