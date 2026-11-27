"""Step 13 - spectra of the observation errors with a red-noise test (R3.5 a).

r = obs - truth over 1995-01-01..2006-12-31 (all 876 ESM2.0 epochs, treated as uniform 5-day steps; the single
4-day epoch 1995-12-27 is ignored for the time axis), alpha 1e18, three missions:
  obs   = <ESM2_ROOT>/<basin>/gravity_signal_cov/<mission>_1e+18_<basin>_v5.hdf5  Grid_1/<mission>/EWH_alpha_1e+18
  truth = <basin>/truth/basin_signal.hdf5 Grid_1/HIS (same Grid_1 units, m -> mm); both de-meaned over 876 epochs.
Series: basin mean (cos-lat area weights of the Grid_1 units, as step 08) and every Grid_1 unit.
Spectrum: Welch, Hann window, segment 292 epochs (1460 d), 50 % overlap (5 segments covering all 876 epochs),
linear detrend per segment; Nyquist bin doubled (as periodic.spectrum).
Background: AR(1) with lag-1 autocorrelation rho of the linearly detrended series,
    B(f) = (1 - rho^2) / (1 - 2 rho cos(2 pi f dt) + rho^2), scaled to the mean Welch power over f > 0;
95 % (99 %) level = B * chi2_{0.95}(nu) / nu with the equivalent dof nu of Welch/Hann/50 % overlap.
Peaks = local maxima above the 95 % level; amplitude = least-squares cos/sin amplitude (with constant + trend),
maximised over trial periods within +-half a Welch bin of the peak (refined period reported). Per unit: share of units above their own 95 % level at each bin.
Reference: the review-26 fits (results/08_periodic_errors summary.csv, stability.csv) for 10/15/30 d.
Note: about 5 % of the bins exceed the 95 % level by chance (~10 of ~146 bins); the list is not corrected for that.
Usage: python 13_residual_spectra.py
"""
import csv
import importlib
import time

import h5py
import numpy as np
from scipy import signal, stats

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import metrics as M

STEP = 5.0
SEG = 292
P08 = importlib.import_module('08_periodic_errors')


def welch(x):
    f, P = signal.welch(x, fs=1 / STEP, window='hann', nperseg=SEG, noverlap=SEG // 2, detrend='linear', axis=0)
    P[-1] *= 2
    return f, P


def welch_dof(n):
    K = (n - SEG) // (SEG // 2) + 1
    w = signal.get_window('hann', SEG)
    # squared correlation of Hann segments shifted by SEG/2 (Percival & Walden 1993, Sect. 6.17): 0.167^2
    rho = np.sum(w[:SEG - SEG // 2] * w[SEG // 2:]) ** 2 / np.sum(w ** 2) ** 2
    nu = 2 * K / (1 + 2 * sum((1 - m / K) * (rho if m == 1 else 0.0) for m in range(1, K)))
    return float(nu), int(K), float(rho)


def ar1_background(x, f, P):
    t = np.arange(x.shape[0])
    d = x - np.polyval(np.polyfit(t, x, 1), t)
    rho = np.sum(d[1:] * d[:-1]) / np.sum(d * d)
    B = (1 - rho ** 2) / (1 - 2 * rho * np.cos(2 * np.pi * f * STEP) + rho ** 2)
    sel = f > 0
    B = B * P[sel].mean() / B[sel].mean()
    return B, float(rho)


def amp_at(x, period):
    t = STEP * np.arange(x.shape[0])
    A = np.stack([np.ones_like(t), t - t.mean(), np.cos(2 * np.pi * t / period), np.sin(2 * np.pi * t / period)], 1)
    if np.isclose(period, 2 * STEP):
        A = A[:, :3]
    c = np.linalg.lstsq(A, x, rcond=None)[0]
    return float(np.hypot(c[2], c[3] if len(c) > 3 else 0.0))


def refine(x, f0, df=1.0 / (SEG * STEP)):
    """period (d) and LS amplitude maximising the amplitude within +-df/2 of the Welch bin f0 (41 trial frequencies)"""
    fr = np.linspace(max(f0 - df / 2, 1e-6), min(f0 + df / 2, 0.5 / STEP), 41)
    a = [amp_at(x, 1 / q) for q in fr]
    i = int(np.argmax(a))
    return float(1 / fr[i]), float(a[i])


def read_csv(fn):
    with open(fn) as f:
        return list(csv.DictReader(f))


def main():
    t0 = time.time()
    args = parser(__doc__).parse_args()
    out = outdir(args, '13_residual_spectra')
    nu, K, rho_w = welch_dof(876)
    q95, q99 = stats.chi2.ppf(0.95, nu) / nu, stats.chi2.ppf(0.99, nu) / nu
    meta = {'welch': {'segment_epochs': SEG, 'segments': K, 'overlap_corr_squared': rho_w, 'dof': nu, 'factor95': q95,
                      'factor99': q99}}
    peak_rows, unit_rows, bg_rows = [], [], []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref = C.OUT_ROOT / '08_periodic_errors'
    s08 = read_csv(ref / 'summary.csv') if (ref / 'summary.csv').exists() else []
    st08 = read_csv(ref / 'stability.csv') if (ref / 'stability.csv').exists() else []
    for basin in args.basins:
        truth, _, w, starts, his = P08.load_basin(basin)
        meta[basin] = {'truth': his, 'n_units': truth.shape[1], 'period': '%s..%s' % (starts[0], starts[-1])}
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
        for im, m in enumerate(C.MISSIONS):
            mf = C.MISSION_FILE[m]
            fn = C.ESM2_ROOT / basin / 'gravity_signal_cov' / ('%s_1e+18_%s_v5.hdf5' % (mf, basin))
            with h5py.File(fn, 'r') as h:
                obs = h['Grid_1/%s/EWH_alpha_1e+18' % mf][:].T * 1000.0
            assert obs.shape == truth.shape, (obs.shape, truth.shape)
            r = M.demean(obs) - M.demean(truth)
            rb = r @ w
            f, Pb = welch(rb)
            B, rho = ar1_background(rb, f, Pb)
            sel = np.flatnonzero(f > 0)
            ratio = Pb / B
            meta[basin][m] = {'obs': str(fn), 'rho_basin': rho, 'rms_basin_mm': float(rb.std()),
                              'rms_units_median_mm': float(np.median(r.std(0)))}
            # per unit
            Pu = welch(r)[1]
            exceed = np.zeros_like(Pu, dtype=bool)
            rhos = []
            for j in range(r.shape[1]):
                Bj, rj = ar1_background(r[:, j], f, Pu[:, j])
                exceed[:, j] = Pu[:, j] > q95 * Bj
                rhos.append(rj)
            share_units = exceed.mean(1)
            meta[basin][m]['rho_units_median'] = float(np.median(rhos))
            # peaks of the basin mean: local maxima (Nyquist edge included) above 95 %
            for i in sel:
                left = Pb[i - 1] if i - 1 >= 1 else -np.inf
                right = Pb[i + 1] if i + 1 < len(f) else -np.inf
                if Pb[i] >= left and Pb[i] >= right and ratio[i] > q95:
                    per = 1 / f[i]
                    peak_rows.append({'basin': basin, 'mission': m, 'series': 'basin_mean', 'period_d': round(per, 1),
                                      'power_mm2d': round(float(Pb[i]), 2), 'ratio_to_AR1': round(float(ratio[i]), 2),
                                      'sig99': bool(ratio[i] > q99), 'refined_period_d': round(refine(rb, f[i])[0], 2),
                                      'amp_ls_mm': round(refine(rb, f[i])[1], 2),
                                      'share_units_sig95': round(float(share_units[i]), 3)})
            # unit-level: bins where most units are significant
            for i in sel:
                if share_units[i] >= 0.5:
                    per = 1 / f[i]
                    unit_rows.append({'basin': basin, 'mission': m, 'period_d': round(per, 1),
                                      'share_units_sig95': round(float(share_units[i]), 3),
                                      'amp_ls_units_median_mm': round(float(np.median([refine(r[:, j], f[i])[1] for j in range(r.shape[1])])), 2)})
            for P in (10.0, 15.0, 30.0, 365.25 / 2, 365.25):
                i = int(np.argmin(np.abs(f - 1 / P)))
                row = {'basin': basin, 'mission': m, 'target_period_d': round(P, 1), 'bin_period_d': round(1 / f[i], 1),
                       'ratio_basin_to_AR1': round(float(ratio[i]), 2), 'sig95_basin': bool(ratio[i] > q95),
                       'share_units_sig95': round(float(share_units[i]), 3), 'amp_ls_basin_mm': round(amp_at(rb, P), 2)}
                for s in s08:
                    if s['basin'] == basin and s['mission'] == m and P in (10.0, 15.0, 30.0):
                        row['amp08_basin_2002_2006_mm'] = round(float(s['amp%g_basinmean' % P]), 2)
                for s in st08:
                    if s['basin'] == basin and s['mission'] == m and P in (10.0, 15.0, 30.0) and float(s['period_d']) == P:
                        row['amp08_basin_1996_2002_mm'] = round(float(s['amp_v5_basinmean']), 2)
                bg_rows.append(row)
            ax = axs[im]
            pp = 1 / f[sel]
            ax.loglog(pp, Pb[sel], color='#1f5fa8', lw=1.4, label='obs - truth, basin mean (Welch)')
            ax.loglog(pp, B[sel], color='0.4', lw=1, label='AR(1) background (rho=%.2f)' % rho)
            ax.loglog(pp, q95 * B[sel], color='0.4', lw=1, ls='--', label='95 % level')
            pk = [p for p in peak_rows if p['basin'] == basin and p['mission'] == m]
            for p in pk:
                ax.annotate('%.0f d' % p['period_d'], (p['period_d'], p['power_mm2d']), textcoords='offset points',
                            xytext=(0, 6), ha='center', fontsize=8)
            ax.set_xlim(1500, 9.5)
            ax.set_xlabel('period (days)')
            ax.set_title('%s %s (alpha 1e18), 1995-2006' % (basin, m.replace('GRACEC', 'GRACE-C')), fontsize=10)
            ax.grid(alpha=0.3, which='both')
            if im == 0:
                ax.set_ylabel('power (mm$^2$ d)')
                ax.legend(fontsize=8, loc='lower left')
        fig.tight_layout()
        fig.savefig(out / ('residual_spectra_%s.png' % basin), dpi=130)
        plt.close(fig)
    write_csv(out / 'peaks_basin_mean.csv', peak_rows)
    write_csv(out / 'units_significant_bins.csv', unit_rows)
    write_csv(out / 'target_periods.csv', bg_rows)
    meta['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'meta.json', meta)
    print('written', out, meta['run_time_s'], 's')


if __name__ == '__main__':
    main()
