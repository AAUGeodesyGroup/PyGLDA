"""Step 8 - periodic (10/15/30-day) errors of the 5-day synthetic observations.

r = obs - truth per 1-degree unit (both de-meaned) over the DA input epochs 2002-09..2006-12 (ESM2.0 epochs 561..875).
obs   = the products the DA assimilated: GRACE/output/<mission>/1e18/<basin>1degree_signal.hdf5 (sub_basin_<k>, mm)
truth = <ESM2_ROOT>/<basin>/truth/basin_signal.hdf5 Grid_1/HIS (same Grid_1 units, mapping checked against the
        products Grid_1/<mission>/EWHA_alpha_1e+18 of the same file).
1. periodogram of r (Hann, linear detrend, zero-padded x8): peaks 10..60 d (mean of unit spectra, basin mean, units);
2. least squares per unit (periodic.fit): const, trend, annual, semi-annual + cos/sin 15 d, 30 d + (-1)^k (10 d);
   p = 10/15/30-d part; share of var(r) explained by p and RMS before/after, full and after 60-d high-pass;
3. stability: same fit on 1995-12-31..2002-08 (epochs 73..560, uniform 5-day steps) from the v5 files;
4. self-check: refit on r - p.
No cleaned products are written.
Usage: python 08_periodic_errors.py [--periods 15 30]
"""
import time
from datetime import date, timedelta

import h5py
import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import metrics as M, stats as ST, periodic as PE

OBS_ROOT = C.V1_DATA_ROOT / 'GRACE' / 'output'
DA_EPOCHS = (561, 876)        # 2002-09-05 .. 2006-12-27 (obs time_epoch = start + 2 d)
V5_EPOCHS = (73, 561)         # 1995-12-31 .. 2002-09-04 (epoch 72 is the 4-day epoch, left out)


def _d(x):
    return date.fromisoformat(x.decode() if isinstance(x, bytes) else str(x))


def load_basin(basin):
    with h5py.File(C.basin_signal_file(basin), 'r') as h:
        g = h['Grid_1']
        his = 'EWHA' if 'EWHA' in g['HIS'] else 'EWHA_HIS'
        truth = g['HIS'][his][:].T * 1000.0                             # (876, n) mm
        prods = {m: g[C.MISSION_FILE[m]]['EWHA_alpha_1e+18'][:].T * 1000.0 for m in C.MISSIONS}
        mask, lat = g['mask'][:], g['mask_lat'][:]
        starts = np.array([_d(x) for x in h['dates'][0]])
    w = (mask * np.cos(np.radians(lat))[None, :, None]).sum((1, 2))   # unit area weight (0.5-deg cells, cos lat)
    return truth, prods, w / w.sum(), starts, 'Grid_1/HIS/' + his


def load_obs(basin, mission, prods, starts):
    fn = OBS_ROOT / mission / '1e18' / ('%s1degree_signal.hdf5' % basin)
    with h5py.File(fn, 'r') as h:
        te = [_d(x) for x in h['time_epoch'][:]]
        keys = sorted([k for k in h if k.startswith('sub_basin_') and k != 'sub_basin_area'], key=lambda s: int(s[10:]))
        obs = np.stack([h[k][:] for k in keys], 1)                      # held in mm (checked below)
    mid = {s + timedelta(days=2): i for i, s in enumerate(starts)}
    ep = np.array([mid[t] for t in te])
    assert np.array_equal(ep, np.arange(*DA_EPOCHS)), 'obs epochs differ from 561..875'
    ref = prods[ep]
    scale = np.median(np.abs(obs)) / np.median(np.abs(ref))
    assert abs(scale - 1) < 0.01, 'obs file not in mm (median ratio to Grid_1 product %.4g)' % scale
    mapping = []
    for j, k in enumerate(keys):
        d = np.abs(ref - obs[:, [j]]).max(0)
        mapping.append({'basin': basin, 'mission': mission, 'key': k, 'grid1_index': int(np.argmin(d)),
                        'max_abs_diff_mm': float(d.min()), 'second_best_mm': float(np.sort(d)[1])})
    idx = [m['grid1_index'] for m in mapping]
    assert len(set(idx)) == len(idx) == ref.shape[1], 'no 1:1 mapping'
    out = np.full_like(obs, np.nan)
    out[:, idx] = obs                                                   # reorder to Grid_1 index
    return out, mapping, str(fn)


def stats_row(x):
    x = np.asarray(x, float)
    return float(np.median(x)), float(np.min(x)), float(np.max(x))


def evaluate(r, k, periods):
    """fit + shares/RMS (per column of r)"""
    F = PE.fit(r, k, periods)
    p = F['p']
    rf, pf = ST.butter_band(r, low_days=60), ST.butter_band(p, low_days=60)
    F['share_full'] = 1 - np.var(r - p, 0) / np.var(r, 0)
    F['share_lt60'] = 1 - np.var(rf - pf, 0) / np.var(rf, 0)
    F['rms_full'], F['rms_full_after'] = np.std(r, 0), np.std(r - p, 0)
    F['rms_lt60'], F['rms_lt60_after'] = np.std(rf, 0), np.std(rf - pf, 0)
    F['self_amp'] = max(float(np.max(a)) for a in PE.fit(r - p, k, periods)['amp'].values())
    return F


def main():
    t0 = time.time()
    ap = parser(__doc__)
    ap.add_argument('--periods', nargs='+', type=float, default=[15.0, 30.0], help='cos/sin periods (d) besides 10 d')
    args = ap.parse_args()
    out = outdir(args, '08_periodic_errors')
    periods = tuple(args.periods)
    per_all = (10.0,) + periods
    k_da = np.arange(*DA_EPOCHS)
    k_v5 = np.arange(*V5_EPOCHS)
    maps, peak_rows, truth_rows, unit_rows, summ, stab, meta = [], [], [], [], [], [], {'periods': per_all}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for basin in args.basins:
        truth_all, prods, w, starts, his = load_basin(basin)
        n = truth_all.shape[1]
        meta[basin] = {'truth': str(C.basin_signal_file(basin)) + ':' + his, 'n_units': n,
                       'da_epochs': [str(starts[DA_EPOCHS[0]]), str(starts[DA_EPOCHS[1] - 1] + timedelta(days=4))],
                       'v5_epochs': [str(starts[V5_EPOCHS[0]]), str(starts[V5_EPOCHS[1] - 1] + timedelta(days=4))]}
        truth = M.demean(truth_all[k_da])
        # truth spectrum: power at 10/15/30 d vs local background
        ft, Pt = PE.spectrum(truth)
        Ptm = Pt.mean(1)
        for P in (10.0, 15.0, 30.0):
            pw, ratio = PE.power_at(ft, Ptm, P)
            truth_rows.append({'basin': basin, 'period_d': P, 'power_unitmean': pw, 'ratio_to_local_bg': round(ratio, 3)})
        for i, (pp, pw, rt) in enumerate(PE.peaks(ft, Ptm)):
            truth_rows.append({'basin': basin, 'period_d': 'peak%d=%.2f' % (i + 1, pp), 'power_unitmean': pw,
                               'ratio_to_local_bg': round(rt, 3)})
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        for im, m in enumerate(C.MISSIONS):
            obs, mp, fn = load_obs(basin, m, prods[m], starts)
            maps += mp
            meta[basin]['obs_' + m] = fn
            r = M.demean(obs) - truth
            rb = r @ w                                                  # basin-mean residual (area weights)
            # periodogram + peaks
            f, P = PE.spectrum(r)
            fb, Pb = PE.spectrum(rb)
            series = [('unit_mean', P.mean(1)), ('basin_mean', Pb)]
            big = np.argsort(r.std(0))[::-1][:2]
            series += [('unit%d' % j, P[:, j]) for j in (*big, int(np.argmin(r.std(0))))]
            for name, spec in series:
                for i, (pp, pw, rt) in enumerate(PE.peaks(f, spec)):
                    peak_rows.append({'basin': basin, 'mission': m, 'series': name, 'rank': i + 1,
                                      'period_d': round(pp, 2), 'power': pw, 'ratio_to_band_median': round(rt, 2)})
            # fits
            F = evaluate(r, k_da, periods)
            Fb = evaluate(rb[:, None], k_da, periods)
            for j in range(n):
                row = {'basin': basin, 'mission': m, 'unit': j, 'weight': float(w[j])}
                for Pp in per_all:
                    row['amp%g_mm' % Pp] = float(F['amp'][Pp][j])
                    row['phase%g_deg' % Pp] = float(F['phase'][Pp][j])
                for key in ('share_full', 'share_lt60', 'rms_full', 'rms_full_after', 'rms_lt60', 'rms_lt60_after'):
                    row[key] = float(F[key][j])
                unit_rows.append(row)
            srow = {'basin': basin, 'mission': m, 'n_units': n}
            for Pp in per_all:
                med, lo, hi = stats_row(F['amp'][Pp])
                srow.update({'amp%g_med' % Pp: med, 'amp%g_min' % Pp: lo, 'amp%g_max' % Pp: hi,
                             'amp%g_basinmean' % Pp: float(Fb['amp'][Pp][0])})
            for key in ('share_full', 'share_lt60', 'rms_full', 'rms_full_after', 'rms_lt60', 'rms_lt60_after'):
                med, lo, hi = stats_row(F[key])
                srow.update({key + '_med': med, key + '_min': lo, key + '_max': hi, key + '_basinmean': float(Fb[key][0])})
            srow['selfcheck_max_amp_mm'] = max(F['self_amp'], Fb['self_amp'])
            summ.append(srow)
            # spectrum before/after
            fa, Pa = PE.spectrum(r - F['p'])
            sel = f > 0
            ax = axs[im]
            ax.loglog(1 / f[sel], Ptm[sel], color='0.6', lw=1, label='truth (unit mean)')
            ax.loglog(1 / f[sel], P.mean(1)[sel], color='#1f5fa8', lw=1.3, label='r before (unit mean)')
            ax.loglog(1 / fa[sel], Pa.mean(1)[sel], color='#d0632b', lw=1.3, label='r - p (unit mean)')
            for Pp in per_all:
                ax.axvline(Pp, color='k', ls=':', lw=0.7)
            ax.axvline(60, color='k', ls='--', lw=0.7)
            ax.set_xlim(1000, 9.8)
            ax.set_title('%s %s' % (basin, m))
            ax.set_xlabel('period (days)')
            if im == 0:
                ax.set_ylabel('power (mm$^2$ d)')
                ax.legend(fontsize=8, loc='lower left')
            # stability (v5 1995-12-31..2002-08)
            mf = C.MISSION_FILE[m]
            v5fn = C.ESM2_ROOT / basin / 'gravity_signal_cov' / ('%s_1e+18_%s_v5.hdf5' % (mf, basin))
            with h5py.File(v5fn, 'r') as h:
                same = np.array_equal(h['Grid_1/mask'][:], h5py.File(C.basin_signal_file(basin), 'r')['Grid_1/mask'][:])
                v5 = h['Grid_1/%s/EWH_alpha_1e+18' % mf][:].T * 1000.0
            v5_vs_bs = float(np.abs(v5 - prods[m]).max())
            meta[basin]['v5_' + m] = {'file': str(v5fn), 'same_grid1_mask': bool(same), 'max_abs_diff_vs_basin_signal_mm': v5_vs_bs}
            if not same:
                continue
            r5 = M.demean(v5[k_v5]) - M.demean(truth_all[k_v5])
            F5 = PE.fit(r5, k_v5, periods)
            F5b = PE.fit((r5 @ w)[:, None], k_v5, periods)
            for Pp in per_all:
                a5, a2 = F5['amp'][Pp], F['amp'][Pp]
                dph = (F5['phase'][Pp] - F['phase'][Pp] + 180) % 360 - 180
                stab.append({'basin': basin, 'mission': m, 'period_d': Pp,
                             'amp_v5_med': float(np.median(a5)), 'amp_da_med': float(np.median(a2)),
                             'amp_ratio_med': float(np.median(a5 / a2)), 'amp_corr_units': float(np.corrcoef(a5, a2)[0, 1]),
                             'abs_dphase_med_deg': float(np.median(np.abs(dph))),
                             'frac_units_dphase_lt30': float(np.mean(np.abs(dph) < 30)),
                             'amp_v5_basinmean': float(F5b['amp'][Pp][0]), 'amp_da_basinmean': float(Fb['amp'][Pp][0]),
                             'phase_v5_basinmean': float(F5b['phase'][Pp][0]), 'phase_da_basinmean': float(Fb['phase'][Pp][0])})
        fig.tight_layout()
        fig.savefig(out / ('spectra_%s.png' % basin), dpi=130)
        plt.close(fig)

    write_csv(out / 'unit_mapping.csv', maps)
    write_csv(out / 'peaks.csv', peak_rows)
    write_csv(out / 'truth_spectrum.csv', truth_rows)
    write_csv(out / 'unit_fit.csv', unit_rows)
    write_csv(out / 'summary.csv', summ)
    write_csv(out / 'stability.csv', stab)
    meta['max_abs_mapping_diff_mm'] = max(m['max_abs_diff_mm'] for m in maps)
    meta['min_second_best_mm'] = min(m['second_best_mm'] for m in maps)
    meta['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'meta.json', meta)
    print('written', out, meta['run_time_s'], 's')


if __name__ == '__main__':
    main()
