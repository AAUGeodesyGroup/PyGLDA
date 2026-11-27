"""Step 22 - 2-member vs 30-member DA ensemble mean at the DA-unit level.

Source: w3ra/res/<case>/Res_DA_<basin><grid>.h5df, basin_k/TWS/<member> ('0'..'30'; 0 = unperturbed OL member,
not updated), 'time' = 1218 daily values (decimal years, 2003-09-01..2006-12-31, checked: step 1/365.25).
basin_0 = whole basin (checked: area-weighted mean of the units within a few mm, see basin0_check_mm).
Per case:
  * difference D = mean(members 1-2) - mean(members 1-30), daily over the whole res period: RMS and max |D| for basin_0
    and per unit (median and max over units);
  * metrics vs truth (paper definition, jog_eval.metrics; series de-meaned), paper period epochs 634..875, DA daily
    averaged over the 5-day windows: units = basin_signal <group>/HIS (Grid_1..Grid_4, subbasins_3), basin_0 =
    basin_signal_whole_basin whole/HIS; spatial (unit) mean NSE/KGE (clipped at -1) and RMSE, for member 0, the
    2-member and the 30-member mean, and the change 30 - 2.
Unit order: res basin_k = obs sub_basin_k = v5 / basin_signal row k-1 (step 21: order identity in all 360 signal
files); checked here: obs file sub_basin_k = basin_signal <group>/<mission>/EWHA_alpha_<a> row k-1, and
res GRACE_DA_<basin><grid>.h5df original/basin_k = obs sub_basin_k + constant over the paper epochs (de-meaned).
unit_match_share (nearest truth unit by centred RMS) is a weak diagnostic: DA errors are of the size of the
differences between neighbouring units.
Note: the res/ folders are not all the published April runs (1-degree: July 2025 reruns with the realistic VCM).
Outputs: results/22_member_count/{member_count.csv, summary.json}
Usage: python 22_member_count.py
"""
import time
from datetime import date, timedelta

import h5py
import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import metrics as M, sources as S

RES = C.V1_DATA_ROOT / 'w3ra' / 'res'
OBS = C.V1_DATA_ROOT / 'GRACE' / 'output'
GROUP = {'1degree': 'Grid_1', '2degree': 'Grid_2', '3degree': 'Grid_3', '4degree': 'Grid_4', '3subbasins': 'subbasins_3'}
DAY0 = date(2003, 9, 1)


def his(g):
    return g['HIS']['EWHA' if 'EWHA' in g['HIS'] else 'EWHA_HIS'][:]


def load_truth(basin, grid, mission, alpha):
    with h5py.File(C.basin_signal_file(basin), 'r') as h:
        g = h[GROUP[grid]]
        t = his(g) * 1000.0                                                # (n, 876)
        prod = g[C.MISSION_FILE[mission]]['EWHA_alpha_%s' % C.alpha_file(alpha)][:] * 1000.0
    with h5py.File(C.basin_signal_file(basin, whole=True), 'r') as h:
        tw = his(h['whole']).ravel() * 1000.0
    return t, prod, tw


def obs_check(basin, grid, mission, alpha, prod):
    """max |obs file sub_basin_k - basin_signal product row k-1| (old folder = April input)"""
    fn = OBS / (mission + '_old') / alpha / ('%s%s_signal.hdf5' % (basin, grid))
    with h5py.File(fn, 'r') as h:
        keys = sorted([k for k in h if k.startswith('sub_basin_') and k != 'sub_basin_area'], key=lambda s: int(s[10:]))
        x = np.stack([h[k][:] for k in keys], 0)
        area = h['sub_basin_area'][:]
        te = [date.fromisoformat(v.decode() if isinstance(v, bytes) else str(v)) for v in h['time_epoch'][:]]
    starts = S.epoch_windows()[0]
    mid = {s + timedelta(days=2): i for i, s in enumerate(starts)}
    ep = [mid[d] for d in te]
    return float(np.abs(x - prod[:, ep]).max()), area


def five_day(x, days, epochs):
    """x (n_days, units) daily from DAY0 -> (len(epochs), units) means over each epoch window"""
    daily = {DAY0 + timedelta(days=i): x[i] for i in range(x.shape[0])}
    y, missing = S.five_day_means(daily, epochs)
    assert not missing, missing[:3]
    return y


def main():
    t0 = time.time()
    args = parser(__doc__).parse_args()
    out = outdir(args, '22_member_count')
    epochs = np.arange(*C.PAPER_EPOCHS)
    rows = []
    for basin in args.basins:
        for mission in C.MISSIONS:
            for alpha in C.ALPHAS:
                for grid in C.GRIDS:
                    case = C.case_name(mission, alpha, grid, basin)
                    fn = RES / case / ('Res_DA_%s%s.h5df' % (basin, grid))
                    row = {'case': case, 'basin': basin, 'mission': mission, 'alpha': alpha, 'grid': grid}
                    with h5py.File(fn, 'r') as h:
                        t = h['time'][:]
                        n = len([k for k in h if k.startswith('basin_')]) - 1
                        X = np.stack([np.stack([h['basin_%d/TWS/%d' % (k, m)][:] for m in range(31)], 1)
                                      for k in range(0, n + 1)], 2)        # (days, member, unit 0..n)
                    assert np.allclose(np.diff(t), 1 / 365.25) and len(t) == 1218, (case, len(t))
                    row['res_mtime'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(fn.stat().st_mtime))
                    row['n_units'] = n
                    truth, prod, tw = load_truth(basin, grid, mission, alpha)
                    assert truth.shape[0] == n, (case, truth.shape, n)
                    row['obs_vs_basin_signal_maxdiff_mm'], area = obs_check(basin, grid, mission, alpha, prod)
                    # unit order of the DA: GRACE_DA original/basin_k = obs sub_basin_k + constant (paper epochs)
                    with h5py.File(RES / case / ('GRACE_DA_%s%s.h5df' % (basin, grid)), 'r') as g:
                        go = np.stack([g['original/basin_%d' % k][:] for k in range(1, n + 1)], 0)
                    po = prod[:, epochs]
                    row['gda_original_vs_obs_demeaned_maxdiff_mm'] = float(np.abs((go - go.mean(1, keepdims=True)) -
                                                                              (po - po.mean(1, keepdims=True))).max())
                    m2, m30, m0 = X[:, 1:3].mean(1), X[:, 1:31].mean(1), X[:, 0]
                    row['basin0_check_mm'] = float(np.abs(m30[:, 0] - (m30[:, 1:] @ area) / area.sum()).max())
                    D = m2 - m30
                    rms = np.sqrt((D ** 2).mean(0))
                    mx = np.abs(D).max(0)
                    row.update({'b0_rms_diff_mm': rms[0], 'b0_max_diff_mm': mx[0],
                                'units_rms_diff_median_mm': float(np.median(rms[1:])), 'units_rms_diff_max_mm': float(rms[1:].max()),
                                'units_max_diff_mm': float(mx[1:].max()),
                                'b0_spread30_mean_mm': float(X[:, 1:31, 0].std(1).mean())})
                    T = truth[:, epochs].T
                    Tw = tw[epochs][:, None]
                    for name, s in (('m0', m0), ('2m', m2), ('30m', m30)):
                        y = five_day(s, len(t), epochs)
                        mu = M.all_metrics(T, y[:, 1:], 'paper')
                        mb = M.all_metrics(Tw, y[:, :1], 'paper')
                        for k in ('nse', 'kge', 'rmse'):
                            row['%s_units_%s' % (k, name)] = M.spatial_mean(mu[k], k)
                            row['%s_b0_%s' % (k, name)] = float(mb[k][0])
                        if name == '30m':
                            # weak diagnostic only: truth unit nearest (centred RMS) to DA unit k is unit k
                            a, b = M.demean(T), M.demean(y[:, 1:])
                            dist = np.sqrt(((a[:, :, None] - b[:, None, :]) ** 2).mean(0))       # (truth j, DA k)
                            row['unit_match_share'] = float(np.mean(dist.argmin(0) == np.arange(n)))
                    for k in ('nse', 'kge', 'rmse'):
                        row['%s_units_change' % k] = row['%s_units_30m' % k] - row['%s_units_2m' % k]
                        row['%s_b0_change' % k] = row['%s_b0_30m' % k] - row['%s_b0_2m' % k]
                    rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
                print('done', basin, mission, round(time.time() - t0, 1), 's', flush=True)
    write_csv(out / 'member_count.csv', rows)
    summ = {'n_cases': len(rows), 'max_obs_vs_basin_signal_mm': max(r['obs_vs_basin_signal_maxdiff_mm'] for r in rows),
            'max_basin0_check_mm': max(r['basin0_check_mm'] for r in rows),
            'max_gda_original_vs_obs_demeaned_mm': max(r['gda_original_vs_obs_demeaned_maxdiff_mm'] for r in rows),
            'min_unit_match_share_weak': min(r['unit_match_share'] for r in rows)}
    for grid in C.GRIDS:
        for basin in args.basins:
            s = [r for r in rows if r['grid'] == grid and r['basin'] == basin]
            summ['%s_%s' % (basin, grid)] = {
                'res_dates': sorted({r['res_mtime'][:10] for r in s}),
                'b0_rms_diff_mm_median': float(np.median([r['b0_rms_diff_mm'] for r in s])),
                'b0_rms_diff_mm_max': float(np.max([r['b0_rms_diff_mm'] for r in s])),
                'units_rms_diff_median_mm': float(np.median([r['units_rms_diff_median_mm'] for r in s])),
                'units_max_diff_mm': float(np.max([r['units_max_diff_mm'] for r in s])),
                **{'%s_units_change_%s' % (k, st): float(f([r['%s_units_change' % k] for r in s]))
                   for k in ('nse', 'kge', 'rmse') for st, f in (('median', np.median), ('min', np.min), ('max', np.max))}}
    summ['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'summary.json', summ)
    print('written', out, summ['run_time_s'], 's')


if __name__ == '__main__':
    main()
