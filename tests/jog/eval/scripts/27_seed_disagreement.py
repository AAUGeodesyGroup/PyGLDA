"""Step 27 - where and when do the seeds of one nominal baseline disagree? (Danube GRACE-C by default)

For mission M in basin B and the available seeds (jog_<M>_<B>_s<k>_nominal, complete products only):
  * per 5-day epoch (634..875): spatial RMS over the 0.5 deg cells of DA_si - DA_sj (all seed pairs) and of DA - truth
    per seed; the epochs with the largest seed differences;
  * per 0.5 deg cell: NSE per seed, range, and the share of cells with range > 0.1 / 0.3;
  * innovations (cases/<case>/DA/innovations.h5): per window the mean NIS over units per seed and per unit the mean NIS
    per seed; correlation of the unit NIS between seeds; units whose NIS differs most.
Outputs: results/27_seed_disagreement/{epoch_series_<B>_<M>.csv, units_nis_<B>_<M>.csv, windows_nis_<B>_<M>.csv,
         seed_disagreement_<B>_<M>.png, summary.json}
Usage: python 27_seed_disagreement.py [--basin Danube] [--mission GRACEC]
"""
import importlib
import itertools
import time

import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import baselines as B, evaluate as E, metrics as M, sources as S

I16 = importlib.import_module('16_innovations')
T24 = importlib.import_module('24_tables')


def main():
    t0 = time.time()
    ap = parser(__doc__)
    ap.add_argument('--basin', default='Danube')
    ap.add_argument('--mission', default='GRACEC')
    args = ap.parse_args()
    out = outdir(args, '27_seed_disagreement')
    b, m = args.basin, args.mission
    epochs = np.arange(*C.PAPER_EPOCHS)
    seeds, names = [], {}
    for k in (1, 2, 3):
        f = B.folders(b, k, missions=(m,))
        if T24.ready(f[m][1]):
            seeds.append(k)
            names['s%d' % k] = f[m]
            names.setdefault('model', f['model'])
    data = E.load_basin(b, epochs, folders=names)
    T = data['truth']
    ok = np.isfinite(T).all(0)
    dates = S.epoch_dates()[epochs]
    ser = [{'epoch': int(e), 'date': str(d)} for e, d in zip(epochs, dates)]
    for k in seeds:
        err = M.demean(data['sims']['s%d' % k]) - M.demean(T)
        for i, r in enumerate(np.sqrt(np.nanmean(err[:, ok] ** 2, 1))):
            ser[i]['rms_err_s%d' % k] = round(float(r), 3)
    pair_rms = {}
    for i, j in itertools.combinations(seeds, 2):
        d = data['sims']['s%d' % i] - data['sims']['s%d' % j]
        d = d - np.nanmean(d, 0)
        r = np.sqrt(np.nanmean(d[:, ok] ** 2, 1))
        pair_rms['s%d-s%d' % (i, j)] = r
        for e, v in enumerate(r):
            ser[e]['rms_diff_s%d_s%d' % (i, j)] = round(float(v), 3)
    write_csv(out / ('epoch_series_%s_%s.csv' % (b, m)), ser)
    nse = {k: M.all_metrics(T, data['sims']['s%d' % k], 'paper')['nse'] for k in seeds}
    rng = np.nanmax([np.clip(v, -1, None) for v in nse.values()], 0) - np.nanmin([np.clip(v, -1, None) for v in nse.values()], 0)
    summ = {'basin': b, 'mission': m, 'seeds': seeds, 'cells': int(ok.sum()),
            'nse_spatial_mean': {k: M.spatial_mean(v, 'nse') for k, v in nse.items()},
            'cells_range_gt_0.1': int((rng[ok] > 0.1).sum()), 'cells_range_gt_0.3': int((rng[ok] > 0.3).sum())}
    for p, r in pair_rms.items():
        top = np.argsort(r)[::-1][:8]
        summ['pair_' + p] = {'median_rms_mm': float(np.median(r)), 'max_rms_mm': float(r.max()),
                             'top_epochs': [str(dates[t]) for t in top],
                             'share_of_sq_diff_in_top10pct_epochs': float(np.sort(r ** 2)[::-1][:len(r) // 10].sum() / (r ** 2).sum())}
    # sensitivity: NSE without the 2 epochs of largest seed difference (any pair)
    allr = np.max(np.stack(list(pair_rms.values())), 0) if pair_rms else np.zeros(len(epochs))
    drop = np.argsort(allr)[::-1][:2]
    keep = np.setdiff1d(np.arange(len(epochs)), drop)
    summ['dropped_epochs'] = [str(dates[d]) for d in drop]
    summ['nse_spatial_mean_without_2_epochs'] = {k: M.spatial_mean(M.all_metrics(T[keep], data['sims']['s%d' % k][keep], 'paper')['nse'], 'nse')
                                                 for k in seeds}
    summ['nse_model_with_without'] = [M.spatial_mean(M.all_metrics(T, data['sims']['model'], 'paper')['nse'], 'nse'),
                                      M.spatial_mean(M.all_metrics(T[keep], data['sims']['model'][keep], 'paper')['nse'], 'nse')]
    # innovations
    nis_w, nis_u = {}, {}
    for k in seeds:
        x = I16.load(C.V1_DATA_ROOT / 'w3ra' / 'cases' / B.case(m, b, k) / 'DA' / 'innovations.h5')
        r, nis, unit_mean = I16.analyse(x)
        nis_w[k], nis_u[k] = nis.mean(1), unit_mean
        summ['nis_s%d' % k] = {'mean': r['nis_mean'], 'ci95': r['nis_ci95'], 'unit_median': r['nis_unit_median'],
                               'unit_max': r['nis_unit_max'], 'rms_d': r['rms_d']}
        wep = x['epoch']
    write_csv(out / ('windows_nis_%s_%s.csv' % (b, m)), [dict(window=i, epoch=str(wep[i]), **{'nis_s%d' % k: round(float(nis_w[k][i]), 3) for k in seeds})
                                                        for i in range(len(wep))])
    write_csv(out / ('units_nis_%s_%s.csv' % (b, m)), [dict(unit=u + 1, **{'nis_s%d' % k: round(float(nis_u[k][u]), 3) for k in seeds})
                                                      for u in range(len(nis_u[seeds[0]]))])
    for i, j in itertools.combinations(seeds, 2):
        du = nis_u[i] - nis_u[j]
        summ['units_s%d_s%d' % (i, j)] = {'corr_unit_nis': float(np.corrcoef(nis_u[i], nis_u[j])[0, 1]),
                                          'units_largest_abs_diff': [int(u) + 1 for u in np.argsort(np.abs(du))[::-1][:5]],
                                          'corr_window_nis': float(np.corrcoef(nis_w[i], nis_w[j])[0, 1])}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))
    cols = ['#1f5fa8', '#d0632b', '#3a9a5b']
    for c, k in zip(cols, seeds):
        axs[0].plot(dates, [s['rms_err_s%d' % k] for s in ser], color=c, lw=1, label='seed %d: RMS(DA - truth)' % k)
        axs[2].plot(np.arange(len(wep)), nis_w[k], color=c, lw=1, label='seed %d' % k)
    for c, (p, r) in zip(cols, pair_rms.items()):
        axs[1].plot(dates, r, color=c, lw=1, label=p)
    axs[0].set_ylabel('mm'); axs[1].set_ylabel('mm'); axs[2].set_ylabel('mean NIS over units')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('DA window (5-day, from %s)' % wep[0])
    axs[0].set_title('%s %s nominal 1e18/1deg: spatial RMS over 0.5 deg cells per epoch' % (b, m))
    axs[1].set_title('seed-to-seed DA difference (de-meaned), spatial RMS')
    axs[2].set_title('innovations: mean NIS per window (log)')
    for a in axs:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / ('seed_disagreement_%s_%s.png' % (b, m)), dpi=120)
    summ['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'summary.json', summ)
    print('written', out, summ['run_time_s'], 's')


if __name__ == '__main__':
    main()
