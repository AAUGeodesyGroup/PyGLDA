"""Step 7 - recompute every published grid metric with the standard definitions (Appendix B).

The published numbers (ESAsing/paper_RSE: metrics.h5, Figures 7-11, Table 1, Section 5.2, Table 2) were computed with
NSE_2D(truth, sim) / KGE_2D(truth, sim), i.e. with reference and evaluated series swapped (NSE normalised by the
evaluated series, KGE ratio sigma_truth / sigma_sim). This step writes, for both definitions:

  metrics_<definition>.h5   same layout as metrics.h5: group <basin>_<MISSION>_F<alpha>_<grid>_<hpf> with
                            nse/kge/rmse _model/_DA maps (0.5 deg), hpf 1 = 60-day high-pass (Figures 10-11)
  summary_<definition>.json Table 1 (fixed alpha 1e18 / 1 deg, and per-metric best over the 30 configurations),
                            abstract numbers, Section 5.2 numbers (MAGIC 1e17 / 1 deg, thresholds 0.5 / 0.9),
                            Figure 9 statistics (share of cells improved over GRACE-C, max RMSE reduction, share > 10 mm),
                            Figure 8 / Appendix D series (spatial means of all 30 configurations per mission)
  check_published.json      max |published_code - metrics.h5| per group (must be ~0)
  comparison.csv            published vs standard, side by side, for every number in the summary

Table 2 (realistic VCM): run again with --results-root <...>/Analysis_cov_residuals --tag cov_residual
(only the 1e18 / 1 deg cases exist there).
Usage: python 07_recompute_all.py [--metrics-h5 <published metrics.h5>] [--results-root R] [--tag T] [--best-only]
"""
import numpy as np
import h5py

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import evaluate as E, metrics as M, sources as S, stats as ST

GRID_KEY = {'1degree': 'G1', '2degree': 'G2', '3degree': 'G3', '4degree': 'G4', '3subbasins': 'Gb'}
DEFS = ('published_code', 'paper')
HPF_CUTOFF_DAYS = 60


def v1_folders(basin, alpha, grid, seed, suffix='_nominal'):
    """rerun products: DA_mean of members 1-30 per case, model = OL mean (member 0) of the shared OL; 0.1 deg masks of the
    published case of the same configuration"""
    w3 = C.V1_DATA_ROOT / 'w3ra'
    cfg = '' if (alpha, grid) == ('1e18', '1degree') else '_%s_%s' % (alpha, grid)
    f = {'model': ('OL', str(w3 / 'domains' / ('jog_%s1degree' % basin) / 'OL' / 'pub_magic1e19' / 'products' / 'OL_mean'),
                   C.case_name(C.MISSIONS[0], alpha, grid, basin))}
    for m in C.MISSIONS:
        case = 'jog_%s_%s%s_s%s%s' % (m, basin, cfg, seed, suffix)
        f[m] = ('DA', str(w3 / 'cases' / case / 'products' / 'DA_mean'), str(w3 / 'cases' / case / 'products' / 'mask' / (case + '.h5')))
    ref = 'jog_%s_%s_s%s_nominal' % (C.MISSIONS[0], basin, seed)      # the shared OL mean covers the 1-degree unit cells
    f['model'] = (f['model'][0], f['model'][1], str(w3 / 'cases' / ref / 'products' / 'mask' / (ref + '.h5')))
    return f


def maps(truth, sim, d):
    m = M.all_metrics(truth, sim, d)
    return {k: m[k] for k in ('nse', 'kge', 'rmse')}


def main():
    p = parser(__doc__)
    p.add_argument('--metrics-h5', default=None)
    p.add_argument('--results-root', default=None, help='override JOG_RESULTS_ROOT (e.g. Analysis_cov_residuals)')
    p.add_argument('--tag', default='published_outputs')
    p.add_argument('--best-only', action='store_true', help='only alpha 1e18 / 1 deg and 1e17 / 1 deg (fast)')
    p.add_argument('--no-hpf', action='store_true')
    p.add_argument('--v1', action='store_true', help='rerun products (30-member DA means, shared OL) of the v1 data layout '
                                                    'instead of the published outputs: cases/jog_<M>_<basin>[_<alpha>_<grid>]_s<seed>_nominal')
    p.add_argument('--seed', default='1')
    p.add_argument('--v1-suffix', default='_nominal', help="case suffix with --v1 ('' = realistic-VCM runs)")
    p.add_argument('--grids', nargs='+', default=None, help='only these grids (e.g. 1degree)')
    args = p.parse_args()
    if args.results_root:
        from pathlib import Path
        C.RESULTS_ROOT = Path(args.results_root)
    out = outdir(args, '07_recompute_' + args.tag)
    epochs = np.arange(*C.PAPER_EPOCHS)
    configs = [(a, g) for a in C.ALPHAS for g in C.GRIDS]
    if args.grids:
        configs = [(a, g) for a, g in configs if g in args.grids]
    if args.best_only:
        configs = [('1e18', '1degree'), ('1e17', '1degree')]
    hpfs = (0,) if args.no_hpf else (0, 1)

    h5out = {d: h5py.File(out / ('metrics_%s.h5' % d), 'w') for d in DEFS}
    means = {d: {} for d in DEFS}        # (basin, mission, alpha, grid, hpf) -> {nse_DA, ..., nse_model, ...}
    keep = {}                            # maps needed for Section 5.2 / Figure 9
    check = {}
    ref = h5py.File(args.metrics_h5, 'r') if args.metrics_h5 else None
    for basin in args.basins:
        for alpha, grid in configs:
            try:
                data = E.load_basin(basin, epochs, alpha=alpha, grid=grid, folders=v1_folders(basin, alpha, grid, args.seed, args.v1_suffix)
                                    if args.v1 else None)
            except FileNotFoundError as e:
                print('skip', basin, alpha, grid, e)
                continue
            t = data['truth']
            for hpf in hpfs:
                tt = ST.butter_band(M.demean(t), low_days=HPF_CUTOFF_DAYS) if hpf else t
                model = data['sims']['model']
                mm = ST.butter_band(M.demean(model), low_days=HPF_CUTOFF_DAYS) if hpf else model
                for mission in C.MISSIONS:
                    sim = data['sims'][mission]
                    ss = ST.butter_band(M.demean(sim), low_days=HPF_CUTOFF_DAYS) if hpf else sim
                    key = '%s_%s_F%s_%s_%d' % (basin, mission, alpha.upper(), GRID_KEY[grid], hpf)
                    for d in DEFS:
                        mo, da = maps(tt, mm, d), maps(tt, ss, d)
                        g = h5out[d].create_group(key)
                        for k in ('nse', 'kge', 'rmse'):
                            g.create_dataset(k + '_model', data=mo[k])
                            g.create_dataset(k + '_DA', data=da[k])
                        means[d][(basin, mission, alpha, grid, hpf)] = {
                            **{k + '_DA': M.spatial_mean(da[k], k) for k in ('nse', 'kge', 'rmse')},
                            **{k + '_model': M.spatial_mean(mo[k], k) for k in ('nse', 'kge', 'rmse')}}
                        if hpf == 0 and grid == '1degree' and alpha in ('1e17', '1e18'):
                            keep[(d, basin, mission, alpha)] = (mo, da)
                        if ref is not None and d == 'published_code' and key in ref:
                            check[key] = {n: float(np.nanmax(np.abs(ref[key][n][:] - g[n][:])))
                                          for n in ('nse_DA', 'kge_DA', 'rmse_DA', 'nse_model', 'kge_model', 'rmse_model')}
            print('done', basin, alpha, grid)
    for f in h5out.values():
        f.close()
    if check:
        write_json(out / 'check_published.json', check)

    summaries = {d: summary(means[d], keep, d, args.basins) for d in DEFS}
    for d in DEFS:
        write_json(out / ('summary_%s.json' % d), summaries[d])
    write_csv(out / 'comparison.csv', compare(summaries['published_code'], summaries['paper']))
    print('written', out)


def summary(means, keep, d, basins):
    s = {'definition': d, 'table1_fixed_1e18_1deg': {}, 'table1_best_per_metric': {}, 'section52_MAGIC_1e17_1deg': {},
         'figure9_vs_GRACEC_1e18_1deg': {}, 'figure8_series': {}}
    for basin in basins:
        fixed, best = {}, {}
        model = None
        for mission in C.MISSIONS:
            k = (basin, mission, '1e18', '1degree', 0)
            if k in means:
                fixed[mission] = {n: means[k][n + '_DA'] for n in ('nse', 'kge', 'rmse')}
                model = {n: means[k][n + '_model'] for n in ('nse', 'kge', 'rmse')}
            cand = {kk: v for kk, v in means.items() if kk[0] == basin and kk[1] == mission and kk[4] == 0}
            if cand:
                best[mission] = {}
                for n in ('nse', 'kge', 'rmse'):
                    pick = (min if n == 'rmse' else max)(cand, key=lambda kk: cand[kk][n + '_DA'])
                    best[mission][n] = {'value': cand[pick][n + '_DA'], 'alpha': pick[2], 'grid': pick[3]}
                s['figure8_series'][basin + '_' + mission] = {'%s_%s' % (kk[2], kk[3]): {n: v[n + '_DA'] for n in ('nse', 'kge', 'rmse')}
                                                              for kk, v in sorted(cand.items())}
        if model:
            fixed['model'] = model
            s['table1_fixed_1e18_1deg'][basin] = {**fixed, 'improvement': improvements(fixed)}
            flat = {m: {n: best[m][n]['value'] for n in ('nse', 'kge', 'rmse')} for m in best}
            flat['model'] = model
            s['table1_best_per_metric'][basin] = {'values': best, 'model': model, 'improvement': improvements(flat)}
        if (d, basin, 'MAGIC', '1e17') in keep:
            mo, da = keep[(d, basin, 'MAGIC', '1e17')]
            s['section52_MAGIC_1e17_1deg'][basin] = section52(mo, da)
        if all((d, basin, m, '1e18') in keep for m in C.MISSIONS):
            s['figure9_vs_GRACEC_1e18_1deg'][basin] = figure9({m: keep[(d, basin, m, '1e18')][1] for m in C.MISSIONS})
    return s


def improvements(t):
    if not all(m in t for m in ('model', 'GRACEC', 'NGGM', 'MAGIC')):
        return {}
    out = {}
    for m in ('NGGM', 'MAGIC'):
        out[m] = {}
        for n in ('nse', 'kge', 'rmse'):
            dlt, pct = M.improvement(t['model'][n], t['GRACEC'][n], t[m][n])
            out[m][n] = {'delta': dlt, 'percent': pct}
    return out


def share(x, thr):
    x = x[np.isfinite(x)]
    return float(100 * np.mean(x > thr)) if x.size else float('nan')


def section52(mo, da):
    r = {}
    for n in ('nse', 'kge'):
        dd = da[n] - mo[n]
        dd = dd[np.isfinite(dd)]
        r[n] = {'model_gt_0.5': share(mo[n], 0.5), 'DA_gt_0.5': share(da[n], 0.5), 'model_gt_0.9': share(mo[n], 0.9),
                'DA_gt_0.9': share(da[n], 0.9), 'improved_share': share(dd, 0), 'mean_gain_improved': float(dd[dd > 0].mean()),
                'mean_loss_degraded': float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0}
    return r


def figure9(da):
    r = {}
    for m in ('NGGM', 'MAGIC'):
        r[m] = {}
        for n in ('nse', 'kge'):
            dd = da[m][n] - da['GRACEC'][n]
            r[m][n] = {'improved_share': share(dd, 0), 'mean_diff': float(np.nanmean(dd))}
        dr = da[m]['rmse'] - da['GRACEC']['rmse']
        r[m]['rmse'] = {'improved_share': share(-dr, 0), 'max_reduction': float(np.nanmin(dr)),
                        'share_reduction_gt_10mm': share(-dr, 10)}
    return r


def compare(a, b, prefix=''):
    rows = []

    def walk(x, y, path):
        if isinstance(x, dict):
            for k in x:
                if k in y:
                    walk(x[k], y[k], path + [str(k)])
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            rows.append({'item': '/'.join(path), 'published_code': round(x, 4), 'standard': round(y, 4),
                         'change': round(y - x, 4)})
    walk(a, b, [])
    return rows


if __name__ == '__main__':
    main()
