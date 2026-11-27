"""Step 31 - variant R01 (observations with HIS C00/C10/C11/S11 and HIS C20 + noise) vs the nominal baselines.

Products (1e18 / 1 degree, paper epochs, paper definition): model = shared OL mean; nominal seeds 1-3
(cases jog_<M>_<basin>_s<k>_nominal) and R01 seed 1 (jog_<M>_<basin>_s1_R01), DA_mean of members 1-30.
Per basin, grid level (0.5 deg spatial mean of cell metrics) and basin level (basin-mean series):
  * NSE / KGE / RMSE of every product; seed spread (min-max over the 3 nominal seeds);
  * R01 - nominal s1 per mission with paired moving-block bootstrap CIs (same seed = same obs-perturbation draws);
  * gap change: (R01_X - R01_GRACEC) - (nom_X - nom_GRACEC) for X = NGGM, MAGIC, with its bootstrap CI;
  * innovation statistics (script 16: mean NIS [CI], share of windows outside the chi2 95 % interval, bias, RMS d) of R01 and
    nominal s1.
Output: results/31_r01_compare/{summary.json, table.md}.
Usage: python 31_r01_compare.py [--basins ...] [--n-boot 1000]
"""
import importlib

import numpy as np

from _common import parser, outdir, write_json, C
from jog_eval import evaluate as E, metrics as M, stats as ST

NAMES = ('nse', 'kge', 'rmse')
I16 = importlib.import_module('16_innovations')


def folders(basin, r01_seeds=(1,)):
    w3 = C.V1_DATA_ROOT / 'w3ra'
    mc = 'MAGIC_1e18_1degree_%s' % basin
    f = {'model': ('OL', str(w3 / 'domains' / ('jog_%s1degree' % basin) / 'OL' / 'pub_magic1e19' / 'products' / 'OL_mean'), mc)}
    for m in C.MISSIONS:
        for s in (1, 2, 3):
            f['nom%d_%s' % (s, m)] = ('DA', str(w3 / 'cases' / ('jog_%s_%s_s%d_nominal' % (m, basin, s)) / 'products' / 'DA_mean'), mc)
        f['R01_%s' % m] = ('DA', str(w3 / 'cases' / ('jog_%s_%s_s1_R01' % (m, basin)) / 'products' / 'DA_mean'), mc)
        for s in r01_seeds:
            if s != 1:
                f['R01s%d_%s' % (s, m)] = ('DA', str(w3 / 'cases' / ('jog_%s_%s_s%d_R01' % (m, basin, s)) / 'products' / 'DA_mean'), mc)
    return f


def seed_summary(point, r01_seeds):
    """3-seed nominal vs R01: mean and range per mission, seed-mean gaps to GRACE-C, paired per-seed R01 - nominal"""
    def key(kind, s, m):
        if kind == 'nom':
            return 'nom%d_%s' % (s, m)
        return 'R01_%s' % m if s == 1 else 'R01s%d_%s' % (s, m)
    out = {}
    for n in NAMES:
        o = {}
        for m in C.MISSIONS:
            nv = [point[key('nom', s, m)][n] for s in (1, 2, 3)]
            rv = [point[key('R01', s, m)][n] for s in r01_seeds]
            o[m] = {'nominal_mean': float(np.mean(nv)), 'nominal_range': [min(nv), max(nv)],
                    'R01_mean': float(np.mean(rv)), 'R01_range': [min(rv), max(rv)],
                    'paired_delta': {str(s): point[key('R01', s, m)][n] - point[key('nom', s, m)][n] for s in r01_seeds}}
        for x in ('NGGM', 'MAGIC'):
            o['gap_%s' % x] = {'nominal': o[x]['nominal_mean'] - o['GRACEC']['nominal_mean'],
                               'R01': o[x]['R01_mean'] - o['GRACEC']['R01_mean']}
            o['gap_%s' % x]['change'] = o['gap_%s' % x]['R01'] - o['gap_%s' % x]['nominal']
        out[n] = o
    return out


def basin_mean(a):
    return np.nanmean(a.reshape(a.shape[0], -1), axis=1)[:, None]


def table(truth, sims, sel):
    out = {}
    for k, s in sims.items():
        mm = M.all_metrics(truth[sel], s[sel], 'paper')
        out[k] = {n: M.spatial_mean(mm[n], n) for n in NAMES}
    return out


def level(truth, sims, n_boot, block):
    T = truth.shape[0]
    point = table(truth, sims, np.arange(T))
    boot = [table(truth, sims, i) for i in ST.block_bootstrap_indices(T, block, n_boot)]

    def ci(f):
        v = np.array([f(b) for b in boot])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    r = {'point': point, 'delta': {}, 'gap_change': {}, 'seed_range': {}}
    for m in C.MISSIONS:
        r['seed_range'][m] = {n: [min(point['nom%d_%s' % (s, m)][n] for s in (1, 2, 3)),
                                  max(point['nom%d_%s' % (s, m)][n] for s in (1, 2, 3))] for n in NAMES}
        r['delta'][m] = {n: {'value': point['R01_' + m][n] - point['nom1_' + m][n],
                             'ci': ci(lambda b, m=m, n=n: b['R01_' + m][n] - b['nom1_' + m][n])} for n in NAMES}
    for x in ('NGGM', 'MAGIC'):
        def g(t, n, x=x):
            return (t['R01_' + x][n] - t['R01_GRACEC'][n]) - (t['nom1_' + x][n] - t['nom1_GRACEC'][n])
        r['gap_change'][x] = {n: {'value': g(point, n), 'ci': ci(lambda b, n=n: g(b, n)),
                                  'gap_nominal': point['nom1_' + x][n] - point['nom1_GRACEC'][n],
                                  'gap_R01': point['R01_' + x][n] - point['R01_GRACEC'][n]} for n in NAMES}
    return r


def main():
    p = parser(__doc__)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--block', type=int, default=18)
    p.add_argument('--r01-seeds', nargs='+', type=int, default=[1])
    args = p.parse_args()
    out = outdir(args, '31_r01_compare')
    epochs = np.arange(*C.PAPER_EPOCHS)
    res, md = {}, ['# R01 vs nominal (1e18 / 1 deg, paper def., %d epochs, block %d, %d bootstraps)' % (len(epochs), args.block, args.n_boot)]
    for basin in args.basins:
        data = E.load_basin(basin, epochs, missions=('MAGIC',), folders=folders(basin, args.r01_seeds), model_case='MAGIC_1e18_1degree_%s' % basin)
        t, sims = data['truth'], data['sims']
        r = {'cells': data['cells']}
        for lev, (tt, ss) in {'grid': (t, sims), 'basin': (basin_mean(t), {k: basin_mean(v) for k, v in sims.items()})}.items():
            r[lev] = level(tt, ss, args.n_boot, args.block)
            r[lev]['seeds'] = seed_summary(r[lev]['point'], args.r01_seeds)
        r['innovations'] = {}
        for m in C.MISSIONS:
            for tag in ('nominal', 'R01'):
                case = 'jog_%s_%s_s1_%s' % (m, basin, tag)
                fn = C.V1_DATA_ROOT / 'w3ra' / 'cases' / case / 'DA' / 'innovations.h5'
                r['innovations'][case] = I16.analyse(I16.load(fn))[0] if fn.exists() else 'missing %s' % fn
        res[basin] = r
        for lev in ('grid', 'basin'):
            L = r[lev]
            md.append('\n## %s, %s level (%d cells)\n' % (basin, lev, r['cells']))
            md.append('| mission | NSE nominal s1 / R01 / delta [CI] / seed range | KGE nominal s1 / R01 / delta [CI] | RMSE nominal s1 / R01 / delta [CI] |')
            md.append('|---|---|---|---|')
            for m in C.MISSIONS:
                cells = []
                for n in NAMES:
                    d = L['delta'][m][n]
                    c = '%.3f / %.3f / %+.3f [%+.3f, %+.3f]' % (L['point']['nom1_' + m][n], L['point']['R01_' + m][n], d['value'], *d['ci'])
                    if n == 'nse':
                        c += ' / %.3f-%.3f' % tuple(L['seed_range'][m][n])
                    cells.append(c)
                md.append('| %s | %s |' % (m, ' | '.join(cells)))
            md.append('| model | %s |' % ' | '.join('%.3f' % L['point']['model'][n] for n in NAMES))
            for x in ('NGGM', 'MAGIC'):
                md.append('| gap %s - GRACEC: nominal / R01 / change [CI] | %s |' % (x, ' | '.join(
                    '%+.3f / %+.3f / %+.3f [%+.3f, %+.3f]' % (L['gap_change'][x][n]['gap_nominal'], L['gap_change'][x][n]['gap_R01'],
                                                           L['gap_change'][x][n]['value'], *L['gap_change'][x][n]['ci']) for n in NAMES)))
            if len(args.r01_seeds) > 1:
                S3 = L['seeds']
                md.append('\n| %s seeds: mean (range) | NSE nominal | NSE R01 | paired R01 - nominal per seed | KGE nominal | KGE R01 |' % (
                    '/'.join(str(x) for x in args.r01_seeds)))
                md.append('|---|---|---|---|---|---|')
                for m in C.MISSIONS:
                    a, k = S3['nse'][m], S3['kge'][m]
                    md.append('| %s | %.3f (%.3f-%.3f) | %.3f (%.3f-%.3f) | %s | %.3f (%.3f-%.3f) | %.3f (%.3f-%.3f) |' % (
                        m, a['nominal_mean'], *a['nominal_range'], a['R01_mean'], *a['R01_range'],
                        ' / '.join('%+.3f' % v for v in a['paired_delta'].values()),
                        k['nominal_mean'], *k['nominal_range'], k['R01_mean'], *k['R01_range']))
                for x in ('NGGM', 'MAGIC'):
                    gn, gk = S3['nse']['gap_%s' % x], S3['kge']['gap_%s' % x]
                    md.append('| gap %s - GRACEC (seed means) | %+.3f | %+.3f | change %+.3f | %+.3f | %+.3f |' % (
                        x, gn['nominal'], gn['R01'], gn['change'], gk['nominal'], gk['R01']))
                md.append('')
        md.append('\n### %s innovations (s1)\n' % basin)
        md.append('| case | mean NIS [CI] | share windows above chi2 95 % | bias d mm | RMS d mm | spread sqrt(HPH) mm | sigma R mm |')
        md.append('|---|---|---|---|---|---|---|')
        for case, a in r['innovations'].items():
            if isinstance(a, str):
                md.append('| %s | %s | | | | | |' % (case, a))
                continue
            md.append('| %s | %.2f [%.2f, %.2f] | %.2f | %+.2f | %.2f | %.2f | %.2f |' % (
                case, a['nis_mean'], *a['nis_ci95'], a['share_windows_above'], a['bias_d'], a['rms_d'], a['spread_hx'], a['sigma_R']))
        print('\n'.join(md[-30:]), flush=True)
    write_json(out / 'summary.json', res)
    (out / 'table.md').write_text('\n'.join(md) + '\n')
    print('written', out)


if __name__ == '__main__':
    main()
