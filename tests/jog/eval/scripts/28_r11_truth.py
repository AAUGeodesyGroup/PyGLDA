"""Step 28 (R1.1) - same-climate control: score the R11 runs against the synthetic W3RA truth.

Truth = member 1 of ensemble r11truth_s<seed> (one extra perturbed W3RA member, own seed), daily TotalWater on the model
cells (cases/jog_R11truth_<basin>_s<seed>/products/truth/<case>/), treated exactly like the DA products: 5-day means over
the ESM2.0 epochs, 0.5 deg block means on the published case mask. The R11 observations are this truth + the published
observation errors (published obs - ESM2.0 HIS, per unit and epoch), so the missions differ only by their error.
Products: model = OL mean (member 0) of the shared ensemble; DA = cases/jog_<M>_<basin>_s<seed>_R11/products/DA_mean.
Epochs = the paper epochs (2003-09-05..2006-12-27). Metrics: paper definition (Appendix B), spatial means of cell maps
(grid) and metrics of the basin-mean series (basin); moving-block bootstrap (paired) CIs of each product and of the mission
differences.
Output: results/28_r11_truth/{summary.json, table.md}.
Harder truth (perturbation std x k, jog_run.py r11-truth --pert-scale k): --truth-tag x<k> --variant R11x<k>.
Usage: python 28_r11_truth.py [--basins ...] [--seed 1] [--truth-seed 101] [--truth-tag ''] [--variant R11] [--n-boot 1000]
"""
import numpy as np

from _common import parser, outdir, write_json, C
from jog_eval import metrics as M, sources as S, stats as ST

PAIRS = (('NGGM', 'GRACEC'), ('MAGIC', 'GRACEC'), ('MAGIC', 'NGGM'), ('GRACEC', 'model'), ('NGGM', 'model'),
         ('MAGIC', 'model'))
NAMES = ('nse', 'kge', 'rmse')


def load(basin, seed, truth_seed, epochs, truth_tag='', variant='R11'):
    w3 = C.V1_DATA_ROOT / 'w3ra'
    mask_case = 'MAGIC_1e18_1degree_%s' % basin                     # 0.1 deg mask of the shared 1-degree domain
    tcase = 'jog_R11truth_%s_s%s%s' % (basin, truth_seed, truth_tag)
    folders = {'truth': ('OL', w3 / 'cases' / tcase / 'products' / 'truth' / tcase),
               'model': ('OL', w3 / 'domains' / ('jog_%s1degree' % basin) / 'OL' / 'pub_magic1e19' / 'products' / 'OL_mean')}
    for m in C.MISSIONS:
        folders[m] = ('DA', w3 / 'cases' / ('jog_%s_%s_s%s_%s' % (m, basin, seed, variant)) / 'products' / 'DA_mean')
    x, missing = {}, {}
    for k, (kind, f) in folders.items():
        x[k], missing[k] = S.product_grid(kind, mask_case, epochs, folder=str(f), mask_case=mask_case)
    ok = np.ones(x['truth'].shape[1:], bool)
    for v in x.values():
        ok &= np.isfinite(v).all(0)
    x = {k: np.where(ok[None], v, np.nan) for k, v in x.items()}
    return x, int(ok.sum()), {k: [str(d) for d in v] for k, v in missing.items() if v}, folders


def load_ol(basin, truth_seed, epochs, truth_tag):
    w3 = C.V1_DATA_ROOT / 'w3ra'
    mask_case = 'MAGIC_1e18_1degree_%s' % basin
    tcase = 'jog_R11truth_%s_s%s%s' % (basin, truth_seed, truth_tag)
    x = {}
    for k, f in {'truth': w3 / 'cases' / tcase / 'products' / 'truth' / tcase,
                 'model': w3 / 'domains' / ('jog_%s1degree' % basin) / 'OL' / 'pub_magic1e19' / 'products' / 'OL_mean'}.items():
        x[k], _ = S.product_grid('OL', mask_case, epochs, folder=str(f), mask_case=mask_case)
    ok = np.isfinite(x['truth']).all(0) & np.isfinite(x['model']).all(0)
    return {k: np.where(ok[None], v, np.nan) for k, v in x.items()}, int(ok.sum()), {}, {}


def basin_mean(a):
    return np.nanmean(a.reshape(a.shape[0], -1), axis=1)[:, None]


def main():
    p = parser(__doc__)
    p.add_argument('--seed', default='1')
    p.add_argument('--truth-seed', default='101')
    p.add_argument('--truth-tag', default='')
    p.add_argument('--variant', default='R11')
    p.add_argument('--ol-only', action='store_true', help='only the OL (model) vs truth numbers, no DA products needed')
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--block', type=int, default=18)
    args = p.parse_args()
    out = outdir(args, '28_r11_truth' + ('_' + args.truth_tag if args.truth_tag else ''))
    epochs = np.arange(*C.PAPER_EPOCHS)
    res, md = {}, ['# R1.1 same-climate control (paper definition, %d epochs, block %d, %d bootstraps)\n' %
                   (len(epochs), args.block, args.n_boot)]
    for basin in args.basins:
        if args.ol_only:
            x, cells, missing, folders = load_ol(basin, args.truth_seed, epochs, args.truth_tag)
            t = x.pop('truth')
            for level, (tt, ss) in {'grid': (t, x['model']), 'basin': (basin_mean(t), basin_mean(x['model']))}.items():
                m = M.all_metrics(tt, ss, 'paper')
                print('OL-only', basin, args.truth_tag or 'x1', level, cells, 'cells',
                      {n: round(M.spatial_mean(m[n], n), 3) for n in NAMES}, flush=True)
            continue
        x, cells, missing, folders = load(basin, args.seed, args.truth_seed, epochs, args.truth_tag, args.variant)
        truth = x.pop('truth')
        r = {'cells': cells, 'missing_epochs': missing, 'folders': {k: str(v[1]) for k, v in folders.items()}}
        for level, (t, sims) in {'grid': (truth, x),
                                 'basin': (basin_mean(truth), {k: basin_mean(v) for k, v in x.items()})}.items():
            r[level] = ST.bootstrap_table(t, sims, NAMES, 'paper', args.block, args.n_boot, pairs=PAIRS)
        res[basin] = r
        md.append('\n## %s (%d cells at 0.5 deg)\n' % (basin, cells))
        for level in ('grid', 'basin'):
            b = r[level]
            md.append('\n%s level | NSE [95%% CI] | KGE [95%% CI] | RMSE mm [95%% CI]\n|---|---|---|---|' % level)
            for k in ('model',) + C.MISSIONS + tuple('%s-%s' % pr for pr in PAIRS):
                cells_ = ['%.3f [%.3f, %.3f]' % ((b['point'][k][n],) + tuple(b['ci'][k][n])) for n in NAMES]
                md.append('| %s | %s |' % (k, ' | '.join(cells_)))
        print(basin, cells, {k: round(v['nse'], 3) for k, v in r['grid']['point'].items()}, flush=True)
    write_json(out / 'summary.json', res)
    (out / 'table.md').write_text('\n'.join(md) + '\n')
    print('written', out)


if __name__ == '__main__':
    main()
