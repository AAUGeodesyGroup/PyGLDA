"""Step 1 - reproduce Table 1 (and metrics.h5) from the published OL/DA products.

Checks, per basin, for the best configuration (alpha 1e18, 1 deg) and for every one of the 30 configurations per mission:
  * spatial-mean NSE / KGE / RMSE with the published code's definition  -> must equal ESAsing/paper_RSE/res/metrics.h5
  * the same with the Appendix-B definition                             -> what the paper text says it computed
  * the paper's Table 1 values, and which configuration each Table-1 number actually comes from
Usage:  python 01_reproduce_table1.py [--metrics-h5 .../metrics.h5] [--all-configs]
"""
import numpy as np
import h5py

from _common import parser, outdir, write_csv, write_json, definitions, C
from jog_eval import evaluate as E, sources as S

TABLE1 = {  # manuscript Table 1 (model, GRACE-C, NGGM, MAGIC)
    'Brahmaputra': {'nse': (0.372, 0.703, 0.796, 0.806), 'kge': (0.504, 0.752, 0.812, 0.806),
                    'rmse': (109.2, 73.2, 61.3, 59.6)},
    'Danube': {'nse': (-0.120, 0.689, 0.783, 0.800), 'kge': (0.319, 0.737, 0.824, 0.808),
               'rmse': (44.0, 31.7, 26.1, 24.3)},
}
H5_KEY = {'1degree': 'G1', '2degree': 'G2', '3degree': 'G3', '4degree': 'G4', '3subbasins': 'Gb'}


def main():
    p = parser(__doc__)
    p.add_argument('--metrics-h5', default=None, help='ESAsing/paper_RSE/res/metrics.h5 for a cell-by-cell check')
    p.add_argument('--all-configs', action='store_true', help='evaluate all 30 configurations per mission')
    args = p.parse_args()
    out = outdir(args, '01_table1')
    epochs = np.arange(*C.PAPER_EPOCHS)
    rows, report = [], {}
    for basin in args.basins:
        configs = [(a, g) for a in C.ALPHAS for g in C.GRIDS] if args.all_configs else [(C.BEST['alpha'], C.BEST['grid'])]
        for alpha, grid in configs:
            data = E.load_basin(basin, epochs, alpha=alpha, grid=grid)
            for d in definitions(args):
                tab = E.table(data, d)
                for prod, vals in tab.items():
                    rows.append({'basin': basin, 'alpha': alpha, 'grid': grid, 'definition': d, 'product': prod,
                                 'cells': data['cells'], **{k: round(v, 4) for k, v in vals.items()}})
            if args.metrics_h5 and (alpha, grid) == (C.BEST['alpha'], C.BEST['grid']):
                report[basin] = check_against_h5(args.metrics_h5, basin, alpha, grid, data)
    write_csv(out / 'table1_recomputed.csv', rows)
    report['table1_paper'] = TABLE1
    report['which_config_matches_table1'] = locate_table1(rows) if args.all_configs else 'run with --all-configs'
    write_json(out / 'table1_check.json', report)
    print('written', out)


def check_against_h5(fn, basin, alpha, grid, data):
    """max |difference| between recomputed per-cell metrics (published definition) and metrics.h5"""
    from jog_eval import metrics as M
    res = {}
    with h5py.File(fn, 'r') as h:
        for m in C.MISSIONS:
            g = h['%s_%s_F%s_%s_0' % (basin, m, alpha.upper().replace('1E', '1E'), H5_KEY[grid])]
            mine = M.all_metrics(data['truth'], data['sims'][m], 'published_code')
            res[m] = {k: float(np.nanmax(np.abs(g['%s_DA' % k][:] - mine[k]))) for k in ('nse', 'kge', 'rmse')}
    return res


def locate_table1(rows):
    """for every Table-1 number: the (definition, alpha, grid) whose spatial mean is closest"""
    names = ['model', 'GRACEC', 'NGGM', 'MAGIC']
    hit = {}
    for basin, t in TABLE1.items():
        for metric, vals in t.items():
            for prod, target in zip(names, vals):
                cand = [r for r in rows if r['basin'] == basin and r['product'] == prod]
                if not cand:
                    continue
                best = min(cand, key=lambda r: abs(r[metric] - target))
                hit['%s %s %s' % (basin, metric, prod)] = {'paper': target, 'closest': best[metric],
                                                           'definition': best['definition'], 'alpha': best['alpha'],
                                                           'grid': best['grid']}
    return hit


if __name__ == '__main__':
    main()
