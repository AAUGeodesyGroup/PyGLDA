"""Step 3 (R1.2) - out-of-sample choice of the best configuration.

For each basin and mission: spatial-mean metrics of all 30 configurations (6 alpha x 5 grids) on period A
(2003-09..2004-12) and B (2005-01..2006-12); the configuration chosen on A is evaluated on B (and vice versa).
Reports the chosen configuration, its score, the best one on the other period and the regret.
Usage: python 03_split_sample.py [--metric kge]
"""
from datetime import date

import numpy as np

from _common import parser, outdir, write_csv, write_json, definitions, C
from jog_eval import evaluate as E, metrics as M, sources as S, stats as ST

PERIODS = {'A': (date(2003, 9, 1), date(2004, 12, 31)), 'B': (date(2005, 1, 1), date(2006, 12, 31))}


def main():
    p = parser(__doc__)
    p.add_argument('--metric', default='kge', choices=['kge', 'nse', 'rmse'])
    args = p.parse_args()
    out = outdir(args, '03_split_sample')
    ep = {k: S.epochs_between(*v) for k, v in PERIODS.items()}
    allep = np.arange(*C.PAPER_EPOCHS)
    rows, res = [], {}
    sign = -1 if args.metric == 'rmse' else 1
    defs = definitions(args)
    for basin in args.basins:
        for m in C.MISSIONS:
            score = {d: {'A': {}, 'B': {}} for d in defs}
            for a in C.ALPHAS:                    # each configuration is loaded once for all definitions
                for g in C.GRIDS:
                    data = E.load_basin(basin, allep, missions=(m,), alpha=a, grid=g)
                    for per, e in ep.items():
                        sel = np.isin(allep, e)
                        for d in defs:
                            v = M.all_metrics(data['truth'][sel], data['sims'][m][sel], d)[args.metric]
                            score[d][per]['%s_%s' % (a, g)] = {args.metric: sign * M.spatial_mean(v, args.metric)}
            for d in defs:
                for x, y in (('A', 'B'), ('B', 'A')):
                    r = ST.split_sample(score[d][x], score[d][y], key=args.metric)
                    res['%s_%s_%s_%s->%s' % (basin, d, m, x, y)] = r
                    rows.append({'basin': basin, 'definition': d, 'mission': m, 'select_on': x, 'evaluate_on': y,
                                 **{k: (round(sign * v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
    write_csv(out / ('split_sample_%s.csv' % args.metric), rows)
    write_json(out / ('split_sample_%s.json' % args.metric), res)
    print('written', out)


if __name__ == '__main__':
    main()
