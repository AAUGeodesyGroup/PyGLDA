"""Step 2 (R1.3) - uncertainty of the headline numbers.

Moving-block bootstrap over the 242 five-day epochs (paired: the same resampled epochs for model, GRACE-C, NGGM, MAGIC),
95 % percentile intervals of the spatial-mean NSE/KGE/RMSE and of the differences NGGM-GRACEC, MAGIC-GRACEC, MAGIC-NGGM,
GRACEC-model, NGGM-model, MAGIC-model.
Block length default 18 epochs (90 days) to keep seasonal autocorrelation; --block 36 / 73 as sensitivity.
--hpf 1: 60-day high-pass (ST.butter_band(M.demean(x), low_days=60), as step 07) on truth and every product before
resampling (Sect. 6.3). --alpha / --grid: configuration (default C.BEST).
Usage: python 02_bootstrap_ci.py [--block 18] [--n-boot 1000] [--hpf 0|1] [--alpha 1e18] [--grid 1degree]
"""
import numpy as np

from _common import parser, outdir, write_csv, write_json, definitions, C
from jog_eval import evaluate as E, metrics as M, stats as ST

PAIRS = [('NGGM', 'GRACEC'), ('MAGIC', 'GRACEC'), ('MAGIC', 'NGGM'), ('GRACEC', 'model'), ('NGGM', 'model'),
         ('MAGIC', 'model')]
HPF_CUTOFF_DAYS = 60


def main():
    p = parser(__doc__)
    p.add_argument('--block', type=int, default=18)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--hpf', type=int, default=0, choices=[0, 1])
    p.add_argument('--alpha', default=C.BEST['alpha'])
    p.add_argument('--grid', default=C.BEST['grid'])
    args = p.parse_args()
    out = outdir(args, '02_bootstrap')
    tag = '%s_%s_hpf%d_block%d' % (args.alpha, args.grid, args.hpf, args.block)
    epochs = np.arange(*C.PAPER_EPOCHS)
    rows, full = [], {}
    for basin in args.basins:
        data = E.load_basin(basin, epochs, alpha=args.alpha, grid=args.grid)
        truth, sims = data['truth'], data['sims']
        if args.hpf:
            truth = ST.butter_band(M.demean(truth), low_days=HPF_CUTOFF_DAYS)
            sims = {k: ST.butter_band(M.demean(v), low_days=HPF_CUTOFF_DAYS) for k, v in sims.items()}
        for d in definitions(args):
            r = ST.bootstrap_table(truth, sims, definition=d, block=args.block, n_boot=args.n_boot, pairs=PAIRS)
            full['%s_%s' % (basin, d)] = r
            for k, ci in r['ci'].items():
                for n, (lo, hi) in ci.items():
                    rows.append({'basin': basin, 'alpha': args.alpha, 'grid': args.grid, 'hpf': args.hpf,
                                 'block': args.block, 'n_boot': args.n_boot, 'definition': d, 'item': k, 'metric': n,
                                 'value': round(r['point'][k][n], 4), 'ci_low': round(lo, 4), 'ci_high': round(hi, 4),
                                 'excludes_zero': (lo > 0 or hi < 0) if '-' in k else ''})
    write_csv(out / ('bootstrap_ci_%s.csv' % tag), rows)
    write_json(out / ('bootstrap_ci_%s.json' % tag), full)
    print('written', out)


if __name__ == '__main__':
    main()
