"""Step 11 - spread of the grid metrics over the 0.5 deg cells (R1.3).

Reads results/07_recompute_published_outputs/metrics_paper.h5 (standard definition, Appendix B) and gives, per basin,
mission, model/DA, configuration (1e18 / 1 deg and 1e17 / 1 deg) and HPF (0 = full signal, 1 = 60-day high-pass):
median, IQR (p25, p75), mean and the paper-style spatial mean (NSE/KGE clipped at -1) of NSE, KGE and RMSE over cells.
'model' is the OL (identical for the three missions of one configuration; listed once per configuration).
Caveat: DA = published DA_mean products (2-member means, members 1-2).
Usage: python 11_cell_spread.py [--definition paper]
"""
import time

import h5py
import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import metrics as M

CONFIGS = (('1e18', '1degree', 'F1E18_G1'), ('1e17', '1degree', 'F1E17_G1'))


def main():
    t0 = time.time()
    ap = parser(__doc__)
    ap.set_defaults(definition='paper')
    ap.add_argument('--src-tag', default='published_outputs', help="07 output tag, e.g. v1_rerun (30-member rerun products)")
    args = ap.parse_args()
    d = 'paper' if args.definition == 'both' else args.definition
    out = outdir(args, '11_cell_spread')
    src = C.OUT_ROOT / ('07_recompute_' + args.src_tag) / ('metrics_%s.h5' % d)
    rows = []
    with h5py.File(src, 'r') as h:
        for basin in args.basins:
            for alpha, grid, tag in CONFIGS:
                for hpf in (0, 1):
                    for prod in ('model',) + C.MISSIONS:
                        mission = 'GRACEC' if prod == 'model' else prod
                        g = h['%s_%s_%s_%d' % (basin, mission, tag, hpf)]
                        suffix = '_model' if prod == 'model' else '_DA'
                        row = {'basin': basin, 'alpha': alpha, 'grid': grid, 'hpf': hpf, 'product': prod}
                        for n in ('nse', 'kge', 'rmse'):
                            v = g[n + suffix][:]
                            v = v[np.isfinite(v)]
                            q50, q25, q75 = np.percentile(v, [50, 25, 75])
                            row.update({n + '_median': round(float(q50), 4), n + '_p25': round(float(q25), 4),
                                        n + '_p75': round(float(q75), 4), n + '_iqr': round(float(q75 - q25), 4),
                                        n + '_mean': round(float(np.mean(v)), 4),
                                        n + '_spatial_mean_paper': round(M.spatial_mean(v, n), 4)})
                        row['n_cells'] = int(v.size)
                        rows.append(row)
    write_csv(out / ('cell_spread_%s.csv' % d), rows)
    write_json(out / 'meta.json', {'source': str(src), 'definition': d, 'run_time_s': round(time.time() - t0, 1),
                                   'caveat': 'DA = published DA_mean products, 2-member means (members 1-2)'})
    print('written', out)


if __name__ == '__main__':
    main()
