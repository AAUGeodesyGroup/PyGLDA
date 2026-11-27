"""Step 12 - response surface over the 30 configurations (R1.11).

From results/07_recompute_published_outputs/summary_paper.json (standard definition, 'figure8_series': spatial means
of the DA metrics, NSE/KGE clipped at -1): heat maps alpha (6) x grid (5) per basin, mission and metric (NSE, KGE,
RMSE), plus NGGM - GRACE-C and MAGIC - GRACE-C. Best cell per panel boxed (max NSE/KGE, min RMSE; for the difference
panels the largest improvement). One PNG per basin + response_surface.csv (long format) + best.csv.
Caveat: DA = published DA_mean products (2-member means, members 1-2).
Usage: python 12_response_surface.py
"""
import json
import time

import numpy as np

from _common import parser, outdir, write_csv, write_json, C

GRID_LABEL = {'1degree': '1°', '2degree': '2°', '3degree': '3°', '4degree': '4°', '3subbasins': '3 sub-basins'}
METRICS = ('nse', 'kge', 'rmse')


def main():
    t0 = time.time()
    ap = parser(__doc__)
    ap.set_defaults(definition='paper')
    ap.add_argument('--src-tag', default='published_outputs', help="07 output tag, e.g. v1_rerun (30-member rerun products)")
    args = ap.parse_args()
    d = 'paper' if args.definition == 'both' else args.definition
    out = outdir(args, '12_response_surface')
    src = C.OUT_ROOT / ('07_recompute_' + args.src_tag) / ('summary_%s.json' % d)
    s = json.loads(src.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows, best = [], []
    for basin in args.basins:
        Z = {}
        for m in C.MISSIONS:
            ser = s['figure8_series'][basin + '_' + m]
            for n in METRICS:
                Z[(m, n)] = np.array([[ser['%s_%s' % (a, g)][n] for g in C.GRIDS] for a in C.ALPHAS])
                for ia, a in enumerate(C.ALPHAS):
                    for ig, g in enumerate(C.GRIDS):
                        rows.append({'basin': basin, 'mission': m, 'metric': n, 'alpha': a, 'grid': g,
                                     'value': round(float(Z[(m, n)][ia, ig]), 4),
                                     'minus_GRACEC': round(float(Z[(m, n)][ia, ig] - Z[('GRACEC', n)][ia, ig]), 4)})
        model = s['table1_fixed_1e18_1deg'][basin]['model']
        cols = list(C.MISSIONS) + ['NGGM-GRACEC', 'MAGIC-GRACEC']
        fig, axs = plt.subplots(3, 5, figsize=(19, 11.5))
        for i, n in enumerate(METRICS):
            vals = np.stack([Z[(m, n)] for m in C.MISSIONS])
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            dif = np.stack([Z[(m, n)] - Z[('GRACEC', n)] for m in ('NGGM', 'MAGIC')])
            dl = np.nanmax(np.abs(dif))
            for j, c in enumerate(cols):
                ax = axs[i, j]
                if j < 3:
                    z = Z[(c, n)]
                    cmap = 'Blues' if n != 'rmse' else 'Blues_r'     # darker = better
                    im = ax.imshow(z, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
                    good = z if n != 'rmse' else -z
                else:
                    z = Z[(c.split('-')[0], n)] - Z[('GRACEC', n)]
                    cmap = 'RdBu' if n != 'rmse' else 'RdBu_r'       # blue = improvement over GRACE-C
                    im = ax.imshow(z, cmap=cmap, vmin=-dl, vmax=dl, aspect='auto')
                    good = z if n != 'rmse' else -z
                ib = np.unravel_index(np.nanargmax(good), z.shape)
                ax.add_patch(Rectangle((ib[1] - 0.5, ib[0] - 0.5), 1, 1, fill=False, ec='#e8a33d', lw=3))
                best.append({'basin': basin, 'panel': c, 'metric': n, 'alpha': C.ALPHAS[ib[0]], 'grid': C.GRIDS[ib[1]],
                             'value': round(float(z[ib]), 4)})
                fmt = '%.2f' if n != 'rmse' else '%.1f'
                for ia in range(z.shape[0]):
                    for ig in range(z.shape[1]):
                        v = z[ia, ig]
                        frac = (v - im.norm.vmin) / (im.norm.vmax - im.norm.vmin)
                        dark = (frac > 0.6) if j < 3 and n != 'rmse' else (frac < 0.4) if j < 3 else abs(frac - 0.5) > 0.3
                        ax.text(ig, ia, fmt % v, ha='center', va='center', fontsize=7.5, color='white' if dark else '#222')
                ax.set_xticks(range(len(C.GRIDS)))
                ax.set_xticklabels([GRID_LABEL[g] for g in C.GRIDS], fontsize=8)
                ax.set_yticks(range(len(C.ALPHAS)))
                ax.set_yticklabels(C.ALPHAS, fontsize=8)
                if j == 0:
                    ax.set_ylabel('%s\nalpha' % n.upper())
                title = c.replace('GRACEC', 'GRACE-C')
                if j == 0:
                    title += '  (model %s = %s)' % (n.upper(), fmt % model[n])
                ax.set_title(title, fontsize=10)
                if j in (2, 4):
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle('%s: DA spatial-mean metrics over alpha x grid, 2003-09-05..2006-12-27 (standard definition; '
                     'published DA_mean = 2-member means). Box = best cell.' % basin, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / ('response_surface_%s.png' % basin), dpi=120)
        plt.close(fig)
    write_csv(out / 'response_surface.csv', rows)
    write_csv(out / 'best.csv', best)
    write_json(out / 'meta.json', {'source': str(src), 'definition': d, 'run_time_s': round(time.time() - t0, 1),
                                   'caveat': 'DA = published DA_mean products, 2-member means (members 1-2)'})
    print('written', out)


if __name__ == '__main__':
    main()
