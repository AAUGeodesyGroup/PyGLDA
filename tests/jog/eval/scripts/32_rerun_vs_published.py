"""Step 32 - published (2-member DA mean, per-configuration OL) vs rerun (30-member DA mean, shared OL) for all 180 configurations.

Inputs: results/07_recompute_published_outputs/summary_paper.json and results/07_recompute_v1_rerun/summary_paper.json (script 07,
paper definition, epochs 634..875; figure8_series = spatial means per basin / mission / alpha_grid).
Output (results/32_rerun_vs_published/):
  configs.csv          basin, mission, alpha, grid, nse/kge/rmse published and rerun, difference
  table.md             per basin and mission: median / max |difference|, Spearman rank correlation of the 30 configurations
                       (Figure 8 ranking), best configuration per metric (published vs rerun), Table 1 A (1e18 / 1 deg),
                       Table 1 B (best alpha at 1 deg per mission; overall best alpha x grid), mission differences per alpha (1 deg)
  scatter.png          published vs rerun NSE / KGE / RMSE, all configurations
Usage: python 32_rerun_vs_published.py [--published <dir>] [--rerun <dir>]
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from _common import parser, outdir, write_csv, write_json, C

NAMES = ('nse', 'kge', 'rmse')


def best(series, n):
    pick = (min if n == 'rmse' else max)(series, key=lambda k: series[k][n])
    return pick, series[pick][n]


def main():
    p = parser(__doc__)
    p.add_argument('--published', default=str(C.OUT_ROOT / '07_recompute_published_outputs'))
    p.add_argument('--rerun', default=str(C.OUT_ROOT / '07_recompute_v1_rerun'))
    args = p.parse_args()
    out = outdir(args, '32_rerun_vs_published')
    pub = json.loads((Path(args.published) / 'summary_paper.json').read_text())
    new = json.loads((Path(args.rerun) / 'summary_paper.json').read_text())
    rows, res, md = [], {}, ['# Published (2-member, per-configuration OL) vs rerun (30-member, shared OL), paper definition\n']
    for basin in args.basins:
        md.append('\n## %s\n' % basin)
        md.append('| mission | metric | median diff (rerun - pub) | max abs diff | Spearman (30 configs) | best pub | best rerun |')
        md.append('|---|---|---|---|---|---|---|')
        res[basin] = {}
        for m in C.MISSIONS:
            sp, sn = pub['figure8_series']['%s_%s' % (basin, m)], new['figure8_series']['%s_%s' % (basin, m)]
            keys = sorted(set(sp) & set(sn))
            assert len(keys) == 30, (basin, m, len(keys))
            for k in keys:
                a, g = k.split('_', 1)
                rows.append(dict(basin=basin, mission=m, alpha=a, grid=g,
                                 **{'%s_pub' % n: round(sp[k][n], 4) for n in NAMES}, **{'%s_rerun' % n: round(sn[k][n], 4) for n in NAMES},
                                 **{'%s_diff' % n: round(sn[k][n] - sp[k][n], 4) for n in NAMES}))
            r = {}
            for n in NAMES:
                x, y = np.array([sp[k][n] for k in keys]), np.array([sn[k][n] for k in keys])
                rho = float(stats.spearmanr(x, y)[0])
                bp, bn = best(sp, n), best(sn, n)
                r[n] = dict(median_diff=float(np.median(y - x)), max_abs_diff=float(np.max(np.abs(y - x))), spearman=rho,
                            best_pub=bp, best_rerun=bn)
                md.append('| %s | %s | %+.3f | %.3f | %.3f | %s %.3f | %s %.3f |' % (m, n, r[n]['median_diff'], r[n]['max_abs_diff'], rho,
                                                                          bp[0], bp[1], bn[0], bn[1]))
            res[basin][m] = r
        # Table 1 A and B
        t1p, t1n = pub['table1_fixed_1e18_1deg'][basin], new['table1_fixed_1e18_1deg'][basin]
        md.append('\n**Table 1 A (1e18 / 1 deg)** NSE / KGE / RMSE, published -> rerun:')
        for m in ('model',) + C.MISSIONS:
            md.append('- %s: %s' % (m, ', '.join('%.3f -> %.3f' % (t1p[m][n], t1n[m][n]) for n in NAMES)))
        md.append('\n**Table 1 B (rerun)**: best alpha at 1 deg per mission | best alpha x grid per mission (per metric):')
        md.append('| mission | NSE best alpha 1 deg | KGE | RMSE | NSE best alpha x grid | KGE | RMSE |')
        md.append('|---|---|---|---|---|---|---|')
        tb = {}
        for m in C.MISSIONS:
            sn = new['figure8_series']['%s_%s' % (basin, m)]
            one = {k: v for k, v in sn.items() if k.endswith('_1degree')}
            tb[m] = {'1deg': {n: best(one, n) for n in NAMES}, 'all': {n: best(sn, n) for n in NAMES}}
            md.append('| %s | %s | %s |' % (m, ' | '.join('%s %.3f' % (tb[m]['1deg'][n][0].split('_')[0], tb[m]['1deg'][n][1]) for n in NAMES),
                                          ' | '.join('%s %.3f' % tb[m]['all'][n] for n in NAMES)))
        res[basin]['table1_B'] = tb
        md.append('\n**Mission differences per alpha at 1 deg (rerun, NSE)**:')
        md.append('| alpha | GRACE-C | NGGM | MAGIC | NGGM - GRACE-C | MAGIC - GRACE-C | MAGIC - NGGM |')
        md.append('|---|---|---|---|---|---|---|')
        for a in C.ALPHAS:
            v = {m: new['figure8_series']['%s_%s' % (basin, m)]['%s_1degree' % a]['nse'] for m in C.MISSIONS}
            md.append('| %s | %.3f | %.3f | %.3f | %+.3f | %+.3f | %+.3f |' % (a, v['GRACEC'], v['NGGM'], v['MAGIC'], v['NGGM'] - v['GRACEC'],
                                                                         v['MAGIC'] - v['GRACEC'], v['MAGIC'] - v['NGGM']))
    write_csv(out / 'configs.csv', rows)
    write_json(out / 'summary.json', res)
    (out / 'table.md').write_text('\n'.join(md) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
        col = {'GRACEC': 'tab:blue', 'NGGM': 'tab:orange', 'MAGIC': 'tab:green'}
        mk = {'Brahmaputra': 'o', 'Danube': 's'}
        for i, n in enumerate(NAMES):
            for r in rows:
                ax[i].scatter(r['%s_pub' % n], r['%s_rerun' % n], s=12, c=col[r['mission']], marker=mk[r['basin']], alpha=0.7)
            lo = min(min(r['%s_pub' % n], r['%s_rerun' % n]) for r in rows)
            hi = max(max(r['%s_pub' % n], r['%s_rerun' % n]) for r in rows)
            ax[i].plot([lo, hi], [lo, hi], 'k-', lw=0.8)
            ax[i].set_xlabel('published (2 members) %s' % n.upper())
            ax[i].set_ylabel('rerun (30 members) %s' % n.upper())
        for m, c in col.items():
            ax[0].scatter([], [], c=c, label=m)
        for b, s in mk.items():
            ax[0].scatter([], [], c='grey', marker=s, label=b)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / 'scatter.png', dpi=150)
    except Exception as e:                                            # figure is optional
        print('no figure:', e)
    print((out / 'table.md').read_text())


if __name__ == '__main__':
    main()
