"""Step 34 (R4.7) - can the innovations alone pick the configuration the truth picks?

For the 180 seed-1 reruns (6 alpha x 5 grids x 3 missions x 2 basins; cases jog_<M>_<basin>[_<alpha>_<grid>]_s1_nominal,
DA/innovations.h5): per configuration the mean NIS = mean over units x windows of (y - hx_mean)^2 / (hx_var + R_diag) and the
innovation RMS = sqrt(mean (y - hx_mean)^2) (script 16's definitions). Truth-based score: NSE (paper definition, spatial mean,
results/07_recompute_v1_rerun/summary_paper.json, figure8_series). Per basin and mission, the 30 configurations are ranked by
  (i) |mean NIS - 1| (closest to a consistent filter first), (ii) innovation RMS (smallest first), and by NSE (largest first);
Spearman rho of (i) and (ii) against the NSE ranking (rho > 0 = the innovation criterion orders the configurations like the
truth); the innovation-best configuration, its NSE and NSE rank, and whether it equals the truth-best or neighbours it (same alpha
or same grid). Note: innovation RMS is computed in each grid's own units, so it mixes the error of the field with the unit size.
Output: results/34_innovation_ranking/{configs.csv, table.md, summary.json}.
Usage: python 34_innovation_ranking.py
"""
import json

import h5py
import numpy as np
from scipy import stats

from _common import parser, outdir, write_csv, write_json, C


def innov(fn):
    with h5py.File(fn, 'r') as f:
        ks = sorted(f.keys())
        y = np.array([f[w]['y'][()] for w in ks])
        hx = np.array([f[w]['hx_mean'][()] for w in ks])
        var = np.array([f[w]['hx_var'][()] + f[w]['R_diag'][()] for w in ks])
    d = y - hx
    return float(np.mean(d ** 2 / var)), float(np.sqrt(np.mean(d ** 2))), len(ks), int(y.shape[1])


def main():
    p = parser(__doc__)
    args = p.parse_args()
    out = outdir(args, '34_innovation_ranking')
    nse = json.loads((C.OUT_ROOT / '07_recompute_v1_rerun' / 'summary_paper.json').read_text())['figure8_series']
    w3 = C.V1_DATA_ROOT / 'w3ra' / 'cases'
    rows, res = [], {}
    md = ['| basin | mission | rho(NIS, NSE) | rho(RMS d, NSE) | truth-best (NSE) | NIS-best: config, NSE, NSE rank | '
          'RMS-best: config, NSE, NSE rank |', '|---|---|---|---|---|---|---|']
    for basin in args.basins:
        for m in C.MISSIONS:
            recs = []
            for a in C.ALPHAS:
                for g in C.GRIDS:
                    cfg = '' if (a, g) == ('1e18', '1degree') else '_%s_%s' % (a, g)
                    fn = w3 / ('jog_%s_%s%s_s1_nominal' % (m, basin, cfg)) / 'DA' / 'innovations.h5'
                    nis, rms, nw, nu = innov(fn)
                    key = '%s_%s' % (a, g)
                    recs.append(dict(basin=basin, mission=m, alpha=a, grid=g, units=nu, windows=nw, nis=round(nis, 3),
                                     rms_d=round(rms, 2), nse=round(nse['%s_%s' % (basin, m)][key]['nse'], 4)))
            nse_rank = stats.rankdata([-r['nse'] for r in recs])
            nis_rank = stats.rankdata([abs(r['nis'] - 1) for r in recs])
            rms_rank = stats.rankdata([r['rms_d'] for r in recs])
            for r, a, b, c in zip(recs, nse_rank, nis_rank, rms_rank):
                r.update(rank_nse=int(a), rank_nis=float(b), rank_rms=float(c))
            rows += recs
            rho_nis = float(stats.spearmanr(nis_rank, nse_rank)[0])
            rho_rms = float(stats.spearmanr(rms_rank, nse_rank)[0])
            tb = recs[int(np.argmin(nse_rank))]

            def pick(rank):
                r = recs[int(np.argmin(rank))]
                rel = 'equal' if (r['alpha'], r['grid']) == (tb['alpha'], tb['grid']) else \
                      ('neighbour' if r['alpha'] == tb['alpha'] or r['grid'] == tb['grid'] else 'different')
                return r, rel
            bn, reln = pick(nis_rank)
            br, relr = pick(rms_rank)
            res['%s_%s' % (basin, m)] = dict(rho_nis_vs_nse=rho_nis, rho_rms_vs_nse=rho_rms,
                                             truth_best='%s_%s' % (tb['alpha'], tb['grid']),
                                             nis_best='%s_%s' % (bn['alpha'], bn['grid']), nis_best_relation=reln,
                                             rms_best='%s_%s' % (br['alpha'], br['grid']), rms_best_relation=relr)
            md.append('| %s | %s | %+.2f | %+.2f | %s_%s %.3f | %s_%s %.3f #%d (%s) | %s_%s %.3f #%d (%s) |' % (
                basin, m, rho_nis, rho_rms, tb['alpha'], tb['grid'], tb['nse'], bn['alpha'], bn['grid'], bn['nse'], bn['rank_nse'],
                reln, br['alpha'], br['grid'], br['nse'], br['rank_nse'], relr))
            print(md[-1], flush=True)
    write_csv(out / 'configs.csv', rows)
    write_json(out / 'summary.json', res)
    (out / 'table.md').write_text('\n'.join(md) + '\n')
    print('written', out)


if __name__ == '__main__':
    main()
