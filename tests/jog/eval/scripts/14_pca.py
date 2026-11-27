"""Step 14 (R3.6) - EOF / PC analysis of truth, model and DA (nominal seed-1 baselines, 1e18 / 1 deg).

Common 0.5 deg / 5-day grid, epochs 634..875 (2003-09-05..2006-12-27), per basin. Each field de-meaned per cell over
the period and weighted by sqrt(cos lat) (area weighting); EOFs from the SVD of the weighted (time x cell) matrix.
Reported for modes 1-3 of every product:
  explained variance (% of its own total variance);
  pattern congruence with the truth EOF of the same rank: sum(w e_p e_t) / sqrt(sum(w e_p^2) sum(w e_t^2)) (w = cos lat;
  the sign of each product mode is set so that this is positive) and the best |congruence| over truth modes 1-3;
  correlation of the PC with the truth PC of the same rank (after that sign alignment);
  correlation of the product field projected on the truth EOF with the truth PC (rank-independent check).
EOF maps are shown as covariance maps (EOF x std of PC, mm). One PNG per basin + pca.csv + meta.json.
Usage: python 14_pca.py [--seed 1]
"""
import time

import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import baselines as B, metrics as M, sources as S

NM = 3


def eof(x, w):
    """x (T, n) de-meaned; w (n,) area weights -> patterns (NM, n) unweighted, PCs (T, NM), explained (NM,)"""
    sw = np.sqrt(w)
    U, s, Vt = np.linalg.svd(x * sw[None], full_matrices=False)
    ev = s ** 2 / np.sum(s ** 2)
    pat = Vt[:NM] / sw[None]
    pcs = U[:, :NM] * s[:NM]
    return pat, pcs, ev[:NM]


def congruence(a, b, w):
    return float(np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b)))


def main():
    t0 = time.time()
    ap = parser(__doc__)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    out = outdir(args, '14_pca')
    epochs = np.arange(*C.PAPER_EPOCHS)
    t = S.epoch_dates()[epochs]
    rows, meta = [], {'seed': args.seed, 'epochs': [int(epochs[0]), int(epochs[-1])]}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for basin in args.basins:
        data = B.load(basin, epochs, seed=args.seed)
        ok = np.isfinite(data['truth']).all(0)
        lat, lon = data['lat'], data['lon']
        w = np.cos(np.radians(np.broadcast_to(lat[:, None], ok.shape)[ok]))
        prods = {'truth': data['truth'], **data['sims']}
        meta[basin] = {'cells': int(ok.sum()), 'missing': {k: len(v) for k, v in data['missing'].items()}}
        res = {}
        for k, f in prods.items():
            x = M.demean(f[:, ok])
            pat, pcs, ev = eof(x, w)
            res[k] = (pat, pcs, ev, x)
        tp, tpc, _, _ = res['truth']
        for k, (pat, pcs, ev, x) in res.items():
            for i in range(NM):
                c = congruence(pat[i], tp[i], w)
                if c < 0 and k != 'truth':
                    pat[i], pcs[:, i], c = -pat[i], -pcs[:, i], -c
                best = max(range(NM), key=lambda j: abs(congruence(pat[i], tp[j], w)))
                proj = (x * w[None]) @ tp[i] / np.sum(w * tp[i] ** 2)          # product projected on truth EOF i
                rows.append({'basin': basin, 'product': k, 'mode': i + 1, 'explained_var_pct': round(100 * ev[i], 2),
                             'congruence_same_rank': round(c, 4),
                             'best_truth_mode': best + 1, 'best_abs_congruence': round(abs(congruence(pat[i], tp[best], w)), 4),
                             'pc_corr_same_rank': round(float(np.corrcoef(pcs[:, i], tpc[:, i])[0, 1]), 4),
                             'proj_on_truth_eof_corr': round(float(np.corrcoef(proj, tpc[:, i])[0, 1]), 4)})
        # figure: rows products, cols EOF 1-3 (covariance maps) + PCs
        names = list(prods)
        fig, axs = plt.subplots(len(names), NM + 1, figsize=(17, 2.6 * len(names)),
                                gridspec_kw={'width_ratios': [1] * NM + [1.6]})
        for r, k in enumerate(names):
            pat, pcs, ev, _ = res[k]
            for i in range(NM):
                m = np.full(ok.shape, np.nan)
                cov = pat[i] * pcs[:, i].std()                                  # mm per 1 std of the PC
                m[ok] = cov
                lim = np.nanmax(np.abs(res['truth'][0][i] * res['truth'][1][:, i].std()))
                ax = axs[r, i]
                im = ax.pcolormesh(lon, lat, m, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='nearest')
                ax.set_aspect('equal')
                ax.tick_params(labelsize=6)
                ax.set_title('%s EOF%d (%.1f %%)%s' % (k.replace('GRACEC', 'GRACE-C'), i + 1, 100 * ev[i],
                             '' if k == 'truth' else ', cong %.2f' % congruence(pat[i], res['truth'][0][i], w)), fontsize=8)
                if r == len(names) - 1:
                    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.25, label='mm per PC std')
            ax = axs[r, NM]
            for i, col in zip(range(NM), ('#1f5fa8', '#d0632b', '#3a9a5b')):
                ax.plot(t, pcs[:, i] / pcs[:, i].std(), color=col, lw=1, label='PC%d' % (i + 1))
                if k != 'truth':
                    tt = res['truth'][1][:, i]
                    ax.plot(t, tt / tt.std(), color=col, lw=0.6, ls=':')
            ax.set_title('%s PCs (standardized; dotted = truth)' % k.replace('GRACEC', 'GRACE-C'), fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.legend(fontsize=7, ncol=3)
        fig.suptitle('%s: EOF 1-3 of truth, model (OL) and DA (nominal seed-%d baselines, 1e18 / 1 deg), '
                     '2003-09-05..2006-12-27' % (basin, args.seed), fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(out / ('pca_%s.png' % basin), dpi=110)
        plt.close(fig)
        print('done', basin, round(time.time() - t0, 1), 's', flush=True)
    write_csv(out / 'pca.csv', rows)
    meta['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'meta.json', meta)
    print('written', out, meta['run_time_s'], 's')


if __name__ == '__main__':
    main()
