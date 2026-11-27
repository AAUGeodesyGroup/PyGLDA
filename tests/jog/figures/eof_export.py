"""EOF 1-3 of truth, model and DA (nominal 1e18 / 1 deg baselines, seed 1) for the appendix EOF figure.

Same computation as 14_pca.py (de-meaned per cell, sqrt(cos lat) weighting, SVD; product modes sign-aligned to the truth mode of
the same rank). Saves per basin <out>/eof_<basin>.npz: lat, lon, ok (valid 0.5 deg cells), dates, and per product k in
(truth, model, GRACEC, NGGM, MAGIC): map_<k> (3, nlat, nlon) covariance maps (EOF x std of the PC, mm), pc_<k> (T, 3) standardized
PCs, ev_<k> (3,) explained variance (%), cong_<k> (3,) congruence with the truth EOF of the same rank.
Usage (from jog_eval/scripts): python eof_export.py --out <dir>
"""
import importlib

import numpy as np

from _common import parser, C
from jog_eval import baselines as B, metrics as M, sources as S

P14 = importlib.import_module('14_pca')


def main():
    p = parser(__doc__)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    epochs = np.arange(*C.PAPER_EPOCHS)
    t = S.epoch_dates()[epochs]
    for basin in args.basins:
        data = B.load(basin, epochs, seed=args.seed)
        ok = np.isfinite(data['truth']).all(0)
        lat, lon = data['lat'], data['lon']
        w = np.cos(np.radians(np.broadcast_to(lat[:, None], ok.shape)[ok]))
        prods = {'truth': data['truth'], **data['sims']}
        res = {k: P14.eof(M.demean(f[:, ok]), w) for k, f in prods.items()}
        tp = res['truth'][0]
        out = {'lat': lat, 'lon': lon, 'ok': ok, 'dates': np.array([str(d) for d in t])}
        for k, (pat, pcs, ev) in res.items():
            pat, pcs = pat.copy(), pcs.copy()
            cong = np.ones(P14.NM)
            for i in range(P14.NM):
                c = P14.congruence(pat[i], tp[i], w)
                if c < 0 and k != 'truth':
                    pat[i], pcs[:, i], c = -pat[i], -pcs[:, i], -c
                cong[i] = c
            maps = np.full((P14.NM,) + ok.shape, np.nan)
            for i in range(P14.NM):
                maps[i][ok] = pat[i] * pcs[:, i].std()
            out['map_%s' % k] = maps
            out['pc_%s' % k] = pcs / pcs.std(0)
            out['ev_%s' % k] = 100 * ev
            out['cong_%s' % k] = cong
        np.savez(args.out + '/eof_%s.npz' % basin, **out)
        print(basin, {k: [round(float(x), 1) for x in out['ev_%s' % k]] for k in prods},
              {k: [round(float(x), 2) for x in out['cong_%s' % k]] for k in prods}, flush=True)


if __name__ == '__main__':
    main()
