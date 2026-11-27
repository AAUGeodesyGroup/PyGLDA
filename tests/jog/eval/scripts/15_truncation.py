"""Step 15 (R3.2) - truncate model and DA (and the observation grids) to SH degree/order 70 like the truth.

The truth (Global_EWH_05_HIS) is a d/o 70 synthesis incl. degrees 0-1 (sh_synthesis.published_factor). An SH analysis
of an EWH grid with the inverse of those degree factors followed by the synthesis with the same factors is the
orthogonal projection of the EWH field on the spherical harmonics of degree 0..70 (the Love/density factors cancel),
so the truncation is done as that projection:
    C_nm, S_nm = 1/(4 pi) sum_cells F(lat, lon) Pnm(sin lat) {cos, sin}(m lon) cos(lat) dlat dlon   (0.5 deg global grid)
    F70 = sum_{n<=70} Pnm (C_nm cos m lon + S_nm sin m lon)              (sh_synthesis.legendre, fully normalized)
Fields are anomalies (de-meaned per cell over the period) and embedded in the global 0.5 deg grid in two ways:
  'zero'  : product inside the evaluated basin cells, 0 everywhere else (the method suggested in the review);
  'truth' : product inside the basin cells, the truth anomaly everywhere else (outside the basin and globally).
'zero' introduces an artificial step at the basin edge (full signal -> 0), whose truncation rings back into the basin;
'truth' only has the step (product - truth), i.e. it emulates a global model that equals the truth outside the basin -
as the truth is band-limited, this equals truncating the error field (product - truth) set to 0 outside the
          basin, which dilutes the error near the edges (optimistic);
'nnbuf' : as 'truth', but the error is extended by nearest neighbour into a 3 deg buffer around the basin cells before
          the truncation (removes most of the edge dilution; preferred estimate).
Self-check: the truth itself (band-limited to 70) truncated with the same code ('truth' embedding = exact global truth).
Table-1 numbers (paper definition, truth unchanged) for model, the 3 nominal seed-1 baselines and the 3 observation
grids (1e18), before and after truncation; ranking of the missions.
Usage: python 15_truncation.py [--seed 1]
"""
import time

import h5py
import numpy as np
from scipy import ndimage

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import baselines as B, metrics as M, sources as S, sh_synthesis as SH

LMAX = 70
BUF = 6                     # 3 deg buffer (about half the d/o-70 half-wavelength of ~2.6 deg)
RES = 0.5
MODES = ('zero', 'truth', 'nnbuf')


class Projector:
    def __init__(self, lmax=LMAX):
        self.lat = np.arange(90 - RES / 2, -90, -RES)                 # north first (as the masks)
        self.lon = np.arange(-180 + RES / 2, 180, RES)
        self.P = SH.legendre(lmax, self.lat)                           # (nlat, n, m)
        m = np.arange(lmax + 1)
        ml = np.deg2rad(np.outer(self.lon, m))                         # (nlon, m)
        self.cos, self.sin = np.cos(ml), np.sin(ml)
        d = np.deg2rad(RES)
        self.w = np.cos(np.deg2rad(self.lat)) * d * d / (4 * np.pi)    # quadrature weight per cell / 4 pi

    def analyse(self, F, rows=None, cols=None):
        """F (T, nrows, ncols) on latitude rows / longitude columns (default all; zero elsewhere) -> C, S (T, n, m)"""
        rows = np.arange(len(self.lat)) if rows is None else rows
        cols = np.arange(len(self.lon)) if cols is None else cols
        a = F @ self.cos[cols]                                         # (T, rows, m)
        b = F @ self.sin[cols]
        Pw = self.P[rows] * self.w[rows][:, None, None]
        return np.einsum('inm,tim->tnm', Pw, a), np.einsum('inm,tim->tnm', Pw, b)

    def synthesise(self, Cc, Ss, rows, cols):
        P = self.P[rows]
        A = np.einsum('inm,tnm->tim', P, Cc)
        Bs = np.einsum('inm,tnm->tim', P, Ss)
        return A @ self.cos[cols].T + Bs @ self.sin[cols].T


def global_truth_anomaly(epochs):
    with h5py.File(C.truth_grid_file(), 'r') as h:
        x = h['EWHA'][epochs[0]:epochs[-1] + 1]
    x = np.flip(x, axis=-2) * 1000.0                                   # north first, mm
    return x - x.mean(0)


def main():
    t0 = time.time()
    ap = parser(__doc__)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    out = outdir(args, '15_truncation')
    epochs = np.arange(*C.PAPER_EPOCHS)
    pj = Projector()
    G = global_truth_anomaly(epochs)
    rows_out, check, meta = [], [], {'lmax': LMAX, 'seed': args.seed}
    for basin in args.basins:
        data = B.load(basin, epochs, seed=args.seed)
        box, lat, lon, full = S.region_mask(basin)
        r, c = np.where(full)
        R = np.arange(r.min(), r.max() + 1)
        Cl = np.arange(c.min(), c.max() + 1)
        assert np.allclose(pj.lat[R], lat) and np.allclose(pj.lon[Cl], lon)
        ok = np.isfinite(data['truth']).all(0)
        truth = data['truth']
        prods = dict(data['sims'])
        for m in C.MISSIONS:
            o, _, _ = S.obs_grid(basin, m, '1e18', epochs)
            prods['obs_' + m] = np.where(ok[None], o, np.nan)
        meta[basin] = {'cells': int(ok.sum())}

        def truncate(x, mode):
            a = np.where(ok[None], M.demean(x), 0.0)
            if mode == 'zero':
                Cc, Ss = pj.analyse(a, R, Cl)
            elif mode == 'nnbuf':
                # error (product - truth) extended by nearest neighbour to a buffer of BUF cells, truth elsewhere
                Rp = np.arange(max(R[0] - BUF, 0), min(R[-1] + BUF + 1, len(pj.lat)))
                Cp = np.arange(Cl[0] - BUF, Cl[-1] + BUF + 1) % len(pj.lon)
                okp = np.zeros((len(Rp), len(Cp)), bool)
                r0, c0 = R[0] - Rp[0], BUF
                okp[r0:r0 + len(R), c0:c0 + len(Cl)] = ok
                dist, (ii, jj) = ndimage.distance_transform_edt(~okp, return_indices=True)
                err = np.zeros((a.shape[0],) + okp.shape)
                e_box = a - G[:, R[0]:R[-1] + 1, Cl[0]:Cl[-1] + 1]
                err[:, r0:r0 + len(R), c0:c0 + len(Cl)] = np.where(ok[None], e_box, 0.0)
                ext = err[:, ii, jj] * (dist <= BUF)[None]
                Fg = G.copy()
                Fg[:, Rp[:, None], Cp[None, :]] += ext
                Cc, Ss = pj.analyse(Fg)
                del Fg
            else:
                Fg = G.copy()
                sub = Fg[:, R[0]:R[-1] + 1, Cl[0]:Cl[-1] + 1]
                sub[:, ok] = a[:, ok]
                Cc, Ss = pj.analyse(Fg)
                del Fg
            y = pj.synthesise(Cc, Ss, R, Cl)
            return np.where(ok[None], y, np.nan)

        # self-check: truncating the (d/o 70) truth
        for mode in MODES:
            tt = truncate(truth, mode)
            d = tt - M.demean(truth)
            mt = M.all_metrics(truth, tt, 'paper')
            check.append({'basin': basin, 'embedding': mode, 'max_abs_diff_mm': round(float(np.nanmax(np.abs(d))), 3),
                          'rms_diff_mm': round(float(np.sqrt(np.nanmean(d ** 2))), 3),
                          'nse': round(M.spatial_mean(mt['nse'], 'nse'), 4), 'rmse': round(M.spatial_mean(mt['rmse'], 'rmse'), 3)})
        for name, x in prods.items():
            base = M.all_metrics(truth, x, 'paper')
            row = {'basin': basin, 'product': name}
            for k in ('nse', 'kge', 'rmse'):
                row[k + '_orig'] = round(M.spatial_mean(base[k], k), 4)
            for mode in MODES:
                xt = truncate(x, mode)
                mt = M.all_metrics(truth, xt, 'paper')
                for k in ('nse', 'kge', 'rmse'):
                    v = M.spatial_mean(mt[k], k)
                    row['%s_%s' % (k, mode)] = round(v, 4)
                    row['d%s_%s' % (k, mode)] = round(v - row[k + '_orig'], 4)
                row['rms_change_field_%s_mm' % mode] = round(float(np.sqrt(np.nanmean((xt - M.demean(x)) ** 2))), 3)
            rows_out.append(row)
            print(basin, name, round(time.time() - t0, 1), 's', flush=True)
        for var in ('orig',) + MODES:
            for k in ('nse', 'kge', 'rmse'):
                vals = {r['product']: r['%s_%s' % (k, var)] for r in rows_out if r['basin'] == basin and r['product'] in C.MISSIONS}
                order = sorted(vals, key=lambda p: vals[p], reverse=(k != 'rmse'))
                meta[basin]['ranking_%s_%s' % (k, var)] = '>'.join(order)
    write_csv(out / 'truncation_table1.csv', rows_out)
    write_csv(out / 'selfcheck_truth.csv', check)
    meta['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'meta.json', meta)
    print('written', out, meta['run_time_s'], 's')


if __name__ == '__main__':
    main()
