"""Step 30 - low degrees (C00, C10, C11, S11, C20) of the observations: error analysis (A) and variant R01 (B).

The ESA-SING release replaces C00, C10, C11, S11 by the HIS reference and C20 by an SLR-like series (HIS C20 + white noise,
sigma 2e-11 per 5-day epoch); the pre-release v5 obs used in the paper have mission-estimated low degrees.
Coefficients: projection of the global 0.5 deg EWH grids (obs Global_EWH_05_<mission>_<alpha>, truth Global_EWH_05_HIS,
epochs 561..875 = the DA obs epochs) on the fully normalized spherical harmonics n <= 2 (cos-lat quadrature), i.e. EWH
coefficients E_nm = f_n C_nm (f_n = published_factor; check: a synthesized Y20 field is recovered).
Unit operator U: cos-lat weighted mean over the 0.5 deg cells of each 1-degree DA unit (shapefile of the unit set), matched to
the published unit series by value (as script 08); check: U(obs grid) vs the published <basin>1degree_signal.hdf5 series.
A. per basin / mission / alpha (1e18, 1e17): RMS of dE20 = E20(obs) - E20(truth) (= RMS over the sphere of its EWH field, mm);
   low-degree error field L = synth(obs - truth; C00, C10, C11, S11, C20) on the units: RMS per unit, basin mean, and its share
   of the total unit error variance var(U(L)) / var(U(obs - truth)) per unit (median, range); series de-meaned.
B. (--write-r01) GRACE/output_R01/<M>/1e18/<basin>1degree_{signal,gridded_signal,cov}.hdf5: copies of the nominal product
   (GRACE/output/<M>_old/1e18), sub_basin_k of the signal file + U(synth(new - obs; low degrees)), new = truth C00/C10/C11/S11
   and truth C20 + N(0, 2e-11) (Stokes; seed 30; one series for all missions and basins); gridded_signal and cov unchanged.
   Check: (R01 - nominal) - U(correction) = 0 and its RMS per unit.
Output: results/30_low_degrees/{summary.json, table.md}.
Usage: python 30_low_degrees.py [--basins ...] [--write-r01]
"""
import hashlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from _common import parser, outdir, write_json, C
from jog_eval import sh_synthesis as SH, sources as S

DA_EPOCHS = (561, 876)
LOW = [(0, 0, 'C'), (1, 0, 'C'), (1, 1, 'C'), (1, 1, 'S'), (2, 0, 'C')]
SIGMA_C20 = 2e-11
SEED = 30
RES = 0.5


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class Low:
    """projection on / synthesis from n <= 2 on the global 0.5 deg grid, latitude south first (as the files)"""

    def __init__(self):
        self.lat = np.arange(-90 + RES / 2, 90, RES)
        self.lon = np.arange(-180 + RES / 2, 180, RES)
        self.P = SH.legendre(2, self.lat)                                 # (nlat, n, m)
        self.w = np.cos(np.deg2rad(self.lat)) * np.deg2rad(RES) ** 2 / (4 * np.pi)
        m = np.arange(3)
        ml = np.deg2rad(np.outer(m, self.lon))
        self.cos, self.sin = np.cos(ml), np.sin(ml)

    def basis(self, n, m, cs):
        return self.P[:, n, m][:, None] * (self.cos[m] if cs == 'C' else self.sin[m])[None, :]

    def project(self, F):
        """F (T, nlat, nlon) -> {(n, m, cs): (T,)}"""
        return {k: np.einsum('tij,ij->t', F, self.basis(*k) * self.w[:, None]) for k in LOW}

    def synth(self, coef):
        return sum(coef[k][:, None, None] * self.basis(*k)[None] for k in LOW)


def unit_operator(basin, low):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))    # repository root
    from src_DA.shp2mask import basin_shp_process
    unit = '%s1degree' % basin
    shp = str(C.V1_DATA_ROOT / 'basin' / 'shp' / 'ESA_SING' / 'Grid_1' / ('%s_subbasins.shp' % unit))
    bs = basin_shp_process(res=RES, basin_name=unit).shp_to_mask(shp_path=shp, issave=False)
    keys = sorted([k for k in bs.mask if k != 0])
    masks = [np.flip(bs.mask[k].astype(bool), axis=0) for k in keys]          # north first -> south first
    cw = np.cos(np.deg2rad(low.lat))[:, None] * np.ones((1, len(low.lon)))
    W = np.array([np.where(mk, cw, 0.0).ravel() for mk in masks])
    W /= W.sum(1, keepdims=True)
    return lambda F: F.reshape(F.shape[0], -1) @ W.T                        # (T, units)


def load_grid(fn, ep):
    with h5py.File(fn, 'r') as h:
        x = h['EWHA'][ep[0]:ep[-1] + 1]
    return np.asarray(x, float) * 1000.0                                    # m -> mm, south first


def load_signal(fn):
    with h5py.File(fn, 'r') as h:
        keys = sorted([k for k in h if k.startswith('sub_basin_') and k != 'sub_basin_area'], key=lambda s: int(s[10:]))
        return keys, np.stack([h[k][:] for k in keys], 1)


def match(U_obs, pub):
    """index of the unit-operator column for each published sub_basin_k (by value) and the max abs difference"""
    idx, worst = [], 0.0
    for j in range(pub.shape[1]):
        d = np.abs(U_obs - pub[:, [j]]).max(0)
        idx.append(int(np.argmin(d)))
        worst = max(worst, float(d.min()))
    assert len(set(idx)) == len(idx), 'no 1:1 unit mapping'
    return np.array(idx), worst


def dm(x):
    return x - x.mean(0)


def main():
    p = parser(__doc__)
    p.add_argument('--write-r01', action='store_true')
    p.add_argument('--alphas', nargs='+', default=['1e18', '1e17'])
    args = p.parse_args()
    out = outdir(args, '30_low_degrees')
    ep = np.arange(*DA_EPOCHS)
    low = Low()
    f2 = SH.published_factor()[2] * 1000.0                                # Stokes -> EWH coefficient (mm) at n = 2
    test = {k: np.zeros(1) for k in LOW}
    test[(2, 0, 'C')] = np.ones(1)
    rec = low.project(low.synth(test))
    res = {'self_check_Y20': {'%d%d%s' % k: float(v[0]) for k, v in rec.items()}, 'factor_n2_mm': f2}
    print('self-check Y20 ->', res['self_check_Y20'], flush=True)
    noise = np.random.default_rng(SEED).normal(0.0, SIGMA_C20, len(ep)) * f2   # EWH coefficient (mm)
    truth = load_grid(C.truth_grid_file(), ep)
    ct = low.project(truth)
    rows = ['| basin | mission | alpha | RMS dE20 mm | low-deg unit RMS mm median (range) | basin-mean RMS mm | share of '
            'unit error var median (range) | U check max diff mm |', '|---|---|---|---|---|---|---|---|']
    for basin in args.basins:
        U = unit_operator(basin, low)
        Ut = U(truth)
        for alpha in args.alphas:
            for m in C.MISSIONS:
                if not C.obs_grid_file(m, alpha).exists():
                    print('skip', basin, m, alpha, ': no obs grid', C.obs_grid_file(m, alpha), flush=True)
                    res['%s_%s_%s' % (basin, m, alpha)] = 'no obs grid on this machine: %s' % C.obs_grid_file(m, alpha)
                    continue
                obs = load_grid(C.obs_grid_file(m, alpha), ep)
                co = low.project(obs)
                Uo = U(obs)
                sig = C.V1_DATA_ROOT / 'GRACE' / 'output' / ('%s_old' % m) / alpha / ('%s1degree_signal.hdf5' % basin)
                keys, pub = load_signal(sig)
                idx, worst = match(dm(Uo), dm(pub))
                d = {k: co[k] - ct[k] for k in LOW}
                L = U(low.synth(d))[:, idx]
                tot = dm(Uo - Ut)[:, idx]
                rms_u = np.sqrt(np.mean(dm(L) ** 2, 0))
                share = np.var(L, 0) / np.var(tot, 0)
                bm = dm(L).mean(1)
                r = {'rms_dE20_mm': float(np.std(d[(2, 0, 'C')])),
                     'rms_dE_mm': {'%d%d%s' % k: float(np.std(v)) for k, v in d.items()},
                     'unit_rms_mm': [float(np.median(rms_u)), float(rms_u.min()), float(rms_u.max())],
                     'basin_mean_rms_mm': float(np.sqrt(np.mean(bm ** 2))),
                     'share_var': [float(np.median(share)), float(share.min()), float(share.max())],
                     'unit_operator_check_max_abs_mm': worst}
                res['%s_%s_%s' % (basin, m, alpha)] = r
                rows.append('| %s | %s | %s | %.2f | %.2f (%.2f-%.2f) | %.2f | %.3f (%.3f-%.3f) | %.2e |' % (
                    basin, m, alpha, r['rms_dE20_mm'], *r['unit_rms_mm'], r['basin_mean_rms_mm'], *r['share_var'], worst))
                print(rows[-1], flush=True)
                if args.write_r01 and alpha == '1e18':
                    new = {k: ct[k].copy() for k in LOW}
                    new[(2, 0, 'C')] = ct[(2, 0, 'C')] + noise
                    corr = U(low.synth({k: new[k] - co[k] for k in LOW}))[:, idx]        # (T, sub_basin order)
                    src = sig.parent
                    dst = C.V1_DATA_ROOT / 'GRACE' / 'output_R01' / m / '1e18'
                    dst.mkdir(parents=True, exist_ok=True)
                    for kind in ('signal', 'gridded_signal', 'cov'):
                        fn = '%s1degree_%s.hdf5' % (basin, kind)
                        if (dst / fn).exists():
                            (dst / fn).unlink()
                        shutil.copy2(src / fn, dst / fn)
                    with h5py.File(dst / ('%s1degree_signal.hdf5' % basin), 'r+') as h:
                        for j, k in enumerate(keys):
                            ds = h[k]
                            ds[...] = (ds[()] + corr[:, j]).astype(ds.dtype)
                    k2, r01 = load_signal(dst / ('%s1degree_signal.hdf5' % basin))
                    assert k2 == keys
                    chk = float(np.max(np.abs((r01 - pub) - corr)))
                    same = all(md5(dst / ('%s1degree_%s.hdf5' % (basin, kd))) == md5(src / ('%s1degree_%s.hdf5' % (basin, kd)))
                               for kd in ('gridded_signal', 'cov'))
                    res['R01_%s_%s' % (basin, m)] = {
                        'dst': str(dst), 'check_max_abs_mm': chk, 'gridded_cov_identical': same,
                        'change_rms_mm_median_range': [float(np.median(np.sqrt(np.mean((r01 - pub) ** 2, 0)))),
                                                       float(np.min(np.sqrt(np.mean((r01 - pub) ** 2, 0)))),
                                                       float(np.max(np.sqrt(np.mean((r01 - pub) ** 2, 0))))],
                        'signal_md5': md5(dst / ('%s1degree_signal.hdf5' % basin))}
                    print('R01', basin, m, res['R01_%s_%s' % (basin, m)], flush=True)
    res['noise'] = {'seed': SEED, 'sigma_stokes': SIGMA_C20, 'sigma_ewh_mm': SIGMA_C20 * f2, 'n_epochs': len(ep),
                    'realized_std_mm': float(np.std(noise))}
    write_json(out / 'summary.json', res)
    (out / 'table.md').write_text('\n'.join(rows) + '\n')
    print('written', out)


if __name__ == '__main__':
    main()
