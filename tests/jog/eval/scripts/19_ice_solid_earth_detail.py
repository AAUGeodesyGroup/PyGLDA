"""Step 19 (R4.1) - detail of the ice (I) + solid-Earth (S) part of the HIS truth and preparation of the R4.1 variant.

Needs the 5-day SH of I and S built by 09_ice_solid_earth.py (--phase build).
Phases
  detail    paper period (634..875): I+S basin trend split I/S x degrees 0-1 / 2-70; I+S effect on the cell RMSE
            (de-meaned, not detrended) of model and DA vs truth when the truth is HIS-(I+S) instead of HIS
  filter    is I+S in the simulated mission observations unattenuated? per-degree gain obs/truth (SH analysis of
            Global_EWH_05_<M>_1e+18 vs Global_EWH_05_HIS) and regression of the obs unit series on H and I+S
  prepare   (1) R4.1 observation products <data>/GRACE/output_R41/<M>/1e18/ = published products with the sub_basin_k
            series minus the I+S unit series (de-meaned over the 315 obs epochs 561..875), I+S of degrees 1..70 unfiltered
            (--is-filter no_deg0, default: the mission grids contain no degree 0); cov, gridded_signal copied
            (2) truth minus I+S: <components>/truth_minus_IS/Global_EWH_05_HIS_minus_IS.hdf5 and
            basin_signal_HIS_minus_IS_<basin>.hdf5 (Grid_1/HIS/EWHA, whole/HIS/EWHA)
Usage: python 19_ice_solid_earth_detail.py --phase detail filter prepare
"""
import hashlib, importlib.util, shutil, time
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import evaluate as E, metrics as MT, sources as S, sh_synthesis as SH

_spec = importlib.util.spec_from_file_location('s09', str(Path(__file__).with_name('09_ice_solid_earth.py')))
S09 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S09)

OBS_EPOCHS = (561, 876)                       # epochs of the published 1-degree obs products (2002-09-05..2006-12-23)
MISSIONS = {'GRACEC': 'GRACE-C-like', 'NGGM': 'NGGM', 'MAGIC': 'MAGIC'}
LMAX = 180


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def t_mid(epochs):
    st, _ = S.epoch_windows()
    return np.array([(st[k] - date(2003, 1, 1)).days + 2.5 for k in epochs]) / 365.25 + 2003.0


def comp_grids(root, epochs, degrees=None, comps=('I', 'S')):
    """{comp: (t, 360, 720) mm, south first} synthesized as the truth; degrees: slice of degrees kept"""
    syn = SH.Synthesis(LMAX); f = SH.published_factor(LMAX)
    if degrees is not None:
        g = np.zeros_like(f); g[degrees] = f[degrees]; f = g
    out = {}
    for comp in comps:
        Cc, Sc = S09.load_5day(root, comp, list(epochs))
        out[comp] = np.array([syn.grid(Cc[i], Sc[i], f) for i in range(len(epochs))]) * 1000.0
    return out


def coslat():
    return np.cos(np.deg2rad(np.arange(-89.75, 90, 0.5)))


def basin_mask(basin):
    return S.region_mask(basin)[3][::-1].astype(bool)          # south first


def wmean(x, mask):
    w = np.broadcast_to(coslat()[:, None], mask.shape)[mask]
    return x[:, mask] @ w / w.sum()


def unit_masks(basin):
    with h5py.File(C.basin_signal_file(basin), 'r') as h:
        m = h['Grid_1']['mask'][:].astype(bool)
        if h['Grid_1']['mask_lat'][0] > h['Grid_1']['mask_lat'][-1]:
            m = m[:, ::-1]
        g = h['Grid_1']['HIS']
        his = g['EWHA' if 'EWHA' in g else 'EWHA_HIS'][:]
    return m, his                                                 # (K, 360, 720) south first, (K, 876) m


def unit_series(x, masks, weighting='coslat'):
    """(t, 360, 720) -> (K, t) area means over each unit mask"""
    out = []
    for m in masks:
        w = np.broadcast_to(coslat()[:, None], m.shape)[m] if weighting == 'coslat' else np.ones(m.sum())
        out.append(x[:, m] @ w / w.sum())
    return np.array(out)


# ---- detail ------------------------------------------------------------------------------------------------------
def phase_detail(args, out):
    root = Path(args.components)
    epochs = np.arange(*C.PAPER_EPOCHS); t = t_mid(epochs)
    rows = []
    split = {}
    for lab, deg in (('all', None), ('deg0-1', slice(0, 2)), ('deg2-70', slice(2, 71))):
        split[lab] = comp_grids(root, epochs, deg)
    for basin in args.basins:
        mask = basin_mask(basin)
        for lab in split:
            for comp in ('I', 'S', 'I+S'):
                x = split[lab]['I'] + split[lab]['S'] if comp == 'I+S' else split[lab][comp]
                bm = wmean(x, mask)
                cells = x[:, mask]
                tr_b = S09.fit(t, (bm - bm.mean())[:, None])[0][0]
                tr_c = S09.fit(t, cells - cells.mean(0))[0]
                rows.append({'basin': basin, 'component': comp, 'degrees': lab,
                             'basin_trend_mm_yr': round(float(tr_b), 3),
                             'basin_RMS_mm': round(float(np.sqrt(((bm - bm.mean()) ** 2).mean())), 3),
                             'cell_trend_median': round(float(np.median(tr_c)), 3),
                             'cell_trend_max_abs': round(float(tr_c[np.argmax(np.abs(tr_c))]), 3)})
    write_csv(out / 'trend_split_I_S_degrees.csv', rows)
    is_all = split['all']['I'] + split['all']['S']
    del split

    # RMSE effect (as the DA metrics: per 0.5 deg cell, de-meaned series, RMSE = std of the difference)
    rrows = []
    for basin in args.basins:
        data = E.load_basin(basin, epochs)
        truth = data['truth']
        box, _, _, full = S.region_mask(basin)
        rws, cls = np.where(full)
        isg = np.flip(is_all, axis=-2)[:, rws.min():rws.max() + 1, cls.min():cls.max() + 1]
        ok = np.isfinite(truth).all(0)
        isg = np.where(ok[None], isg, np.nan)
        lat = data['lat']
        w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], ok.shape)[ok]
        rms_is = np.sqrt(np.nanmean(MT.demean(isg) ** 2, axis=0))[ok]
        rrows.append({'basin': basin, 'sim': 'I+S itself (RMS of de-meaned I+S)', 'cells': int(ok.sum()),
                      'mean_RMSE_HIS': round(float(rms_is.mean()), 3), 'mean_RMSE_HISminusIS': '',
                      'delta_mean': '', 'delta_median': round(float(np.median(rms_is)), 3),
                      'delta_max_abs': round(float(rms_is.max()), 3), 'delta_min': '',
                      'basin_series_RMSE_HIS': round(float(np.sqrt(((lambda s: s - s.mean())(isg[:, ok] @ w / w.sum()) ** 2).mean())), 3),
                      'basin_series_RMSE_HISminusIS': '', 'cells_RMSE_down': ''})
        t2 = truth - isg
        for name, sim in data['sims'].items():
            r1 = MT.rmse(truth, sim)[ok]; r2 = MT.rmse(t2, sim)[ok]
            d = r2 - r1
            b_sim = sim[:, ok] @ w / w.sum(); b1 = truth[:, ok] @ w / w.sum(); b2 = t2[:, ok] @ w / w.sum()
            rb1 = float(np.std((b_sim - b_sim.mean()) - (b1 - b1.mean())))
            rb2 = float(np.std((b_sim - b_sim.mean()) - (b2 - b2.mean())))
            rrows.append({'basin': basin, 'sim': name, 'cells': int(ok.sum()),
                          'mean_RMSE_HIS': round(float(r1.mean()), 3), 'mean_RMSE_HISminusIS': round(float(r2.mean()), 3),
                          'delta_mean': round(float(r2.mean() - r1.mean()), 3),
                          'delta_median': round(float(np.median(d)), 3),
                          'delta_max_abs': round(float(d[np.argmax(np.abs(d))]), 3), 'delta_min': round(float(d.min()), 3),
                          'basin_series_RMSE_HIS': round(rb1, 3), 'basin_series_RMSE_HISminusIS': round(rb2, 3),
                          'cells_RMSE_down': int((d < 0).sum())})
            print(rrows[-1], flush=True)
    write_csv(out / 'rmse_effect_truth_minus_IS.csv', rrows)


# ---- filter check ----------------------------------------------------------------------------------------------
_PC = {}
GAIN_EPOCHS = list(range(634, 876, 4))                            # 61 epochs of the paper period


def degree_gain(mission, eps=GAIN_EPOCHS, lmax=70):
    """per-degree gain of the simulated mission grid on the truth: sum_m,t (obs*truth) / sum_m,t truth^2 (n = 0..lmax);
    used clipped to [0, 1] (regularisation cannot amplify; degree 0 is absent from the mission solutions)"""
    num = np.zeros(lmax + 1); den = np.zeros(lmax + 1)
    with h5py.File(C.truth_grid_file(), 'r') as ht, h5py.File(C.obs_grid_file(mission, '1e18'), 'r') as ho:
        for k in eps:
            ct, st_ = analysis(ht['EWHA'][k], lmax); co, so = analysis(ho['EWHA'][k], lmax)
            num += (co * ct + so * st_).sum(1); den += (ct ** 2 + st_ ** 2).sum(1)
    return num / den


def filtered_factor(g):
    f = SH.published_factor(LMAX)
    out = np.zeros_like(f); out[:len(g)] = f[:len(g)] * np.clip(g, 0.0, 1.0)
    return out


def is_factor(mode, gain=None):
    """degree factors for the I+S removed from the observations: 'none' all degrees 0..70 (as the truth),
    'no_deg0' degrees 1..70 (the mission grids contain no degree 0), 'deg2' degrees 2..70, 'gain' x clipped degree gain"""
    f = SH.published_factor(LMAX).copy()
    if mode == 'no_deg0':
        f[0] = 0.0
    elif mode == 'deg2':
        f[:2] = 0.0
    elif mode == 'gain':
        f = filtered_factor(gain)
    return f


def comp_grids_factor(root, epochs, factor):
    syn = SH.Synthesis(LMAX)
    tot = 0.0
    for comp in ('I', 'S'):
        Cc, Sc = S09.load_5day(root, comp, list(epochs))
        tot = tot + np.array([syn.grid(Cc[i], Sc[i], factor) for i in range(len(epochs))]) * 1000.0
    return tot


def analysis(grid, lmax=70):
    """quadrature SH analysis of a (360, 720) south-first EWH grid -> (C, S) of the EWH field (no Love factors)"""
    lat = np.arange(-89.75, 90, 0.5); lon = np.deg2rad(np.arange(-179.75, 180, 0.5))
    if lmax not in _PC:
        _PC[lmax] = SH.legendre(lmax, lat)
    P = _PC[lmax]
    m = np.arange(lmax + 1)
    A = grid @ np.cos(np.outer(m, lon)).T * np.deg2rad(0.5)       # (lat, m)
    B = grid @ np.sin(np.outer(m, lon)).T * np.deg2rad(0.5)
    wl = np.cos(np.deg2rad(lat)) * np.deg2rad(0.5) / (4 * np.pi)
    return np.einsum('i,inm,im->nm', wl, P, A), np.einsum('i,inm,im->nm', wl, P, B)


def phase_filter(args, out):
    root = Path(args.components)
    eps = list(range(634, 876, 11))                               # 22 epochs over the paper period
    with h5py.File(C.truth_grid_file(), 'r') as h:
        tr = {k: h['EWHA'][k] for k in eps}
    isg = comp_grids(root, eps)
    isg = {k: (isg['I'][i] + isg['S'][i]) / 1000.0 for i, k in enumerate(eps)}
    spec_t = {k: analysis(tr[k]) for k in eps}
    spec_is = {k: analysis(isg[k]) for k in eps}
    rows = []
    bands = {'0': (0, 1), '1': (1, 2), '2-10': (2, 11), '11-30': (11, 31), '31-50': (31, 51), '51-70': (51, 71)}
    for M, MF in MISSIONS.items():
        with h5py.File(C.obs_grid_file(M, '1e18'), 'r') as h:
            ob = {k: h['EWHA'][k] for k in eps}
        spec_o = {k: analysis(ob[k]) for k in eps}
        for b, (n0, n1) in bands.items():
            num = sum((spec_o[k][0][n0:n1] * spec_t[k][0][n0:n1] + spec_o[k][1][n0:n1] * spec_t[k][1][n0:n1]).sum() for k in eps)
            den = sum((spec_t[k][0][n0:n1] ** 2 + spec_t[k][1][n0:n1] ** 2).sum() for k in eps)
            pis = sum((spec_is[k][0][n0:n1] ** 2 + spec_is[k][1][n0:n1] ** 2).sum() for k in eps)
            ptot = sum((spec_is[k][0][:71] ** 2 + spec_is[k][1][:71] ** 2).sum() for k in eps)
            rows.append({'mission': M, 'degrees': b, 'gain_obs_on_truth': round(float(num / den), 4),
                         'IS_power_share': round(float(pis / ptot), 4)})
    write_csv(out / 'filter_gain_per_degree_band.csv', rows)
    gains = {M: degree_gain(M) for M in MISSIONS}
    np.save(out / 'filter_gain_per_degree.npy', np.array([gains[M] for M in MISSIONS]))
    write_csv(out / 'filter_gain_per_degree.csv', [{'degree': n, **{M: round(float(gains[M][n]), 4) for M in MISSIONS}}
                                                   for n in range(71)])
    # regression of obs unit series on H = HIS-(I+S) and I+S (epochs 561..875)
    epochs = np.arange(*OBS_EPOCHS)
    g = comp_grids(root, epochs)
    isx = g['I'] + g['S']; del g
    modes = ('none', 'no_deg0', 'deg2', 'gain')
    isf = {(M, md): comp_grids_factor(root, epochs, is_factor(md, gains[M])) for M in MISSIONS for md in modes[1:]}
    reg = []
    for basin in args.basins:
        masks, his = unit_masks(basin)
        his_u = his[:, OBS_EPOCHS[0]:OBS_EPOCHS[1]] * 1000.0
        is_raw = unit_series(isx, masks)                          # mm
        for (M, MF), filt in [(x, f) for f in modes for x in MISSIONS.items()]:
            is_u = unit_series(isf[(M, filt)], masks) if filt != 'none' else is_raw
            with h5py.File(C.basin_signal_file(basin), 'r') as h:
                ob = h['Grid_1'][MF]['EWHA_alpha_1e+18'][:, OBS_EPOCHS[0]:OBS_EPOCHS[1]] * 1000.0
            y = (ob - ob.mean(1, keepdims=True)).ravel()
            hh = (his_u - is_raw); hh = (hh - hh.mean(1, keepdims=True)).ravel()       # H (unfiltered)
            hs = (his_u - his_u.mean(1, keepdims=True)).ravel()
            ii = (is_u - is_u.mean(1, keepdims=True)).ravel()
            X = np.column_stack([hh, ii])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ coef
            cov = np.linalg.inv(X.T @ X) * res.var()
            reg.append({'basin': basin, 'mission': M, 'IS': filt,
                        'units': masks.shape[0], 'gain_H': round(float(coef[0]), 4),
                        'gain_IS': round(float(coef[1]), 4), 'se_gain_IS_iid': round(float(np.sqrt(cov[1, 1])), 4),
                        'RMS_IS_units_mm': round(float(np.sqrt((ii ** 2).mean())), 3),
                        'RMS_obs_minus_HIS_mm': round(float(np.sqrt(((y - hs) ** 2).mean())), 3)})
            print(reg[-1], flush=True)
    write_csv(out / 'filter_regression_units.csv', reg)


# ---- degrees 0 and 1 in truth, observations and raw SH ---------------------------------------------------------------
def phase_lowdeg(args, out):
    """EWH coefficients (mm) of degrees 0 and 1 (C00, C10, C11, S11) from a d/o 70 quadrature analysis of the global
    grids (truth, missions 1e18) and from the raw SH (HIS 5-day files in HIS_check, I+S 5-day means x published factor)"""
    root = Path(args.components)
    st, _ = S.epoch_windows()
    eps = sorted(set(int(k) for k in np.linspace(634, 875, 10).round()) | set(S09.his_check_files(root)))
    f = SH.published_factor(LMAX)
    his_raw = S09.his_check_files(root)
    rows = []
    def lo(Cc, Ss):
        return [Cc[0, 0], Cc[1, 0], Cc[1, 1], Ss[1, 1]]
    files = {'truth HIS grid': C.truth_grid_file(), **{'%s grid' % M: C.obs_grid_file(M, '1e18') for M in MISSIONS}}
    hs = {k: h5py.File(v, 'r') for k, v in files.items()}
    for k in eps:
        Cc, Sc = S09.load_5day(root, 'I', [k]); Ci, Si = Cc[0], Sc[0]
        Cc, Sc = S09.load_5day(root, 'S', [k]); Cs, Ss_ = Cc[0], Sc[0]
        src = {name: analysis(h['EWHA'][k], 70) for name, h in hs.items()}
        src['raw I x factor'] = (Ci * f[:, None], Si * f[:, None])
        src['raw S x factor'] = (Cs * f[:, None], Ss_ * f[:, None])
        src['raw I+S x factor'] = ((Ci + Cs) * f[:, None], (Si + Ss_) * f[:, None])
        if k in his_raw:
            a, b = SH.read_icgem(his_raw[k]); src['raw HIS x factor'] = (a * f[:, None], b * f[:, None])
        for name, (a, b) in src.items():
            v = np.array(lo(a, b)) * 1000.0
            rows.append({'epoch': k, 'start': st[k].isoformat(), 'source': name,
                         'C00_mm': round(float(v[0]), 3), 'C10_mm': round(float(v[1]), 3),
                         'C11_mm': round(float(v[2]), 3), 'S11_mm': round(float(v[3]), 3)})
    write_csv(out / 'lowdeg_coefficients.csv', rows)
    # summary: per source, RMS over epochs of the coefficient and of its difference to the truth grid
    summ = []
    for name in sorted({r['source'] for r in rows}):
        for q in ('C00_mm', 'C10_mm', 'C11_mm', 'S11_mm'):
            v = np.array([r[q] for r in rows if r['source'] == name])
            d = np.array([r[q] - t[q] for r in rows if r['source'] == name
                          for t in rows if t['source'] == 'truth HIS grid' and t['epoch'] == r['epoch']])
            summ.append({'source': name, 'coef': q, 'n_epochs': len(v), 'rms': round(float(np.sqrt((v ** 2).mean())), 3),
                         'rms_minus_truth_grid': round(float(np.sqrt((d ** 2).mean())), 3),
                         'max_abs_minus_truth_grid': round(float(np.abs(d).max()), 3)})
    write_csv(out / 'lowdeg_summary.csv', summ)


# ---- prepare -----------------------------------------------------------------------------------------------------
def h5_items(fn):
    out = {}
    with h5py.File(fn, 'r') as h:
        def v(n, o):
            if isinstance(o, h5py.Dataset):
                out[n] = (o[()], o.dtype.str, o.shape, o.chunks, o.compression, dict(o.attrs))
        h.visititems(v)
        out['/attrs'] = dict(h.attrs)
    return out


def phase_prepare(args, out):
    root = Path(args.components)
    data_root = Path(args.data_root)
    st, en = S.epoch_windows()
    epochs = np.arange(*OBS_EPOCHS)
    g = comp_grids(root, epochs)
    isx = g['I'] + g['S']; del g                                  # (315, 360, 720) mm
    checks, files = [], []
    # weights of the published unit series: reproduce Grid_1/HIS from the truth grid (= the synthesis to 1e-5 mm)
    with h5py.File(C.truth_grid_file(), 'r') as h:
        his_grid = h['EWHA'][OBS_EPOCHS[0]:OBS_EPOCHS[1]]
        bg = h['EWH_background'][:]
    syn = SH.Synthesis(LMAX); f = SH.published_factor(LMAX)
    is_units = {}
    for basin in args.basins:
        masks, his = unit_masks(basin)
        for wgt in ('coslat', 'none'):
            u = unit_series(his_grid, masks, wgt)
            checks.append({'check': 'Grid_1/HIS unit series from truth grid, weighting %s' % wgt, 'basin': basin,
                           'max_abs_diff_mm': float(np.abs(u - his[:, OBS_EPOCHS[0]:OBS_EPOCHS[1]]).max() * 1000)})
        for k, fn in sorted(S09.his_check_files(root).items()):
            if not OBS_EPOCHS[0] <= k < OBS_EPOCHS[1]:
                continue
            gs = (syn.grid(*SH.read_icgem(fn), f) - bg)[None]
            u = unit_series(gs, masks)[:, 0]
            checks.append({'check': 'Grid_1/HIS unit series from own synthesis of %s, coslat' % fn.name, 'basin': basin,
                           'max_abs_diff_mm': float(np.abs(u - his[:, k]).max() * 1000)})
        print(checks[-3:], flush=True)
        is_units[basin] = unit_series(isx, masks)                 # (K, 315) mm
        for M, MF in MISSIONS.items():                            # obs unit series = same operator on the mission grid?
            with h5py.File(C.obs_grid_file(M, '1e18'), 'r') as h, h5py.File(C.basin_signal_file(basin), 'r') as b:
                u = unit_series(h['EWHA'][OBS_EPOCHS[0]:OBS_EPOCHS[1]], masks)
                ref = b['Grid_1'][MF]['EWHA_alpha_1e+18'][:, OBS_EPOCHS[0]:OBS_EPOCHS[1]]
            checks.append({'check': 'Grid_1/%s unit series from Global_EWH_05_%s_1e+18, coslat' % (MF, MF), 'basin': basin,
                           'max_abs_diff_mm': float(np.abs(u - ref).max() * 1000)})
    gains = {M: degree_gain(M) for M in MISSIONS}
    is_obs = {}
    for M in MISSIONS:
        xf = isx if args.is_filter == 'none' else comp_grids_factor(root, epochs, is_factor(args.is_filter, gains[M]))
        for basin in args.basins:
            is_obs[(M, basin)] = unit_series(xf, unit_masks(basin)[0])
    # (1) R4.1 observation products
    for M in MISSIONS:
        src = data_root / 'GRACE' / 'output' / M / '1e18'
        dst = data_root / 'GRACE' / 'output_R41' / M / '1e18'
        dst.mkdir(parents=True, exist_ok=True)
        for basin in args.basins:
            base = '%s1degree' % basin
            for kind in ('cov', 'gridded_signal', 'signal'):
                shutil.copy2(src / ('%s_%s.hdf5' % (base, kind)), dst / ('%s_%s.hdf5' % (base, kind)))
            fs, fd = src / (base + '_signal.hdf5'), dst / (base + '_signal.hdf5')
            u = is_obs[(M, basin)]
            u = u - u.mean(1, keepdims=True)
            with h5py.File(fd, 'r+') as h:
                te = [x.decode() if isinstance(x, bytes) else str(x) for x in h['time_epoch'][:]]
                exp = [(st[k] + timedelta(days=2)).isoformat() for k in epochs]
                if te != exp:
                    raise RuntimeError('time_epoch of %s does not match epochs 561..875' % fs)
                K = u.shape[0]
                for k in range(1, K + 1):
                    h['sub_basin_%d' % k][...] = h['sub_basin_%d' % k][...] - u[k - 1]
            a, b = h5_items(fs), h5_items(fd)
            same_other = all(np.array_equal(a[n][0], b[n][0]) and a[n][1:] == b[n][1:] for n in a
                             if not (n.startswith('sub_basin_') and n != 'sub_basin_area') and n != '/attrs')
            meta_same = set(a) == set(b) and a['/attrs'] == b['/attrs'] and all(a[n][1:] == b[n][1:] for n in a if n != '/attrs')
            dmax = max(float(np.abs((a['sub_basin_%d' % k][0] - b['sub_basin_%d' % k][0]) - u[k - 1]).max()) for k in range(1, K + 1))
            for kind in ('cov', 'gridded_signal'):
                checks.append({'check': '%s %s %s md5 differs from published (0 = identical)' % (M, basin, kind), 'basin': basin,
                               'max_abs_diff_mm': float(md5(src / ('%s_%s.hdf5' % (base, kind))) != md5(dst / ('%s_%s.hdf5' % (base, kind))))})
            checks.append({'check': '%s %s signal: non-sub_basin datasets identical=%s, names/dtypes/shapes/attrs identical=%s; '
                                    'max |(old-new) - I+S unit|' % (M, basin, same_other, meta_same),
                           'basin': basin, 'max_abs_diff_mm': dmax})
            print(checks[-1], flush=True)
            for kind in ('cov', 'gridded_signal', 'signal'):
                fn = dst / ('%s_%s.hdf5' % (base, kind))
                files.append({'file': str(fn), 'md5': md5(fn), 'bytes': fn.stat().st_size})
    # (2) truth minus I+S
    tdir = root / 'truth_minus_IS'
    tdir.mkdir(exist_ok=True)
    allep = np.arange(S09.FIRST_EPOCH, 876)
    with h5py.File(C.truth_grid_file(), 'r') as h, h5py.File(tdir / 'Global_EWH_05_HIS_minus_IS.hdf5', 'w') as o:
        ew = o.create_dataset('EWHA', shape=h['EWHA'].shape, dtype=h['EWHA'].dtype)
        ew[:S09.FIRST_EPOCH] = np.nan
        # anomaly of I+S w.r.t. its mean over epochs 73..875 (1995 epochs 0..72 not built: NaN)
        mean_is = None
        um = {b: unit_masks(b)[0] for b in args.basins}
        is_all_units = {b: [] for b in args.basins}; is_all_whole = {b: [] for b in args.basins}
        chunks = [allep[i:i + 100] for i in range(0, len(allep), 100)]
        for pas in (0, 1):
            acc = 0.0
            for ch in chunks:
                g = comp_grids(root, ch)
                x = (g['I'] + g['S']) / 1000.0
                if pas == 0:
                    acc = acc + x.sum(0)
                else:
                    ew[ch[0]:ch[-1] + 1] = h['EWHA'][ch[0]:ch[-1] + 1] - (x - mean_is)
                    for b in args.basins:
                        is_all_units[b].append(unit_series(x - mean_is, um[b]))
                        is_all_whole[b].append(unit_series(x - mean_is, basin_mask(b)[None])[0])
            if pas == 0:
                mean_is = acc / len(allep)
        for k in ('EWH_background', 'latitude', 'longitude', 'time'):
            h.copy(k, o)
        o.create_dataset('description', data=(
            'HIS minus (I+S) of ESM2.0, same layout as Global_EWH_05_HIS.hdf5. EWHA [m] = EWHA(HIS) - (I+S - mean(I+S over '
            'epochs 73..875)); I+S synthesized like the truth (degrees 0..70, Wang Love numbers, R=6378136.3, rho_e=5517); '
            'epochs 0..72 (1995) = NaN (I+S not built). EWH_background copied from HIS. jog_eval 19_ice_solid_earth_detail.py'))
    files.append({'file': str(tdir / 'Global_EWH_05_HIS_minus_IS.hdf5'), 'md5': md5(tdir / 'Global_EWH_05_HIS_minus_IS.hdf5'),
                  'bytes': (tdir / 'Global_EWH_05_HIS_minus_IS.hdf5').stat().st_size})
    if True:
        for basin in args.basins:
            masks, his = unit_masks(basin)
            u = his[:, S09.FIRST_EPOCH:] - np.concatenate(is_all_units[basin], axis=1)    # m, epochs 73..875
            fn = tdir / ('basin_signal_HIS_minus_IS_%s.hdf5' % basin)
            with h5py.File(C.basin_signal_file(basin), 'r') as b, h5py.File(fn, 'w') as o:
                full = np.full(his.shape, np.nan); full[:, S09.FIRST_EPOCH:] = u
                o.create_dataset('Grid_1/HIS/EWHA', data=full)
                for k in ('mask', 'mask_lat', 'mask_lon'):
                    b.copy(b['Grid_1'][k], o['Grid_1'], k)
                b.copy('dates', o)
                his_w, _ = S09.his_whole(basin)
                wser = np.full(876, np.nan); wser[S09.FIRST_EPOCH:] = his_w[S09.FIRST_EPOCH:] / 1000.0 - np.concatenate(is_all_whole[basin])
                o.create_dataset('whole/HIS/EWHA', data=wser[None])
                o.create_dataset('description', data=[(
                    'HIS minus (I+S) [m]: Grid_1/HIS/EWHA = published basin_signal.hdf5 Grid_1/HIS minus the cos-lat weighted '
                    'means of the I+S anomaly grid (as in Global_EWH_05_HIS_minus_IS.hdf5) over the Grid_1 masks; whole/HIS/EWHA = '
                    'published basin_signal_whole_basin.hdf5 whole/HIS minus I+S over globe/global_mask/<basin>/0.5; '
                    'epochs 0..72 NaN').encode()])
                # check: (HIS - this) = I+S units
                d = (his[:, OBS_EPOCHS[0]:OBS_EPOCHS[1]] - full[:, OBS_EPOCHS[0]:OBS_EPOCHS[1]]) * 1000
                d = d - d.mean(1, keepdims=True)
                r = is_units[basin] - is_units[basin].mean(1, keepdims=True)
                checks.append({'check': 'basin_signal HIS - HIS_minus_IS (de-meaned) vs I+S units', 'basin': basin,
                               'max_abs_diff_mm': float(np.abs(d - r).max())})
            files.append({'file': str(fn), 'md5': md5(fn), 'bytes': fn.stat().st_size})
    write_csv(out / 'prepare_checks.csv', checks)
    write_csv(out / 'prepare_files.csv', files)
    write_json(out / 'prepare_settings.json', {'is_filter': args.is_filter,
                                               'gains_deg0_1': {M: [float(g) for g in gains[M][:2]] for M in MISSIONS}})
    write_json(out / 'prepare_IS_unit_series_rms_obs_removed.json',
               {'%s_%s' % mb: {'RMS_mm_per_unit_median': float(np.median(np.sqrt(((u - u.mean(1, keepdims=True)) ** 2).mean(1)))),
                               'RMS_mm_per_unit_max': float(np.max(np.sqrt(((u - u.mean(1, keepdims=True)) ** 2).mean(1))))}
                for mb, u in is_obs.items()})
    write_json(out / 'prepare_IS_unit_series_rms.json',
               {b: {'RMS_mm_per_unit_median': float(np.median(np.sqrt(((u - u.mean(1, keepdims=True)) ** 2).mean(1)))),
                    'RMS_mm_per_unit_max': float(np.max(np.sqrt(((u - u.mean(1, keepdims=True)) ** 2).mean(1))))}
                for b, u in is_units.items()})


def main():
    p = parser(__doc__)
    p.add_argument('--phase', nargs='+', default=['detail'], choices=['detail', 'filter', 'prepare', 'lowdeg'])
    p.add_argument('--components', default=str(C.ESM2_ROOT / 'components'))
    p.add_argument('--data-root', default=str(C.V1_DATA_ROOT))
    p.add_argument('--is-filter', default='no_deg0', choices=['no_deg0', 'none', 'deg2', 'gain'],
                   help='I+S removed from the obs: degrees 1..70 (default; the mission grids have no degree 0), all '
                        'degrees, degrees 2..70, or x per-degree gain')
    args = p.parse_args()
    out = outdir(args, '19_ice_solid_earth_detail')
    for ph in args.phase:
        t0 = time.time()
        {'detail': phase_detail, 'filter': phase_filter, 'prepare': phase_prepare, 'lowdeg': phase_lowdeg}[ph](args, out)
        print('phase %s: %.1f s' % (ph, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
