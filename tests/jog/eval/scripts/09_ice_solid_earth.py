"""Step 9 (reviewer: share of the ice (I) and solid-Earth (S) components in the HIS truth).

ESM2.0 components I and S (6-hourly, d/o 180, ICGEM, yearly tar archives mtm_<year>_<I|S>.tar.gz from GFZ ISDC,
DOI 10.5880/GFZ.1.3.2014.001) are averaged to the 5-day epochs of the truth and synthesized to 0.5 deg EWH with the
method that reproduces Global_EWH_05_HIS.hdf5 (jog_eval.sh_synthesis.published_factor: degrees 0..70, Wang load Love
numbers, R = 6378136.3 m, rho_e = 5517, rho_w = 1000; no smoothing). 5-day mean of epoch [s, s+5 d) of the truth =
mean of the 21 six-hourly fields s 00:00 .. s+5 d 00:00 (reproduces the ESM2.0 5-day HIS files exactly).

Phases
  build     tar archives -> <components>/<I|S>/ESM2_<I|S>_5day_SH_d180.h5 (epochs 73..875 = 1995-12-31..2006-12-23)
  check     H+I+S vs HIS 5-day (needs <components>/H_check/mtm_2004_H.tar.gz and HIS_check/ 2004-05 files)
  analyse   validation of the synthesis (HIS_check files vs the truth grid), basin / cell statistics for the paper
            period (epochs 634..875), shares of I+S in HIS, trend maps
Usage: python 09_ice_solid_earth.py --phase build check analyse [--nproc 8]
"""
import io, json, re, sys, tarfile, time
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import sources as S, sh_synthesis as SH

LMAX = 180
FIRST_EPOCH = 73                     # 1995-12-31; epochs 0..72 (1995) follow another file convention, not needed
COMPS = ('I', 'S')
MAP_BOX = {'Brahmaputra': (75, 110, -5, 40), 'Danube': (0, 45, 35, 72)}   # lon0, lon1, lat0, lat1
TS_RE = re.compile(r'_(\d{8})_(\d{2})\.gfc$')


def comp_root(args):
    return Path(args.components)


# ---- build -----------------------------------------------------------------------------------------------------
def parse_gfc(text):
    head, _, body = text.partition('end_of_head')
    body = body.split('\n', 1)[1].replace('gfc', ' ')
    a = np.array(body.split(), dtype=float).reshape(-1, 4)
    n, m = a[:, 0].astype(int), a[:, 1].astype(int)
    Cn = np.zeros((LMAX + 1, LMAX + 1)); Sn = np.zeros((LMAX + 1, LMAX + 1))
    Cn[n, m] = a[:, 2]; Sn[n, m] = a[:, 3]
    return Cn, Sn


def epoch_table():
    s, e = S.epoch_windows()
    ks = np.arange(FIRST_EPOCH, len(s))
    return ks, s[ks], e[ks]


def _worker(task):
    tarfn, wanted = task                               # wanted: {epoch: start date}
    acc = {}
    with tarfile.open(tarfn, 'r:gz') as tf:
        for mem in tf:
            mt = TS_RE.search(mem.name)
            if not mem.isfile() or not mt:
                continue
            t = datetime.strptime(mt.group(1) + mt.group(2), '%Y%m%d%H')
            hits = [k for k, s0 in wanted.items()
                    if datetime(s0.year, s0.month, s0.day) <= t <= datetime(s0.year, s0.month, s0.day) + timedelta(days=5)]
            if not hits:
                continue
            Cn, Sn = parse_gfc(tf.extractfile(mem).read().decode())
            for k in hits:
                if k not in acc:
                    acc[k] = [np.zeros_like(Cn), np.zeros_like(Sn), []]
                acc[k][0] += Cn; acc[k][1] += Sn; acc[k][2].append(t.strftime('%Y%m%d%H'))
    return tarfn, acc


def five_day_from_tars(tars_by_year, wanted, nproc):
    """{epoch: (C, S, stamps)} from the yearly tar archives; each epoch must have 21 stamps"""
    tasks = []
    for y, fn in sorted(tars_by_year.items()):
        w = {k: s0 for k, s0 in wanted.items() if date(y, 1, 1) <= s0 + timedelta(days=5) and s0 <= date(y, 12, 31)}
        if w:
            tasks.append((str(fn), w))
    tot = {}
    with Pool(nproc) as p:
        for fn, acc in p.imap_unordered(_worker, tasks):
            print('  read', Path(fn).name, len(acc), 'epochs', datetime.now().strftime('%H:%M:%S'), flush=True)
            for k, (c, s, st) in acc.items():
                if k not in tot:
                    tot[k] = [np.zeros_like(c), np.zeros_like(s), []]
                tot[k][0] += c; tot[k][1] += s; tot[k][2] += st
    out = {}
    for k, (c, s, st) in tot.items():
        if len(st) != 21:
            raise RuntimeError('epoch %d has %d six-hourly fields' % (k, len(st)))
        out[k] = (c / 21.0, s / 21.0, sorted(st))
    return out


def phase_build(args, out):
    root = comp_root(args)
    ks, st, en = epoch_table()
    wanted = {int(k): s0 for k, s0 in zip(ks, st)}
    info = {}
    for comp in COMPS:
        t0 = time.time()
        tars = {int(p.name.split('_')[1]): p for p in sorted((root / comp).glob('mtm_*_%s.tar.gz' % comp))}
        print(comp, 'archives', sorted(tars), flush=True)
        res = five_day_from_tars(tars, wanted, args.nproc)
        miss = sorted(set(wanted) - set(res))
        if miss:
            raise RuntimeError('%s: epochs without data %s' % (comp, miss[:5]))
        fn = root / comp / ('ESM2_%s_5day_SH_d180.h5' % comp)
        with h5py.File(fn, 'w') as h:
            h.create_dataset('C', data=np.array([res[k][0] for k in ks]), compression='gzip', compression_opts=1)
            h.create_dataset('S', data=np.array([res[k][1] for k in ks]), compression='gzip', compression_opts=1)
            h.create_dataset('epoch_index', data=ks)
            h.create_dataset('start', data=[d.isoformat() for d in st])
            h.create_dataset('end', data=[d.isoformat() for d in en])
            h.attrs['description'] = ('ESM2.0 component %s, fully normalized Stokes coefficients d/o 180, 5-day means of '
                                      'the 21 six-hourly fields start 00:00 .. start+5d 00:00 of each truth epoch '
                                      '(epoch_index = index in the 876 truth epochs). Source: GFZ ISDC ESAESM2/mtmshc, '
                                      'DOI 10.5880/GFZ.1.3.2014.001. Built by jog_eval scripts/09_ice_solid_earth.py' % comp)
        info[comp] = {'file': str(fn), 'epochs': len(ks), 'first': st[0].isoformat(), 'last': st[-1].isoformat(),
                      'seconds': round(time.time() - t0, 1)}
        print(info[comp], flush=True)
    write_json(out / 'build_info.json', info)


def load_5day(root, comp, epochs):
    with h5py.File(root / comp / ('ESM2_%s_5day_SH_d180.h5' % comp), 'r') as h:
        idx = list(h['epoch_index'][:])
        sel = [idx.index(k) for k in epochs]
        return h['C'][sel[0]:sel[-1] + 1][np.array(sel) - sel[0]], h['S'][sel[0]:sel[-1] + 1][np.array(sel) - sel[0]]


# ---- check H + I + S = HIS -------------------------------------------------------------------------------------------
def his_check_files(root):
    """HIS 5-day files in HIS_check/ -> {epoch index: path}; file label = truth epoch start + 1 day"""
    s, _ = S.epoch_windows()
    pos = {d: i for i, d in enumerate(s)}
    out = {}
    for fn in sorted((root / 'HIS_check').glob('mtmshc_HIS_*.180')):
        lab = datetime.strptime(fn.name.split('_')[2], '%Y%m%d').date() - timedelta(days=1)
        out[pos[lab]] = fn
    return out


def phase_check(args, out):
    root = comp_root(args)
    files = {k: f for k, f in his_check_files(root).items() if S.epoch_windows()[0][k].year == 2004}
    s, _ = S.epoch_windows()
    wanted = {k: s[k] for k in files}
    resH = five_day_from_tars({2004: root / 'H_check' / 'mtm_2004_H.tar.gz'}, wanted, args.nproc)
    syn = SH.Synthesis(LMAX); f = SH.published_factor(LMAX)
    rows = []
    for k in sorted(files):
        Ch, Sh = SH.read_icgem(files[k])
        CI, SI = [a[0] for a in load_5day(root, 'I', [k])]
        CS, SS = [a[0] for a in load_5day(root, 'S', [k])]
        dC = resH[k][0] + CI + CS - Ch; dS = resH[k][1] + SI + SS - Sh
        g = syn.grid(dC, dS, f) * 1000
        rows.append({'epoch': k, 'start': s[k].isoformat(), 'HIS_file': files[k].name,
                     'max_abs_coef_diff': float(max(np.abs(dC).max(), np.abs(dS).max())),
                     'max_abs_coef_HIS': float(max(np.abs(Ch).max(), np.abs(Sh).max())),
                     'max_abs_EWH_diff_mm': float(np.abs(g).max())})
        print(rows[-1], flush=True)
    write_csv(out / 'check_H_plus_I_plus_S.csv', rows)


# ---- analyse ---------------------------------------------------------------------------------------------------
def fit(t, y):
    """OLS offset, trend, annual, semi-annual along axis 0; returns trend (/yr), annual amplitude, residual"""
    w = 2 * np.pi * t
    A = np.column_stack([np.ones_like(t), t, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    Y = y.reshape(len(t), -1)
    x, *_ = np.linalg.lstsq(A, Y, rcond=None)
    shp = y.shape[1:]
    return x[1].reshape(shp), np.hypot(x[2], x[3]).reshape(shp), (Y - A @ x).reshape(y.shape)


def stats(t, y):
    y = y - y.mean(0)
    tr, amp, _ = fit(t, y)
    return {'RMS': np.sqrt((y ** 2).mean(0)), 'annual_amp': amp, 'trend': tr}


def his_whole(basin):
    """whole-basin HIS series (mm, 876) and mask (south-first) of basin_signal_whole_basin.hdf5
    (dataset 'EWHA' for Brahmaputra, 'EWHA_HIS' for Danube)"""
    with h5py.File(C.basin_signal_file(basin, whole=True), 'r') as h:
        g = h['whole']['HIS']
        v = g['EWHA' if 'EWHA' in g else 'EWHA_HIS'][:]
        m = h['whole']['mask'][0].astype(bool)
        if h['whole']['mask_lat'][0] > h['whole']['mask_lat'][-1]:
            m = m[::-1]
    return np.ravel(v) * 1000.0, m


def phase_analyse(args, out):
    root = comp_root(args)
    syn = SH.Synthesis(LMAX); f = SH.published_factor(LMAX)
    tfile = C.truth_grid_file()
    summary = {'synthesis': 'degrees 0..70, Wang load Love numbers, R=6378136.3 m, rho_e=5517, rho_w=1000, no filter',
               'truth_file': str(tfile)}
    # 1 validation against the published truth grid
    val = []
    with h5py.File(tfile, 'r') as h:
        bg = h['EWH_background'][:]
        for k, fn in sorted(his_check_files(root).items()):
            if S.epoch_windows()[0][k].year == 2004 and S.epoch_windows()[0][k].month == 5:
                continue
            g = syn.grid(*SH.read_icgem(fn), f)
            tr = h['EWHA'][k] + bg
            val.append({'epoch': k, 'start': S.epoch_windows()[0][k].isoformat(), 'file': fn.name,
                        'max_abs_diff_mm': float(np.abs(g - tr).max() * 1000),
                        'max_abs_truth_mm': float(np.abs(tr).max() * 1000)})
            print(val[-1], flush=True)
    write_csv(out / 'validation_HIS_synthesis.csv', val)
    summary['validation'] = val

    # 2 I, S on the 0.5 deg grid, paper period
    epochs = np.arange(*C.PAPER_EPOCHS)
    st, en = S.epoch_windows()
    tmid = np.array([(st[k] - date(2003, 1, 1)).days + 2.5 for k in epochs]) / 365.25 + 2003.0
    grids = {}
    for comp in COMPS:
        Cc, Sc = load_5day(root, comp, list(epochs))
        grids[comp] = np.array([syn.grid(Cc[i], Sc[i], f) for i in range(len(epochs))], dtype=np.float32) * 1000
    grids['I+S'] = grids['I'] + grids['S']
    with h5py.File(tfile, 'r') as h:
        grids['HIS'] = h['EWHA'][epochs[0]:epochs[-1] + 1].astype(np.float32) * 1000
    grids['HIS-(I+S)'] = grids['HIS'] - grids['I+S']
    names = ['I', 'S', 'I+S', 'HIS', 'HIS-(I+S)']

    lat, lon = syn.lat, syn.lon
    coslat = np.cos(np.deg2rad(lat))
    basin_rows, cell_rows, share_rows, series_rows = [], [], [], []
    trend_maps = {}
    for basin in args.basins:
        _, _, _, full = S.region_mask(basin)             # north-up
        mask = full[::-1].astype(bool)                   # grid here is south-first
        n_cells = int(mask.sum())
        w = np.broadcast_to(coslat[:, None], mask.shape)[mask]
        # basin mean weighting: compare with basin_signal_whole_basin.hdf5 (published basin series)
        his_cells = grids['HIS'][:, mask]
        bm_w = his_cells @ w / w.sum(); bm_u = his_cells.mean(1)
        ref_all, wmask = his_whole(basin)
        ref = ref_all[epochs]
        dw = float(np.abs((bm_w - bm_w.mean()) - (ref - ref.mean())).max())
        du = float(np.abs((bm_u - bm_u.mean()) - (ref - ref.mean())).max())
        use_w = dw <= du
        summary.setdefault('basin_mean_check', {})[basin] = {'n_cells': n_cells,
                                                              'n_cells_whole_mask': int(wmask.sum()),
                                                              'masks_equal': bool((wmask == mask).all()),
                                                              'maxdiff_coslat_weighted_mm': dw, 'maxdiff_unweighted_mm': du,
                                                              'used': 'cos-lat weighted' if use_w else 'unweighted'}
        series = {}
        for nm in names:
            cells = grids[nm][:, mask].astype(float)
            series[nm] = cells @ w / w.sum() if use_w else cells.mean(1)
            b = stats(tmid, series[nm][:, None])
            c = stats(tmid, cells)
            basin_rows.append({'basin': basin, 'component': nm, 'RMS_mm': round(float(b['RMS'][0]), 3),
                               'annual_amp_mm': round(float(b['annual_amp'][0]), 3),
                               'trend_mm_yr': round(float(b['trend'][0]), 3)})
            for q in ('RMS', 'annual_amp', 'trend'):
                v = c[q]
                imax = int(np.argmax(np.abs(v)))
                cell_rows.append({'basin': basin, 'component': nm, 'quantity': q, 'n_cells': n_cells,
                                  'median': round(float(np.median(v)), 3), 'max_abs': round(float(v[imax]), 3),
                                  'median_abs': round(float(np.median(np.abs(v))), 3)})
            series[nm + '_cells'] = cells
        # shares of I+S in HIS
        def shares(his, is_):
            his = his - his.mean(0); is_ = is_ - is_.mean(0)
            tr_h, _, _ = fit(tmid, his); tr_i, _, _ = fit(tmid, is_)
            return {'trend_share': tr_i / tr_h, 'var_ratio': is_.var(0) / his.var(0),
                    'var_explained': 1 - (his - is_).var(0) / his.var(0)}
        sb = shares(series['HIS'][:, None], series['I+S'][:, None])
        sc = shares(series['HIS_cells'], series['I+S_cells'])
        for q in sb:
            share_rows.append({'basin': basin, 'quantity': q, 'basin_mean': round(float(sb[q][0]), 4),
                               'cell_median': round(float(np.median(sc[q])), 4),
                               'cell_max_abs': round(float(sc[q][np.argmax(np.abs(sc[q]))]), 4)})
        for i, k in enumerate(epochs):
            series_rows.append({'basin': basin, 'epoch': int(k), 'start': st[k].isoformat(),
                                **{nm: round(float(series[nm][i]), 4) for nm in names}})
        # Brahmaputra/Danube HIS trend variants
        tv = {}
        for lab, (d0, d1) in {'paper_2003-09_2006-12': (date(2003, 9, 1), date(2006, 12, 31)),
                              'calendar_2003-01_2006-12': (date(2003, 1, 1), date(2006, 12, 31))}.items():
            ks = S.epochs_between(d0, d1)
            tt = np.array([(st[k] - date(2003, 1, 1)).days + 2.5 for k in ks]) / 365.25 + 2003.0
            his_k = ref_all[ks]
            A1 = np.column_stack([np.ones_like(tt), tt])
            tv[lab] = {'epochs': [int(ks[0]), int(ks[-1])], 'n': int(len(ks)),
                       'HIS_trend_with_seasonal_mm_yr': round(float(fit(tt, his_k[:, None])[0][0]), 3),
                       'HIS_trend_linear_only_mm_yr': round(float(np.linalg.lstsq(A1, his_k, rcond=None)[0][1]), 3)}
        tv['source'] = 'basin_signal_whole_basin.hdf5 whole/HIS'
        summary.setdefault('HIS_trend_variants', {})[basin] = tv
        # Sumatra-Andaman (2004-12-26) step in S: mean of 6 epochs after minus 6 before
        i0 = int(np.searchsorted(np.array([st[k] for k in epochs]), date(2004, 12, 26)))
        summary.setdefault('S_step_2004_12_26_mm', {})[basin] = round(float(series['S'][i0:i0 + 6].mean() - series['S'][i0 - 6:i0].mean()), 3)
        trend_maps[basin] = mask
    write_csv(out / 'basin_mean_stats.csv', basin_rows)
    write_csv(out / 'cell_stats.csv', cell_rows)
    write_csv(out / 'shares_IS_in_HIS.csv', share_rows)
    write_csv(out / 'basin_series.csv', series_rows)

    # 3 trend maps
    tr = {nm: fit(tmid, grids[nm] - grids[nm].mean(0))[0] for nm in ('I', 'S', 'I+S', 'HIS')}
    np.savez_compressed(out / 'trend_maps_mm_yr.npz', lat=lat, lon=lon, **{k.replace('+', 'p'): v for k, v in tr.items()})
    plot_maps(out, tr, lat, lon, trend_maps)
    plot_series(out, series_rows, args.basins)
    write_json(out / 'summary.json', summary)


def plot_maps(out, tr, lat, lon, masks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for basin, mask in masks.items():
        x0, x1, y0, y1 = MAP_BOX[basin]
        ii = (lat >= y0) & (lat <= y1); jj = (lon >= x0) & (lon <= x1)
        fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
        for ax, nm in zip(axs, ('I', 'S', 'I+S', 'HIS')):
            v = tr[nm][np.ix_(ii, jj)]
            lim = max(float(np.nanpercentile(np.abs(v), 99)), 1e-3)
            pc = ax.pcolormesh(lon[jj], lat[ii], v, cmap='RdBu', vmin=-lim, vmax=lim, shading='nearest')
            ax.contour(lon[jj], lat[ii], mask[np.ix_(ii, jj)].astype(float), levels=[0.5], colors='k', linewidths=1.2)
            ax.set_title('%s trend 2003-09..2006-12' % nm, fontsize=10)
            ax.set_aspect('equal'); ax.set_xlabel('lon'); ax.set_ylabel('lat')
            fig.colorbar(pc, ax=ax, shrink=0.8, label='mm/yr')
        fig.suptitle('%s: ESM2.0 trends (0.5 deg, d/o 70 synthesis as the truth); black = basin' % basin)
        fig.savefig(out / ('map_trend_IS_%s.png' % basin), dpi=130)
        plt.close(fig)


def plot_series(out, rows, basins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(basins), 1, figsize=(11, 3.6 * len(basins)), constrained_layout=True)
    for ax, basin in zip(np.atleast_1d(axs), basins):
        r = [x for x in rows if x['basin'] == basin]
        d = [datetime.fromisoformat(x['start']) for x in r]
        for nm, col in (('HIS', '#333333'), ('I+S', '#c0392b'), ('I', '#2e86c1'), ('S', '#8e44ad')):
            y = np.array([x[nm] for x in r]); ax.plot(d, y - y.mean(), color=col, lw=1.5, label=nm)
        ax.set_title('%s basin mean (de-meaned), mm' % basin); ax.legend(ncol=4, frameon=False); ax.grid(alpha=0.3)
    fig.savefig(out / 'basin_series_IS_HIS.png', dpi=130)
    plt.close(fig)


def main():
    p = parser(__doc__)
    p.add_argument('--phase', nargs='+', default=['analyse'], choices=['build', 'check', 'analyse'])
    p.add_argument('--components', default=str(C.ESM2_ROOT / 'components'))
    p.add_argument('--nproc', type=int, default=8)
    args = p.parse_args()
    out = outdir(args, '09_ice_solid_earth')
    for ph in args.phase:
        t0 = time.time()
        {'build': phase_build, 'check': phase_check, 'analyse': phase_analyse}[ph](args, out)
        print('phase %s: %.1f s' % (ph, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
