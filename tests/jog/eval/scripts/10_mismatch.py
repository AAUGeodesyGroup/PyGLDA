"""Step 10 - model mismatch (R1.1 (1)): OL - truth, OL - real GRACE, truth - real GRACE.

Decomposition of a difference series d(t) (both members de-meaned over the period) by least squares
    d = c + b (t - tm) + annual (cos/sin) + semi-annual (cos/sin) + residual,      t in years (epoch centre dates)
reported as trend (mm/yr), annual and semi-annual amplitude (mm), residual RMS (mm) and the variance share
var(part) / var(d) of each part (parts are nearly, not exactly, orthogonal: the shares need not add up to 1).

A. 5-day, paper period 2003-09-05..2006-12-27 (epochs 634..875):
   truth = ESM2.0 HIS (Global_EWH_05_HIS, 0.5 deg), OL = published OL_mean of GRACEC_1e18_1degree_<basin>
   (= OL member 0, same for every case), 0.5 deg by 5x5 block means; per cell and basin mean (cos-lat weights over
   the common valid cells). Also truth and OL alone (trend, annual amplitude) and a check of draft numbers
   with several fit / averaging variants (draft_check.csv).
B. monthly: real GRACE = CSR RL06.3 mascons (0.25 deg, cm), 2x2 block means to the 0.5 deg grid (lon shifted to
   -180..180), same cells; truth and OL averaged over the 5-day epochs whose centre date falls in the month;
   GRACE month = calendar month of its time stamp (mid-interval). Common period 2003-09..2006-12 (OL limit);
   truth - GRACE additionally over 2002-04..2006-12 (all GRACE months before 2007).
Usage: python 10_mismatch.py
"""
import time
from datetime import date, timedelta

import h5py
import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import evaluate as E, sources as S

MASCON = C.V1_DATA_ROOT / 'GRACE' / 'SaGEA' / 'signal_Mascon' / 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
YEAR = 365.25


def decimal_year(d):
    d = np.atleast_1d(d)
    return np.array([x.year + (x - date(x.year, 1, 1)).days / (366 if x.year % 4 == 0 else 365) for x in d])


def design(t, seasonal=True):
    tm = t - t.mean()
    cols = [np.ones_like(t), tm]
    if seasonal:
        for P in (1.0, 0.5):
            cols += [np.cos(2 * np.pi * t / P), np.sin(2 * np.pi * t / P)]
    return np.stack(cols, 1)


def decompose(d, t, seasonal=True):
    """d (n, units) -> dict of arrays (units,)"""
    d = d - np.nanmean(d, 0)
    A = design(t, seasonal)
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    fitted = A @ coef
    res = d - fitted
    v = np.var(d, 0)
    out = {'trend_mm_yr': coef[1], 'std_mm': np.sqrt(v), 'resid_rms_mm': np.std(res, 0)}
    parts = {'trend': A[:, [1]] @ coef[[1]]}
    if seasonal:
        out['annual_amp_mm'] = np.hypot(coef[2], coef[3])
        out['semiannual_amp_mm'] = np.hypot(coef[4], coef[5])
        out['annual_phase_doy'] = (np.degrees(np.arctan2(coef[3], coef[2])) % 360) / 360 * YEAR   # day of maximum
        parts['annual'] = A[:, 2:4] @ coef[2:4]
        parts['semiannual'] = A[:, 4:6] @ coef[4:6]
    parts['resid'] = res
    for k, p in parts.items():
        out['share_' + k] = np.var(p, 0) / v
    return out


def basin_mean(x, w):
    """x (n, lat, lon) with NaN outside, w (lat, lon) -> (n,)"""
    ok = np.isfinite(x).all(0)
    ww = np.where(ok, w, 0.0)
    return np.nansum(np.where(ok[None], x, 0) * ww[None], (1, 2)) / ww.sum()


def cells(x):
    ok = np.isfinite(x).all(0)
    return x[:, ok], ok


def row_basin(basin, pair, res, period, kind):
    r = {'basin': basin, 'pair': pair, 'period': period, 'level': kind}
    r.update({k: round(float(np.asarray(v).ravel()[0]), 4) for k, v in res.items()})
    return r


def row_cells(basin, pair, res, period):
    r = {'basin': basin, 'pair': pair, 'period': period, 'level': 'cells_median[p25,p75]'}
    for k, v in res.items():
        q = np.nanpercentile(v, [50, 25, 75])
        r[k] = '%.3f [%.3f,%.3f]' % tuple(q)
    return r


def mascon_grid(lat05, lon05, box, first, last):
    """mascon months in [first, last] on the 0.5 deg cells of the basin box (global 2x2 block means), mm"""
    import netCDF4
    with netCDF4.Dataset(MASCON) as n:
        t = n.variables['time'][:].data
        days = [date(2002, 1, 1) + timedelta(days=float(x)) for x in t]
        lat = n.variables['lat'][:].data
        lon = n.variables['lon'][:].data
        months = [(d.year, d.month) for d in days]
        sel = [i for i, m in enumerate(months) if first <= m <= last]
        x = np.ma.filled(n.variables['lwe_thickness'][sel].astype(float), np.nan) * 10.0      # cm -> mm
    lon = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon)
    x, lon = x[:, :, order], lon[order]
    g = x.reshape(len(sel), 360, 2, 720, 2).mean((2, 4))
    glat = lat.reshape(360, 2).mean(1)
    glon = lon.reshape(720, 2).mean(1)
    ia = [int(np.argmin(np.abs(glat - v))) for v in lat05]
    io = [int(np.argmin(np.abs(glon - v))) for v in lon05]
    assert np.allclose(glat[ia], lat05) and np.allclose(glon[io], lon05), 'mascon / 0.5 deg grid misaligned'
    g = g[:, ia][:, :, io]
    g = np.where(box[None], g, np.nan)
    return g, [months[i] for i in sel], [days[i] for i in sel]


def monthly(x, centres, months):
    keys = np.array([d.year * 12 + d.month for d in centres])
    out = np.full((len(months),) + x.shape[1:], np.nan)
    nn = []
    for i, m in enumerate(months):
        s = keys == m[0] * 12 + m[1]
        nn.append(int(s.sum()))
        if s.any():
            out[i] = x[s].mean(0)
    return out, nn


def mid_month(months):
    return decimal_year([date(y, m, 15) for y, m in months])


def main():
    t0 = time.time()
    args = parser(__doc__).parse_args()
    out = outdir(args, '10_mismatch')
    epochs = np.arange(*C.PAPER_EPOCHS)
    starts, ends = S.epoch_windows()
    rows, alone, draft, meta = [], [], [], {'mascon': str(MASCON)}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(11, 7))
    for ib, basin in enumerate(args.basins):
        case = C.case_name('GRACEC', '1e18', '1degree', basin)
        data = E.load_basin(basin, epochs, folders={'OL': ('OL', None, case)})
        truth, ol = data['truth'], data['sims']['OL']
        lat, lon = data['lat'], data['lon']
        box = np.isfinite(truth).all(0)
        w = np.cos(np.radians(lat))[:, None] * np.ones(len(lon))[None]
        cen = np.array([s + timedelta(days=(e - s).days / 2) for s, e in zip(starts[epochs], ends[epochs])])
        t = decimal_year(cen)
        per = '%s..%s (5-day, %d)' % (starts[epochs[0]], ends[epochs[-1]] - timedelta(days=1), len(epochs))
        meta[basin] = {'cells': int(box.sum()), 'OL_case': case, 'missing_OL_epochs': len(data['missing']['OL'])}
        tb, ob = basin_mean(truth, w), basin_mean(ol, w)
        # A. OL - truth
        rows.append(row_basin(basin, 'OL-truth', decompose((ob - ob.mean() - (tb - tb.mean()))[:, None], t), per, 'basin_mean'))
        tc, _ = cells(truth)
        oc, _ = cells(ol)
        rows.append(row_cells(basin, 'OL-truth', decompose((oc - oc.mean(0)) - (tc - tc.mean(0)), t), per))
        for name, xb, xc in (('truth', tb, tc), ('OL', ob, oc)):
            alone.append(row_basin(basin, name, decompose(xb[:, None], t), per, 'basin_mean'))
            alone.append(row_cells(basin, name, decompose(xc, t), per))
        # draft-number check: variants
        tu, ou = np.nanmean(truth, (1, 2)), np.nanmean(ol, (1, 2))
        variants = {'cos-weighted basin mean, trend+annual+semi': (tb, ob, True),
                    'unweighted basin mean, trend+annual+semi': (tu, ou, True),
                    'cos-weighted basin mean, trend only': (tb, ob, False)}
        for vn, (a, b, seas) in variants.items():
            ra, rb = decompose(a[:, None], t, seas), decompose(b[:, None], t, seas)
            draft.append({'basin': basin, 'variant': vn, 'period': 'paper 2003-09-05..2006-12-27',
                          'truth_trend': round(float(ra['trend_mm_yr'][0]), 2), 'OL_trend': round(float(rb['trend_mm_yr'][0]), 2),
                          'truth_annual_amp': round(float(ra.get('annual_amp_mm', [np.nan])[0]), 2),
                          'OL_annual_amp': round(float(rb.get('annual_amp_mm', [np.nan])[0]), 2),
                          'truth_std': round(float(ra['std_mm'][0]), 2), 'OL_std': round(float(rb['std_mm'][0]), 2)})
        ra, rb = decompose(tc, t), decompose(oc, t)
        draft.append({'basin': basin, 'variant': 'mean over cells of per-cell fit', 'period': 'paper 2003-09-05..2006-12-27',
                      'truth_trend': round(float(np.mean(ra['trend_mm_yr'])), 2), 'OL_trend': round(float(np.mean(rb['trend_mm_yr'])), 2),
                      'truth_annual_amp': round(float(np.mean(ra['annual_amp_mm'])), 2),
                      'OL_annual_amp': round(float(np.mean(rb['annual_amp_mm'])), 2),
                      'truth_std': round(float(np.mean(ra['std_mm'])), 2), 'OL_std': round(float(np.mean(rb['std_mm'])), 2)})
        # truth over longer periods (OL not available outside the paper period)
        for p0, p1, lab in ((date(2002, 4, 1), date(2006, 12, 31), 'truth 2002-04..2006-12'),
                            (date(1995, 1, 1), date(2006, 12, 31), 'truth 1995-2006'),
                            (date(2003, 1, 1), date(2006, 12, 31), 'truth 2003-2006')):
            ek = S.epochs_between(p0, p1)
            tl, _, _ = S.truth_grid(basin, ek)
            tl = np.where(box[None], tl, np.nan)
            tt = decimal_year(np.array([s + timedelta(days=(e - s).days / 2) for s, e in zip(starts[ek], ends[ek])]))
            for seas in (True, False):
                ra = decompose(basin_mean(tl, w)[:, None], tt, seas)
                draft.append({'basin': basin, 'variant': 'cos-weighted basin mean, ' + ('trend+annual+semi' if seas else 'trend only'),
                              'period': lab, 'truth_trend': round(float(ra['trend_mm_yr'][0]), 2), 'OL_trend': '',
                              'truth_annual_amp': round(float(ra.get('annual_amp_mm', [np.nan])[0]), 2), 'OL_annual_amp': '',
                              'truth_std': round(float(ra['std_mm'][0]), 2), 'OL_std': ''})
        # B. monthly vs real GRACE
        g, gm, gdays = mascon_grid(lat, lon, box, (2002, 4), (2006, 12))
        common = [m for m in gm if m >= (2003, 9)]
        gi = [gm.index(m) for m in common]
        gc = g[gi]
        tm_, n_t = monthly(truth, cen, common)
        om_, n_o = monthly(ol, cen, common)
        ok = box & np.isfinite(gc).all(0)
        meta[basin].update({'grace_months_common': len(common), 'epochs_per_month_min': min(n_t),
                            'cells_with_grace': int(ok.sum()), 'grace_months_2002_04_2006_12': len(gm)})
        tmon = mid_month(common)
        perm = '%d-%02d..%d-%02d (monthly, %d)' % (common[0] + common[-1] + (len(common),))
        mk = lambda x: np.where(ok[None], x, np.nan)
        G, T, O = mk(gc), mk(tm_), mk(om_)
        Gb, Tb, Ob = (basin_mean(x, w) for x in (G, T, O))
        dm = lambda a, b: (a - a.mean(0)) - (b - b.mean(0))
        for pair, a, b, ac, bc in (('OL-truth', Ob, Tb, O, T), ('OL-GRACE', Ob, Gb, O, G), ('truth-GRACE', Tb, Gb, T, G)):
            rows.append(row_basin(basin, pair, decompose(dm(a, b)[:, None], tmon), perm, 'basin_mean'))
            rows.append(row_cells(basin, pair, decompose(dm(cells(ac)[0], cells(bc)[0]), tmon), perm))
        for name, xb in (('GRACE', Gb), ('truth', Tb), ('OL', Ob)):
            alone.append(row_basin(basin, name, decompose(xb[:, None], tmon), perm, 'basin_mean'))
        # truth - GRACE over 2002-04..2006-12
        ek = S.epochs_between(date(2002, 4, 1), date(2006, 12, 31))
        tl, _, _ = S.truth_grid(basin, ek)
        cl = np.array([s + timedelta(days=(e - s).days / 2) for s, e in zip(starts[ek], ends[ek])])
        tlm, _ = monthly(np.where(ok[None], tl, np.nan), cl, gm)
        Gl = np.where(ok[None], g, np.nan)
        perl = '%d-%02d..%d-%02d (monthly, %d)' % (gm[0] + gm[-1] + (len(gm),))
        rows.append(row_basin(basin, 'truth-GRACE', decompose(dm(basin_mean(tlm, w), basin_mean(Gl, w))[:, None], mid_month(gm)),
                              perl, 'basin_mean'))
        rows.append(row_cells(basin, 'truth-GRACE', decompose(dm(cells(tlm)[0], cells(Gl)[0]), mid_month(gm)), perl))
        for name, xb in (('GRACE', basin_mean(Gl, w)), ('truth', basin_mean(tlm, w))):
            alone.append(row_basin(basin, name, decompose(xb[:, None], mid_month(gm)), perl, 'basin_mean'))
        ax = axs[ib]
        ax.plot(t, tb - tb.mean(), color='0.3', lw=1, label='ESM2.0 HIS (5-day)')
        ax.plot(t, ob - ob.mean(), color='#1f5fa8', lw=1, label='W3RA OL (5-day)')
        ax.plot(tmon, Gb - Gb.mean(), 'o-', color='#d0632b', ms=3, lw=1, label='CSR RL06.3 mascons (monthly)')
        ax.set_title('%s basin mean, de-meaned over the period' % basin)
        ax.set_ylabel('TWS anomaly (mm)')
        ax.grid(alpha=0.3)
        if ib == 0:
            ax.legend(fontsize=8)
        print('done', basin, round(time.time() - t0, 1), 's')
    fig.tight_layout()
    fig.savefig(out / 'basin_series.png', dpi=130)
    write_csv(out / 'decomposition.csv', rows)
    write_csv(out / 'truth_ol_grace_alone.csv', alone)
    write_csv(out / 'draft_check.csv', draft)
    meta['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'meta.json', meta)
    print('written', out, meta['run_time_s'], 's')


if __name__ == '__main__':
    main()
