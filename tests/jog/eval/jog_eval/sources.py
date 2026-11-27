"""Readers for truth, synthetic observations and OL/DA products, all returned on the common 0.5 deg / 5-day grid.

The alignment reproduces ESAsing/paper_RSE/spatial_analysis.py:
* truth and observations: global 0.5 deg files 'EWHA' (epoch, lat, lon) in m, flipped along lat, cut to the basin mask
  of globe/global_mask/<basin>/0.5/mask.h5 and to its bounding box; m -> mm;
* OL / DA: daily 0.1 deg ensemble means 'TotalWater.<year>.h5' (one dataset per day), averaged over each 5-day truth
  epoch [d, d+5) and up-scaled to 0.5 deg by 5x5 block means after cutting the 0.1 deg box [2:-3, 2:-3]
  (the published code; this aligns the 0.1 deg box with the 0.5 deg cells).
Epochs are handled by date (the published code used index offsets: truth 634..875, daily blocks from day 4 of
2003-09-01; both agree for 2003-09-05..2006-12-27). `epoch_windows()` gives the 876 ESM2.0 epochs (start, end).
"""
from datetime import date, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from . import config as C


# ---- time ----------------------------------------------------------------------------------------------------------
def epoch_windows():
    """start and end (exclusive) of the 876 ESM2.0 epochs, read from basin_signal.hdf5 ('dates': start, end).
    All epochs are 5 days except 1995-12-27..1995-12-30 (4 days); fallback: that rule."""
    for basin in C.BASINS:
        fn = C.basin_signal_file(basin)
        if fn.exists():
            with h5py.File(fn, 'r') as h:
                d = h['dates'][:]
            return (np.array([_parse_day(x) for x in d[0]]), np.array([_parse_day(x) for x in d[1]]))
    starts = [date(1995, 1, 1) + timedelta(days=5 * k) for k in range(73)]
    starts += [date(1995, 12, 31) + timedelta(days=5 * k) for k in range(803)]
    ends = starts[1:] + [starts[-1] + timedelta(days=5)]
    return np.array(starts), np.array(ends)


def epoch_dates():
    return epoch_windows()[0]


def epochs_between(first, last):
    """indices of the epochs lying completely in [first, last] (dates)"""
    s, e = epoch_windows()
    return np.flatnonzero((s >= first) & (e - timedelta(days=1) <= last))


def _parse_day(key):
    k = key.decode() if isinstance(key, bytes) else str(key)
    for fmt in ('%Y-%m-%d', '%Y%m%d', '%Y.%m.%d'):
        try:
            return datetime.strptime(k, fmt).date()
        except ValueError:
            pass
    raise ValueError('unknown date key %r' % key)


# ---- masks -----------------------------------------------------------------------------------------------------
def region_mask(basin):
    """0.5 deg basin mask cut to its bounding box: (mask_box, lat, lon, full_mask)"""
    with h5py.File(C.region_mask_file(basin), 'r') as h:
        full = h['basin'][:].astype(bool)
    res, err = 0.5, 0.05
    lat0 = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon0 = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
    rows, cols = np.where(full)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return full[r0:r1, c0:c1], lat0[r0:r1], lon0[c0:c1], full


# ---- truth and observations -----------------------------------------------------------------------------------------
def _global_grid(fn, basin, epochs):
    box, lat, lon, full = region_mask(basin)
    with h5py.File(fn, 'r') as h:
        x = h['EWHA'][epochs[0]:epochs[-1] + 1]
    x = np.flip(x, axis=-2) * 1000.0                      # m -> mm, north up as in the masks
    rows, cols = np.where(full)
    x = x[:, rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    x = np.where(box[None], x, np.nan)
    sel = np.asarray(epochs) - epochs[0]
    return x[sel], lat, lon


_TRUTH_CACHE = {}


def truth_grid(basin, epochs):
    """cached: the split-sample and bootstrap steps call it many times with the same epochs"""
    key = (basin, tuple(int(e) for e in epochs))
    if key not in _TRUTH_CACHE:
        _TRUTH_CACHE.clear()
        _TRUTH_CACHE[key] = _global_grid(C.truth_grid_file(), basin, epochs)
    t, lat, lon = _TRUTH_CACHE[key]
    return t.copy(), lat, lon


def obs_grid(basin, mission, alpha, epochs):
    return _global_grid(C.obs_grid_file(mission, alpha), basin, epochs)


def basin_series(basin, what='HIS', grid='subbasins_3', mission=None, alpha=None, whole=False):
    """basin series in mm from basin_signal*.hdf5: what='HIS' (truth) or 'OBS' (mission, alpha required)"""
    with h5py.File(C.basin_signal_file(basin, whole), 'r') as h:
        g = h['whole'] if whole else h[grid]
        if what == 'HIS':
            v = g['HIS']['EWHA'][:]
        else:
            v = g[C.MISSION_FILE[mission]]['EWHA_alpha_%s' % C.alpha_file(alpha)][:]
        dates = [_parse_day(x) for x in h['dates'][0]]
    return np.asarray(v) * 1000.0, np.array(dates)


# ---- OL / DA products -------------------------------------------------------------------------------------------
def read_daily(folder, years=None):
    """{date: 2-D or 1-D field} from TotalWater.<year>.h5 files of one product folder"""
    folder = Path(folder)
    files = sorted(folder.glob('TotalWater.*.h5'))
    if years is not None:
        files = [f for f in files if int(f.name.split('.')[1]) in years]
    out = {}
    for fn in files:
        with h5py.File(fn, 'r') as h:
            for k in h.keys():
                out[_parse_day(k)] = h[k][:]
    if not out:
        raise FileNotFoundError('no TotalWater.<year>.h5 in %s' % folder)
    return out


def five_day_means(daily, epochs):
    """average daily fields over each epoch [start, end); epochs without all their days are NaN (and reported)"""
    starts, ends = epoch_windows()
    sample = next(iter(daily.values()))
    out = np.full((len(epochs),) + sample.shape, np.nan)
    missing = []
    for i, k in enumerate(epochs):
        n = (ends[k] - starts[k]).days
        days = [starts[k] + timedelta(days=j) for j in range(n)]
        if all(d in daily for d in days):
            out[i] = np.mean([daily[d] for d in days], axis=0)
        else:
            missing.append(starts[k])
    return out, missing


def upscale_to_half_degree(x01, mask01):
    """0.1 deg (time, ...) on the published case mask -> 0.5 deg block means, as in spatial_analysis.OL_DA"""
    if x01.ndim == 2:                                       # (time, cells) -> (time, lat, lon)
        full = np.full((x01.shape[0],) + mask01.shape, np.nan)
        full[:, mask01] = x01
        x01 = full
    m = mask01[2:-3, 2:-3].astype(float)
    rows, cols = m.shape
    x = x01[:, 2:-3, 2:-3]
    new = np.nanmean(x.reshape(-1, rows // 5, 5, cols // 5, 5), axis=(2, 4))
    new_mask = np.nanmean(m.reshape(rows // 5, 5, cols // 5, 5), axis=(1, 3)) > 0.01
    return np.where(new_mask[None], new, np.nan), new_mask


def case_mask(case):
    with h5py.File(C.case_mask_file(case), 'r') as h:
        return h['mask'][:].astype(bool), h['lat'][:], h['lon'][:]


def product_grid(kind, case, epochs, folder=None, mask_case=None):
    """OL or DA of a case on the 0.5 deg / 5-day truth grid.
    folder: product folder (default: published layout); mask_case: case whose 0.1 deg mask applies (default: case)."""
    folder = folder or C.published_daily_dir(kind, case)
    yrs = sorted({d.year for d in epoch_dates()[epochs]} | {d.year for d in epoch_windows()[1][epochs]})
    daily = read_daily(folder, years=set(yrs))
    x, missing = five_day_means(daily, epochs)
    mask01, _, _ = case_mask(mask_case or case)
    up, _ = upscale_to_half_degree(x, mask01)
    return up, missing


def case_products(case, domain, ensemble, data_root=None):
    """product folders of a run made with the v1 data layout: DA mean in cases/<case>/products/DA_mean,
    OL mean in domains/<domain>/OL/<ensemble>/products/OL_mean"""
    root = Path(data_root or C.V1_DATA_ROOT) / 'w3ra'
    return {'DA': root / 'cases' / case / 'products' / 'DA_mean',
            'OL': root / 'domains' / domain / 'OL' / ensemble / 'products' / 'OL_mean'}


def on_truth_cells(sim, truth):
    """common valid cells of two (time, lat, lon) arrays of equal shape"""
    if sim.shape != truth.shape:
        raise ValueError('grid mismatch %s vs %s: check masks/boxes' % (sim.shape, truth.shape))
    ok = np.isfinite(sim).all(0) & np.isfinite(truth).all(0)
    return np.where(ok[None], sim, np.nan), np.where(ok[None], truth, np.nan), ok
