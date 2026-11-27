"""Assemble the Zenodo v2 package (no upload): /work/data_for_w3/zenodo_v2/<item>/.

Daily 0.1 deg TWS products (TotalWater.<year>.h5, one dataset per day over the product mask cells) -> NetCDF4 (zlib 5), variable
tws (time, lat, lon) int16 with scale_factor 0.1 mm and a per-file add_offset (CF packing; max abs packing error recorded, <= 0.05 mm),
float32 if the value span does not fit. Global 0.5 deg 5-day grids -> NetCDF4 float32 zlib. Other items copied (copy2, never moved).
Writes <root>/conversion_log.json, <root>/ITEMS.txt (files, GB per item) and <root>/MANIFEST.md5.
Usage: python assemble.py [--workers 28] [--only 05_openloop ...]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

D = Path(os.environ.get('PYGLDA_V1_DATA', '/work/data_for_w3'))
W3, GR, EV = D / 'w3ra', D / 'GRACE', D / 'w3ra' / 'evaluation' / 'JoG'
OUT = D / 'zenodo_v2'
MISSIONS = ('GRACEC', 'NGGM', 'MAGIC')
MNAME = {'GRACEC': 'GRACE-C-like', 'NGGM': 'NGGM', 'MAGIC': 'MAGIC'}
BASINS = ('Brahmaputra', 'Danube')
ALPHAS = ('1e14', '1e15', '1e16', '1e17', '1e18', '1e19')
GRIDS = ('1degree', '2degree', '3degree', '4degree', '3subbasins')
COMMIT = 'acf7a42 + pending revision commit (see 13_repro/README)'


def day(k):
    k = k.decode() if isinstance(k, bytes) else str(k)
    return datetime.strptime(k, '%Y%m%d').date()


def find_product(folder):
    hits = sorted({f.parent for f in Path(folder).rglob('TotalWater.*.h5')})
    return hits[0] if hits else None


def case_mask(case):
    return W3 / 'cases' / case / 'products' / 'mask' / (case + '.h5')


def convert(job):
    src, mask_fn, out_fn, attrs = job
    import netCDF4
    t0 = time.time()
    folder = find_product(src)
    if folder is None:
        return {'out': str(out_fn), 'error': 'no product in %s' % src}
    with h5py.File(mask_fn, 'r') as h:
        mask = h['mask'][()].astype(bool)
        lat, lon = h['lat'][()], h['lon'][()]
    days, vals = [], []
    for f in sorted(folder.glob('TotalWater.*.h5')):
        with h5py.File(f, 'r') as h:
            for k in sorted(h.keys()):
                days.append(day(k))
                vals.append(h[k][()])
    v = np.array(vals, dtype=np.float64)
    if v.shape[1] != mask.sum():
        return {'out': str(out_fn), 'error': 'cells %d != mask %d (%s)' % (v.shape[1], mask.sum(), mask_fn)}
    lo, hi = np.nanmin(v), np.nanmax(v)
    out_fn.parent.mkdir(parents=True, exist_ok=True)
    nc = netCDF4.Dataset(out_fn, 'w', format='NETCDF4')
    nc.createDimension('time', len(days))
    nc.createDimension('lat', len(lat))
    nc.createDimension('lon', len(lon))
    t = nc.createVariable('time', 'i4', ('time',))
    t.units, t.calendar, t.long_name = 'days since 2000-01-01', 'standard', 'time (daily)'
    t[:] = [(d - date(2000, 1, 1)).days for d in days]
    y = nc.createVariable('lat', 'f8', ('lat',)); y.units, y.long_name = 'degrees_north', 'latitude'; y[:] = lat
    x = nc.createVariable('lon', 'f8', ('lon',)); x.units, x.long_name = 'degrees_east', 'longitude'; x[:] = lon
    span = hi - lo
    packed = span / 0.1 < 65000
    if packed:
        off = round((hi + lo) / 2.0, 1)
        var = nc.createVariable('tws', 'i2', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=np.int16(-32768),
                                chunksizes=(min(366, len(days)), len(lat), len(lon)))
        var.scale_factor, var.add_offset = 0.1, off
    else:
        var = nc.createVariable('tws', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=np.float32(np.nan),
                                chunksizes=(min(366, len(days)), len(lat), len(lon)))
    var.units, var.long_name = 'mm', 'terrestrial water storage (sum of all W3RA stores), ensemble mean'
    full = np.full((len(days),) + mask.shape, np.nan)
    full[:, mask] = v
    var[:] = np.ma.masked_invalid(full)
    for k, val in attrs.items():
        setattr(nc, k, val)
    nc.Conventions = 'CF-1.8'
    nc.pipeline_commit = COMMIT
    nc.close()
    err = 0.0
    if packed:
        q = np.round((v - off) / 0.1) * 0.1 + off
        err = float(np.nanmax(np.abs(q - v)))
    return {'out': str(out_fn), 'days': len(days), 'first': str(days[0]), 'last': str(days[-1]), 'cells': int(mask.sum()),
            'packed_int16': packed, 'max_abs_packing_error_mm': err, 'min_mm': float(lo), 'max_mm': float(hi),
            'MB': round(out_fn.stat().st_size / 1e6, 2), 's': round(time.time() - t0, 1)}


def convert_global(job):
    src, out_fn, attrs = job
    import netCDF4
    t0 = time.time()
    with h5py.File(src, 'r') as h:
        keys = list(h.keys())
        x = h['EWHA']
        nt, ny, nxx = x.shape
        out_fn.parent.mkdir(parents=True, exist_ok=True)
        nc = netCDF4.Dataset(out_fn, 'w', format='NETCDF4')
        nc.createDimension('epoch', nt); nc.createDimension('lat', ny); nc.createDimension('lon', nxx)
        la = nc.createVariable('lat', 'f8', ('lat',)); la.units = 'degrees_north'; la[:] = np.arange(-90 + 0.25, 90, 0.5)
        lo = nc.createVariable('lon', 'f8', ('lon',)); lo.units = 'degrees_east'; lo[:] = np.arange(-180 + 0.25, 180, 0.5)
        v = nc.createVariable('ewha', 'f4', ('epoch', 'lat', 'lon'), zlib=True, complevel=5, chunksizes=(1, ny, nxx))
        v.units, v.long_name = 'm', 'equivalent water height anomaly (5-day epoch, 0.5 deg; rows south to north as in the source)'
        for i in range(0, nt, 50):
            v[i:i + 50] = x[i:i + 50].astype(np.float32)
        for kk, val in attrs.items():
            setattr(nc, kk, val)
        nc.Conventions = 'CF-1.8'
        nc.close()
    return {'out': str(out_fn), 'MB': round(out_fn.stat().st_size / 1e6, 1), 's': round(time.time() - t0, 1), 'source': str(src)}


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        shutil.copy2(src, dst)
    else:
        return 'missing %s' % src


def jobs_products():
    J = []
    for b in BASINS:
        ref = 'jog_GRACEC_%s_s1_nominal' % b
        J.append((W3 / 'domains' / ('jog_%s1degree' % b) / 'OL' / 'pub_magic1e19' / 'products' / 'OL_mean', case_mask(ref),
                  OUT / '05_openloop' / ('OL_mean_%s.nc' % b),
                  dict(title='Open loop (W3RA), ensemble member 0 = published open loop', basin=b, period='2002-09-01..2006-12-31')))
        for m in MISSIONS:
            for a in ALPHAS:
                for g in GRIDS:
                    cfg = '' if (a, g) == ('1e18', '1degree') else '_%s_%s' % (a, g)
                    c = 'jog_%s_%s%s_s1_nominal' % (m, b, cfg)
                    J.append((W3 / 'cases' / c / 'products' / 'DA_mean', case_mask(c),
                              OUT / '06_da_main' / b / ('DA_mean_%s_%s_%s_%s_s1.nc' % (b, m, a, g)),
                              dict(title='EnKF DA ensemble mean (30 members), nominal covariance', basin=b, mission=MNAME[m],
                                   alpha=a, da_unit=g, seed=1, case=c)))
            for s in (2, 3):
                c = 'jog_%s_%s_s%d_nominal' % (m, b, s)
                J.append((W3 / 'cases' / c / 'products' / 'DA_mean', case_mask(c),
                          OUT / '09_seeds' / ('DA_mean_%s_%s_1e18_1degree_s%d.nc' % (b, m, s)),
                          dict(title='EnKF DA ensemble mean, nominal covariance, other seed', basin=b, mission=MNAME[m],
                               alpha='1e18', da_unit='1degree', seed=s, case=c)))
            for a in ALPHAS:
                c = ('jog_%s_%s_s1' % (m, b)) if a == '1e18' else ('jog_%s_%s_%s_1degree_s1' % (m, b, a))
                J.append((W3 / 'cases' / c / 'products' / 'DA_mean', case_mask(c),
                          OUT / '10_realistic' / ('DA_mean_%s_%s_%s_1degree_s1_realisticVCM.nc' % (b, m, a)),
                          dict(title='EnKF DA ensemble mean, realistic (residual-based) covariance', basin=b,
                               mission=MNAME[m], alpha=a, da_unit='1degree', seed=1, case=c)))
            for v, seeds, note in (('R01', (1, 2, 3), 'observations with reference degree 0/1 and SLR-like C20 (low-degree test)'),
                                   ('R34', (1,), 'observations without the fitted 10/15/30-day error terms'),
                                   ('R11', (1,), 'same-climate control: synthetic obs from a W3RA truth member (x1)'),
                                   ('R11x3', (1,), 'same-climate control: synthetic obs from a W3RA truth member (perturbation x3)')):
                for s in seeds:
                    c = 'jog_%s_%s_s%d_%s' % (m, b, s, v)
                    J.append((W3 / 'cases' / c / 'products' / 'DA_mean', case_mask(c),
                              OUT / '11_sensitivity' / v / ('DA_mean_%s_%s_1e18_1degree_s%d_%s.nc' % (b, m, s, v)),
                              dict(title='EnKF DA ensemble mean, sensitivity experiment: ' + note, basin=b,
                                   mission=MNAME[m], alpha='1e18', da_unit='1degree', seed=s, case=c, experiment=v)))
        for tag in ('s101', 's103x3'):
            c = 'jog_R11truth_%s_%s' % (b, tag)
            J.append((W3 / 'cases' / c / 'products' / 'truth', case_mask(ref),
                      OUT / '11_sensitivity' / 'R11_truths' / ('W3RA_truth_%s_%s.nc' % (b, tag)),
                      dict(title='synthetic W3RA truth member for the same-climate control', basin=b, case=c)))
    for m in ('GRACEC', 'MAGIC'):
        c = 'jog_R35_%s_s1' % m
        J.append((W3 / 'cases' / c / 'products' / 'DA_mean', case_mask(c),
                  OUT / '11_sensitivity' / 'R35_1996_2006' / ('DA_mean_Brahmaputra_%s_1e18_1degree_s1_1996_2006.nc' % m),
                  dict(title='EnKF DA ensemble mean, extended period 1996-2006', basin='Brahmaputra', mission=MNAME[m],
                       alpha='1e18', da_unit='1degree', seed=1, case=c)))
    return J


def copies():
    C = []
    C.append((D / 'basin' / 'shp' / 'ESA_SING', OUT / '01_shapes' / 'ESA_SING'))
    for b in BASINS:
        C.append((D / 'basin' / 'shp' / ('%s3subbasins' % b), OUT / '01_shapes' / ('%s3subbasins' % b)))
        C.append((GR / 'ESA_SING' / 'ESM2.0' / 'globe' / 'global_mask' / b, OUT / '02_masks' / '0.5deg' / b))
        for g in GRIDS:
            cfg = '' if g == '1degree' else '_1e18_%s' % g
            c = 'jog_GRACEC_%s%s_s1_nominal' % (b, cfg)
            C.append((case_mask(c), OUT / '02_masks' / '0.1deg' / ('mask_%s_%s.h5' % (b, g))))
        for m in MISSIONS:
            for a in ALPHAS:
                for g in GRIDS:
                    C.append((GR / 'output' / ('%s_old' % m) / a / ('%s%s_cov.hdf5' % (b, g)),
                              OUT / '08_covariances' / 'nominal' / m / a / ('%s%s_cov.hdf5' % (b, g))))
                C.append((GR / 'output' / m / a / ('%s1degree_cov.hdf5' % b),
                          OUT / '08_covariances' / 'realistic' / m / a / ('%s1degree_cov.hdf5' % b)))
    for tag in ('07_recompute_v1_rerun', '07_recompute_v1_rerun_s2', '07_recompute_v1_rerun_s3', '07_recompute_v1_realistic',
                '33_seed_mean_maps'):
        C.append((EV / 'results' / tag, OUT / '07_metrics' / tag))
    for tag in ('24_tables', '28_r11_truth', '28_r11_truth_x3', '29_two_member_grid', '30_low_degrees', '31_r01_compare',
                '32_rerun_vs_published', '34_innovation_ranking', '16_innovations', '17_increments', '14_pca', '15_truncation',
                '27_seed_disagreement', 'r35_full', 'r35_paperperiod', '20_r35_obs', '25_r35_snowmask'):
        C.append((EV / 'results' / tag, OUT / '12_derived' / tag))
    C.append((EV / 'results_v1', OUT / '12_derived' / 'results_v1'))
    C.append((EV / 'scripts', OUT / '13_repro' / 'evaluation_scripts'))
    C.append((EV / 'jog_eval', OUT / '13_repro' / 'evaluation_package_jog_eval'))
    return C


def settings_copy():
    n = 0
    for c in sorted((W3 / 'cases').glob('jog_*')):
        sd = c / 'settings'
        if sd.is_dir():
            dst = OUT / '13_repro' / 'run_settings' / c.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in sd.glob('*.json'):
                shutil.copy2(f, dst / f.name)
                n += 1
    return n


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=28)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--skip-global', action='store_true')
    ap.add_argument('--manifest-only', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = {}
    if not a.manifest_only:
        t0 = time.time()
        J = [j for j in jobs_products() if not a.only or j[2].relative_to(OUT).parts[0] in a.only]
        print('products to convert:', len(J), flush=True)
        with Pool(a.workers) as p:
            res = p.map(convert, J, chunksize=1)
        log['products'] = res
        print('products done %.0f s; errors: %s' % (time.time() - t0, [r for r in res if 'error' in r]), flush=True)
        if not a.skip_global and (not a.only or '03_truth' in a.only or '04_obs_prerelease' in a.only):
            G = [(GR / 'ESA_SING' / 'ESM2.0' / 'globe' / 'Global_EWHA_all' / 'Global_EWH_05_HIS.hdf5',
                  OUT / '03_truth' / 'ESM2_HIS_EWHA_5day_0.5deg_1995_2006.nc',
                  dict(title='ESA Earth System Model 2.0 HIS (truth), d/o 70 incl. degrees 0-1, 5-day, 0.5 deg'))]
            for m in MISSIONS:
                G.append((GR / 'ESA_SING' / 'ESM2.0' / 'globe' / 'Global_EWHA_all' / ('Global_EWH_05_%s_1e+18.hdf5' % MNAME[m]),
                          OUT / '04_obs_prerelease' / ('obs_%s_1e18_EWHA_5day_0.5deg.nc' % MNAME[m]),
                          dict(title='pre-release closed-loop simulated observations, filtered (alpha 1e18), 5-day, 0.5 deg',
                               mission=MNAME[m], alpha='1e18')))
            with Pool(4) as p:
                log['global'] = p.map(convert_global, G)
            print('global grids done', flush=True)
        miss = [cp(s, d) for s, d in copies()]
        log['copy_missing'] = [m for m in miss if m]
        log['settings_files'] = settings_copy()
        (OUT / 'conversion_log.json').write_text(json.dumps(log, indent=1, default=str))
        print('copies done; missing:', log['copy_missing'], flush=True)
    files = sorted(f for f in OUT.rglob('*') if f.is_file() and f.name != 'MANIFEST.md5')
    with Pool(a.workers) as p:
        sums = p.map(md5, files, chunksize=4)
    with open(OUT / 'MANIFEST.md5', 'w') as f:
        for s, fn in zip(sums, files):
            f.write('%s  %s\n' % (s, fn.relative_to(OUT)))
    items = {}
    for fn in files:
        it = fn.relative_to(OUT).parts[0]
        n, sz = items.get(it, (0, 0))
        items[it] = (n + 1, sz + fn.stat().st_size)
    lines = ['%-22s %6d files %8.2f GB' % (k, n, sz / 1e9) for k, (n, sz) in sorted(items.items())]
    lines.append('%-22s %6d files %8.2f GB' % ('TOTAL', sum(n for n, _ in items.values()), sum(s for _, s in items.values()) / 1e9))
    (OUT / 'ITEMS.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
