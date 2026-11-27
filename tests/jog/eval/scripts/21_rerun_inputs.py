"""Step 21 - input table for a full rerun of the 180 published configurations.

For mission x alpha x basin x grid and both obs folders GRACE/output/<M>_old/<alpha>/ and GRACE/output/<M>/<alpha>/:
files <basin><grid>_{signal,gridded_signal,cov}.hdf5: existence, mtime, md5, size; whether the two folders hold the
same file (md5); cov: shape, diagonal-only?, max |off-diagonal|, max |cov - VCM_v5 x 1e6| (full) and
max |diag(cov) - diag(VCM_v5) x 1e6|; signal: number of sub_basin_k, max |signal - EWH_v5 x 1000| over the obs epochs
(direct order k -> row k-1, and best 1:1 match); v5 group = the group of <mission>_<a>_<basin>_v5.hdf5 matching best.
Also: DA unit shapefiles (basin/shp/ESA_SING/Grid_*/, record count from the .dbf header) and the res/<case> mtimes.
Outputs: results/21_rerun_inputs/{input_table.csv, shapefiles.csv, res_dates.csv, summary.json}
Usage: python 21_rerun_inputs.py
"""
import hashlib
import struct
import time
from datetime import date, datetime, timedelta

import h5py
import numpy as np

from _common import parser, outdir, write_csv, write_json, C

OUT = C.V1_DATA_ROOT / 'GRACE' / 'output'
SHP = C.V1_DATA_ROOT / 'basin' / 'shp'
RES = C.V1_DATA_ROOT / 'w3ra' / 'res'
APRIL_END = datetime(2025, 4, 23, 23, 59, 59)
KINDS = ('signal', 'gridded_signal', 'cov')


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def mtime(fn):
    return datetime.fromtimestamp(fn.stat().st_mtime)


def v5_groups(mission, alpha, basin):
    fn = C.ESM2_ROOT / basin / 'gravity_signal_cov' / ('%s_%s_%s_v5.hdf5' % (C.MISSION_FILE[mission], C.alpha_file(alpha), basin))
    mf = C.MISSION_FILE[mission]
    out = {}
    with h5py.File(fn, 'r') as h:
        for g in h:
            if isinstance(h[g], h5py.Group) and mf in h[g]:
                a = C.alpha_file(alpha)
                out[g] = (h[g][mf]['EWH_alpha_%s' % a][:] * 1000.0, h[g][mf]['VCM_alpha_%s' % a][:] * 1e6)
    return fn, out


def dbf_records(fn):
    with open(fn, 'rb') as f:
        head = f.read(8)
    return struct.unpack('<I', head[4:8])[0]


def obs_epochs(te, starts):
    mid = {s + timedelta(days=2): i for i, s in enumerate(starts)}
    return np.array([mid[date.fromisoformat(x.decode() if isinstance(x, bytes) else str(x))] for x in te])


def check_signal(fn, v5, starts):
    with h5py.File(fn, 'r') as h:
        keys = sorted([k for k in h if k.startswith('sub_basin_') and k != 'sub_basin_area'], key=lambda s: int(s[10:]))
        x = np.stack([h[k][:] for k in keys], 0)
        ep = obs_epochs(h['time_epoch'][:], starts)
    res = {'n_units': len(keys), 'n_epochs': x.shape[1], 'epochs': '%d..%d' % (ep[0], ep[-1])}
    best = None
    for g, (ewh, _) in v5.items():
        if ewh.shape[0] != x.shape[0]:
            continue
        ref = ewh[:, ep]
        direct = float(np.abs(ref - x).max())
        d = np.abs(ref[None, :, :] - x[:, None, :]).max(2)           # (prod unit, v5 row)
        perm = d.argmin(1)
        matched = float(d[np.arange(len(perm)), perm].max()) if len(set(perm)) == len(perm) else np.inf
        if best is None or min(direct, matched) < best[1]:
            best = (g, min(direct, matched), direct, matched, bool(np.all(perm == np.arange(len(perm)))))
    if best:
        res.update({'v5_group': best[0], 'sig_maxdiff_direct_mm': best[2], 'sig_maxdiff_matched_mm': best[3],
                    'sig_order_identity': best[4]})
    return res, keys


def check_cov(fn, v5, group):
    with h5py.File(fn, 'r') as h:
        c = h['data'][:]
    off = c - np.diag(np.diag(c))
    r = {'cov_shape': 'x'.join(map(str, c.shape)), 'cov_max_offdiag': float(np.abs(off).max()),
         'cov_diag_only': bool(np.abs(off).max() == 0), 'cov_diag_median': float(np.median(np.diag(c)))}
    cands = [group] if group in v5 else [g for g in v5 if v5[g][1].shape == c.shape]
    for g in cands:
        V = v5[g][1]
        if V.shape != c.shape:
            continue
        r.update({'cov_v5_group': g, 'cov_maxdiff_full_mm2': float(np.abs(c - V).max()),
                  'cov_maxreldiff_full': float(np.abs(c - V).max() / np.abs(V).max()),
                  'cov_maxdiff_diag_mm2': float(np.abs(np.diag(c) - np.diag(V)).max()),
                  'cov_equals_diag_v5': bool(np.allclose(c, np.diag(np.diag(V)), rtol=1e-9, atol=0)),
                  'cov_equals_full_v5': bool(np.allclose(c, V, rtol=1e-9, atol=0))})
        break
    return r


def main():
    t0 = time.time()
    args = parser(__doc__).parse_args()
    out = outdir(args, '21_rerun_inputs')
    from jog_eval import sources as S
    starts = S.epoch_windows()[0]
    rows = []
    for basin in args.basins:
        for mission in C.MISSIONS:
            for alpha in C.ALPHAS:
                v5fn, v5 = v5_groups(mission, alpha, basin)
                for grid in C.GRIDS:
                    md5s = {}
                    for folder in (mission + '_old', mission):
                        group = None
                        for kind in KINDS:
                            fn = OUT / folder / alpha / ('%s%s_%s.hdf5' % (basin, grid, kind))
                            row = {'basin': basin, 'mission': mission, 'alpha': alpha, 'grid': grid, 'folder': folder,
                                   'kind': kind, 'exists': fn.exists(), 'path': str(fn)}
                            if fn.exists():
                                mt = mtime(fn)
                                row.update({'mtime': mt.strftime('%Y-%m-%d %H:%M:%S'), 'le_23apr2025': mt <= APRIL_END,
                                            'size': fn.stat().st_size, 'md5': md5(fn)})
                                md5s[(folder, kind)] = row['md5']
                                if kind == 'signal':
                                    r, _ = check_signal(fn, v5, starts)
                                    group = r.get('v5_group')
                                    row.update(r)
                                elif kind == 'cov':
                                    row.update(check_cov(fn, v5, group))
                                elif kind == 'gridded_signal':
                                    with h5py.File(fn, 'r') as h:
                                        row['n_units'] = 'tws ' + 'x'.join(map(str, h['tws'].shape))
                            row['v5_file'] = str(v5fn)
                            rows.append(row)
                    for r in rows[-6:]:
                        other = (mission if r['folder'].endswith('_old') else mission + '_old', r['kind'])
                        r['same_md5_other_folder'] = (md5s.get((r['folder'], r['kind'])) == md5s.get(other)
                                                      if (r['folder'], r['kind']) in md5s and other in md5s else '')
            print('done', basin, mission, round(time.time() - t0, 1), 's')
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    rows = [{k: r.get(k, '') for k in keys} for r in rows]
    write_csv(out / 'input_table.csv', rows)
    # shapefiles
    shp = []
    for d in sorted((SHP / 'ESA_SING').glob('Grid_*')) + sorted({p.parent for p in SHP.rglob('*3subbasins*.dbf')} |
                                                                  {p.parent for p in SHP.rglob('*subbasins_3*.dbf')}):
        for f in sorted(d.glob('*.dbf')):
            shp.append({'folder': str(d.relative_to(SHP)), 'file': f.name, 'records': dbf_records(f),
                        'mtime': mtime(f).strftime('%Y-%m-%d %H:%M:%S')})
    write_csv(out / 'shapefiles.csv', shp)
    # res/ dates
    res = []
    for basin in args.basins:
        for mission in C.MISSIONS:
            for alpha in C.ALPHAS:
                for grid in C.GRIDS:
                    case = C.case_name(mission, alpha, grid, basin)
                    fn = RES / case / ('Res_DA_%s%s.h5df' % (basin, grid))
                    row = {'case': case, 'grid': grid, 'exists': fn.exists()}
                    if fn.exists():
                        row['res_da_mtime'] = mtime(fn).strftime('%Y-%m-%d %H:%M:%S')
                        cov_new = OUT / mission / alpha / ('%s%s_cov.hdf5' % (basin, grid))
                        row['res_after_23apr2025'] = mtime(fn) > APRIL_END
                    res.append(row)
    write_csv(out / 'res_dates.csv', res)
    # summary
    summ = {'n_rows': len(rows), 'missing': [r['path'] for r in rows if not r['exists']]}
    for folder in ('old', 'new'):
        sel = [r for r in rows if r['exists'] and r['folder'].endswith('_old') == (folder == 'old')]
        for kind in KINDS:
            s = [r for r in sel if r['kind'] == kind]
            summ['%s_%s_n' % (folder, kind)] = len(s)
            summ['%s_%s_all_le_23apr' % (folder, kind)] = all(r['le_23apr2025'] for r in s)
            summ['%s_%s_after_23apr' % (folder, kind)] = sorted({'%s/%s %s' % (r['mission'], r['alpha'], r['basin'] + r['grid'])
                                                                 for r in s if not r['le_23apr2025']})
    for grid in C.GRIDS:
        for folder in ('old', 'new'):
            s = [r for r in rows if r['exists'] and r['kind'] == 'cov' and r['grid'] == grid and r['folder'].endswith('_old') == (folder == 'old')]
            summ['cov_%s_%s' % (grid, folder)] = {
                'n': len(s), 'diag_only': sum(bool(r['cov_diag_only']) for r in s),
                'equals_full_v5': sum(r.get('cov_equals_full_v5') is True for r in s),
                'equals_diag_v5': sum(r.get('cov_equals_diag_v5') is True for r in s),
                'max_reldiff_full': max([r['cov_maxreldiff_full'] for r in s if r.get('cov_maxreldiff_full') != ''] or [np.nan]),
                'max_diag_diff_mm2': max([r['cov_maxdiff_diag_mm2'] for r in s if r.get('cov_maxdiff_diag_mm2') != ''] or [np.nan]),
                'same_md5_as_other_folder': sum(r['same_md5_other_folder'] is True for r in s)}
            sg = [r for r in rows if r['exists'] and r['kind'] == 'signal' and r['grid'] == grid and r['folder'].endswith('_old') == (folder == 'old')]
            summ['signal_%s_%s' % (grid, folder)] = {
                'n': len(sg), 'v5_groups': sorted({str(r.get('v5_group')) for r in sg}),
                'n_units': sorted({'%s:%s' % (r['basin'], r['n_units']) for r in sg}),
                'max_diff_direct_mm': max([r['sig_maxdiff_direct_mm'] for r in sg if r.get('sig_maxdiff_direct_mm') != ''] or [np.nan]),
                'max_diff_matched_mm': max([r['sig_maxdiff_matched_mm'] for r in sg if r.get('sig_maxdiff_matched_mm') != ''] or [np.nan]),
                'order_identity': sum(r.get('sig_order_identity') is True for r in sg),
                'same_md5_as_other_folder': sum(r['same_md5_other_folder'] is True for r in sg)}
    summ['res_dates'] = {g: sorted({r['res_da_mtime'][:10] for r in res if r['grid'] == g and r['exists']}) for g in C.GRIDS}
    summ['res_missing'] = [r['case'] for r in res if not r['exists']]
    summ['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'summary.json', summ)
    print('written', out, summ['run_time_s'], 's')


if __name__ == '__main__':
    main()
