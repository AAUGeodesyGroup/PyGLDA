"""Step 18 (R3.4 part 2) - observation products without the fitted 10/15/30-day error terms.

For every mission and basin (alpha 1e18, 1-degree units):
  copy GRACE/output/<M>/1e18/<basin>1degree_{signal,gridded_signal,cov}.hdf5 to GRACE/output_R34/<M>/1e18/ (copy2),
  fit script 08's model (periodic.fit: const, trend, annual, semi-annual, 15 d, 30 d, (-1)^k) on r = obs - truth
  (both de-meaned, epochs 561..875) and subtract ONLY the periodic part p (10/15/30 d) from each sub_basin_<k> series
  of the copied signal file (in place, same dataset, dtype and attributes).
  The nuisance terms and the unfitted lines (12.08 d, 10.81 d, 12.73 d, ~40 d, ~52 d) stay in the products.
Checks: md5 of cov / gridded_signal equal to the published files; dataset-by-dataset comparison of the signal file
(only sub_basin_<k> values differ); refit on cleaned - truth gives periodic amplitudes ~0.
Report: results/18_clean_periodic/{check.json, datasets.csv}.
Usage: python 18_clean_periodic.py [--dest <root>] [--overwrite]
"""
import hashlib
import importlib
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import metrics as M, periodic as PE

S08 = importlib.import_module('08_periodic_errors')
KINDS = ('signal', 'gridded_signal', 'cov')


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def compare(src, dst, basin, mission):
    """every object of src vs dst: type, shape, dtype, attributes, values"""
    rows = []
    with h5py.File(src, 'r') as a, h5py.File(dst, 'r') as b:
        def attrs_eq(x, y):
            return set(x.attrs) == set(y.attrs) and all(np.array_equal(np.asarray(x.attrs[k]), np.asarray(y.attrs[k]))
                                                        for k in x.attrs)
        rows.append({'basin': basin, 'mission': mission, 'object': '/', 'kind': 'file', 'shape': '', 'dtype': '',
                     'attrs_equal': attrs_eq(a, b), 'values_equal': '', 'max_abs_diff': ''})
        names = []
        a.visit(names.append)
        nb = []
        b.visit(nb.append)
        assert sorted(names) == sorted(nb), 'object lists differ'
        for n in names:
            x, y = a[n], b[n]
            if isinstance(x, h5py.Group):
                rows.append({'basin': basin, 'mission': mission, 'object': n, 'kind': 'group', 'shape': '', 'dtype': '',
                             'attrs_equal': attrs_eq(x, y), 'values_equal': '', 'max_abs_diff': ''})
                continue
            assert x.shape == y.shape and x.dtype == y.dtype, n
            u, v = x[()], y[()]
            eq = bool(np.array_equal(u, v))
            diff = float(np.max(np.abs(u - v))) if np.issubdtype(x.dtype, np.number) and x.size else ''
            rows.append({'basin': basin, 'mission': mission, 'object': n, 'kind': 'dataset', 'shape': str(x.shape),
                         'dtype': str(x.dtype), 'attrs_equal': attrs_eq(x, y), 'values_equal': eq, 'max_abs_diff': diff})
    return rows


def main():
    t0 = time.time()
    ap = parser(__doc__)
    ap.add_argument('--dest', default=str(C.V1_DATA_ROOT / 'GRACE' / 'output_R34'))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    out = outdir(args, '18_clean_periodic')
    k = np.arange(*S08.DA_EPOCHS)
    periods = (15.0, 30.0)
    rows, check = [], {'dest': args.dest, 'periods_removed': [10.0, 15.0, 30.0], 'files': {}}
    for basin in args.basins:
        truth_all, prods, w, starts, his = S08.load_basin(basin)
        truth = M.demean(truth_all[k])
        for m in C.MISSIONS:
            src_dir = S08.OBS_ROOT / m / '1e18'
            dst_dir = Path(args.dest) / m / '1e18'
            dst_dir.mkdir(parents=True, exist_ok=True)
            fns = {kd: '%s1degree_%s.hdf5' % (basin, kd) for kd in KINDS}
            for kd, fn in fns.items():
                if (dst_dir / fn).exists() and not args.overwrite:
                    raise SystemExit('%s exists (use --overwrite)' % (dst_dir / fn))
                shutil.copy2(src_dir / fn, dst_dir / fn)
            obs, mapping, _ = S08.load_obs(basin, m, prods[m], starts)        # Grid_1 order, mm
            r = M.demean(obs) - truth
            F = PE.fit(r, k, periods)
            p = F['p']
            with h5py.File(dst_dir / fns['signal'], 'r+') as h:
                te = [S08._d(x) for x in h['time_epoch'][:]]
                assert len(te) == len(k)
                for mp in mapping:
                    d = h[mp['key']]
                    d[...] = (d[()] - p[:, mp['grid1_index']]).astype(d.dtype)
            with h5py.File(dst_dir / fns['signal'], 'r') as h:
                cl = np.full_like(obs, np.nan)
                for mp in mapping:
                    cl[:, mp['grid1_index']] = h[mp['key']][()]
            F2 = PE.fit(M.demean(cl) - truth, k, periods)
            refit = {('%g' % P): float(np.max(a)) for P, a in F2['amp'].items()}
            rows += compare(src_dir / fns['signal'], dst_dir / fns['signal'], basin, m)
            sub = [r_ for r_ in rows if r_['basin'] == basin and r_['mission'] == m]
            changed = [r_['object'] for r_ in sub if r_['kind'] == 'dataset' and not r_['values_equal']]
            check['files']['%s_%s' % (m, basin)] = {
                'md5': {kd: {'published': md5(src_dir / fn), 'R34': md5(dst_dir / fn)} for kd, fn in fns.items()},
                'cov_gridded_identical': all(md5(src_dir / fns[kd]) == md5(dst_dir / fns[kd]) for kd in ('cov', 'gridded_signal')),
                'n_changed_datasets': len(changed),
                'changed_only_sub_basin_series': all(c.startswith('sub_basin_') and c != 'sub_basin_area' for c in changed),
                'n_sub_basin_series': len(mapping),
                'all_attrs_equal': all(r_['attrs_equal'] for r_ in sub),
                'max_abs_change_mm': float(np.max(np.abs(p))),
                'max_abs_cleaned_minus_(obs-p)_mm': float(np.nanmax(np.abs(cl - (obs - p)))),
                'refit_amp_max_mm': refit,
                'amp_removed_median_mm': {('%g' % P): float(np.median(a)) for P, a in F['amp'].items()},
            }
            print(m, basin, check['files']['%s_%s' % (m, basin)]['n_changed_datasets'], refit, flush=True)
    check['run_time_s'] = round(time.time() - t0, 1)
    write_csv(out / 'datasets.csv', rows)
    write_json(out / 'check.json', check)
    print('written', out, check['run_time_s'], 's')


if __name__ == '__main__':
    main()
