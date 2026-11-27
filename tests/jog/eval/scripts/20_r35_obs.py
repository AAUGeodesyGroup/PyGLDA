"""Step 20 (R3.5) - observation products for the Brahmaputra 1996-2006 extension (alpha 1e18, 1-degree units).

For M in MAGIC, GRACEC writes to <dest>/<M>/1e18/ (default GRACE/output_R35):
  Brahmaputra1degree_signal.hdf5          sub_basin_<k> (mm) = v5 Grid_1/<mission>/EWH_alpha_1e+18 row k-1 (m -> mm),
                                          sub_basin_area copied from the published file, time_epoch = start + 2 d
  Brahmaputra1degree_cov.hdf5             NOMINAL: data = diag(diag(v5 VCM_alpha_1e+18)) (m2 -> mm2), time_epoch
  Brahmaputra1degree_gridded_signal.hdf5  tws (epoch, cell) from Global_EWH_05_<mission>_1e+18.hdf5 (flipped along lat,
                                          m -> mm) on the 0.5-deg cells of the published file (found by value matching)
Epochs: ESM2.0 indices 73..875 (1995-12-31 .. 2006-12-27, all 5-day windows; the 4-day epoch 1995-12-27..30 is left out).
Validation: on the 315 published epochs the signal and gridded_signal equal GRACE/output/<M>/1e18 and the cov equals
GRACE/output/<M>_old/1e18 (max |diff|); epoch list vs GRACE/aux/ESA_5daily (obs_auxiliary.aux_ESAsing_5daily rule).
Report: results/20_r35_obs/check.json.
Usage: python 20_r35_obs.py [--dest <root>] [--overwrite]
"""
import hashlib
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from _common import parser, outdir, write_json, C

BASIN = 'Brahmaputra'
MISSIONS = ('MAGIC', 'GRACEC')
FIRST, LAST = 73, 875
GRACE = C.V1_DATA_ROOT / 'GRACE'
STR = h5py.string_dtype(encoding='utf-8')


def md5(fn):
    return hashlib.md5(Path(fn).read_bytes()).hexdigest()


def epochs():
    with h5py.File(C.basin_signal_file(BASIN), 'r') as h:
        d = h['dates'][:]
    s = [datetime.strptime(x.decode(), '%Y-%m-%d').date() for x in d[0]]
    e = [datetime.strptime(x.decode(), '%Y-%m-%d').date() for x in d[1]]
    idx = np.arange(FIRST, LAST + 1)
    assert all((e[i] - s[i]).days == 5 for i in idx)
    return idx, [(s[i] + timedelta(days=2)).isoformat() for i in idx], (s[FIRST], e[LAST] - timedelta(days=1))


def aux_epochs(day_begin, day_end):
    """obs_auxiliary.aux_ESAsing_5daily.setTimeReference, read-only copy of its rule"""
    out = []
    for year in range(1995, 2007):
        for ff in sorted(os.listdir(GRACE / 'aux' / 'ESA_5daily' / str(year))):
            a, b = ff.split('_')[4].split('-')
            s1, s2 = datetime.strptime(a, '%Y%m%d').date(), datetime.strptime(b, '%Y%m%d').date() - timedelta(days=1)
            if s1 >= day_begin and s2 <= day_end:
                out.append((s1 + timedelta(days=2)).isoformat())
    return out


def write_te(h, te):
    h.create_dataset('time_epoch', data=np.array(te, dtype=object), dtype=STR)


def find_cells(pub_tws, field0, field1):
    """indices (flat, row-major) of the 0.5-deg cells whose values equal the published columns"""
    flat0, flat1 = field0.ravel(), field1.ravel()
    cells = []
    for j in range(pub_tws.shape[1]):
        c = np.flatnonzero(flat0 == pub_tws[0, j])
        if len(c) > 1:
            c = c[flat1[c] == pub_tws[1, j]]
        assert len(c) == 1, 'cell %d: %d matches' % (j, len(c))
        cells.append(int(c[0]))
    return np.array(cells)


def main():
    t0 = time.time()
    ap = parser(__doc__)
    ap.add_argument('--dest', default=str(GRACE / 'output_R35'))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    out = outdir(args, '20_r35_obs')
    idx, te, (d0, d1) = epochs()
    aux = aux_epochs(d0, d1)
    check = {'n_epochs': len(te), 'first': te[0], 'last': te[-1], 'index_range': [FIRST, LAST],
             'aux_ESA_5daily_same_epochs': aux == te, 'aux_n': len(aux),
             'aux_full_1995_2006_n': len(aux_epochs(date(1995, 1, 1), date(2006, 12, 31)))}
    for m in MISSIONS:
        mf = C.MISSION_FILE[m]
        pub = GRACE / 'output' / m / '1e18'
        old = GRACE / 'output' / (m + '_old') / '1e18'
        dst = Path(args.dest) / m / '1e18'
        dst.mkdir(parents=True, exist_ok=True)
        names = {k: '%s1degree_%s.hdf5' % (BASIN, k) for k in ('signal', 'cov', 'gridded_signal')}
        for fn in names.values():
            if (dst / fn).exists() and not args.overwrite:
                raise SystemExit('%s exists (use --overwrite)' % (dst / fn))
        v5 = C.ESM2_ROOT / BASIN / 'gravity_signal_cov' / ('%s_1e+18_%s_v5.hdf5' % (mf, BASIN))
        with h5py.File(v5, 'r') as h:
            ewh = h['Grid_1/%s/EWH_alpha_1e+18' % mf][:, idx] * 1000.0            # (units, epochs) mm
            vcm = h['Grid_1/%s/VCM_alpha_1e+18' % mf][:] * 1e6                      # mm2
        n = ewh.shape[0]
        with h5py.File(pub / names['signal'], 'r') as p:
            pte = [x.decode() for x in p['time_epoch'][:]]
            area = p['sub_basin_area'][:]
            pser = np.stack([p['sub_basin_%d' % (k + 1)][:] for k in range(n)], 0)
            assert len([k for k in p if k.startswith('sub_basin_') and k != 'sub_basin_area']) == n
        pos = np.array([te.index(t) for t in pte])
        # signal
        with h5py.File(dst / names['signal'], 'w') as h:
            for k in range(n):
                h.create_dataset('sub_basin_%d' % (k + 1), data=ewh[k])
            h.create_dataset('sub_basin_area', data=area)
            write_te(h, te)
        # cov (nominal, diagonal only)
        cov = np.diag(np.diag(vcm))
        with h5py.File(dst / names['cov'], 'w') as h:
            h.create_dataset('data', data=cov)
            write_te(h, te)
        with h5py.File(old / names['cov'], 'r') as o:
            ocov = o['data'][:]
        # gridded signal
        gfile = C.ESM2_ROOT / 'globe' / 'Global_EWHA_all' / ('Global_EWH_05_%s_1e+18.hdf5' % mf)
        with h5py.File(pub / names['gridded_signal'], 'r') as p:
            ptws = p['tws'][:]
        with h5py.File(gfile, 'r') as g:
            gt = [x.decode() for x in g['time'][0]]
            assert len(gt) == 876
            f0 = np.flip(g['EWHA'][idx[pos[0]]], axis=-2) * 1000.0
            f1 = np.flip(g['EWHA'][idx[pos[1]]], axis=-2) * 1000.0
            cells = find_cells(ptws, f0, f1)
            rows, cols = np.unravel_index(cells, f0.shape)
            tws = np.stack([np.flip(g['EWHA'][i], axis=-2)[rows, cols] * 1000.0 for i in idx], 0)
        with h5py.File(dst / names['gridded_signal'], 'w') as h:
            h.create_dataset('tws', data=tws)
            write_te(h, te)
        check[m] = {
            'v5_file': str(v5), 'n_units': n, 'published_epochs_in_new': int(len(pos)),
            'signal_max_abs_diff_vs_published_mm': float(np.max(np.abs(ewh[:, pos] - pser))),
            'cov_max_abs_diff_vs_old_mm2': float(np.max(np.abs(cov - ocov))),
            'cov_equal_old_exactly': bool(np.array_equal(cov, ocov)),
            'cov_v5_offdiag_dropped_max_mm2': float(np.max(np.abs(vcm - np.diag(np.diag(vcm))))),
            'gridded_cells': int(len(cells)), 'gridded_cells_row_major': bool(np.all(np.diff(cells) > 0)),
            'gridded_max_abs_diff_vs_published_mm': float(np.max(np.abs(tws[pos] - ptws))),
            'md5': {k: md5(dst / fn) for k, fn in names.items()},
        }
        print(m, check[m], flush=True)
    check['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'check.json', check)
    print('written', out, check['run_time_s'], 's')


if __name__ == '__main__':
    main()
