"""Step 25 (R3.5 sensitivity) - 1996-2006 Brahmaputra extension without the perennial-snow cells.

Perennial snow: 0.1 deg cells whose Fhru-weighted DrySnow (sum over HRUs of Fhru * DrySnow) stays >= 1 mm on every
day of OL member 0 (domains/<domain>/OL/<ensemble>/ens_00/state/state.YYYYMMDD.h5, 1996-01-01..2006-12-31; Fhru from
domains/<domain>/input/par/par.h5 data/Fhru). State cells = non-zero cells of input/mask/mask.h5 (row-major).
Every 0.5 deg cell that contains such a 0.1 deg cell (same 5x5 blocks as sources.upscale_to_half_degree) is excluded.
Metrics as step 05 (paper and published_code definitions, Table-1 numbers + 95 % block-bootstrap CIs, block 18,
pair MAGIC - GRACEC) for all cells and without the excluded cells, for 1996-2006 (epochs 74..875, as r35_full) and the
paper period (634..875, as r35_paperperiod).
Usage: python 25_r35_snowmask.py --domain jog_R35_Brahmaputra1degree --ensemble r35_s1
                                 --cases GRACEC=jog_R35_GRACEC_s1 MAGIC=jog_R35_MAGIC_s1
"""
import time
from pathlib import Path

import h5py
import numpy as np

from _common import parser, outdir, write_csv, write_json, C
from jog_eval import evaluate as E, metrics as M, sources as S, stats as ST

PERIODS = {'1996-2006': (74, 876), 'paper_period': C.PAPER_EPOCHS}


def perennial(domain, ensemble):
    root = C.V1_DATA_ROOT / 'w3ra' / 'domains' / domain
    with h5py.File(root / 'input' / 'par' / 'par.h5', 'r') as h:
        fhru = h['data/Fhru'][:]
    with h5py.File(root / 'input' / 'mask' / 'mask.h5', 'r') as h:
        dmask = h['mask'][:] != 0
    files = sorted((root / 'OL' / ensemble / 'ens_00' / 'state').glob('state.*.h5'))
    low = None
    for fn in files:
        with h5py.File(fn, 'r') as h:
            s = np.sum(fhru * h['DrySnow'][:], 0)
        low = s if low is None else np.minimum(low, s)
    assert dmask.sum() == low.size, (dmask.sum(), low.size)
    per = np.zeros(dmask.shape, bool)
    per[dmask] = low >= 1.0
    return per, len(files), files[0].name, files[-1].name


def main():
    t0 = time.time()
    p = parser(__doc__)
    p.add_argument('--domain', required=True)
    p.add_argument('--ensemble', required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--mask-case', default=C.case_name('MAGIC', '1e18', '1degree', 'Brahmaputra'))
    p.add_argument('--n-boot', type=int, default=1000)
    args = p.parse_args()
    out = outdir(args, '25_r35_snowmask')
    basin = 'Brahmaputra'
    per01, ndays, d0, d1 = perennial(args.domain, args.ensemble)
    mask01, lat01, lon01 = S.case_mask(args.mask_case)
    assert per01.shape == mask01.shape
    sub = per01[2:-3, 2:-3]
    r, c = sub.shape
    excl = sub.reshape(r // 5, 5, c // 5, 5).any(axis=(1, 3))
    meta = {'state_days': ndays, 'first': d0, 'last': d1, 'perennial_01_domain': int(per01.sum()),
            'perennial_01_in_basin_mask': int((per01 & mask01).sum())}
    print('perennial', meta, round(time.time() - t0, 1), 's', flush=True)
    cases = dict(x.split('=') for x in args.cases)
    first = S.case_products(next(iter(cases.values())), args.domain, args.ensemble)
    folders = {'model': ('OL', first['OL'], args.mask_case)}
    for m, cs in cases.items():
        folders[m] = ('DA', S.case_products(cs, args.domain, args.ensemble)['DA'], args.mask_case)
    pairs = [(m, 'GRACEC') for m in cases if m != 'GRACEC']
    rows = []
    for per, (e0, e1) in PERIODS.items():
        epochs = np.arange(e0, e1)
        data = E.load_basin(basin, epochs, folders=folders)
        ok = np.isfinite(data['truth']).all(0)
        assert excl.shape == ok.shape, (excl.shape, ok.shape)
        meta[per] = {'cells_all': int(ok.sum()), 'cells_excluded': int((ok & excl).sum()), 'epochs': [e0, e1 - 1],
                     'missing': {k: len(v) for k, v in data['missing'].items()}}
        for sel_name, sel in (('all_cells', ok), ('without_perennial_snow', ok & ~excl)):
            truth = np.where(sel[None], data['truth'], np.nan)
            sims = {k: np.where(sel[None], v, np.nan) for k, v in data['sims'].items()}
            for d in ('paper', 'published_code'):
                bt = ST.bootstrap_table(truth, sims, definition=d, block=18, n_boot=args.n_boot, pairs=pairs)
                for k in list(sims) + ['%s-%s' % pr for pr in pairs]:
                    for n in ('nse', 'kge', 'rmse'):
                        pt = bt['point'][k][n] if k in bt['point'] else bt['point'][k.split('-')[0]][n] - bt['point'][k.split('-')[1]][n]
                        lo, hi = bt['ci'][k][n]
                        rows.append({'period': per, 'cells': sel_name, 'n_cells': int(sel.sum()), 'definition': d,
                                     'product': k, 'metric': n, 'value': round(pt, 4), 'ci_low': round(lo, 4),
                                     'ci_high': round(hi, 4)})
            print(per, sel_name, round(time.time() - t0, 1), 's', flush=True)
    write_csv(out / 'snowmask_metrics.csv', rows)
    np.save(out / 'excluded_cells_05deg.npy', excl)
    meta['run_time_s'] = round(time.time() - t0, 1)
    write_json(out / 'meta.json', meta)
    print('written', out, meta['run_time_s'], 's')


if __name__ == '__main__':
    main()
