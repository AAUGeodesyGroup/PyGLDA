"""Step 17 (R4.5) - share of the TWS increment per storage compartment (DA runs with output.save_innovations).

inc_<compartment> in cases/<case>/DA/innovations.h5 = ensemble-mean increment of each W3RA store in a 5-day window, mapped to the
observation units with the DA's own operator (Fhru-weighted, Mleaf x4, unit means; their sum = hxa_mean - hx_mean = the TWS
increment). Basin series = unit increments weighted by the unit areas (sub_basin_area of the observation product, if given via
--areas-from, else equal weights). Per run: per compartment the share of the absolute increment, sum_t |inc_c| / sum_t sum_c |inc_c|,
the signed share mean(inc_c) / mean(inc_TWS) (net water added), and the time series (CSV).
Usage: python 17_increments.py --cases jog_MAGIC_Brahmaputra_s1 [...] [--data-root ...] [--areas-from <..._signal.hdf5>]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from _common import write_csv, write_json, C

COMP = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--data-root', default=str(C.V1_DATA_ROOT))
    p.add_argument('--out', default=str(C.OUT_ROOT))
    p.add_argument('--areas-from', nargs='*', default=[], help='per case (same order) an obs signal file with sub_basin_area')
    a = p.parse_args()
    out = Path(a.out) / '17_increments'
    out.mkdir(parents=True, exist_ok=True)
    table, full = [], {}
    for i, case in enumerate(a.cases):
        fn = Path(a.data_root) / 'w3ra' / 'cases' / case / 'DA' / 'innovations.h5'
        with h5py.File(fn, 'r') as f:
            keys = sorted(f.keys())
            inc = {c: np.array([f[w]['inc_%s' % c][()] for w in keys]) for c in COMP if 'inc_%s' % c in f[keys[0]]}
            epochs = [f[w].attrs['epoch'] for w in keys]
        m = next(iter(inc.values())).shape[1]
        w = np.ones(m)
        if i < len(a.areas_from):
            with h5py.File(a.areas_from[i], 'r') as g:
                w = np.ravel(g['sub_basin_area'][()]).astype(float)
        w = w / w.sum()
        basin = {c: v @ w for c, v in inc.items()}
        tws = sum(basin.values())
        absum = sum(np.abs(v).sum() for v in basin.values())
        res = {c: dict(abs_share=float(np.abs(v).sum() / absum), signed_share=float(v.mean() / tws.mean()) if tws.mean() else None,
                       mean_mm=float(v.mean()), rms_mm=float(np.sqrt((v ** 2).mean()))) for c, v in basin.items()}
        full[case] = dict(windows=len(epochs), tws_mean_mm=float(tws.mean()), tws_rms_mm=float(np.sqrt((tws ** 2).mean())),
                          compartments=res)
        for c, r in res.items():
            table.append(dict(case=case, compartment=c, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
        write_csv(out / ('%s_series.csv' % case), [dict(epoch=e, TWS=round(float(tws[t]), 4),
                                                        **{c: round(float(basin[c][t]), 4) for c in basin})
                                                   for t, e in enumerate(epochs)])
    write_csv(out / 'increments_shares.csv', table)
    write_json(out / 'increments_shares.json', full)
    print('written', out)


if __name__ == '__main__':
    main()
