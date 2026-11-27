"""Snow accumulation in W3RA (no glacier dynamics) - numbers for the R3.5 extension.

    python snow_check.py --data-root /work/data_for_w3

(1) spin-up: snow drift per recycling cycle (spinup/drift.json) and the 1995-12-31 snow state (basin mean, perennial cells);
(2) OL DrySnow trend (basin mean, mm/yr; least squares with annual + semi-annual terms, and trend-only) of member 0 in the R3.5 OL
    (1996-2006 and 2003-09..2006-12) and in the shared published-ensemble OL (2002-09..2006-12, 2003-09..2006-12);
perennial cells = model cells whose DrySnow (Fhru-weighted) stays >= 1 mm on every day of the OL (member 0); count and area.
Writes <data_root>/w3ra/runs/jog/snow_check.json.
"""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np

UNIT = 'Brahmaputra1degree'


def trend(t, y, seasonal=True):
    cols = [np.ones_like(t), t]
    if seasonal:
        for f in (1, 2):
            cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    c = np.linalg.lstsq(np.array(cols).T, y, rcond=None)[0]
    return float(c[1])


def series(fn, start):
    with h5py.File(fn, 'r') as h:
        y = h['basin_0']['DrySnow'][:]
    d = [start + timedelta(days=i) for i in range(len(y))]
    t = np.array([(x - date(2000, 1, 1)).days / 365.25 for x in d])
    return d, t, y


def perennial(state_dir, par_fn, mask_fn, step=5):
    with h5py.File(par_fn, 'r') as p:
        fhru = p['data']['Fhru'][:]
    with h5py.File(mask_fn, 'r') as m:
        lat = m['lat'][()] if 'lat' in m else None
        mask = m['mask'][()].astype(bool)
    files = sorted(Path(state_dir).glob('state.*.h5'))[::step]
    mn = None
    for f in files:
        with h5py.File(f, 'r') as s:
            v = np.sum(s['DrySnow'][:] * fhru, axis=0)
        mn = v if mn is None else np.fmin(mn, v)
    per = np.isfinite(mn) & (mn >= 1.0)
    if lat is not None:
        latc = np.repeat(lat[:, None], mask.shape[1], axis=1)[mask]
        area = float(np.sum((111.195 * 0.1) ** 2 * np.cos(np.deg2rad(latc[per]))))
    else:
        area = None
    return int(per.sum()), int(np.isfinite(mn).sum()), area, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    a = ap.parse_args()
    w3 = Path(a.data_root) / 'w3ra'
    out = {}
    d35 = w3 / 'domains' / ('jog_R35_%s' % UNIT)
    drift = json.loads((d35 / 'spinup' / 'drift.json').read_text())
    out['spinup_cycles'] = [dict(cycle=c['cycle'], snow_change=round(c['change']['DrySnow'] + c['change']['FreeWater'], 3),
                                 non_snow_change=round(c['change'].get('non_snow', float('nan')), 3))
                            for c in drift if 'change' in c]
    out['spinup_end_1995_12_31'] = {k: round(v, 2) for k, v in drift[-1]['end'].items()}
    runs = {'R35_OL_1996_2006': (d35 / 'OL' / 'r35_s1', date(1996, 1, 1)),
            'shared_OL_2002_2006': (w3 / 'domains' / ('jog_%s' % UNIT) / 'OL' / 'pub_magic1e19', date(2002, 9, 1))}
    for name, (ol, start) in runs.items():
        d, t, y = series(ol / 'ens_00' / 'output' / 'basin_ts_OL.h5', start)
        sel = np.array([x >= date(2003, 9, 5) for x in d])
        out[name] = dict(full=dict(period='%s..%s' % (d[0], d[-1]), trend_seasonal=round(trend(t, y), 2),
                                   trend_only=round(trend(t, y, False), 2), mean=round(float(y.mean()), 2)),
                         paper_period=dict(period='2003-09-05..%s' % d[-1], trend_seasonal=round(trend(t[sel], y[sel]), 2),
                                           trend_only=round(trend(t[sel], y[sel], False), 2)))
        dom = ol.parents[1]
        n, ncell, area, nd = perennial(ol / 'ens_00' / 'state', dom / 'input' / 'par' / 'par.h5',
                                       dom / 'input' / 'mask' / 'mask.h5')
        out[name]['perennial_cells'] = dict(cells=n, of=ncell, area_km2=round(area, 0) if area else None,
                                            days_sampled=nd)
    (w3 / 'runs' / 'jog' / 'snow_check.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
