"""R1.1 same-climate control: synthetic truth and observations from one extra W3RA member.

    cd <repo>/tests/jog
    python r11_prepare.py --data-root /work/data_for_w3 --basin Brahmaputra [--seed 101] [--tag x3]

--tag: harder truth of jog_run.py r11-truth --pert-scale (ensemble r11truth_s<seed><tag>) -> GRACE/output_R11<tag>.

Inputs: the truth member = member 1 of ensemble r11truth_s<seed> (jog_run.py r11-truth: one extra perturbed member, own seed,
OL 2002-09-01..2006-12-31); its daily unit series basin_ts_OL.h5 (stores Fhru-weighted, Mleaf x4, cos-lat unit means = the
vertical/horizontal aggregation of the DA operator H; TWS = sum of the stores).
1. truth units: 5-day means of the truth's unit TWS over the ESM2.0 epoch windows of the obs epochs (the published obs
   time_epoch = window start + 2 d, ESM2.0 epochs 561..875; windows from basin_signal 'dates').
2. synthetic obs per mission: truth units + (published obs units - ESM2.0 HIS units) -> GRACE/output_R11/<M>/1e18/: the
   nominal set (GRACE/output/<M>_old/1e18) copied, only the sub_basin_k series of <unit>_signal.hdf5 replaced; gridded_signal
   and cov byte-identical to the nominal files. Check: synthetic - truth = published - HIS for every unit and epoch.
   (The anomaly reference does not matter: the DA removes the temporal mean of the obs and adds the OL mean.)
3. truth on the model grid for the evaluation: daily TotalWater.<year>.h5 of the truth member (DataManager, member 1, the
   basin cells) in cases/jog_R11truth_<basin>_s<seed>/products/truth/<prefix>/ (upscaled to 0.5 deg by the evaluation like the
   model).
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
MISSIONS = ('GRACEC', 'NGGM', 'MAGIC')


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def s2d(x):
    return datetime.strptime(x.decode() if isinstance(x, bytes) else str(x), '%Y-%m-%d')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--basin', required=True, choices=['Brahmaputra', 'Danube'])
    ap.add_argument('--seed', default='101')
    ap.add_argument('--tag', default='', help="truth tag of r11-truth --pert-scale (e.g. 'x3')")
    ap.add_argument('--skip-grid', action='store_true')
    a = ap.parse_args()
    root = Path(a.data_root)
    unit = '%s1degree' % a.basin
    w3, gr = root / 'w3ra', root / 'GRACE'
    log = []

    '''truth: daily unit TWS of member 1'''
    fn = w3 / 'domains' / ('jog_%s' % unit) / 'OL' / ('r11truth_s%s%s' % (a.seed, a.tag)) / 'ens_01' / 'output' / 'basin_ts_OL.h5'
    with h5py.File(fn, 'r') as h:
        n_units = len([k for k in h if k.startswith('basin_')]) - 1
        tws = np.array([np.sum([h['basin_%d' % k][s][:] for s in h['basin_%d' % k]], axis=0)
                        for k in range(1, n_units + 1)])                     # (units, days)
    day0 = datetime(2002, 9, 1)
    days = [day0 + timedelta(days=i) for i in range(tws.shape[1])]
    log.append('truth %s: %d units x %d days (%s..%s)' % (fn, n_units, len(days), days[0].date(), days[-1].date()))

    '''ESM2.0 windows and HIS units'''
    bs = gr / 'ESA_SING' / 'ESM2.0' / a.basin / 'truth' / 'basin_signal.hdf5'
    with h5py.File(bs, 'r') as h:
        g = h['Grid_1/HIS']
        his = g['EWHA'][()] if 'EWHA' in g else g['EWHA_HIS'][()]
        starts = [s2d(x) for x in h['dates'][0]]
        ends = [s2d(x) for x in h['dates'][1]]
    his = his * 1000.0                                                    # m -> mm (obs files are in mm)
    assert his.shape[0] == n_units, (his.shape, n_units)

    nominal = {m: gr / 'output' / ('%s_old' % m) / '1e18' for m in MISSIONS}
    with h5py.File(nominal['MAGIC'] / ('%s_signal.hdf5' % unit), 'r') as h:
        te = [s2d(x) for x in h['time_epoch'][()]]
    idx = [starts.index(t - timedelta(days=2)) for t in te]                 # ESM2.0 epoch of each obs epoch
    win = []
    for j in idx:
        n = (ends[j] - starts[j]).days
        last = ends[j] - timedelta(days=1) if n == 5 else ends[j]          # 5-day windows
        win.append([d for d in days if starts[j] <= d <= last])
    assert all(len(w) == 5 for w in win), set(len(w) for w in win)
    pos = {d: i for i, d in enumerate(days)}
    truth_units = np.array([[tws[k, [pos[d] for d in w]].mean() for w in win] for k in range(n_units)])
    log.append('obs epochs %d (%s..%s) = ESM2.0 epochs %d..%d; 5-day windows from basin_signal dates' %
               (len(te), te[0].date(), te[-1].date(), idx[0], idx[-1]))

    out = gr / ('output_R11%s' % a.tag)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(out / ('truth_units_%s_s%s%s.hdf5' % (unit, a.seed, a.tag)), 'w') as h:
        for k in range(n_units):
            h['sub_basin_%d' % (k + 1)] = truth_units[k]
        h['time_epoch'] = np.array([t.strftime('%Y-%m-%d') for t in te], dtype='S10')
        h.attrs['source'] = str(fn)
        h.attrs['note'] = 'R1.1 synthetic truth: 5-day means (mm) of W3RA member 1 of ensemble r11truth_s%s%s' % (a.seed, a.tag)

    worst = 0.0
    for m in MISSIONS:
        dst = out / m / '1e18'
        dst.mkdir(parents=True, exist_ok=True)
        for kind in ('signal', 'gridded_signal', 'cov'):
            src = nominal[m] / ('%s_%s.hdf5' % (unit, kind))
            tgt = dst / ('%s_%s.hdf5' % (unit, kind))
            if tgt.exists():
                tgt.unlink()
            shutil.copy2(src, tgt)
        with h5py.File(dst / ('%s_signal.hdf5' % unit), 'r+') as h:
            for k in range(n_units):
                obs = h['sub_basin_%d' % (k + 1)][()]
                syn = truth_units[k] + (obs - his[k, idx])
                h['sub_basin_%d' % (k + 1)][...] = syn
                worst = max(worst, float(np.max(np.abs((syn - truth_units[k]) - (obs - his[k, idx])))))
        for kind in ('gridded_signal', 'cov'):
            assert md5(dst / ('%s_%s.hdf5' % (unit, kind))) == md5(nominal[m] / ('%s_%s.hdf5' % (unit, kind)))
        log.append('%s: %s (signal md5 %s; gridded/cov = nominal)' % (m, dst, md5(dst / ('%s_signal.hdf5' % unit))))
    log.append('check: max |(synthetic - truth) - (published - HIS)| = %.3e mm' % worst)
    print('\n'.join(log))

    if not a.skip_grid:
        '''truth on the model grid (member 1, daily)'''
        from src_FlowControl.Regional_DA import RDA
        from src_FlowControl import run_context as rc
        from src_postprocessing.DataManager import dataManager_ensemble_member, data_var
        from src_DA.shp2mask import basin_shp_process
        case = 'jog_R11truth_%s_s%s%s' % (a.basin, a.seed, a.tag)
        RDA.data_root, RDA.legacy_layout = str(root), False
        RDA.setting_dir = str(w3 / 'runs' / 'jog_settings' / case)
        RDA.case, RDA.basin, RDA.shp_path, RDA.ens = case, unit, 'ESA_SING/Grid_1/%s_subbasins.shp' % unit, 1
        RDA.domain, RDA.ensemble, RDA.unit_set = 'jog_%s' % unit, 'r11truth_s%s%s' % (a.seed, a.tag), unit
        RDA.grace_product = 'ESA_SING_MAGIC_1e18'
        RDA.warm_begin_time, RDA.warm_end_time = '2002-09-01', '2006-12-31'
        sd = RDA.activate()
        gmask = basin_shp_process(res=0.1, basin_name=unit, save_dir=None).shp_to_mask(
            shp_path=rc.resolve_shp(RDA.shp_path)).mask[0]
        o = dataManager_ensemble_member(ens=1, stage='OL').configure(
            setting_fn=str(Path(sd) / 'setting.json'), out_dir=str(w3 / 'cases' / case / 'products' / 'truth'),
            variable=data_var.TotalWater)
        o.reduce_datasize(global_basin_mask=gmask)
        o.aggregation_daily(date_begin='2002-09-01', date_end='2006-12-31')
        print('truth grid products -> %s' % o._out_dir)
    (w3 / 'runs' / 'jog' / ('r11_prepare_%s%s.json' % (a.basin, a.tag))).write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
