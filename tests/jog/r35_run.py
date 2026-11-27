"""R3.5 Brahmaputra 1996-2006 extension, published method (EnKF, inflation 1.0, forcing_day same,
nominal cov), new data layout.

    cd <repo>/tests/jog
    python r35_run.py crop       --data-root /work/data_for_w3          # ERA5 1994-01..2006-12 -> new domain (Metview)
    python r35_run.py check-crop --data-root /work/data_for_w3          # vs the published crop_input of the JoG runs
    python r35_run.py spinup     --data-root /work/data_for_w3 [--max-cycles 10 --tol 1.0]
    mpiexec -n 31 python r35_run.py ol --data-root /work/data_for_w3 --seed 1
    mpiexec -n 31 python r35_run.py da --data-root /work/data_for_w3 --mission MAGIC --seed 1
    python r35_run.py products   --data-root /work/data_for_w3 --mission MAGIC --seed 1

Domain jog_R35_Brahmaputra1degree (same 1-degree unit shapefile and box as the JoG Brahmaputra runs). Spin-up: cold
1994-01-01..1995-12-31, then recycled over 1994-1995 (the 1995-12-31 state restarts 1994-01-01) until the basin-mean TWS of
1995-12-31 of the non-snow stores (TWS - DrySnow - FreeWater; snow in perennially frozen cells grows every cycle, W3RA has
no glacier dynamics) changes by less than --tol mm between cycles (drift of every cycle in spinup/drift.json) -> initial state
1995-12-31. Ensemble r35_s<seed>: new seeded perturbation (run.seed, the JoG perturbation settings, 30 members) 1996-01..2006-12;
OL 1996-01-01..2006-12-31; DA 1996-01-05 (first 5-day window start after 1996-01-01)..2006-12-31 with the observation product
ESA_SING_<M>_1e18_R35 = GRACE/output_R35/<M>/1e18 (v5 series 1996-2006, nominal cov), imported by the 'da' step.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

BASIN, UNIT = 'Brahmaputra', 'Brahmaputra1degree'
DOMAIN = 'jog_R35_%s' % UNIT
SHP = 'ESA_SING/Grid_1/%s_subbasins.shp' % UNIT
ENS = 30
PERIODS = dict(cold=('1994-01-01', '1995-12-31'), warm=('1996-01-01', '2006-12-31'), resume=('1996-01-05', '2006-12-31'))


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def comm_rank():
    try:
        from mpi4py import MPI
        c = MPI.COMM_WORLD
        return (c if c.Get_size() > 1 else None), c.Get_rank()
    except ImportError:
        return None, 0


def settings_copy(a, name, mission='MAGIC', seed=None, innovations=False):
    """settings/JoG_ESA_SING/<M>_Brahmaputra_nominal copied with run.seed / output.save_innovations (one writer)"""
    src = REPO / 'settings' / 'JoG_ESA_SING' / ('%s_%s_nominal' % (mission, BASIN))
    dst = Path(a.data_root) / 'w3ra' / 'runs' / 'jog_settings' / name
    comm, rank = comm_rank()
    if rank == 0:
        dst.mkdir(parents=True, exist_ok=True)
        for f in ('DA_setting.json', 'perturbation.json', 'pre_process.json', 'setting.json'):
            txt = (src / f).read_text()
            if f == 'setting.json':
                st = json.loads(txt)
                st['run']['seed'] = None if seed is None else int(seed)
                st['output']['save_innovations'] = bool(innovations)
                txt = json.dumps(st, indent=4)
            if f == 'DA_setting.json':
                d = json.loads(txt)
                d['obs']['GRACE']['preprocess_res'] = '${DATA_ROOT}/GRACE/output_R35/%s/1e18' % mission
                txt = json.dumps(d, indent=4)
            (dst / (f + '.tmp')).write_text(txt)
            os.replace(dst / (f + '.tmp'), dst / f)
    if comm is not None:
        comm.barrier()
    return str(dst)


def setup(a, case, setting_dir, ensemble=None, product='ESA_SING_MAGIC_1e18_R35', ens=ENS):
    from src_FlowControl.Regional_DA import RDA
    RDA.data_root, RDA.legacy_layout, RDA.setting_dir = a.data_root, False, setting_dir
    RDA.case, RDA.basin, RDA.shp_path, RDA.ens = case, UNIT, SHP, ens
    RDA.domain, RDA.ensemble, RDA.unit_set, RDA.grace_product = DOMAIN, ensemble or 'r35_s%s' % a.seed, UNIT, product
    RDA.cold_begin_time, RDA.cold_end_time = PERIODS['cold']
    RDA.warm_begin_time, RDA.warm_end_time = PERIODS['warm']
    RDA.resume_begin_time, RDA.resume_end_time = PERIODS['resume']
    RDA.isSet = False
    return RDA


def do_crop(a):
    RDA = setup(a, 'jog_R35_crop', settings_copy(a, 'jog_R35_crop'))
    RDA.prepare_Forcing()


def do_check_crop(a):
    """the new crop vs the published crop_input/MAGIC_1e19_1degree_Brahmaputra (forcing months 2000-2006, par, clim, mask)"""
    import h5py
    import numpy as np
    w3 = Path(a.data_root) / 'w3ra'
    new, pub = w3 / 'domains' / DOMAIN / 'input', w3 / 'crop_input' / 'MAGIC_1e19_1degree_Brahmaputra'
    rows, bad = [], 0
    for sub in ('forcing', 'clim', 'par', 'mask'):
        for f in sorted((pub / sub).glob('*.h5')):
            g = new / sub / f.name
            if not g.exists():
                continue
            same = md5(f) == md5(g)
            if not same:                        # compare the arrays (file metadata may differ)
                with h5py.File(f, 'r') as x, h5py.File(g, 'r') as y:
                    d = []
                    x.visititems(lambda n, o: d.append(n) if isinstance(o, h5py.Dataset) else None)
                    same = all(np.array_equal(x[n][()], y[n][()], equal_nan=True) for n in d)
            bad += not same
            rows.append('%s %s %s' % (sub, f.name, 'equal' if same else 'DIFFERENT'))
    txt = 'new crop vs published crop: %d files compared, %d different' % (len(rows), bad)
    print(txt)
    (w3 / 'domains' / DOMAIN / 'check_crop.txt').write_text(txt + '\n' + '\n'.join(rows) + '\n')


def basin_mean_tws(state_fn, par_fn, mask_fn):
    """area-mean (cos-lat) TWS and stores of a state file over the model cells (Fhru-weighted, Mleaf x4), mm"""
    import h5py
    import numpy as np
    with h5py.File(par_fn, 'r') as p:
        fhru = p['data']['Fhru'][:]
    with h5py.File(mask_fn, 'r') as m:
        lat = m['lat'][()] if 'lat' in m else None
        mask = m['mask'][()].astype(bool)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(mask.shape) if lat is not None else np.ones(mask.shape)
    w = w[mask]
    out = {}
    with h5py.File(state_fn, 'r') as s:
        tws = 0
        for k in s:
            v = np.sum(s[k][:] * fhru, axis=0) * (4 if k == 'Mleaf' else 1)
            ok = np.isfinite(v)                                  # state arrays hold NaN outside the model land cells
            out[k] = float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))
            tws = tws + np.where(ok, v, np.nan)
        ok = np.isfinite(tws)
        out['TWS'] = float(np.sum(tws[ok] * w[ok]) / np.sum(w[ok]))
    return out


def do_spinup(a):
    from src_FlowControl import run_context as rc
    from src_FlowControl.SingleModel import SingleModel
    from src_GHM.EnumType import init_mode
    RDA = setup(a, 'jog_R35_spinup', settings_copy(a, 'jog_R35_spinup'))
    RDA.set_box()
    sd = RDA._settings()
    L = rc.get_layout()
    end = L.spinup_dir / 'state' / 'state.19951231.h5'
    par, mask = L.input_dir / 'par' / 'par.h5', L.input_dir / 'mask' / 'mask.h5'
    drift = []
    # cycle 1: cold start from the default states
    demo = SingleModel(case=RDA.case, setting_dir=sd)
    demo.configure_time(begin_time=PERIODS['cold'][0], end_time=PERIODS['cold'][1])
    demo.configure_area(box=RDA.box, basin=RDA.basin)
    demo.generate_settings(mode=init_mode.cold)
    demo.model_run()
    prev = basin_mean_tws(end, par, mask)
    drift.append(dict(cycle=1, end=prev))
    print('cycle 1 (cold): TWS 1995-12-31 = %.2f mm' % prev['TWS'])
    L.init_dir.mkdir(parents=True, exist_ok=True)
    for cyc in range(2, a.max_cycles + 1):
        shutil.copy2(end, L.init_dir / ('%s_state.19931231.h5' % L.domain))      # recycle: end state restarts 1994-01-01
        demo = SingleModel(case=RDA.case, setting_dir=sd)
        demo.configure_time(begin_time=PERIODS['cold'][0], end_time=PERIODS['cold'][1])
        demo.configure_area(box=RDA.box, basin=RDA.basin)
        demo.generate_settings(mode=init_mode.warm)
        demo.model_run()
        cur = basin_mean_tws(end, par, mask)
        d = {k: cur[k] - prev[k] for k in cur}
        drift.append(dict(cycle=cyc, end=cur, change=d))
        d['non_snow'] = d['TWS'] - d.get('DrySnow', 0.0) - d.get('FreeWater', 0.0)
        print('cycle %d: TWS 1995-12-31 = %.2f mm, change %.3f mm (non-snow stores %.3f, snow %.3f)'
              % (cyc, cur['TWS'], d['TWS'], d['non_snow'], d.get('DrySnow', 0.0) + d.get('FreeWater', 0.0)))
        prev = cur
        '''criterion on the non-snow stores: W3RA has no glacier dynamics, snow in perennially frozen cells accumulates
        with every cycle (never converges); soil, routing and groundwater stores do'''
        if abs(d['non_snow']) < a.tol:
            break
    shutil.copy2(end, L.init_dir / ('%s_state.19951231.h5' % L.domain))
    (L.spinup_dir / 'drift.json').write_text(json.dumps(drift, indent=1))
    rc.made_with_write(L.spinup_dir, 'spin-up', dict(cold_begin=PERIODS['cold'][0], cold_end=PERIODS['cold'][1],
                                                     recycled_cycles=len(drift), tol_mm=a.tol))
    print('initial state %s (after %d cycles)' % (L.init_dir / ('%s_state.19951231.h5' % L.domain), len(drift)))


def do_ol(a):
    case = 'jog_R35_OL_s%s' % a.seed
    RDA = setup(a, case, settings_copy(a, case, seed=a.seed))
    RDA.OL_run(skip_perturbation=False)                   # new seeded ensemble r35_s<seed>, perturbed by this run


def import_product(a, mission):
    """GRACE/output_R35/<M>/1e18 -> units/<unit>/grace/ESA_SING_<M>_1e18_R35 (rank 0, md5)"""
    from src_FlowControl import run_context as rc
    comm, rank = comm_rank()
    if rank == 0:
        L = rc.get_layout()
        src_dir = Path(a.data_root) / 'GRACE' / 'output_R35' / mission / '1e18'
        dst_dir = L.units_dir / 'grace' / ('ESA_SING_%s_1e18_R35' % mission)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for kind in ('signal', 'gridded_signal', 'cov'):
            src, dst = src_dir / ('%s_%s.hdf5' % (UNIT, kind)), dst_dir / ('%s_%s.hdf5' % (UNIT, kind))
            if not dst.exists():
                shutil.copy2(src, dst)
            assert md5(src) == md5(dst), dst
    if comm is not None:
        comm.barrier()


def do_da(a):
    case = 'jog_R35_%s_s%s' % (a.mission, a.seed)
    RDA = setup(a, case, settings_copy(a, case, mission=a.mission, seed=a.seed, innovations=True),
                product='ESA_SING_%s_1e18_R35' % a.mission)
    comm, _ = comm_rank()
    RDA._settings(comm)                                   # one writer (rank 0) + barrier: all ranks would race otherwise
    import_product(a, a.mission)
    RDA.OL_run(skip_perturbation=False)                   # reuses ensemble + OL (made_with), writes this case's OL mean
    RDA.isSet = False
    RDA.DA_run(skip_obs_perturbation=False)


def do_products(a):
    sys.path.insert(0, str(HERE))
    import jog_run
    from src_postprocessing.DataManager import dataManager_ensemble_statistic, dataManager_ensemble_member, data_var
    from src_DA.shp2mask import basin_shp_process
    from src_FlowControl import run_context as rc
    case = 'jog_R35_%s_s%s' % (a.mission, a.seed)
    RDA = setup(a, case, str(Path(a.data_root) / 'w3ra' / 'runs' / 'jog_settings' / case),
                product='ESA_SING_%s_1e18_R35' % a.mission)
    sd = RDA.activate()
    L = rc.get_layout()
    gmask = basin_shp_process(res=0.1, basin_name=UNIT, save_dir=None).shp_to_mask(shp_path=rc.resolve_shp(SHP)).mask[0]
    s = str(Path(sd) / 'setting.json')
    d = dataManager_ensemble_statistic(ens_size=ENS).configure(setting_fn=s, variable=data_var.TotalWater)
    d.reduce_datasize(global_basin_mask=gmask, save_mask=True)
    d.aggregation_daily(date_begin=PERIODS['resume'][0], date_end=PERIODS['resume'][1])
    if not any(L.products_dir('OL_mean').glob('TotalWater.*.h5')):
        o = dataManager_ensemble_member(ens=0, stage='OL').configure(setting_fn=s, variable=data_var.TotalWater)
        o.reduce_datasize(global_basin_mask=gmask)
        o.aggregation_daily(date_begin=PERIODS['resume'][0], date_end=PERIODS['resume'][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['crop', 'check-crop', 'spinup', 'ol', 'da', 'products'])
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--mission', choices=['MAGIC', 'GRACEC'], default='MAGIC')
    ap.add_argument('--seed', default='1')
    ap.add_argument('--max-cycles', type=int, default=10)
    ap.add_argument('--tol', type=float, default=0.5, help='spin-up: stop when the basin-mean change of the non-snow '
                                                               'stores (TWS - DrySnow - FreeWater) < tol mm per cycle')
    a = ap.parse_args()
    a.data_root = str(Path(a.data_root).resolve())
    {'crop': do_crop, 'check-crop': do_check_crop, 'spinup': do_spinup, 'ol': do_ol, 'da': do_da,
     'products': do_products}[a.step](a)


if __name__ == '__main__':
    main()
