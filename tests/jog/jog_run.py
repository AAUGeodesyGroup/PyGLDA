"""JoG revision runs (ESA-SING, alpha 1e18, 1-degree units) with the published method.

    cd <repo>/tests/jog
    python jog_run.py import   --data-root /work/data_for_w3 --basin Brahmaputra
    mpiexec -n 31 python jog_run.py ol --data-root /work/data_for_w3 --basin Brahmaputra
    python jog_run.py check-ol --data-root /work/data_for_w3 --basin Brahmaputra
    mpiexec -n 31 python jog_run.py da --data-root /work/data_for_w3 --basin Brahmaputra --mission MAGIC --seed 1
    python jog_run.py products --data-root /work/data_for_w3 --basin Brahmaputra --mission MAGIC --seed 1

Layout (configuration.md section 7): domain jog_<basin>1degree, ensemble pub_magic1e19 = the published shared ensemble
crop_input/MAGIC_1e19_1degree_<basin> (imported, never perturbed again), unit set <basin>1degree, GRACE product
ESA_SING_<MISSION>_1e18 = the published GRACE/output/<MISSION>/1e18 products (with the 27 Jul 2025 cov). Spin-up state =
initial_states/GRACEC_1e18_1degree_<basin>_state.20020831.h5 (identical for every published case of a basin).
Open loop case jog_OL_<basin>; DA cases jog_<MISSION>_<basin>_s<seed>[_<tag>]. Periods as published (demo_ESA_SING):
OL 2002-09-01..2006-12-31, DA 2003-09-05..2006-12-31 (see PERIODS). Settings: settings/JoG_ESA_SING/<MISSION>_<basin> (EnKF, inflation
1.0, forcing_day same) copied with run.seed and output.save_innovations into <data_root>/w3ra/runs/jog_settings/<case>/.
Products: DA ensemble mean of members 1..30 (cases/<case>/products/DA_mean) and OL member 0
(domains/<domain>/OL/<ensemble>/products/OL_mean), daily TotalWater.<year>.h5 over the basin cells, as the evaluation package
(jog_eval.sources.case_products) reads them. The published save_data() used dataManager_ensemble_statistic() with its
default ens_size=2; here ens_size is set to the ensemble size explicitly.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

MISSIONS = ('GRACEC', 'NGGM', 'MAGIC')
ENS = 30
# DA from 2003-09-05: the published runs set 2003-09-01, but no observation window starts before 2003-09-05 (5-day ESA-SING
# epochs), so the members followed the open loop until then; the DA start check asks for a window start; the paper's
# evaluation starts at epoch 634 = 2003-09-05 as well
PERIODS = dict(cold=('2000-01-01', '2002-08-31'), warm=('2002-09-01', '2006-12-31'), resume=('2003-09-05', '2006-12-31'))


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def shapefile(basin, grid):
    """DA unit shapefile of a grid (relative to <data_root>/basin/shp); all grids of a basin have the same bounds"""
    if grid == '3subbasins':
        return '%s3subbasins/%s3subbasins_subbasins.shp' % (basin, basin)
    return 'ESA_SING/Grid_%s/%s%s_subbasins.shp' % (grid[0], basin, grid)


def names(a, mission=None):
    alpha, grid = getattr(a, 'alpha', None) or '1e18', getattr(a, 'grid', None) or '1degree'
    other = (alpha, grid) != ('1e18', '1degree')            # rerun of the 180 published configurations (nominal only)
    unit = '%s%s' % (a.basin, grid)
    m = mission or a.mission or 'GRACEC'
    var = getattr(a, 'variant', None)
    tag = a.tag or (var or '')
    cfg = ('_%s_%s' % (alpha, grid)) if other else ''
    case = 'jog_OL_%s' % a.basin if mission is None and a.step in ('import', 'import-variant', 'ol', 'check-ol') else \
        'jog_%s_%s%s_s%s%s' % (m, a.basin, cfg, a.seed, ('_' + tag) if tag else '')
    vs = ('_' + var) if (var and mission is not None) else ''
    return dict(unit=unit, domain='jog_%s1degree' % a.basin, ensemble='pub_magic1e19',
                product='ESA_SING_%s_%s%s' % (m, alpha, vs), case=case, mission=m, alpha=alpha, grid=grid,
                crop='MAGIC_1e19_1degree_%s' % a.basin, variant=var, published='GRACEC_1e18_1degree_%s' % a.basin,
                shp=shapefile(a.basin, grid),
                settings=str(REPO / 'settings' / 'JoG_ESA_SING' / ('%s_%s%s' % (m, a.basin, vs))),
                obs_src=('${DATA_ROOT}/GRACE/output/%s%s/%s' % (m, '_old' if var == 'nominal' else '', alpha)) if other else None)


def run_settings(a, n):
    """template copy with run.seed / output.save_innovations (never written into the repository)"""
    src = Path(n['settings'])
    if a.step in ('import', 'import-variant', 'ol', 'check-ol'):
        return str(src)
    import os
    dst = Path(a.data_root) / 'w3ra' / 'runs' / 'jog_settings' / n['case']
    comm = None
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None
    except ImportError:
        pass
    if comm is None or comm.Get_rank() == 0:             # one writer; atomic replace; the other ranks wait
        dst.mkdir(parents=True, exist_ok=True)
        for f in ('DA_setting.json', 'perturbation.json', 'pre_process.json', 'setting.json'):
            txt = (src / f).read_text()
            if f == 'setting.json':
                st = json.loads(txt)
                st['run']['seed'] = None if a.seed in (None, 'none') else int(a.seed)
                st['output']['save_innovations'] = not a.no_innovations
                txt = json.dumps(st, indent=4)
            if f == 'DA_setting.json' and n.get('obs_src'):         # record the product the run reads
                d = json.loads(txt)
                d['obs']['GRACE']['preprocess_res'] = n['obs_src']
                txt = json.dumps(d, indent=4)
            tmp = dst / (f + '.tmp')
            tmp.write_text(txt)
            os.replace(tmp, dst / f)
    if comm is not None:
        comm.barrier()
    return str(dst)


def setup(a, n):
    from src_FlowControl.Regional_DA import RDA
    RDA.data_root = a.data_root
    RDA.legacy_layout = False
    RDA.setting_dir = run_settings(a, n)
    RDA.case, RDA.basin, RDA.shp_path, RDA.ens = n['case'], n['unit'], n['shp'], ENS
    RDA.domain, RDA.ensemble, RDA.unit_set, RDA.grace_product = n['domain'], n['ensemble'], n['unit'], n['product']
    RDA.cold_begin_time, RDA.cold_end_time = PERIODS['cold']
    RDA.warm_begin_time, RDA.warm_end_time = PERIODS['warm']
    RDA.resume_begin_time, RDA.resume_end_time = PERIODS['resume']
    RDA.isSet = False
    return RDA


def do_import(a):
    """read-only copies of the published inputs into the layout (refuses to overwrite)"""
    n = names(a)
    RDA = setup(a, n)
    RDA._settings()
    from src_FlowControl import run_context as rc
    L = rc.get_layout()
    w3, gr = Path(a.data_root) / 'w3ra', Path(a.data_root) / 'GRACE'
    crop = w3 / 'crop_input' / n['crop']
    pairs = []
    for f in sorted((crop / 'forcing').glob('*.h5')):
        if '2002-08' <= f.stem <= '2007-01':
            pairs.append((f, L.input_dir / 'forcing' / f.name))
    for sub in ('clim', 'par', 'mask'):
        pairs += [(f, L.input_dir / sub / f.name) for f in sorted((crop / sub).iterdir())]
    for sub in ('ens_forcing', 'ens_par'):
        pairs += [(f, L.ensemble_dir / sub / f.name) for f in sorted((crop / sub).glob('*.h5'))]
    pairs.append((w3 / 'initial_states' / ('%s_state.20020831.h5' % n['published']),
                  L.init_dir / ('%s_state.20020831.h5' % L.domain)))
    for m in MISSIONS:
        gdir = L.units_dir / 'grace' / ('ESA_SING_%s_1e18' % m)
        for kind in ('signal', 'gridded_signal', 'cov'):
            pairs.append((gr / 'output' / m / '1e18' / ('%s_%s.hdf5' % (n['unit'], kind)),
                          gdir / ('%s_%s.hdf5' % (n['unit'], kind))))
    lines = []
    for src, dst in pairs:
        if dst.exists():
            ms, md = md5(src), md5(dst)
            if ms != md:
                raise RuntimeError('%s exists with other content' % dst)
            lines.append('%s  %s  ->  %s  (present)' % (ms, src, dst))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        ms, md = md5(src), md5(dst)
        assert ms == md, (src, dst)
        lines.append('%s  %s  ->  %s' % (ms, src, dst))
    (L.domain_dir / 'import_md5.txt').write_text('\n'.join(lines) + '\n')
    rc.note('imported %d published files into %s (md5 in %s)' % (len(pairs), L.domain_dir, L.domain_dir / 'import_md5.txt'))
    print('imported %d files' % len(pairs))


def do_import_variant(a):
    """obs products of a variant -> units/<unit>/grace/ESA_SING_<M>_1e18_<variant> (read-only copies, md5): nominal =
    GRACE/output/<M>_old/1e18; R34 (cleaned of the 10/15/30-d errors), R41 (minus ice + solid Earth) = their signal from
    GRACE/output_<variant>/<M>/1e18 with the nominal cov"""
    from src_FlowControl import run_context as rc
    n = names(a)
    RDA = setup(a, n)
    RDA._settings()
    L = rc.get_layout()
    out = Path(a.data_root) / 'GRACE'
    lines = []
    for m in MISSIONS:
        if a.variant == 'realistic':                                 # July 2025 residual-based VCM: product without suffix
            gdir = L.units_dir / 'grace' / ('ESA_SING_%s_%s' % (m, n['alpha']))
            nominal = sig = out / 'output' / m / n['alpha']
        else:
            gdir = L.units_dir / 'grace' / ('ESA_SING_%s_%s_%s' % (m, n['alpha'], a.variant))
            nominal = out / 'output' / ('%s_old' % m) / n['alpha']  # April 2025 nominal VCM (1-deg diagonal, others full)
            sig = nominal if a.variant == 'nominal' else out / ('output_%s' % a.variant) / m / n['alpha']
        for kind in ('signal', 'gridded_signal', 'cov'):
            src_dir = nominal if kind == 'cov' else sig                # variants: their signal + the NOMINAL cov
            src, dst = src_dir / ('%s_%s.hdf5' % (n['unit'], kind)), gdir / ('%s_%s.hdf5' % (n['unit'], kind))
            if dst.exists():
                if md5(dst) != md5(src):
                    raise RuntimeError('%s exists with other content' % dst)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            assert md5(src) == md5(dst)
            lines.append('%s  %s  ->  %s' % (md5(dst), src, dst))
    with open(L.units_dir / ('import_%s_md5.txt' % a.variant), 'a') as f:
        f.write('\n'.join(lines) + '\n')
    print('imported %d variant files (%s)' % (len(lines), a.variant))


def do_r11_truth(a):
    """R1.1 same-climate control: ONE extra perturbed member with its own seed (same perturbation settings as the
    ensemble, not one of the 30) run as an open loop 2002-09-01..2006-12-31 in its own ensemble r11truth_s<seed> of the domain;
    its member 1 is the synthetic truth (2 ranks: member 0 = unperturbed control, member 1 = truth).
    --pert-scale k (harder truth): every perturbed coefficient x k in the run copy of perturbation.json, multiplicative
    factors bounded at 0 (min_factor) and the additive ssrd field at 0 (min_value); ensemble/case tag 'x<k>'"""
    n = names(a)
    from src_FlowControl.Regional_DA import RDA
    tag = ('x%g' % a.pert_scale) if a.pert_scale != 1 else ''
    case = 'jog_R11truth_%s_s%s%s' % (a.basin, a.seed, tag)
    src = Path(n['settings'])
    dst = Path(a.data_root) / 'w3ra' / 'runs' / 'jog_settings' / case
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        import os
        dst.mkdir(parents=True, exist_ok=True)
        for f in ('DA_setting.json', 'perturbation.json', 'pre_process.json', 'setting.json'):
            txt = (src / f).read_text()
            if f == 'setting.json':
                st = json.loads(txt)
                st['run']['seed'] = int(a.seed)
                txt = json.dumps(st, indent=4)
            if f == 'perturbation.json' and tag:
                pt = json.loads(txt)
                for grp in ('forcing', 'par'):
                    for key, c in pt[grp].items():
                        if not c['is_perturbed']:
                            continue
                        c['coefficients'] = [v * a.pert_scale for v in c['coefficients']]
                        if c['perturb_method'] == 'multiplicative':
                            c['min_factor'] = 0.0
                        elif key == 'ssrd':
                            c['min_value'] = 0.0
                txt = json.dumps(pt, indent=4)
            (dst / (f + '.tmp')).write_text(txt)
            os.replace(dst / (f + '.tmp'), dst / f)
    comm.barrier()
    RDA.data_root, RDA.legacy_layout, RDA.setting_dir = a.data_root, False, str(dst)
    RDA.case, RDA.basin, RDA.shp_path, RDA.ens = case, n['unit'], n['shp'], 1
    RDA.domain, RDA.ensemble, RDA.unit_set, RDA.grace_product = n['domain'], 'r11truth_s%s%s' % (a.seed, tag), n['unit'], \
        n['product']
    RDA.cold_begin_time, RDA.cold_end_time = PERIODS['cold']
    RDA.warm_begin_time, RDA.warm_end_time = PERIODS['warm']
    RDA.resume_begin_time, RDA.resume_end_time = PERIODS['resume']
    RDA.isSet = False
    RDA.OL_run(skip_perturbation=False)


def do_ol(a):
    RDA = setup(a, names(a))
    RDA.OL_run(skip_perturbation=True)


def do_check_ol(a):
    """new Res_OL (all members, all days) vs the published one"""
    import h5py
    import numpy as np
    n = names(a)
    w3 = Path(a.data_root) / 'w3ra'
    new = w3 / 'cases' / n['case'] / 'res' / ('Res_OL_%s.h5df' % n['unit'])
    pub = w3 / 'res' / n['published'] / ('Res_OL_%s.h5df' % n['unit'])
    with h5py.File(new, 'r') as hn, h5py.File(pub, 'r') as hp:
        tn, tp = hn['time'][:], hp['time'][:]
        assert len(tn) == len(tp) and np.array_equal(tn, tp), (len(tn), len(tp))
        worst, nd = 0.0, 0
        for u in [k for k in hn if k.startswith('basin_')]:
            for s in hn[u]:
                for k in hn[u][s]:
                    d = float(np.max(np.abs(hn[u][s][k][:] - hp[u][s][k][:])))
                    worst = max(worst, d)
                    nd += 1
    txt = 'Res_OL %s vs published %s: %d series (units x stores x members), %d days, max |diff| %.3e -> %s' % (
        new, pub, nd, len(tn), worst, 'BIT-IDENTICAL' if worst == 0 else 'differs')
    print(txt)
    (w3 / 'cases' / n['case'] / 'log' / 'check_ol.txt').write_text(txt + '\n')


GRIDS = ('1degree', '2degree', '3degree', '4degree', '3subbasins')


def ol_unit_mean_file(a, grid):
    return Path(a.data_root) / 'w3ra' / 'runs' / 'jog' / 'ol_unit_mean' / ('%s%s.hdf5' % (a.basin, grid))


def do_ol_unit_mean(a):
    """OL temporal mean per DA unit for every grid of a basin, from the shared OL states (mpiexec -n 31: one member per rank).
    The shared OL's basin_ts_OL.h5 (and so the case OL mean written by OL_run) holds the 1-degree units only; the DA of another
    grid needs the OL mean over its own units. Same arithmetic as OpenLoop/BasinSignalAnalysis.get_basin_average (Fhru-weighted
    stores, Mleaf x4, cos-lat unit means over the non-NaN cells of the 2nd day, period PERIODS['warm']) with the mean over time
    taken before the unit mean (linear, same weights); mean_ensemble = mean of members 1..30, mean_unperturbed = member 0.
    Check: the 1-degree result vs the OL mean from basin_ts_OL.h5 (max |diff| printed and stored)."""
    import h5py
    import numpy as np
    from datetime import datetime, timedelta
    from mpi4py import MPI
    from src_DA.shp2mask import basin_shp_process
    from src_FlowControl import run_context as rc
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    a.grid, a.alpha = '1degree', '1e18'
    n = names(a, 'MAGIC')
    RDA = setup(a, n)
    RDA.activate()
    L = rc.get_layout()
    ol = L.ol_dir
    inp = L.domain_dir / 'input'
    with h5py.File(inp / 'par' / 'par.h5', 'r') as h:
        fhru = h['data']['Fhru'][:]
    d0 = datetime.strptime(PERIODS['warm'][0], '%Y-%m-%d')
    days = [d0 + timedelta(days=i) for i in range((datetime.strptime(PERIODS['warm'][1], '%Y-%m-%d') - d0).days + 1)]
    keys = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']
    mine = {}
    for m in range(rank, ENS + 1, size):
        sd = ol / ('ens_%02d' % m) / 'state'
        acc = None
        for d in days:
            with h5py.File(sd / ('state.%s.h5' % d.strftime('%Y%m%d')), 'r') as h:
                tws = sum(np.sum(h[k][:] * fhru, axis=0) * (4 if k == 'Mleaf' else 1) for k in keys)
            acc = tws if acc is None else acc + tws
        mine[m] = acc / len(days)
    allm = comm.gather(mine, root=0)
    if rank != 0:
        return
    tmean = {}
    for dct in allm:
        tmean.update(dct)
    assert sorted(tmean) == list(range(ENS + 1))
    with h5py.File(ol / 'ens_00' / 'state' / ('state.%s.h5' % days[1].strftime('%Y%m%d')), 'r') as h:
        fw = np.sum(h['FreeWater'][:] * fhru, axis=0)
    out = ol_unit_mean_file(a, '1degree').parent
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for grid in GRIDS:
        unit = '%s%s' % (a.basin, grid)
        bs = basin_shp_process(res=0.1, basin_name=unit).shp_to_mask(shp_path=rc.resolve_shp(shapefile(a.basin, grid)),
                                                                     issave=False)
        bs.mask_to_vec(model_mask_global=str(inp / 'mask' / 'mask_global.h5'))
        mask_2D, mask_1D, lat, lon, res = bs.collect_mask
        ub = sorted([k for k in mask_1D if k.startswith('basin_') and k != 'basin_0'], key=lambda k: int(k.split('_')[1]))
        means = np.zeros((ENS + 1, len(ub)))
        for j, k in enumerate(ub):
            bm = mask_1D[k]
            w = np.cos(np.deg2rad(lat[mask_2D[k]]))
            ok = ~np.isnan(fw[bm])
            for m in range(ENS + 1):
                means[m, j] = np.sum(tmean[m][bm][ok] * w[ok]) / np.sum(w[ok])
        fn = ol_unit_mean_file(a, grid)
        with h5py.File(fn, 'w') as h:
            h['mean_unperturbed'] = means[0]
            h['mean_ensemble'] = np.mean(means[1:], axis=0)
            h.attrs['source'] = str(ol)
            h.attrs['period'] = '%s..%s' % PERIODS['warm']
        line = '%s: %d units -> %s' % (unit, len(ub), fn)
        if grid == '1degree':
            with h5py.File(L.case_dir / 'grace' / 'ol_mean.hdf5', 'r') as h:      # written from basin_ts_OL.h5 by OL_run
                ref_e, ref_u = h['mean_ensemble'][:], h['mean_unperturbed'][:]
            line += '; check vs basin_ts OL mean: max |diff| ensemble %.3e, unperturbed %.3e mm' % (
                np.max(np.abs(ref_e - np.mean(means[1:], axis=0))), np.max(np.abs(ref_u - means[0])))
        log.append(line)
        print(line, flush=True)
    (out / ('%s_log.txt' % a.basin)).write_text('\n'.join(log) + '\n')


def do_da(a):
    n = names(a, a.mission)
    RDA = setup(a, n)
    RDA.OL_run(skip_perturbation=True)                # reuses the shared OL; writes this case's OL mean + NaN sample
    if n['grid'] != '1degree':                        # the shared OL mean is over the 1-degree units: use this grid's units
        from mpi4py import MPI
        from src_FlowControl import run_context as rc
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            src, dst = ol_unit_mean_file(a, n['grid']), rc.get_layout().ol_mean_file
            shutil.copy2(src, dst)
            rc.note('OL mean of the %s units from %s (md5 %s)' % (n['grid'], src, md5(dst)))
        comm.barrier()
    RDA.isSet = False
    RDA.DA_run(skip_obs_perturbation=False)


def do_products(a):
    from src_postprocessing.DataManager import dataManager_ensemble_statistic, dataManager_ensemble_member, data_var
    from src_DA.shp2mask import basin_shp_process
    from src_FlowControl import run_context as rc
    n = names(a, a.mission)
    RDA = setup(a, n)
    sd = RDA.activate()
    L = rc.get_layout()
    gmask = basin_shp_process(res=0.1, basin_name=n['unit'], save_dir=None).shp_to_mask(
        shp_path=rc.resolve_shp(n['shp'])).mask[0]
    s = str(Path(sd) / 'setting.json')
    d = dataManager_ensemble_statistic(ens_size=ENS).configure(setting_fn=s, variable=data_var.TotalWater)
    d.reduce_datasize(global_basin_mask=gmask, save_mask=True)
    d.aggregation_daily(date_begin=PERIODS['resume'][0], date_end=PERIODS['resume'][1])
    print('DA_mean (members 1..%d) -> %s' % (ENS, d._out_dir))
    ol_dir = L.products_dir('OL_mean')
    if not any(ol_dir.glob('TotalWater.*.h5')):
        o = dataManager_ensemble_member(ens=0, stage='OL').configure(setting_fn=s, variable=data_var.TotalWater)
        o.reduce_datasize(global_basin_mask=gmask)
        o.aggregation_daily(date_begin=PERIODS['resume'][0], date_end=PERIODS['resume'][1])
        print('OL_mean (member 0) -> %s' % o._out_dir)


PROTECTED = {'jog_%s_%s_s1_nominal' % (m, b) for m in MISSIONS for b in ('Brahmaputra', 'Danube')}


def do_check_clean(a):
    """after a finished DA + products : (1) products complete (DA_mean/DA_std
    TotalWater.2003-2006 present) -> md5 list products_md5.txt in the case; (2) a 07-style evaluation of the DA_mean product
    against the truth passes (paper definition, finite spatial-mean NSE/KGE/RMSE, epochs 634..875); (3) only then the daily
    member state folders cases/<case>/DA/ens_kk/state are deleted (never for the 6 nominal seed-1 baselines), logged with path,
    size, time in cases/<case>/log/deleted_states.txt. Nothing else is touched."""
    import time
    import numpy as np
    n = names(a, a.mission)
    case = n['case']
    cdir = Path(a.data_root) / 'w3ra' / 'cases' / case
    prod = cdir / 'products'
    need = [prod / k / ('TotalWater.%d.h5' % y) for k in ('DA_mean', 'DA_std') for y in range(2003, 2007)]
    miss = [str(x) for x in need if not x.is_file()]
    if miss:
        sys.exit('check-clean %s: products missing %s' % (case, miss))
    lines = ['%s  %s' % (md5(f), f.relative_to(cdir)) for f in sorted(prod.rglob('*.h5'))]
    (cdir / 'log' / 'products_md5.txt').write_text('\n'.join(lines) + '\n')
    sys.path.insert(0, str(Path(a.data_root) / 'w3ra' / 'evaluation' / 'JoG'))
    from jog_eval import evaluate as E, config as C
    epochs = np.arange(*C.PAPER_EPOCHS)
    grid = getattr(a, 'grid', None) or '1degree'
    alpha = getattr(a, 'alpha', None) or '1e18'
    mask_case = '%s_%s_%s_%s' % (n['mission'], alpha, grid, a.basin)
    data = E.load_basin(a.basin, epochs, missions=(n['mission'],),
                        folders={n['mission']: ('DA', str(prod / 'DA_mean'), mask_case)})
    tab = E.table(data, 'paper')[n['mission']]
    ok = all(np.isfinite(v) for v in tab.values()) and data['cells'] > 0
    print('check-clean %s: %d cells, paper-def %s -> %s' % (case, data['cells'],
          ', '.join('%s %.4f' % kv for kv in tab.items()), 'OK' if ok else 'FAILED'))
    (cdir / 'log' / 'check_eval.txt').write_text(json.dumps(dict(cells=data['cells'], missing=str(data['missing']),
                                                                **{k: float(v) for k, v in tab.items()}), indent=1))
    if not ok:
        sys.exit('check-clean %s: evaluation failed, nothing deleted' % case)
    if case in PROTECTED or a.keep_states:
        print('check-clean %s: states kept (%s)' % (case, 'protected baseline' if case in PROTECTED else '--keep-states'))
        return
    total = 0
    with open(cdir / 'log' / 'deleted_states.txt', 'a') as log:
        for d in sorted((cdir / 'DA').glob('ens_*/state')):
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            shutil.rmtree(d)
            total += size
            log.write('%s\t%s\t%d bytes\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), d, size))
    print('check-clean %s: deleted daily member states, %.1f GB' % (case, total / 1e9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['import', 'import-variant', 'ol', 'check-ol', 'da', 'products', 'r11-truth',
                                     'check-clean', 'ol-unit-mean'])
    ap.add_argument('--keep-states', action='store_true', help='check-clean: check products, do not delete states')
    ap.add_argument('--alpha', default='1e18', choices=['1e14', '1e15', '1e16', '1e17', '1e18', '1e19'])
    ap.add_argument('--grid', default='1degree', choices=['1degree', '2degree', '3degree', '4degree', '3subbasins'])
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--basin', required=True, choices=['Brahmaputra', 'Danube'])
    ap.add_argument('--mission', choices=MISSIONS)
    ap.add_argument('--seed', default='1', help='run.seed of the DA case (int, or none)')
    ap.add_argument('--tag', default='', help='suffix of the DA case name (variants)')
    ap.add_argument('--no-innovations', action='store_true')
    ap.add_argument('--pert-scale', type=float, default=1.0, help='r11-truth: perturbation std factor (harder truth)')
    ap.add_argument('--variant', choices=['nominal', 'R34', 'R41', 'R11', 'R11x3', 'R11x5', 'R01', 'realistic'],
                    help='obs product ESA_SING_<M>_1e18_<variant> (settings <M>_<basin>_<variant>, case tag <variant>): '
                         'nominal = GRACE/output/<M>_old/1e18 (Table 1: diagonal of the propagated noise VCM); R34/R41 = '
                         'GRACE/output_<variant>/<M>/1e18 signal with the nominal cov. Without --variant: GRACE/output/<M>/1e18 '
                         '= the realistic cov of 27 Jul 2025 (Table 2)')
    a = ap.parse_args()
    a.data_root = str(Path(a.data_root).resolve())
    if a.step in ('da', 'products', 'check-clean') and not a.mission:
        sys.exit('--mission needed')
    if a.step == 'import-variant' and not a.variant:
        sys.exit('--variant needed')
    {'import': do_import, 'import-variant': do_import_variant, 'ol': do_ol, 'r11-truth': do_r11_truth,
     'check-clean': do_check_clean, 'check-ol': do_check_ol, 'ol-unit-mean': do_ol_unit_mean, 'da': do_da, 'products': do_products}[a.step](a)


if __name__ == '__main__':
    main()
