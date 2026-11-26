"""Regression harness for PyGLDA v1: runs the Danube DRB chain with fixed random seeds, so that two runs (and a run of
changed code against a run of the old code) can be compared bit by bit with compare.py.

    cd <repo>/tests/regression
    python run_reference.py --case reg_base_a --data-root /work/data_for_w3 [--settings reference_settings]
                            [--ens 4] [--stages forcing,grace,spinup,ol,da] [--legacy-layout]

Chain (Danube DRB, 3 sub-basins, CSR mascons): forcing crop 2002-01..12, GRACE preparation 2002-04..12, spin-up
2002-01-01..03-31, open loop 2002-04-01..12-31, DA 2002-05-01..12-31 (first window that can start after the OL start).
Before every stage each rank calls np.random.seed(20021 + 1000*rank + stage_no): the perturbation code uses numpy's
global generator (src_DA/Perturbation.py, src_OBS/GRACE_perturbation.py, src_DA/EnKF.py). Only the harness seeds; the
model code itself is not seeded.

Settings: the four JSON files of --settings (placeholders ${DATA_ROOT} and ${CASE}) are copied, with the placeholders
substituted, to <data-root>/w3ra/runs/<case>/harness_settings/ and the run uses that copy, so tracked files are never
written. Works with the old code (module variables res_output/figure_output) and with code that has RDA.data_root.
Each stage runs as its own process (mpiexec -n ens+1 for ol and da) with cwd = <repo>/demo; logs and a resources
line per stage go to <data-root>/w3ra/runs/<case>/harness_log/.
"""
import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGES = ['forcing', 'grace', 'spinup', 'ol', 'da']
STAGE_NO = {s: i + 1 for i, s in enumerate(STAGES)}
BASIN = 'DRB'
SHP = 'DRB_3_shapefiles/DRB_subbasins.shp'                    # relative to <data_root>/basin/shp
PERIODS = dict(cold=('2002-01-01', '2002-03-31'), warm=('2002-04-01', '2002-12-31'),
               resume=('2002-05-01', '2002-12-31'))


def rank():
    for k in ('PMI_RANK', 'OMPI_COMM_WORLD_RANK'):
        if k in os.environ:
            return int(os.environ[k])
    return 0


def prepare_settings(src: Path, data_root: str, case: str, make_obs_dirs=True) -> Path:
    dst = Path(data_root) / 'w3ra' / 'runs' / case / 'harness_settings'
    dst.mkdir(parents=True, exist_ok=True)
    for f in ('DA_setting.json', 'perturbation.json', 'pre_process.json', 'setting.json'):
        txt = (src / f).read_text().replace('${DATA_ROOT}', data_root).replace('${CASE}', case)
        json.loads(txt)                                   # still valid JSON
        (dst / f).write_text(txt)
    if make_obs_dirs:                                     # old places; the layout makes its own folders
        a = json.loads((dst / 'DA_setting.json').read_text())
        for d in (a['obs']['dir'], a['obs']['GRACE']['preprocess_res'], a['obs']['GRACE']['OL_mean']):
            Path(d).mkdir(parents=True, exist_ok=True)
    return dst


def run_stage(stage, settings_dir, data_root, case, ens, legacy, names=None):
    """executed inside each process (rank)"""
    import numpy as np
    np.random.seed(20021 + 1000 * rank() + STAGE_NO[stage])
    sys.path.insert(0, str(REPO))
    import src_FlowControl.Regional_DA as mm
    from src_FlowControl.Regional_DA import RDA
    if hasattr(RDA, 'data_root'):                         # new code
        RDA.data_root = data_root
        if hasattr(RDA, 'legacy_layout'):
            RDA.legacy_layout = legacy
        for k, v in (names or {}).items():                # domain / ensemble / unit_set / grace_product (layout)
            if v and hasattr(RDA, k):
                setattr(RDA, k, v)
    else:                                                 # old code: module variables (issue 13)
        mm.res_output = data_root + '/w3ra/res'
        mm.figure_output = data_root + '/w3ra/figure'
    RDA.setting_dir = str(settings_dir)
    RDA.ens = ens
    RDA.case = case
    RDA.basin = BASIN
    RDA.shp_path = SHP
    RDA.cold_begin_time, RDA.cold_end_time = PERIODS['cold']
    RDA.warm_begin_time, RDA.warm_end_time = PERIODS['warm']
    RDA.resume_begin_time, RDA.resume_end_time = PERIODS['resume']
    RDA.isSet = False
    if stage == 'forcing':
        RDA.prepare_Forcing()
    elif stage == 'grace':
        g = json.loads((Path(settings_dir) / 'DA_setting.json').read_text())['obs']['GRACE']
        RDA.prepare_GRACE_Mascon(dir_in=g['EWH_grid_dir'], cov_dir_in=g['cov_dir'], dir_out=g['preprocess_res'],
                                 is_diagonal=False)
    elif stage == 'spinup':
        RDA.single_run(skip_croping_data=True, skip_signal_extraction=True)
    elif stage == 'ol':
        RDA.OL_run(skip_perturbation=False)
    elif stage == 'da':
        RDA.DA_run(skip_obs_perturbation=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--settings', default=str(HERE / 'reference_settings'))
    ap.add_argument('--ens', type=int, default=4)
    ap.add_argument('--stages', default=','.join(STAGES))
    ap.add_argument('--legacy-layout', action='store_true')
    ap.add_argument('--domain', help='layout: shared domain name (default: case)')
    ap.add_argument('--ensemble', help='layout: shared ensemble name (default: n<ens>)')
    ap.add_argument('--unit-set', help='layout: shared unit set name (default: basin)')
    ap.add_argument('--product', help='layout: GRACE product name (default from obs.GRACE.kind)')
    ap.add_argument('--stage', help=argparse.SUPPRESS)              # internal: run one stage in this process
    a = ap.parse_args()
    data_root = str(Path(a.data_root).resolve())

    if a.stage:
        t0 = time.time()
        try:
            run_stage(a.stage, Path(data_root) / 'w3ra' / 'runs' / a.case / 'harness_settings', data_root, a.case,
                      a.ens, a.legacy_layout, names=dict(domain=a.domain, ensemble=a.ensemble, unit_set=a.unit_set,
                                                         grace_product=a.product))
        except Exception as exc:
            # errors outside a guarded stage (imports, settings): central error log if the code under test has one
            try:
                import traceback
                sys.path.insert(0, str(REPO))
                from src_FlowControl.run_context import record_error
                record_error('harness_' + a.stage, traceback.format_exc(), exc, case=a.case, data_root=data_root,
                             run_settings=str(Path(data_root) / 'w3ra' / 'runs' / a.case / 'harness_settings'))
            except ImportError:
                pass
            raise
        r = rank()
        log = Path(data_root) / 'w3ra' / 'runs' / a.case / 'harness_log'
        with open(log / 'resources_harness.csv', 'a') as f:
            f.write('%s,%d,%.2f,%.1f\n' % (a.stage, r, time.time() - t0,
                                           resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        return

    new_layout = (REPO / 'src_FlowControl' / 'run_context.py').is_file() and not a.legacy_layout
    if new_layout:
        new_layout = 'class Layout' in (REPO / 'src_FlowControl' / 'run_context.py').read_text()
    settings = prepare_settings(Path(a.settings), data_root, a.case, make_obs_dirs=not new_layout)
    log = Path(data_root) / 'w3ra' / 'runs' / a.case / 'harness_log'
    log.mkdir(parents=True, exist_ok=True)
    (log / 'resources_harness.csv').write_text('stage,rank,wall_s,maxrss_MB\n')
    me = [sys.executable, '-u', str(Path(__file__).resolve()), '--case', a.case, '--data-root', data_root,
          '--ens', str(a.ens)] + (['--legacy-layout'] if a.legacy_layout else [])
    for opt, val in (('--domain', a.domain), ('--ensemble', a.ensemble), ('--unit-set', a.unit_set),
                     ('--product', a.product)):
        if val:
            me += [opt, val]
    os.environ['PYGLDA_V1_SEED_INFO'] = 'tests/regression harness: np.random.seed(20021 + 1000*rank + stage_no)'
    for s in a.stages.split(','):
        cmd = (['mpiexec', '-n', str(a.ens + 1)] if s in ('ol', 'da') else []) + me + ['--stage', s]
        t0 = time.time()
        print('[%s] stage %s start' % (time.strftime('%H:%M:%S'), s), flush=True)
        with open(log / ('%s.log' % s), 'w') as out:
            try:                                          # an MPI rank that dies can leave the others hanging
                rc = subprocess.run(cmd, cwd=str(REPO / 'demo'), stdout=out, stderr=subprocess.STDOUT,
                                    timeout=3600).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        print('[%s] stage %s end rc=%d wall=%.1f s' % (time.strftime('%H:%M:%S'), s, rc, time.time() - t0),
              flush=True)
        if rc != 0:
            sys.exit('harness stopped: stage %s failed (rc %d), see %s' % (s, rc, log / ('%s.log' % s)))
    print('HARNESS_DONE %s settings=%s' % (a.case, settings), flush=True)


if __name__ == '__main__':
    main()
