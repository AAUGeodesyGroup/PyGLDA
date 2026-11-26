"""Danube example of PyGLDA v1 (W3RA, 0.1 deg, CSR RL06.3 mascons, Mascon_monthly), basin DRB (3 sub-basins) or
GDRB (19 units, 3-deg grid clipped to the basin). Runs on the desktop and on UCloud.

    cd py-w3ra/demo
    export PYGLDA_V1_DATA=/work/data_for_w3                 # or --data-root
    python demo_danube.py --basin DRB --stages forcing,grace,spinup
    mpiexec -n 31 python demo_danube.py --basin DRB --stages ol,da        # ens + 1 ranks

Settings: settings/Danube_<basin>/ (paths as ${DATA_ROOT}/...). Each stage works on a run copy in
<data_root>/w3ra/cases/<case>/settings/; shared data in w3ra/domains/<domain> and w3ra/units/<unit_set>
(see doc/configuration.md section 7). Default periods: spin-up
2000-01-01..2001-12-31, open loop 2002-01-01..2006-12-31, DA 2002-04-01..2006-12-31 (first CSR month).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append('../')
from src_FlowControl.Regional_DA import RDA

SHP = {'DRB': 'DRB_3_shapefiles/DRB_subbasins.shp',            # relative to <data_root>/basin/shp
       'GDRB': 'GDRB_shapefiles/GDRB_subbasins.shp'}


def main():
    ap = argparse.ArgumentParser(description='PyGLDA v1 Danube example')
    ap.add_argument('--basin', choices=sorted(SHP), default='DRB')
    ap.add_argument('--case', help='case name (default Danube_<basin>)')
    ap.add_argument('--ens', type=int, default=30, help='perturbed members (MPI stages need ens + 1 ranks)')
    ap.add_argument('--spinup', nargs=2, default=['2000-01-01', '2001-12-31'], metavar=('BEGIN', 'END'))
    ap.add_argument('--ol', nargs=2, default=['2002-01-01', '2006-12-31'], metavar=('BEGIN', 'END'))
    ap.add_argument('--da', nargs=2, default=['2002-04-01', '2006-12-31'], metavar=('BEGIN', 'END'))
    ap.add_argument('--stages', default='forcing,grace,spinup', help='comma list of forcing,grace,spinup,ol,da')
    ap.add_argument('--data-root', help='default: environment variable PYGLDA_V1_DATA')
    ap.add_argument('--settings', help='settings folder (default ../settings/Danube_<basin>)')
    ap.add_argument('--legacy-layout', action='store_true', help='old shared OL/DA folders')
    a = ap.parse_args()

    RDA.data_root = a.data_root
    RDA.legacy_layout = a.legacy_layout
    RDA.setting_dir = a.settings or '../settings/Danube_%s' % a.basin
    RDA.case = a.case or 'Danube_%s' % a.basin
    RDA.basin = a.basin
    RDA.shp_path = SHP[a.basin]
    RDA.ens = a.ens
    RDA.cold_begin_time, RDA.cold_end_time = a.spinup
    RDA.warm_begin_time, RDA.warm_end_time = a.ol
    RDA.resume_begin_time, RDA.resume_end_time = a.da
    RDA.isSet = False

    for stage in a.stages.split(','):
        if stage == 'forcing':
            RDA.prepare_Forcing()
        elif stage == 'grace':
            g = json.loads((Path(RDA._settings()) / 'DA_setting.json').read_text())['obs']
            for d in (g['dir'], g['GRACE']['preprocess_res'], g['GRACE']['OL_mean']):
                Path(d).mkdir(parents=True, exist_ok=True)
            RDA.prepare_GRACE_Mascon(dir_in=g['GRACE']['EWH_grid_dir'], cov_dir_in=g['GRACE']['cov_dir'],
                                     dir_out=g['GRACE']['preprocess_res'], is_diagonal=False)
        elif stage == 'spinup':
            RDA.single_run(skip_croping_data=True, skip_signal_extraction=True)
        elif stage == 'ol':
            RDA.OL_run(skip_perturbation=False)
        elif stage == 'da':
            RDA.DA_run(skip_obs_perturbation=False)
        else:
            sys.exit('unknown stage %s' % stage)


if __name__ == '__main__':
    main()
