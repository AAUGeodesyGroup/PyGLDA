"""Step 29 - 2-member vs 30-member DA ensemble mean on the grid (0.5 deg), the 6 nominal seed-1 baselines.

The published DA_mean products averaged members 1-2 only (dataManager_ensemble_statistic default ens_size=2); the reruns
average members 1-30. Per baseline case jog_<M>_<basin>_s1_nominal (states kept):
  1. --build: daily 2-member mean product (members 1-2, as the published save_data) -> cases/<case>/products/DA_mean2
     (DA_std2), with the same basin mask and period as the 30-member DA_mean (tests/jog/jog_run.py products);
  2. evaluation vs the ESM2.0 truth, paper definition, epochs 634..875, 0.5 deg: spatial-mean NSE/KGE/RMSE of the 2- and the
     30-member mean, the change 30 - 2 with paired block-bootstrap CIs, and the grid RMS / max of D = mean2 - mean30 (mm).
Output: results/29_two_member_grid/{summary.json, table.md}.
Usage: python 29_two_member_grid.py [--build] [--basins ...] [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from _common import parser, outdir, write_json, C
from jog_eval import evaluate as E, stats as ST

REPO = Path(__file__).resolve().parents[4]          # repository root
NAMES = ('nse', 'kge', 'rmse')


def build(basin, mission):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / 'tests' / 'jog'))
    import jog_run as J
    from src_postprocessing.DataManager import dataManager_ensemble_statistic, data_var
    from src_DA.shp2mask import basin_shp_process
    from src_FlowControl import run_context as rc
    a = argparse.Namespace(step='products', basin=basin, mission=mission, seed='1', variant='nominal', tag='', alpha='1e18',
                           grid='1degree', data_root=str(C.V1_DATA_ROOT), no_innovations=False)
    n = J.names(a, mission)
    RDA = J.setup(a, n)
    sd = RDA.activate()
    gmask = basin_shp_process(res=0.1, basin_name=n['unit'], save_dir=None).shp_to_mask(
        shp_path=rc.resolve_shp(n['shp'])).mask[0]
    prod = C.V1_DATA_ROOT / 'w3ra' / 'cases' / n['case'] / 'products'
    d = dataManager_ensemble_statistic(ens_size=2).configure(
        setting_fn=str(Path(sd) / 'setting.json'), out_dir_mean=str(prod / 'DA_mean2'), out_dir_std=str(prod / 'DA_std2'),
        variable=data_var.TotalWater)
    d.reduce_datasize(global_basin_mask=gmask)
    d.aggregation_daily(date_begin=J.PERIODS['resume'][0], date_end=J.PERIODS['resume'][1])
    return n['case'], d._out_dir


def folder(case, name):
    p = C.V1_DATA_ROOT / 'w3ra' / 'cases' / case / 'products' / name
    hits = sorted({f.parent for f in p.rglob('TotalWater.2004.h5')})
    assert len(hits) == 1, (p, hits)
    return str(hits[0])


def main():
    p = parser(__doc__)
    p.add_argument('--build', action='store_true')
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--block', type=int, default=18)
    args = p.parse_args()
    out = outdir(args, '29_two_member_grid')
    epochs = np.arange(*C.PAPER_EPOCHS)
    res, md = {}, ['# 2- vs 30-member DA mean, 0.5 deg grid, paper definition (%d epochs, block %d, %d bootstraps)\n' %
                   (len(epochs), args.block, args.n_boot),
                   '| case | cells | NSE 2 / 30 / 30-2 [CI] | KGE 2 / 30 / 30-2 [CI] | RMSE 2 / 30 / 30-2 [CI] | RMS(D) / max abs D mm |',
                   '|---|---|---|---|---|---|']
    for basin in args.basins:
        for m in C.MISSIONS:
            case = 'jog_%s_%s_s1_nominal' % (m, basin)
            if args.build:
                print('built', build(basin, m), flush=True)
            mask_case = '%s_1e18_1degree_%s' % (m, basin)
            data = E.load_basin(basin, epochs, missions=('m2', 'm30'),
                                folders={'m2': ('DA', folder(case, 'DA_mean2'), mask_case),
                                         'm30': ('DA', folder(case, 'DA_mean'), mask_case)})
            b = ST.bootstrap_table(data['truth'], data['sims'], NAMES, 'paper', args.block, args.n_boot, pairs=(('m30', 'm2'),))
            D = data['sims']['m2'] - data['sims']['m30']
            rmsd = float(np.sqrt(np.nanmean(D ** 2)))
            mxd = float(np.nanmax(np.abs(D)))
            res[case] = {'cells': data['cells'], 'bootstrap': b, 'rms_D_mm': rmsd, 'max_abs_D_mm': mxd, 'missing': data['missing']}
            cell = lambda n: '%.3f / %.3f / %+.3f [%+.3f, %+.3f]' % (b['point']['m2'][n], b['point']['m30'][n],
                                                                   b['point']['m30-m2'][n], *b['ci']['m30-m2'][n])
            md.append('| %s | %d | %s | %s | %s | %.2f / %.1f |' % (case, data['cells'], cell('nse'), cell('kge'), cell('rmse'),
                                                                 rmsd, mxd))
            print(md[-1], flush=True)
    write_json(out / 'summary.json', res)
    (out / 'table.md').write_text('\n'.join(md) + '\n')
    print('written', out)


if __name__ == '__main__':
    main()
