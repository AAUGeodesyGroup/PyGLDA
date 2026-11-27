"""Step 33 - metric maps at 1e18 / 1 deg as the cell-wise mean over the nominal seeds (figures), other configurations seed 1.

Inputs: results/07_recompute_v1_rerun/metrics_paper.h5 (seed 1, all 180 configurations) and
results/07_recompute_v1_rerun_s<k>/metrics_paper.h5 (07 --v1 --best-only --seed k, k = 2, 3).
Output: results/33_seed_mean_maps/metrics_paper_seedmean.h5 - a copy of the seed-1 file in which every group
<basin>_<MISSION>_F1E18_G1_<hpf> holds the cell-wise mean over the seeds of nse/kge/rmse _DA (the _model maps are the same
OL for every seed and are checked to be identical); attribute 'seed_mean' lists the seeds; groups.json lists the replaced groups.
Usage: python 33_seed_mean_maps.py [--seeds 1 2 3]
"""
import json
import shutil

import h5py
import numpy as np

from _common import parser, outdir, C


def main():
    p = parser(__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    args = p.parse_args()
    out = outdir(args, '33_seed_mean_maps')
    src = {s: C.OUT_ROOT / ('07_recompute_v1_rerun' if s == 1 else '07_recompute_v1_rerun_s%d' % s) / 'metrics_paper.h5'
           for s in args.seeds}
    dst = out / 'metrics_paper_seedmean.h5'
    shutil.copy2(src[1], dst)
    files = {s: h5py.File(f, 'r') for s, f in src.items()}
    done = {}
    with h5py.File(dst, 'r+') as h:
        for g in h:
            if '_F1E18_G1_' not in g:
                continue
            for k in ('nse', 'kge', 'rmse'):
                model = [files[s][g][k + '_model'][()] for s in args.seeds]
                assert all(np.array_equal(model[0], x, equal_nan=True) for x in model), (g, k, 'model maps differ')
                h[g][k + '_DA'][...] = np.nanmean(np.stack([files[s][g][k + '_DA'][()] for s in args.seeds]), axis=0)
            h[g].attrs['seed_mean'] = np.array(args.seeds)
            done[g] = {k: [float(np.nanmean(files[s][g][k + '_DA'][()])) for s in args.seeds] for k in ('nse', 'kge', 'rmse')}
    (out / 'groups.json').write_text(json.dumps(done, indent=1))
    print('replaced %d groups -> %s' % (len(done), dst))
    for g, v in done.items():
        print(g, {k: [round(x, 3) for x in vv] for k, vv in v.items()})


if __name__ == '__main__':
    main()
