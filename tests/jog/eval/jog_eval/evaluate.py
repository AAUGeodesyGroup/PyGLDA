"""Building blocks shared by the scripts: load truth + products of one basin and compute the Table-1 style numbers."""
import os

import numpy as np

from . import config as C
from . import metrics as M
from . import sources as S


def v1_folders(basin, alpha, grid, missions=C.MISSIONS, seed=None):
    """rerun products of the v1 data layout: DA_mean (members 1-30) of cases jog_<M>_<basin>[_<alpha>_<grid>]_s<seed>_nominal with
    each case's own products/mask, model = OL mean of the shared OL with the 1-degree mask (as 07_recompute_all --v1)"""
    seed = seed or os.environ.get('JOG_V1_SEED', '1')
    w3 = C.V1_DATA_ROOT / 'w3ra'
    cfg = '' if (alpha, grid) == ('1e18', '1degree') else '_%s_%s' % (alpha, grid)
    ref = 'jog_%s_%s_s%s_nominal' % (C.MISSIONS[0], basin, seed)
    f = {'model': ('OL', str(w3 / 'domains' / ('jog_%s1degree' % basin) / 'OL' / 'pub_magic1e19' / 'products' / 'OL_mean'),
                   str(w3 / 'cases' / ref / 'products' / 'mask' / (ref + '.h5')))}
    for m in missions:
        case = 'jog_%s_%s%s_s%s_nominal' % (m, basin, cfg, seed)
        f[m] = ('DA', str(w3 / 'cases' / case / 'products' / 'DA_mean'), str(w3 / 'cases' / case / 'products' / 'mask' / (case + '.h5')))
    return f


def load_basin(basin, epochs, missions=C.MISSIONS, alpha=C.BEST['alpha'], grid=C.BEST['grid'], folders=None,
               model_case=None):
    """truth and {model, <missions>} on common valid 0.5 deg cells.
    folders: optional {name: (kind, folder, mask_case)} to read other runs (e.g. revision variants);
    by default the published cases <mission>_<alpha>_<grid>_<basin> are used, the model = OL of the first mission case."""
    if folders is None and os.environ.get('JOG_V1_RERUN'):              # environment switch: evaluate the rerun products
        folders = v1_folders(basin, alpha, grid, missions)
    truth, lat, lon = S.truth_grid(basin, epochs)
    sims, missing = {}, {}
    model_case = model_case or C.case_name(missions[0], alpha, grid, basin)
    if folders is None:
        folders = {'model': ('OL', None, model_case)}
        for m in missions:
            folders[m] = ('DA', None, C.case_name(m, alpha, grid, basin))
    for name, (kind, folder, case) in folders.items():
        sims[name], missing[name] = S.product_grid(kind, case, epochs, folder=folder)
    ok = np.isfinite(truth).all(0)
    for v in sims.values():
        ok &= np.isfinite(v).all(0)
    truth = np.where(ok[None], truth, np.nan)
    sims = {k: np.where(ok[None], v, np.nan) for k, v in sims.items()}
    return {'truth': truth, 'sims': sims, 'lat': lat, 'lon': lon, 'cells': int(ok.sum()), 'missing': missing}


def table(data, definition='paper', names=('nse', 'kge', 'rmse')):
    """spatial-mean metrics {product: {metric: value}} with the paper's clipping"""
    out = {}
    for k, s in data['sims'].items():
        m = M.all_metrics(data['truth'], s, definition)
        out[k] = {n: M.spatial_mean(m[n], n) for n in names}
    return out


def improvements(tab, ref='GRACEC', model='model', others=('NGGM', 'MAGIC'), names=('nse', 'kge', 'rmse')):
    out = {}
    for o in others:
        out[o] = {n: M.improvement(tab[model][n], tab[ref][n], tab[o][n]) for n in names}
    return out
