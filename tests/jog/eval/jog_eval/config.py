"""Paths, names and conventions of the JoG ESA-SING evaluation (JOGE-D-26-00146, ESM2.0 only).

Every path is built from two roots that can be set by environment variables:
    JOG_ESM2_ROOT   ESM2.0 inputs (truth, synthetic observations, masks, basin series)
                    default: /work/data_for_w3/GRACE/ESA_SING/ESM2.0         (UCloud)
    JOG_RESULTS_ROOT  published OL/DA products (daily 0.1 deg ensemble means, 'TotalWater.<year>.h5')
                    default: <JOG_ESM2_ROOT>/Analysis   (desktop layout: Analysis/<basin>/{OL,DA}_mean/<case>/)
New runs made with the v1 data layout are read through `sources.case_products()` (cases/<case>/products/...).

Naming follows the published cases: <MISSION>_<alpha>_<grid>_<basin>, e.g. MAGIC_1e18_1degree_Brahmaputra.
"""
import os
from pathlib import Path

ESM2_ROOT = Path(os.environ.get('JOG_ESM2_ROOT', '/work/data_for_w3/GRACE/ESA_SING/ESM2.0'))
RESULTS_ROOT = Path(os.environ.get('JOG_RESULTS_ROOT', str(ESM2_ROOT / 'Analysis')))
V1_DATA_ROOT = Path(os.environ.get('PYGLDA_V1_DATA', '/work/data_for_w3'))
OUT_ROOT = Path(os.environ.get('JOG_OUT', str(Path(__file__).resolve().parents[1] / 'results')))

BASINS = ('Brahmaputra', 'Danube')
MISSIONS = ('GRACEC', 'NGGM', 'MAGIC')          # case names
MISSION_FILE = {'GRACEC': 'GRACE-C-like', 'NGGM': 'NGGM', 'MAGIC': 'MAGIC'}   # names in the ESA-SING files
ALPHAS = ('1e14', '1e15', '1e16', '1e17', '1e18', '1e19')
GRIDS = ('1degree', '2degree', '3degree', '4degree', '3subbasins')
BEST = {'alpha': '1e18', 'grid': '1degree'}    # best configuration in the paper (Table 1 text)

# evaluation period of the paper: 5-day epochs of the truth, 2003-09-05 .. 2006-12 (indices 634..875, 242 epochs)
PAPER_EPOCHS = (634, 634 + 242)
EPOCH_DAYS = 5


def case_name(mission, alpha, grid, basin):
    return '%s_%s_%s_%s' % (mission, alpha, grid, basin)


def alpha_file(alpha):
    """'1e18' -> '1e+18' (spelling in the ESA-SING files)"""
    return alpha.replace('e', 'e+')


# ---- ESM2.0 inputs ----------------------------------------------------------------------------------------------
def truth_grid_file():
    return ESM2_ROOT / 'globe' / 'Global_EWHA_all' / 'Global_EWH_05_HIS.hdf5'


def obs_grid_file(mission, alpha):
    return ESM2_ROOT / 'globe' / 'Global_EWHA_all' / ('Global_EWH_05_%s_%s.hdf5' % (MISSION_FILE[mission], alpha_file(alpha)))


def region_mask_file(basin, res='0.5'):
    return ESM2_ROOT / 'globe' / 'global_mask' / basin / res / 'mask.h5'


def case_mask_file(case):
    """0.1 deg mask of a published case (mask, lat, lon); a path to an .h5 mask file (rerun products) is used as is"""
    if str(case).endswith('.h5'):
        return Path(case)
    return RESULTS_ROOT / 'Mask' / (case + '.h5')


def basin_signal_file(basin, whole=False):
    """basin series of truth (<grid>/HIS/EWHA) and observations (<grid>/<mission>/EWHA_alpha_<a>), 876 epochs 1995-2006"""
    name = 'basin_signal_whole_basin.hdf5' if whole else 'basin_signal.hdf5'
    p = ESM2_ROOT / basin / 'truth' / name                      # UCloud copy
    return p if p.exists() else RESULTS_ROOT / basin / 'Basin' / name   # desktop layout


def published_daily_dir(kind, case):
    """kind 'OL' or 'DA' -> folder with TotalWater.<year>.h5 (daily 0.1 deg ensemble mean)"""
    basin = case.rsplit('_', 1)[-1]
    return RESULTS_ROOT / basin / ('%s_mean' % kind) / case
