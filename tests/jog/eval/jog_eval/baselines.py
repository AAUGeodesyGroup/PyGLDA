"""Revision baselines (v1 data layout): seeded 30-member DA means at alpha 1e18 / 1 deg.

nominal = published method with the nominal VCM: cases jog_<M>_<basin>_s<seed>_nominal
realistic = residual-based VCM (Table 2, Brahmaputra only): cases jog_<M>_Brahmaputra_s<seed>
OL = member 0 of domains/jog_<basin>1degree/OL/pub_magic1e19 (identical to the published OL).
0.1 deg masks: the published case of the same mission (all 1e18 / 1 deg masks of a basin are identical).
"""
from . import config as C
from . import evaluate as E
from . import sources as S

ENSEMBLE = 'pub_magic1e19'


def case(mission, basin, seed=1, cov='nominal'):
    return 'jog_%s_%s_s%d%s' % (mission, basin, seed, '_nominal' if cov == 'nominal' else '')


def folders(basin, seed=1, cov='nominal', missions=C.MISSIONS):
    dom = 'jog_%s1degree' % basin
    out = {}
    for m in missions:
        p = S.case_products(case(m, basin, seed, cov), dom, ENSEMBLE)
        out.setdefault('model', ('OL', p['OL'], C.case_name('MAGIC', '1e18', '1degree', basin)))
        out[m] = ('DA', p['DA'], C.case_name(m, '1e18', '1degree', basin))
    return out


def load(basin, epochs, seed=1, cov='nominal', missions=C.MISSIONS):
    """truth + {model, missions} on the common 0.5 deg / 5-day grid (evaluate.load_basin)"""
    return E.load_basin(basin, epochs, folders=folders(basin, seed, cov, missions))
