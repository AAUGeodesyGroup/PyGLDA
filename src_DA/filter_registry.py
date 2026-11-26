"""Filter classes selectable with DA_setting.json method.fusion_method (issue 4).

    EnKF                      plain perturbed-observation EnKF (base update; no shrinkage/localization). With
                              method.inflation = 1.0 it is the filter of the published ESA-SING runs (2025,
                              src_DA/data_assimilation_flexible_time.DataAssimilation)
    EnKF_localization_v1      shrinkage of H P H^T (0.10) and R (0.50) towards their diagonal - default (today's runs)
    EnKF_localization_v2, EnKF_domain_localization, EnKF_adaptive   other variants in src_DA/EnKF.py
    EnSQRA, EnSQRA_V2         square-root variants in src_DA/EnSQRA.py

The old enum names EnKF_v0, EnKF_v1, EnKF_v2 (never used to choose a filter) map to EnKF_localization_v1 with a
warning, so existing settings run as before. Classes are imported only when a filter is requested.
"""
import importlib
import warnings

FILTERS = {
    'EnKF': ('src_DA.EnKF', 'EnKF'),
    'EnKF_localization_v1': ('src_DA.EnKF', 'EnKF_localization_v1'),
    'EnKF_localization_v2': ('src_DA.EnKF', 'EnKF_localization_v2'),
    'EnKF_domain_localization': ('src_DA.EnKF', 'EnKF_domain_localization'),
    'EnKF_adaptive': ('src_DA.EnKF', 'EnKF_adaptive'),
    'EnSQRA': ('src_DA.EnSQRA', 'EnSQRA'),
    'EnSQRA_V2': ('src_DA.EnSQRA', 'EnSQRA_V2'),
}
DEFAULT = 'EnKF_localization_v1'
LEGACY = {'EnKF_v0': DEFAULT, 'EnKF_v1': DEFAULT, 'EnKF_v2': DEFAULT}


def resolve_name(name) -> str:
    """canonical filter name; old enum names -> EnKF_localization_v1 (warning); unknown -> ValueError"""
    name = getattr(name, 'name', name)                # an old FusionMethod enum member
    if name in FILTERS:
        return name
    if name in LEGACY:
        warnings.warn('method.fusion_method "%s" is an old name that never selected a filter; it runs %s as before. '
                      'Write "%s" (or another of %s) in DA_setting.json.' % (name, LEGACY[name], LEGACY[name],
                                                                              ', '.join(FILTERS)), UserWarning)
        return LEGACY[name]
    raise ValueError('PyGLDA v1: unknown method.fusion_method "%s"; valid: %s (old names %s = %s)'
                     % (name, ', '.join(FILTERS), ', '.join(LEGACY), DEFAULT))


def get_filter(name):
    module, cls = FILTERS[resolve_name(name)]
    return getattr(importlib.import_module(module), cls)
