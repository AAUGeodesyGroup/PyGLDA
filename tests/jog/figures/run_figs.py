import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402
import sys, traceback
import numpy as np
import pygmt
_plot = pygmt.Figure.plot
def _plot_compat(self, *a, **k):          # pygmt >= 0.13: scalar x / y / direction must be sequences
    for key in ('x', 'y'):
        if key in k and np.ndim(k[key]) == 0:
            k[key] = [k[key]]
    if 'direction' in k and np.ndim(k['direction']) == 1 and np.ndim(k['direction'][0]) == 0 and 'x' in k and len(k['x']) == 1:
        k['direction'] = [[k['direction'][0]], [k['direction'][1]]]
    return _plot(self, *a, **k)
pygmt.Figure.plot = _plot_compat
import plot_fig_revised as P
jobs = [('metric_spatial_map', 'MetricIntro.pdf'), ('metric_spatial_map_HPF', 'MetricIntro60days.pdf'),
        ('contribution_fig_combination', 'metric_comparison.pdf'), ('scatter_plot_Brahmaputra', 'mission_com_metric_Brahmaputra_v2.pdf'),
        ('scatter_plot_Danube', 'mission_com_metric_Danube_v2.pdf'), ('scatter_plot_Brahmaputra_HPF', 'mission_com_metric_Brahmaputra_60days_v2.pdf')]
only = sys.argv[1:]
for fn, name in jobs:
    if only and fn not in only: continue
    P._OUT[0] = (FP.OUT + '/') + name
    try:
        getattr(P, fn)(); print('OK', fn, name, flush=True)
    except Exception:
        print('FAIL', fn); traceback.print_exc()
