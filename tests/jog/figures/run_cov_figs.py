"""realistic-VCM figures (Fig. 12 / C1) with the original functions: nominal = seed-1 rerun maps, realistic = realistic-VCM rerun maps"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402

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
P._H5_PATH = (FP.INPUTS + '/metrics_paper.h5')
P._H5_COV_PATH = (FP.INPUTS + '/metrics_paper_realistic.h5')
O = (FP.OUT + '/')
for fn, name in (('scatter_plot_cov_comparison_Brahmaputra', 'realistic_cov_DA.pdf'),
                 ('scatter_plot_cov_comparison_Danube', 'realistic_cov_DA_Danube.pdf')):
    P._OUT[0] = O + name
    getattr(P, fn)()
    print('OK', fn, name, flush=True)
