import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402
import csv
import numpy as np
import pygmt
rows = list(csv.DictReader(open((FP.RESULTS_V1 + '/04_band_errors/band_errors.csv'))))
bands = ['<30 d', '30-90 d', '90-365 d', '>365 d']
M = ['GRACEC', 'NGGM', 'MAGIC']
light = {'GRACEC': '168/218/185', 'NGGM': '170/210/250', 'MAGIC': '255/236/150'}
dark = {'GRACEC': 'seagreen', 'NGGM': 'dodgerblue', 'MAGIC': 'gold'}
fig = pygmt.Figure()
pygmt.config(FONT_TITLE="15p,5", MAP_TITLE_OFFSET='4p', MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5',
             FONT_LABEL='14p,5', MAP_TICK_LENGTH='6p')
w = 0.11
for bi, basin in enumerate(['Brahmaputra', 'Danube']):
    v = {(r['mission'], r['band']): r for r in rows if r['basin'] == basin}
    fig.basemap(region=[0.4, 4.6, 1, 100], projection='X16c/6cl',
                frame=['WSne+t(%s) %s' % ('ab'[bi], basin), 'xf1', 'ya1pf3+lRMS error [mm]'])
    for j, b in enumerate(bands):
        x0 = j + 1
        items = [('model', float(v[('GRACEC', b)]['model_err']), 'gray60')]
        items += [('obs ' + m, float(v[(m, b)]['obs_err']), light[m]) for m in M]
        items += [('DA ' + m, float(v[(m, b)]['DA_err']), dark[m]) for m in M]
        for k, (lab, val, col) in enumerate(items):
            xk = x0 + (k - 3) * w
            fig.plot(data=[[xk, 1, xk + w * 0.9, val]], style='r+s', fill=col, pen='0.5p,black')
    fig.text(x=[1, 2, 3, 4], y=[1] * 4, text=bands, font='13p,5', justify='CT', offset='0/-0.25c', no_clip=True)
    if bi == 0:
        fig.shift_origin(yshift='-8.2c')
# legend
fig.shift_origin(yshift='-2.3c')
fig.basemap(region=[0, 16, 0, 1], projection='X16c/1.2c', frame='+n')
leg = [('Model', 'gray60'), ('GRACE-C obs', light['GRACEC']), ('NGGM obs', light['NGGM']), ('MAGIC obs', light['MAGIC']),
       ('GRACE-C DA', dark['GRACEC']), ('NGGM DA', dark['NGGM']), ('MAGIC DA', dark['MAGIC'])]
for i, (t, c) in enumerate(leg):
    xx = 0.2 + (i % 4) * 4.0; yy = 0.75 if i < 4 else 0.25
    fig.plot(data=[[xx, yy - 0.15, xx + 0.4, yy + 0.15]], style='r+s', fill=c, pen='0.5p,black')
    fig.text(x=xx + 0.6, y=yy, text=t, font='12p,5', justify='LM')
fig.savefig((FP.OUT + '/band_errors.pdf'))
fig.savefig((FP.OUT + '/band_errors.png'), dpi=150)
print('ok')
