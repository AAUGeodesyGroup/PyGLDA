"""Robustness of the mission ranking: NSE (paper definition, spatial mean of 0.5 deg cells) in six experiments, both basins."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402

import json
import csv
import numpy as np
import pygmt

J = (FP.RESULTS + '/')
OUT = (FP.OUT + '/robustness')
M = ['GRACEC', 'NGGM', 'MAGIC']
COL = {'GRACEC': 'seagreen', 'NGGM': 'dodgerblue', 'MAGIC': 'gold'}
LAB = {'GRACEC': 'GRACE-C', 'NGGM': 'NGGM', 'MAGIC': 'MAGIC'}

t1 = {r['basin']: r for r in csv.DictReader(open(J + '24_tables/table1A.csv')) if r['metric'] == 'NSE'}
r01 = json.load(open(J + '31_r01_compare/summary.json'))
tr = {(r['basin'], r['product']): r for r in csv.DictReader(open(
    (FP.RESULTS + '/15_truncation/truncation_table1.csv')))}

# per basin, per panel: model value, and per mission dict(v=..., rng=(lo,hi) or None, ci=(lo,hi) or None)
D = {}
for b in ('Brahmaputra', 'Danube'):
    t = t1[b]
    P = {}
    P['a'] = (float(t['model_mean']), {m: dict(v=float(t[m + '_mean']), rng=tuple(sorted(float(x) for x in t[m + '_seeds'].split()))[::2],
                                            ci=(float(t[m + '_ci_low']), float(t[m + '_ci_high']))) for m in M})
    r34 = {'Brahmaputra': {'GRACEC': 0.720, 'NGGM': 0.789, 'MAGIC': 0.794}, 'Danube': {'GRACEC': 0.565, 'NGGM': 0.832, 'MAGIC': 0.842}}[b]
    P['b'] = (float(t['model_mean']), {m: dict(v=r34[m], rng=None, ci=None) for m in M})
    S = r01[b]['grid']['seeds']['nse']
    P['c'] = (float(t['model_mean']), {m: dict(v=S[m]['R01_mean'], rng=tuple(S[m]['R01_range']), ci=None) for m in M})
    P['d'] = (float(tr[(b, 'model')]['nse_nnbuf']), {m: dict(v=float(tr[(b, m)]['nse_nnbuf']), rng=None, ci=None) for m in M})
    x3 = {'Brahmaputra': (0.861, {'GRACEC': (0.830, 0.704, 0.877), 'NGGM': (0.941, 0.892, 0.955), 'MAGIC': (0.955, 0.920, 0.966)}),
          'Danube': (0.439, {'GRACEC': (-0.341, -0.820, -0.045), 'NGGM': (0.539, 0.257, 0.569), 'MAGIC': (0.703, 0.523, 0.723)})}[b]
    P['e'] = (x3[0], {m: dict(v=x3[1][m][0], rng=None, ci=x3[1][m][1:]) for m in M})
    if b == 'Brahmaputra':
        P['f'] = (0.357, {'GRACEC': dict(v=0.638, rng=None, ci=(0.598, 0.672)), 'MAGIC': dict(v=0.722, rng=None, ci=(0.691, 0.756))})
    D[b] = P

TITLES = {'a': 'Main experiment', 'b': 'No periodic errors', 'c': 'Release low degrees',
          'd': 'Truncated d/o 70', 'e': 'Control (truth x3)', 'f': '1996-2006'}
YR = {'Brahmaputra': (0.3, 1.0, 'a0.1f0.05'), 'Danube': (-0.9, 1.0, 'a0.2f0.1')}

fig = pygmt.Figure()
pygmt.config(FONT_TITLE="11p,5", MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='10p,5',
             FONT_LABEL='12p,5', MAP_TICK_LENGTH='4p')
W, H = 3.9, 5.5
for bi, b in enumerate(('Brahmaputra', 'Danube')):
    y0, y1, yf = YR[b]
    for pi, p in enumerate('abcdef'):
        letter = '(%s)' % 'abcdefghijkl'[bi * 6 + pi]
        side = 'W' if pi == 0 else 'w'
        frame = ['%sSne+t%s %s' % (side, letter, TITLES[p] if bi == 0 else ''), 'xf1', 'y%s%s' % (yf, '+lNSE (%s)' % b if pi == 0 else '')]
        fig.basemap(region=[0.4, 3.6, y0, y1], projection='X%sc/%sc' % (W, H), frame=frame)
        if p not in D[b]:
            fig.text(x=2, y=(y0 + y1) / 2, text='not run', font='10p,6,gray40')
        else:
            mod, vals = D[b][p]
            fig.plot(x=[0.4, 3.6], y=[mod, mod], pen='1.5p,red3,--')
            for k, m in enumerate(M):
                x = k + 1
                if m not in vals:
                    fig.text(x=x, y=y0 + 0.85 * (y1 - y0), text='not run', font='8p,6,gray40')
                    continue
                d = vals[m]
                if d['ci']:
                    lo, hi = max(d['ci'][0], y0), min(d['ci'][1], y1)
                    fig.plot(x=[x, x], y=[lo, hi], pen='1p,black')
                if d['rng']:
                    fig.plot(x=[x, x], y=list(d['rng']), pen='5p,%s@40' % COL[m])
                fig.plot(x=[x], y=[max(d['v'], y0)], style='c0.28c', fill=COL[m], pen='0.8p,black')
        fig.text(x=[1, 2, 3], y=[y0] * 3, text=[LAB[m] for m in M], font='7.5p,5', justify='CT', offset='0/-0.15c', no_clip=True)
        if pi < 5:
            fig.shift_origin(xshift='%sc' % (W + 0.25))
    fig.shift_origin(xshift='-%sc' % (5 * (W + 0.25)), yshift='-%sc' % (H + 1.3))
# legend
fig.shift_origin(yshift='%sc' % (H - 0.2))
fig.basemap(region=[0, 20, 0, 1], projection='X20c/0.9c', frame='+n')
items = [('c', 'seagreen', 'GRACE-C'), ('c', 'dodgerblue', 'NGGM'), ('c', 'gold', 'MAGIC')]
for i, (_, c, t) in enumerate(items):
    fig.plot(x=[0.4 + i * 2.6], y=[0.5], style='c0.28c', fill=c, pen='0.8p,black')
    fig.text(x=0.7 + i * 2.6, y=0.5, text=t, font='10p,5', justify='LM')
fig.plot(x=[8.3, 9.1], y=[0.5, 0.5], pen='1.5p,red3,--'); fig.text(x=9.3, y=0.5, text='Model (open loop)', font='10p,5', justify='LM')
fig.plot(x=[13.0, 13.0], y=[0.2, 0.8], pen='5p,gray50@40'); fig.text(x=13.3, y=0.5, text='Seed range (3 seeds)', font='10p,5', justify='LM')
fig.plot(x=[17.3, 17.3], y=[0.2, 0.8], pen='1p,black'); fig.text(x=17.5, y=0.5, text='95% CI', font='10p,5', justify='LM')
fig.savefig(OUT + '.pdf')
fig.savefig(OUT + '.png', dpi=150)
rows = []
for b in D:
    for p in 'abcdef':
        if p in D[b]:
            mod, vals = D[b][p]
            rows.append((b, p, mod, {m: vals[m] for m in vals}))
json.dump([(b, p, mod, {m: d for m, d in v.items()}) for b, p, mod, v in rows], open((FP.OUT + '/robustness_values.json'), 'w'), indent=1)
print('ok')
