"""Share of the absolute DA increment per storage compartment, one stacked bar per basin x mission (mean of seeds 1-3)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402

import json

import pygmt

z = json.load(open((FP.INPUTS + '/inc_shares.json')))
OUT = (FP.OUT + '/increments')
COL = ['darkorange', 'gold', 'sienna', 'royalblue', 'lightskyblue', 'purple', 'forestgreen']
NAME = {'GRACEC': 'GRACE-C', 'NGGM': 'NGGM', 'MAGIC': 'MAGIC'}
bars = []
y = 7.4
for b in ('Brahmaputra', 'Danube'):
    for m in ('GRACEC', 'NGGM', 'MAGIC'):
        y -= 1
        bars.append((y, '%s %s' % (b, NAME[m]), z['shares_pct']['%s_%s' % (b, m)]['mean']))
    y -= 0.6

fig = pygmt.Figure()
pygmt.config(FONT_TITLE="15p,5", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5', FONT_LABEL='14p,5', MAP_TICK_LENGTH='5p')
fig.basemap(region=[0, 100, 0, 7.0], projection='X15c/8.2c', frame=['S', 'xa10f5+lShare of absolute increment [%]'])
for y, label, sh in bars:
    x0 = 0.0
    for v, c in zip(sh, COL):
        if v > 0:
            fig.plot(data=[[x0, y - 0.38, x0 + v, y + 0.38]], style='r+s', fill=c, pen='0.4p,black')
            if v >= 8:
                fig.text(x=x0 + v / 2, y=y, text='%.0f' % v, font='11p,5,white' if c in ('sienna', 'royalblue', 'purple', 'forestgreen') else '11p,5,black')
        x0 += v
    fig.text(x=-1.5, y=y, text=label, font='13p,5,black', justify='RM', no_clip=True)
# legend below (two rows)
fig.shift_origin(yshift='-3.4c')
fig.basemap(region=[0, 15, 0, 2], projection='X15c/1.6c', frame='+n')
for n, (name, c) in enumerate(zip(z['compartments'], COL)):
    xx, yy = (n % 4) * 3.9, 1.5 - (n // 4) * 0.9
    fig.plot(data=[[xx, yy - 0.25, xx + 0.6, yy + 0.25]], style='r+s', fill=c, pen='0.4p,black')
    fig.text(x=xx + 0.75, y=yy, text=name, font='12p,5,black', justify='LM')
fig.savefig(OUT + '.pdf')
fig.savefig(OUT + '.png', dpi=150)
print('written', OUT)
