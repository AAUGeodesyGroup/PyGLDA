"""Supporting-information panels from the desktop results: record length (extension), periodic-error removal, seed ranges,
open-loop mismatch decomposition."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402

import csv
import numpy as np
import pygmt

O = (FP.OUT + '/')
COL = {'GRACEC': 'seagreen', 'NGGM': 'dodgerblue', 'MAGIC': 'gold'}
LAB = {'GRACEC': 'GRACE-C', 'NGGM': 'NGGM', 'MAGIC': 'MAGIC'}
M = ['GRACEC', 'NGGM', 'MAGIC']


def cfg():
    pygmt.config(FONT_TITLE="13p,5", MAP_TITLE_OFFSET='4p', MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='11p,5',
                 FONT_LABEL='12p,5', MAP_TICK_LENGTH='5p')


def save(fig, name):
    fig.savefig(O + name + '.pdf')
    fig.savefig(O + name + '.png', dpi=150)


# --- SI_extension_b: KGE vs length of the evaluation record (Brahmaputra 1996-2006, last N epochs) ---
N = np.array([146, 219, 292, 365, 511, 657, 802])
yrs = N * 5 / 365.25
kge = {'GRACEC': [0.690, 0.719, 0.724, 0.723, 0.729, 0.728, 0.719], 'MAGIC': [0.766, 0.785, 0.789, 0.780, 0.775, 0.779, 0.774]}
fig = pygmt.Figure(); cfg()
fig.basemap(region=[1.5, 11.5, 0.65, 0.82], projection='X12c/7c',
            frame=['WSne+t(b) Brahmaputra 1996-2006: KGE vs record length', 'xa1f1+lLength of the evaluation record [years]',
                   'ya0.05f0.01+lKGE'])
for m, v in kge.items():
    fig.plot(x=yrs, y=v, pen='2p,%s' % COL[m])
    fig.plot(x=yrs, y=v, style='c0.25c', fill=COL[m], pen='0.8p,black')
fig.plot(x=[3, 3], y=[0.65, 0.82], pen='1p,gray40,--')
fig.text(x=3.1, y=0.81, text='3 years', font='10p,6,gray30', justify='LT')
fig.legend if False else None
for i, m in enumerate(kge):
    fig.plot(x=[8.6], y=[0.69 - i * 0.015], style='c0.25c', fill=COL[m], pen='0.8p,black')
    fig.text(x=8.85, y=0.69 - i * 0.015, text=LAB[m], font='11p,5', justify='LM')
save(fig, 'SI_extension_b')

# --- SI_periodic_b: NSE change after removing the 10/15/30-day observation errors (seed 1) ---
nom = {'Brahmaputra': {'GRACEC': 0.724, 'NGGM': 0.775, 'MAGIC': 0.779}, 'Danube': {'GRACEC': 0.298, 'NGGM': 0.761, 'MAGIC': 0.808}}
r34 = {'Brahmaputra': {'GRACEC': 0.720, 'NGGM': 0.789, 'MAGIC': 0.794}, 'Danube': {'GRACEC': 0.565, 'NGGM': 0.832, 'MAGIC': 0.842}}
fig = pygmt.Figure(); cfg()
fig.basemap(region=[0.3, 7.7, -0.03, 0.30], projection='X12c/7c',
            frame=['WSne+t(b) NSE change after removing the 10/15/30-day errors', 'xf1', 'ya0.05f0.01+l@~D@~NSE'])
fig.plot(x=[0.3, 7.7], y=[0, 0], pen='0.8p,black')
x = 0.7
for b in ('Brahmaputra', 'Danube'):
    for m in M:
        d = round(r34[b][m] - nom[b][m], 3)
        fig.plot(data=[[x - 0.3, 0, x + 0.3, d]], style='r+s', fill=COL[m], pen='0.6p,black')
        fig.text(x=x, y=d, text='%+.3f' % d, font='9p,5', justify='CB' if d >= 0 else 'CT', offset='0/0.1c' if d >= 0 else '0/-0.1c')
        x += 1
    fig.text(x=x - 2, y=-0.03, text=b, font='12p,5', justify='CT', offset='0/-0.2c', no_clip=True)
    x += 0.6
for i, m in enumerate(M):
    fig.plot(data=[[0.6, 0.27 - i * 0.025, 0.85, 0.28 - i * 0.025]], style='r+s', fill=COL[m], pen='0.6p,black')
    fig.text(x=0.95, y=0.275 - i * 0.025, text=LAB[m], font='10p,5', justify='LM')
save(fig, 'SI_periodic_b')

# --- SI_seeds_b: NSE of the three seeds, 1e18 / 1 deg ---
t1 = {r['basin']: r for r in csv.DictReader(open((FP.RESULTS + '/24_tables/table1A.csv')))
      if r['metric'] == 'NSE'}
fig = pygmt.Figure(); cfg()
fig.basemap(region=[0.3, 7.7, 0.2, 0.85], projection='X12c/7c',
            frame=['WSne+t(b) Seed spread of NSE (1e18, 1@.)', 'xf1', 'ya0.1f0.05+lNSE'])
x = 0.7
for b in ('Brahmaputra', 'Danube'):
    for m in M:
        v = [float(s) for s in t1[b][m + '_seeds'].split()]
        fig.plot(x=[x, x], y=[min(v), max(v)], pen='5p,%s@40' % COL[m])
        for k, s in enumerate(v):
            fig.plot(x=[x + (k - 1) * 0.12], y=[s], style='d0.22c', fill=COL[m], pen='0.6p,black')
        fig.plot(x=[x - 0.25, x + 0.25], y=[np.mean(v)] * 2, pen='1.5p,black')
        x += 1
    fig.text(x=x - 2, y=0.2, text=b, font='12p,5', justify='CT', offset='0/-0.2c', no_clip=True)
    x += 0.6
fig.text(x=7.6, y=0.24, text='diamonds: seeds 1-3; bar: mean', font='9p,6,gray30', justify='RB')
save(fig, 'SI_seeds_b')

# --- SI_mismatch_shares: variance shares of open loop - truth (basin mean, 5-day, 2003-09..2006-12) ---
rows = [r for r in csv.DictReader(open((FP.RESULTS + '/10_mismatch/decomposition.csv')))
        if r['pair'] == 'OL-truth' and r['level'] == 'basin_mean' and '5-day' in r['period']]
comp = [('share_trend', 'Trend', 'gray30'), ('share_annual', 'Annual', 'orange'), ('share_semiannual', 'Semi-annual', 'skyblue'),
        ('share_resid', 'Residual', 'gray80')]
fig = pygmt.Figure(); cfg()
fig.basemap(region=[0, 100, 0.3, 2.7], projection='X12c/4.5c',
            frame=['WSne+tOpen loop minus truth: variance shares (basin mean, 5-day)', 'xa20f10+lShare of variance [%]', 'yf1'])
for i, r in enumerate(rows):
    y = 2 - i
    left = 0.0
    for k, lab, col in comp:
        w = 100 * float(r[k])
        fig.plot(data=[[left, y - 0.3, left + w, y + 0.3]], style='r+s', fill=col, pen='0.6p,black')
        if w > 6:
            fig.text(x=left + w / 2, y=y, text='%.0f' % w, font='10p,5')
        left += w
    fig.text(x=0, y=y, text=r['basin'], font='11p,5', justify='RM', offset='-0.2c/0', no_clip=True)
fig.shift_origin(yshift='-1.6c')
fig.basemap(region=[0, 12, 0, 1], projection='X12c/0.8c', frame='+n')
for i, (_, lab, col) in enumerate(comp):
    fig.plot(data=[[0.2 + i * 3, 0.3, 0.6 + i * 3, 0.7]], style='r+s', fill=col, pen='0.6p,black')
    fig.text(x=0.75 + i * 3, y=0.5, text=lab, font='11p,5', justify='LM')
save(fig, 'SI_mismatch_shares')
for r in rows:
    print(r['basin'], {k: r[k] for k in ('trend_mm_yr', 'std_mm', 'annual_amp_mm', 'semiannual_amp_mm', 'resid_rms_mm',
                                          'share_trend', 'share_annual', 'share_semiannual', 'share_resid')})
print('ok')
