import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402
import csv, sys
import numpy as np
import pygmt
R = (FP.RESULTS + '/')
cfg = list(csv.DictReader(open(R + '32_rerun_vs_published/configs.csv')))
t1 = list(csv.DictReader(open(R + '24_tables/table1A.csv')))
B = ['Brahmaputra', 'Danube']; M = ['GRACEC', 'NGGM', 'MAGIC']; ML = {'GRACEC': 'GRACE-C', 'NGGM': 'NGGM', 'MAGIC': 'MAGIC'}
A = ['1e14', '1e15', '1e16', '1e17', '1e18', '1e19']; G = ['1degree', '2degree', '3degree', '4degree', '3subbasins']
GL = ['1@.', '2@.', '3@.', '4@.', 'Bw']
def table(metric):
    v = {}
    for r in cfg:
        v[(r['basin'], r['mission'], r['alpha'], r['grid'])] = float(r['%s_rerun' % metric])
    model = {}
    for r in t1:
        if r['metric'].lower() == metric:
            model[r['basin']] = float(r['model_mean'])
            for m in M:
                v[(r['basin'], m, '1e18', '1degree')] = float(r['%s_mean' % m])     # 1e18 / 1 deg = 3-seed mean
    return v, model
def draw(fig, metrics, out):
    pygmt.config(FONT_TITLE="13p,5", MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='10p,5',
                 FONT_LABEL='11p,5', MAP_TICK_LENGTH='0p')
    letters = iter('abcdefghijklmnop')
    first = True
    for metric in metrics:
        v, model = table(metric)
        for basin in B:
            vals = np.array([[v[(basin, m, a, g)] for g in G for a in A] for m in M])
            d = np.nanmax(np.abs(vals - model[basin])); d = float(np.ceil(d * (100 if metric != 'rmse' else 1)) / (100 if metric != 'rmse' else 1))
            cpt = 'polar+h0' if metric != 'rmse' else 'polar+h0'
            pygmt.makecpt(cmap='polar', series=[model[basin] - d, model[basin] + d], reverse=(metric == 'rmse'), background='o')
            if not first:
                fig.shift_origin(xshift='-9.4c', yshift='-6.9c')
            first = False
            for j, m in enumerate(M):
                fig.basemap(region=[-0.5, 4.5, -0.5, 5.5], projection='X4.2c/5.04c',
                            frame=['%s%s+t(%s) %s %s' % ('W' if j == 0 else 'w', 'S', next(letters), basin, ML[m]),
                                   'xa1', 'ya1'] if False else
                            ['%sSne+t(%s) %s' % ('W' if j == 0 else 'w', next(letters), ML[m]), 'xf1', 'yf1'])
                xs, ys, zs = [], [], []
                for gi, g in enumerate(G):
                    for ai, a in enumerate(A):
                        xs.append(gi); ys.append(ai); zs.append(v[(basin, m, a, g)])
                fig.plot(x=xs, y=ys, fill=zs, cmap=True, style='s0.84c', pen='0.3p,gray40')
                for x, y, z in zip(xs, ys, zs):
                    fig.text(x=x, y=y, text=('%.2f' % z) if metric != 'rmse' else ('%.0f' % z), font='8p,5')
                k = int(np.argmin(zs) if metric == 'rmse' else np.argmax(zs))
                fig.plot(x=[xs[k]], y=[ys[k]], style='s0.84c', pen='2p,black')
                fig.text(x=list(range(5)), y=[-0.5] * 5, text=GL, font='10p,5', justify='CT', offset='0/-0.15c', no_clip=True)
                if j == 0:
                    fig.text(x=[-0.5] * 6, y=list(range(6)), text=['1E%s' % a[2:] for a in A], font='10p,5', justify='RM',
                             offset='-0.15c/0', no_clip=True)
                    fig.text(x=-2.1, y=2.5, text=basin, angle=90, font='14p,5', justify='CB', no_clip=True)
                if j < 2:
                    fig.shift_origin(xshift='4.7c')
            lab = {'kge': 'KGE', 'nse': 'NSE', 'rmse': 'RMSE [mm]'}[metric]
            fig.colorbar(position='JMR+o0.6c/0c+w5c/0.3c', frame=['xa%sf%s+l%s' % (((0.1, 0.05) if metric != 'rmse' else (10, 5)) + (lab,))])
    fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=150)
O = (FP.OUT + '/')
draw(pygmt.Figure(), ['kge'], O + 'response_surface.pdf')
draw(pygmt.Figure(), ['nse', 'rmse'], O + 'response_surface_nse_rmse.pdf')
print('ok')
