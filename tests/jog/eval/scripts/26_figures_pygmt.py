"""Step 26 - Figures 7-11 and D1 with the standard metrics, in the style of the published pygmt figures
(ESAsing/paper_RSE/plot_fig_2.py: metric_spatial_map, metric_spatial_map_HPF, contribution_fig, scatter_plot_*).

Input: a metrics file with the layout of the published metrics.h5, by default the standard-definition file of step 07
(results/07_recompute_published_outputs/metrics_paper.h5: published OL/DA products, DA = 2-member means), groups
<basin>_<MISSION>_F<alpha>_<G>_<hpf> with nse/kge/rmse _model/_DA maps on the 0.5 deg basin box.
  Fig 7  : NSE, KGE maps of model, DA (MAGIC 1e17 / 1 deg) and DA - model, both basins (published: MAGIC F1E17 G1)
  Fig 8  : Brahmaputra spatial means over the 30 scenarios (6 alpha x 5 grids) for the 3 missions, model dashed
  Fig 9  : NGGM - GRACE-C and MAGIC - GRACE-C maps of NSE, KGE, RMSE at 1e18 / 1 deg, both basins
  Fig 10 : as Fig 7 for the 60-day high-pass (hpf 1)
  Fig 11 : as Fig 8 for hpf 1
  Fig D1 : as Fig 8 for the Danube (2-degree points not connected, as published)
Spatial means: NSE/KGE clipped at -1 (as published). Needs pygmt + geopandas (desktop env pyglda_vis) and
JOG_ESM2_ROOT pointing to a folder with globe/global_mask/<basin>/0.5/mask.h5 (basin box coordinates).
Usage: python 26_figures_pygmt.py --metrics-h5 <metrics_paper.h5> --shp-dir <dir with <basin>3subbasins/> --out <dir>
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jog_eval import config as C, metrics as M, sources as S  # noqa: E402

GK = {'1degree': 'G1', '2degree': 'G2', '3degree': 'G3', '4degree': 'G4', '3subbasins': 'Gb'}
GL = {'1degree': '1@.', '2degree': '2@.', '3degree': '3@.', '4degree': '4@.', '3subbasins': 'Bw'}
STY = {'MAGIC': ('gold', 't0.4c', 'gold'), 'NGGM': ('dodgerblue@20', 'c0.25c', 'dodgerblue@30'),
       'GRACEC': ('seagreen', 's0.3c', 'seagreen@30')}


def key(basin, m, alpha, grid, hpf):
    return '%s_%s_F%s_%s_%d' % (basin, m, alpha.upper(), GK[grid], hpf)


def outline(shp_dir, basin):
    import geopandas as gpd
    f = sorted((Path(shp_dir) / ('%s3subbasins' % basin)).glob('*.shp'))[0]
    return gpd.GeoSeries(gpd.read_file(f).unary_union.boundary)


def put_map(fig, value, lat, lon, title, shp, proj='Q6c'):
    import pygmt
    region = [lon.min(), lon.max(), lat.min(), lat.max()]
    region2 = [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]
    LO, LA = np.meshgrid(lon, lat)
    g = pygmt.xyz2grd(x=LO.ravel(), y=LA.ravel(), z=value.ravel(), spacing=0.5, region=region)
    fig.grdimage(grid=g, cmap=True, frame=['+t%s' % title], projection=proj, region=region2)
    fig.plot(data=shp, pen='1p,black', region=region2, projection=proj)


def maps_fig(h, shp_dir, hpf, out, name):
    """Fig 7 / 10: rows NSE, KGE; columns model, DA (MAGIC 1e17 1 deg), DA - model; Brahmaputra then Danube"""
    import pygmt
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='12p,5', MAP_TITLE_OFFSET='4p', MAP_FRAME_TYPE='plain', FONT_ANNOT_PRIMARY='9p,5',
                 FONT_LABEL='10p,5', COLOR_NAN='white')
    lab = iter('abcdefghijkl')
    stats = {}
    for ib, basin in enumerate(C.BASINS):
        box, lat, lon, _ = S.region_mask(basin)
        g = h[key(basin, 'MAGIC', '1e17', '1degree', hpf)]
        shp = outline(shp_dir, basin)
        width = 6.0 if basin == 'Brahmaputra' else 8.0
        hgt = width * (np.ptp(lat) + 1.0) / (np.ptp(lon) + 1.0) + 1.4     # map height + title (cm)
        for ic, what in enumerate(('Model', 'DA', 'DA - Model')):
            for ir, n in enumerate(('nse', 'kge')):
                mo, da = g[n + '_model'][:], g[n + '_DA'][:]
                v = {'Model': mo, 'DA': da, 'DA - Model': da - mo}[what]
                if what == 'DA - Model':
                    pygmt.makecpt(cmap='polar+h0', series=[-0.5, 1.5], background='o')
                else:
                    pygmt.makecpt(cmap='wysiwyg', series=[-1, 1], background='o')
                put_map(fig, v, lat, lon, '(%s) %s: %s' % (next(lab), n.upper(), what), shp, 'Q%.1fc' % width)
                ok = np.isfinite(v)
                stats['%s_%s_%s' % (basin, n, what)] = {'mean_clipped': M.spatial_mean(v, n) if what != 'DA - Model' else float(np.nanmean(v)),
                                                        'share_gt_0.5': float(np.mean(v[ok] > 0.5)),
                                                        'share_gt_0': float(np.mean(v[ok] > 0))}
                fig.shift_origin(yshift='%.1fc' % (-hgt if ir == 0 else hgt))
            if ic in (1, 2):
                fig.colorbar(frame='xa0.5f0.1', position='JBC+w4c/0.3c+h+o0c/%.1fc' % (hgt + 0.8))
            fig.shift_origin(xshift='%.1fc' % (width + 1.0))
        fig.shift_origin(xshift='%.1fc' % (-3 * (width + 1.0)), yshift='%.1fc' % (-2 * hgt - 1.5))
    fig.savefig(str(Path(out) / ('%s.png' % name)), dpi=200)
    return stats


def fig9(h, shp_dir, out):
    """rows NSE, KGE, RMSE; columns Brahmaputra NGGM, MAGIC (left) | Danube NGGM, MAGIC (right), minus GRACE-C, 1e18/1deg.
    NSE/KGE: red = mission better than GRACE-C; RMSE: blue = better (as published)"""
    import pygmt
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='10p,5', MAP_TITLE_OFFSET='4p', MAP_FRAME_TYPE='plain', FONT_ANNOT_PRIMARY='9p,5',
                 COLOR_NAN='white')
    lab = iter('abcdefghijkl')
    stats = {}
    W = {'Brahmaputra': 5.5, 'Danube': 7.0}
    geo = {b: S.region_mask(b) for b in C.BASINS}
    H = max(W[b] * (np.ptp(geo[b][1]) + 1) / (np.ptp(geo[b][2]) + 1) for b in C.BASINS) + 1.6
    total = sum(2 * (W[b] + 0.8) for b in C.BASINS)
    for ir, n in enumerate(('nse', 'kge', 'rmse')):
        if n == 'rmse':
            pygmt.makecpt(cmap='polar', series=[-40, 40], background='o')
        else:
            pygmt.makecpt(cmap='polar', series=[-0.5, 0.5], background='o')
        for basin in C.BASINS:
            box, lat, lon, _ = geo[basin]
            shp = outline(shp_dir, basin)
            ref = h[key(basin, 'GRACEC', '1e18', '1degree', 0)][n + '_DA'][:]
            for m in ('NGGM', 'MAGIC'):
                v = h[key(basin, m, '1e18', '1degree', 0)][n + '_DA'][:] - ref
                put_map(fig, v, lat, lon, '(%s) %s: %s - GRACE-C' % (next(lab), n.upper(), m), shp,
                        'Q%.1fc' % W[basin])
                ok = np.isfinite(v)
                better = v[ok] < 0 if n == 'rmse' else v[ok] > 0
                stats['%s_%s_%s' % (basin, m, n)] = {'share_better': float(better.mean()), 'mean_diff': float(v[ok].mean())}
                if basin == C.BASINS[-1] and m == 'MAGIC':
                    fig.colorbar(frame='xa%s+l%s' % ('20' if n == 'rmse' else '0.25',
                                                     n.upper() + (' [mm]' if n == 'rmse' else '')),
                                 position='JMR+w3.5c/0.3c+o0.5c/0c')
                fig.shift_origin(xshift='%.1fc' % (W[basin] + 0.8))
        fig.shift_origin(xshift='%.1fc' % -total, yshift='%.1fc' % -H)
    fig.savefig(str(Path(out) / 'fig09.png'), dpi=200)
    return stats


def series(h, basin, hpf):
    out, model = {}, None
    for m in C.MISSIONS:
        out[m] = {n: [] for n in ('nse', 'kge', 'rmse')}
        for a in C.ALPHAS:
            for gr in C.GRIDS:
                g = h[key(basin, m, a, gr, hpf)]
                for n in ('nse', 'kge', 'rmse'):
                    out[m][n].append(M.spatial_mean(g[n + '_DA'][:], n))
                if model is None:
                    model = {n: M.spatial_mean(g[n + '_model'][:], n) for n in ('nse', 'kge', 'rmse')}
    return out, model


def series_fig(h, basin, hpf, out, name, skip_2deg=False):
    import pygmt
    s, model = series(h, basin, hpf)
    x = np.arange(1, 31)
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='12p,5', MAP_FRAME_TYPE='plain', FONT_ANNOT_PRIMARY='10p,5', FONT_LABEL='11p,5',
                 COLOR_NAN='white')
    best = {}
    for ip, n in enumerate(('rmse', 'kge', 'nse')):
        allv = np.concatenate([s[m][n] for m in C.MISSIONS] + [[model[n]]])
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        pad = 0.08 * (hi - lo)
        lo, hi = (max(lo - pad, -1.0), min(hi + pad, 1.0)) if n != 'rmse' else (lo - pad, hi + pad)
        fig.basemap(frame=['WSne' if ip == 0 else 'Wsne', 'xa1f1g1' if ip == 0 else 'xf1g1',
                           'ya%sg%s+l%s' % ((('20', '20') if n == 'rmse' else ('0.2', '0.1')) + (n.upper() + (' [mm]' if n == 'rmse' else ''),))],
                    region=[0, 31, lo, hi], projection='X17c/5.5c')
        for k in range(6):
            if k % 2 == 0:
                fig.plot(x=[0.5 + 5 * k, 5.5 + 5 * k, 5.5 + 5 * k, 0.5 + 5 * k], y=[lo, lo, hi, hi], fill='lightgray@40', close=True)
            fig.text(x=3 + 5 * k, y=(lo + 0.07 * (hi - lo)) if n == 'rmse' else (hi - 0.07 * (hi - lo)),
                     text='%s' % C.ALPHAS[k].upper(), font='9p,7')
        for m in ('MAGIC', 'NGGM', 'GRACEC'):
            pen, sty, fill = STY[m]
            y = np.array(s[m][n])
            if skip_2deg:
                for k in range(6):
                    sel = [5 * k, 5 * k + 2, 5 * k + 3, 5 * k + 4]
                    fig.plot(x=x[sel], y=y[sel], pen='2p,%s' % pen)
            else:
                fig.plot(x=x, y=y, pen='2p,%s' % pen)
            fig.plot(x=x, y=y, style=sty, pen='1p', fill=fill)
            j = int(np.nanargmin(y) if n == 'rmse' else np.nanargmax(y))
            fig.plot(x=x[j], y=y[j], style='s0.55c', pen='1.5p,purple')
            best['%s_%s' % (m, n)] = {'alpha': C.ALPHAS[j // 5], 'grid': C.GRIDS[j % 5], 'value': float(y[j])}
        fig.plot(x=[0, 31], y=[model[n], model[n]], pen='2p,red3,--')
        fig.shift_origin(yshift='6c')
    fig.basemap(frame=['WSne+glavender'], region=[0, 31, 0, 1], projection='X17c/0.7c')
    for i, m in enumerate(('MAGIC', 'NGGM', 'GRACEC')):
        pen, sty, fill = STY[m]
        fig.plot(x=[1 + 8 * i, 4 + 8 * i], y=[0.5, 0.5], pen='2p,%s' % pen, no_clip=True)
        fig.plot(x=2.5 + 8 * i, y=0.5, style=sty, pen='1p', fill=fill)
        fig.text(x=6.2 + 8 * i, y=0.5, text=m.replace('GRACEC', 'GRACE-C'))
    fig.plot(x=[25, 27], y=[0.5, 0.5], pen='2p,red3,--')
    fig.text(x=29, y=0.5, text='Model')
    fig.savefig(str(Path(out) / ('%s.png' % name)), dpi=200)
    gl = ' '.join(GL[g] for g in C.GRIDS)
    return {'best': best, 'model': model, 'scenario_order': 'alpha-major (%s), grids %s' % (','.join(C.ALPHAS), gl),
            'series': s}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics-h5', required=True)
    p.add_argument('--shp-dir', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    import json
    res = {'metrics_h5': a.metrics_h5}
    with h5py.File(a.metrics_h5, 'r') as h:
        res['fig07'] = maps_fig(h, a.shp_dir, 0, a.out, 'fig07')
        res['fig10'] = maps_fig(h, a.shp_dir, 1, a.out, 'fig10')
        res['fig09'] = fig9(h, a.shp_dir, a.out)
        res['fig08'] = series_fig(h, 'Brahmaputra', 0, a.out, 'fig08')
        res['fig11'] = series_fig(h, 'Brahmaputra', 1, a.out, 'fig11')
        res['figD1'] = series_fig(h, 'Danube', 0, a.out, 'figD1', skip_2deg=True)
    Path(a.out, 'figures_stats.json').write_text(json.dumps(res, indent=1, default=float))
    print('written', a.out)


if __name__ == '__main__':
    main()
