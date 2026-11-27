"""EOF 1-3 of truth, model and DA (nominal 1e18 / 1 deg, seed 1): maps per product + PCs, one PDF per basin."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402

import sys
from datetime import date

import geopandas as gpd
import numpy as np
import pygmt

SHP = {'Brahmaputra': (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'),
       'Danube': (FP.SHP + '/DRB_3_shapefiles')}
IN = (FP.INPUTS + '/eof/eof_%s.npz')
OUT = (FP.OUT + '/eof3_%s')
PRODS = [('truth', 'Truth', 'black'), ('model', 'Model', 'gray50'), ('GRACEC', 'GRACE-C', 'seagreen'),
         ('NGGM', 'NGGM', 'dodgerblue'), ('MAGIC', 'MAGIC', 'gold')]
GAP = 1.0
LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def decyear(s):
    y, m, d = (int(x) for x in s.split('-'))
    return y + (date(y, m, d) - date(y, 1, 1)).days / 365.25


def main(basin):
    z = np.load(IN % basin)
    lat, lon = z['lat'], z['lon']
    t = np.array([decyear(s) for s in z['dates']])
    LON, LAT = np.meshgrid(lon, lat)
    region = [lon.min(), lon.max(), lat.min(), lat.max()]
    region2 = [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5]
    mw = 5.0 if basin == 'Brahmaputra' else 7.0
    ps = 'Q%.1fc' % mw
    rh = mw * 0.75 if basin == 'Brahmaputra' else 3.6
    cpos, cjust, coff = ('BR', 'BR', '-0.1c/0.1c') if basin == 'Brahmaputra' else ('TL', 'TL', '0.1c/-0.1c')
    outline = gpd.GeoSeries(gpd.read_file(SHP[basin]).unary_union.boundary)
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE="16p,5", MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5',
                 FONT_LABEL='14p,5', MAP_TICK_LENGTH='5p', COLOR_NAN='white')
    k = 0
    for i in range(3):
        lim = float(np.nanmax(np.abs(z['map_truth'][i])))
        lim = float(np.ceil(lim / 5.0) * 5.0)
        pygmt.makecpt(cmap='polar', series=[-lim, lim], background='o')
        for j, (key, name, _) in enumerate(PRODS):
            v = z['map_%s' % key][i]
            g = pygmt.xyz2grd(x=LON.flatten(), y=LAT.flatten(), z=v.flatten(), spacing=0.5, region=region)
            title = '(%s) %s, %.1f %%' % (LETTERS[k], name, z['ev_%s' % key][i])
            fig.grdimage(grid=g, cmap=True, projection=ps, region=region2, frame=['+t%s' % title])
            fig.plot(data=outline, pen='0.8p,black', region=region2, projection=ps)
            if key != 'truth':
                if basin == 'Brahmaputra':
                    fig.text(position=cpos, offset=coff, text='c = %.2f' % z['cong_%s' % key][i], font='14p,5,black',
                             justify=cjust, region=region2, projection=ps)
                else:                                                     # below the map (the outline fills the corners)
                    fig.text(x=0.5 * (region2[0] + region2[1]), y=region2[2], text='c = %.2f' % z['cong_%s' % key][i],
                             font='14p,5,black', justify='TC', offset='0c/-0.15c', no_clip=True, region=region2, projection=ps)
            else:
                fig.text(x=region2[0], y=0.5 * (region2[2] + region2[3]), text='Mode %d' % (i + 1), font='18p,5,black',
                         justify='RM', offset='-0.5c/0c', angle=90, no_clip=True, region=region2, projection=ps)
            k += 1
            if j == len(PRODS) - 1:
                step = 10 ** np.floor(np.log10(lim))
                step = step * (5 if lim / step >= 5 else (2 if lim / step >= 2 else 1)) / (2 if lim / step >= 5 else 1)
                with pygmt.config(FONT_ANNOT_PRIMARY='12p,5', FONT_LABEL='12p,5'):
                    fig.colorbar(position='JMR+o0.3c/0c+w%.1fc/0.3c' % (mw * 0.7), frame=['xa%gf%g+lEWH [mm]' % (step, step / 2)])
            fig.shift_origin(xshift='%.2fc' % (mw + GAP))
        # PC panel
        fig.shift_origin(xshift='1.6c')
        fig.basemap(region=[2003.6, 2007.05, -3.6, 3.6], projection='X9c/%.1fc' % rh,
                    frame=['WSne+t(%s) PC%d, standardized' % (LETTERS[k], i + 1), 'xa1f0.5', 'ya2f1'])
        for key, name, col in PRODS:
            pen = ('1.4p,%s' if key == 'truth' else '0.9p,%s') % col
            fig.plot(x=t, y=z['pc_%s' % key][:, i], pen=pen)
        k += 1
        fig.shift_origin(xshift='-%.2fc' % (5 * (mw + GAP) + 1.6), yshift='-%.1fc' % (rh + 2.0))
    fig.shift_origin(yshift='%.1fc' % (rh + 0.2))
    W = 5 * (mw + GAP) + 1.6 + 9
    fig.basemap(region=[0, W, 0, 1], projection='X%.1fc/1c' % W, frame='+n')
    x0 = W / 2 - 5 * 4.4 / 2
    for n, (key, name, col) in enumerate(PRODS):
        xx = x0 + n * 4.4
        fig.plot(x=[xx, xx + 0.9], y=[0.5, 0.5], pen=('2.5p,%s' % col))
        fig.text(x=xx + 1.05, y=0.5, text=name + ('' if key in ('truth', 'model') else ' DA'), font='14p,5,black', justify='LM')
    fig.savefig(OUT % basin + '.pdf')
    fig.savefig(OUT % basin + '.png', dpi=110)
    print('written', OUT % basin)


if __name__ == '__main__':
    for b in (sys.argv[1:] or ['Brahmaputra', 'Danube']):
        main(b)
