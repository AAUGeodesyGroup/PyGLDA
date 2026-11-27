import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402
import numpy as np, h5py, pygmt
from pathlib import Path
_plot = pygmt.Figure.plot
def _plot_compat(self, *a, **k):
    for key in ('x', 'y'):
        if key in k and np.ndim(k[key]) == 0:
            k[key] = [k[key]]
    return _plot(self, *a, **k)
pygmt.Figure.plot = _plot_compat
CONDS = []
OUT = (FP.OUT + '/Brahma_cov.pdf')
def fig_Bramputra_covariance():
    import pygmt
    import geopandas as gpd
    import pandas as pd

    fig = pygmt.Figure()

    pygmt.config(FONT_TITLE="32p,5", MAP_TITLE_OFFSET="-0.2p", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5',
                 FONT_LABEL='15p,5', MAP_TICK_LENGTH='10p')

    region = [87, 98, 21, 32]
    # region = 'g'
    pj = 'Q12c'

    "-----------------------------------------------------------"
    res = 1
    fig.basemap(region=region, projection=pj,
                frame=['WSne+t1-degree', 'xa2f1', 'ya2f1'])

    # fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    fig.coast(rivers='r/1p,cornflowerblue', shorelines="1/0.2p",
              resolution='f', region=region, projection=pj, water="skyblue")

    gdf = gpd.read_file(
        (FP.ESA_SING + '/shapefiles/Brahmaputra/Grid_%s/Brahmaputra_%s.shp') % (res, res))

    fig.plot(data=gdf.boundary, pen="2p,lightred", projection=pj)
    fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="2p,lightblack", projection=pj)

    gdf = gpd.read_file(
        (FP.ESA_SING + '/shapefiles/Brahmaputra/Grid_%s/Brahmaputra_%s.shp') % (res, res))
    for i in range(gdf.shape[0]):
        nn = gdf.ID[i]
        xy = gdf[gdf.ID == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='15p,5')

    "-----------------------------------------------------------"
    fig.shift_origin(xshift='14c')
    res = 2
    fig.basemap(region=region, projection=pj,
                frame=['WSne+t2-degree', 'xa2f1', 'yf1'])

    # fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    fig.coast(rivers='r/1p,cornflowerblue', shorelines="1/0.2p",
              resolution='f', region=region, projection=pj, water="skyblue")

    gdf = gpd.read_file(
        (FP.ESA_SING + '/shapefiles/Brahmaputra/Grid_%s/Brahmaputra_%s.shp') % (res, res))

    fig.plot(data=gdf.boundary, pen="2p,lightred", projection=pj)
    fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="2p,lightblack", projection=pj)

    gdf = gpd.read_file(
        (FP.ESA_SING + '/shapefiles/Brahmaputra/Grid_%s/Brahmaputra_%s.shp') % (res, res))
    for i in range(gdf.shape[0]):
        nn = gdf.ID[i]
        xy = gdf[gdf.ID == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='15p,5')

    "-----------------------------------------------------------"
    fig.shift_origin(xshift='14c')
    fig.basemap(region=region, projection=pj,
                frame=['WSne+tbasin-wise', 'xa2f1', 'yf1'])
    # grid = pygmt.datasets.load_earth_relief("01m", registration="gridline",region=region)
    # fig.grdimage(grid=grid, shading=True, cmap='geo')

    # fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    fig.coast(rivers='r/1p,cornflowerblue', shorelines="1/0.2p",
              resolution='f', region=region, projection=pj, water="skyblue")

    gdf = gpd.read_file((FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    fig.plot(data=gdf.boundary, pen="2p,lightred", projection=pj)
    fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="2p,black", projection=pj)

    gdf = gpd.read_file((FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))
    for i in range(gdf.shape[0]):
        nn = gdf.ID[i]
        xy = gdf[gdf.ID == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='15p,5')

    "-----------------------------------------------------------"
    fig.shift_origin(xshift='-28c', yshift='-14.5c')
    pygmt.makecpt(cmap='polar', series=[-1, 1], background='o')
    pj = 'X12c/-12c'
    mission_list = ['GRACE-C-like', 'NGGM', 'MAGIC']
    grid_list = ['Grid_1', 'Grid_2', 'subbasins_3']
    label1 = ['GRACE-C', 'NGGM', 'MAGIC']

    lim = [50, 15, 3]
    folder = (FP.ESA_SING + '/Brahmaputra/gravity_signal_cov/')

    filter = '1e+17'
    pygmt.config(FONT_TITLE="32p,5", MAP_TITLE_OFFSET="-0.2p", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='20p,5',
                 FONT_LABEL='20p,5', MAP_TICK_LENGTH='10p')
    for yy, mission in enumerate(mission_list):
        file = Path(folder) / ('%s_%s_Brahmaputra_v5.hdf5' % (mission, filter))
        h5 = h5py.File(name=file, mode='r')
        for zz, grid in enumerate(grid_list):
            cov = h5[grid][mission]['VCM_alpha_%s' % filter][:]
            cond = np.linalg.cond(cov)
            print(cond)
            corr = correlation_from_covariance(cov)

            b_lim = lim[zz]
            region = [0, b_lim, 0, b_lim]

            x, y = np.meshgrid(np.arange(1, b_lim + 1), np.arange(1, b_lim + 1))
            grid = pygmt.xyz2grd(
                x=x.flatten(), y=y.flatten(), z=corr.flatten(), spacing=(1.0, 1.0), region=region
            )

            region = [0.5, b_lim + 0.5, 0.5, b_lim + 0.5]
            CONDS.append((mission, grid_list[zz], cond))
            fig.basemap(region=region, projection=pj,
                        frame=['WSne', 'xaf1g1+lDA unit index', 'yaf1g1+lDA unit index'])
            print(yy)
            if zz == 0:
                fig.basemap(region=region, projection=pj,
                            frame=['WSne', 'xa10f1g1+lDA unit index', 'ya10f1g1+lDA unit index'])

            fig.grdimage(
                grid=grid,
                projection=pj,
                region=region,
                # interpolation='n'
            )

            m_ = '%.1e' % cond
            fig.text(x=region[0] + 0.03 * b_lim, y=region[2] + 0.06 * b_lim, text='cond = %se%d' % (m_.split('e')[0], int(m_.split('e')[1])),
                     font='22p,5,black', justify='LB', fill='white@15', clearance='0.1c/0.1c')
            if zz == 0:
                fig.text(x=region[0] + 0.03 * b_lim, y=region[3] - 0.04 * b_lim, text='DA uses diagonal only',
                         font='22p,5,black', justify='LT', fill='white@15', clearance='0.1c/0.1c')
            if zz == 0:
                with pygmt.config(FONT_ANNOT_PRIMARY='35p,5'):
                    fig.text(x=-17, y=25, text=label1[yy], angle=90, justify="TC", no_clip=True)

            if yy == 2 and zz == 1:
                with pygmt.config(FONT_ANNOT_PRIMARY='30p,5'):
                    fig.colorbar(frame=['xa0.5f0.25+lCorrelation coefficient'], position="JBC+w18c/0.65c+h+o0c/2.6c")

            fig.shift_origin(xshift='14c')

            if zz == 2:
                fig.shift_origin(xshift='-42c', yshift='-14.5c')

    fig.savefig(OUT)

    pass


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation
if __name__ == '__main__':
    fig_Bramputra_covariance()
    for c in CONDS: print('COND', *c)
