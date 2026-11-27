import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import fig_paths as FP  # noqa: E402
import h5py
import numpy as np
import scipy
import numpy as np
import sys

sys.path.append('../')

import numpy as np
from pathlib import Path
import pygmt
import h5py

import Analysis2
from get_h5_structure import get_hdf5_structure
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta


_H5_PATH = (FP.INPUTS + '/metrics_paper_seedmean.h5')
_H5 = h5py.File(_H5_PATH, 'r')
_OUT = [None]
_H5_COV_PATH = None                     # realistic-VCM metrics (set by the driver)


def metric_spatial_map():
    from paper_RSE.PCA import PCA_EOF
    from spatial_analysis import truth, obs, OL_DA
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data

    i = -1
    fig_label = ['(a)', '(d)', '(b)', '(e)', '(c)', '(f)', '(g)', '(j)', '(h)', '(k)', '(i)', '(l)']

    basin = 'Brahmaputra'
    # basin = 'Danube'
    mm = mission.MAGIC
    ff = filter.F1E17
    gg = grid.G1
    dd = data.OL

    tt2 = truth(region=basin)
    b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    his_mask, lat, lon = tt2.shrink_mask()
    a = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
    a[:, his_mask] = b

    tt3 = None
    b = np.zeros_like(a)

    tt4 = None
    c = np.zeros_like(a)

    '''de-mean'''
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)
    c = c - np.mean(c, axis=0)

    '''figure'''
    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="17p,5", MAP_TITLE_OFFSET=-0.5, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='27p,5',
                 FONT_LABEL='25p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')

    ps = 'Q6c'
    region = [min(lon), max(lon), min(lat), max(lat)]
    region2 = [min(lon) - 0.5, max(lon) + 0.5, min(lat) - 0.5, max(lat) + 0.5]
    lon, lat = np.meshgrid(lon, lat)

    if basin == 'Danube':
        gdf = gpd.read_file((FP.SHP + '/DRB_3_shapefiles'))
    else:
        gdf = gpd.read_file(
            (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    OB_list = [b, c, 0]

    for ob_ind, OB in enumerate(OB_list):
        if ob_ind == 2:
            pygmt.makecpt(cmap='polar+h0', series=[-0.5, 1.5], background='o')
            nse = _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['nse_DA'][:] - _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['nse_model'][:]
            kge = _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['kge_DA'][:] - _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['kge_model'][:]
        else:
            pygmt.makecpt(cmap='wysiwyg', series=[-1, 1], background='o')
            nse = _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['nse_model' if ob_ind == 0 else 'nse_DA'][:]
            kge = _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['kge_model' if ob_ind == 0 else 'kge_DA'][:]
        dataset = ['Model', 'DA', 'DA - Model']
        index = [nse, kge]
        label_index = ['NSE', 'KGE']

        for ind, value in enumerate(index):
            i += 1

            h = value[~np.isnan(value)]
            po = len(h[h > 0]) / len(h)
            # print(fig_label[i], po)
            print(fig_label[i], np.mean(h[h > 0]), np.nanmean(h[h < 0]))

            ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=value.flatten(),
                               spacing=0.5, region=region)

            fig.grdimage(
                grid=ee,
                cmap=True,
                frame=['+t%s %s: %s' % (fig_label[i], label_index[ind], dataset[ob_ind])],
                # dpi=100,
                projection=ps,
                region=region2,
                # interpolation='b',
                # nan_transparent=True
            )
            # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
            fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)

            if ind == 1:
                # fig.colorbar(frame='a0.5f0.1')
                fig.shift_origin(yshift='7c', xshift='7c')
            else:
                fig.shift_origin(yshift='-7c')

    """=============================================Danube==============================================="""
    fig.shift_origin(xshift='-21c', yshift='-11c')

    # basin = 'Brahmaputra'
    basin = 'Danube'

    tt2 = truth(region=basin)
    b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    his_mask, lat, lon = tt2.shrink_mask()
    a = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
    a[:, his_mask] = b

    tt3 = None
    b = np.zeros_like(a)

    tt4 = None
    c = np.zeros_like(a)

    '''de-mean'''
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)
    c = c - np.mean(c, axis=0)

    region = [min(lon), max(lon), min(lat), max(lat)]
    region2 = [min(lon) - 0.5, max(lon) + 0.5, min(lat) - 0.5, max(lat) + 0.5]
    lon, lat = np.meshgrid(lon, lat)

    if basin == 'Danube':
        gdf = gpd.read_file((FP.SHP + '/DRB_3_shapefiles'))
    else:
        gdf = gpd.read_file(
            (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    OB_list = [b, c, 0]

    for ob_ind, OB in enumerate(OB_list):
        if ob_ind == 2:
            pygmt.makecpt(cmap='polar+h0', series=[-0.5, 1.5], background='o')
            nse = _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['nse_DA'][:] - _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['nse_model'][:]
            kge = _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['kge_DA'][:] - _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['kge_model'][:]
        else:
            pygmt.makecpt(cmap='wysiwyg', series=[-1, 1], background='o')
            nse = _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['nse_model' if ob_ind == 0 else 'nse_DA'][:]
            kge = _H5['%s_%s_%s_%s_0' % (basin, mm.name, ff.name, gg.name)]['kge_model' if ob_ind == 0 else 'kge_DA'][:]
        dataset = ['Model', 'DA', 'DA - Model']
        index = [nse, kge]
        label_index = ['NSE', 'KGE']

        for ind, value in enumerate(index):

            i += 1
            h = value[~np.isnan(value)]
            po = len(h[h > 0]) / len(h)
            # print(fig_label[i], po)
            print(fig_label[i], np.mean(h[h > 0]), np.mean(h[h < 0]))

            ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=value.flatten(),
                               spacing=0.5, region=region)

            fig.grdimage(
                grid=ee,
                cmap=True,
                frame=['+t%s %s: %s' % (fig_label[i], label_index[ind], dataset[ob_ind])],
                # dpi=100,
                projection=ps,
                region=region2,
                # interpolation='b',
                # nan_transparent=True
            )
            # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
            fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)

            if ind == 1:
                fig.colorbar(frame='xa0.5f0.1+lDimensionless', position="JBC+w5c/0.35c+h+o0c/0.5c")
                fig.shift_origin(yshift='4c', xshift='7c')
            else:
                fig.shift_origin(yshift='-4c')

    fig.savefig(_OUT[0])

    pass


def metric_spatial_map_HPF():
    from spatial_analysis import truth, obs, OL_DA
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data

    i = -1
    fig_label = ['(a)', '(d)', '(b)', '(e)', '(c)', '(f)', '(g)', '(j)', '(h)', '(k)', '(i)', '(l)']

    basin = 'Brahmaputra'
    # basin = 'Danube'
    mm = mission.MAGIC
    ff = filter.F1E17
    gg = grid.G1
    dd = data.OL
    cf = 60

    tt2 = truth(region=basin)
    b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    his_mask, lat, lon = tt2.shrink_mask()
    a = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
    a[:, his_mask] = b

    tt3 = None
    b = np.zeros_like(a)

    # tt2 = obs(region=basin)
    # b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    # his_mask, lat, lon = tt2.shrink_mask()
    # aa = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
    # aa[:, his_mask] = b
    # b = aa

    tt4 = None
    c = np.zeros_like(a)

    '''de-mean'''
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)
    c = c - np.mean(c, axis=0)

    a = butterworth_2D(a, cutoff=cf, sr=1 / 5)
    b = butterworth_2D(b, cutoff=cf, sr=1 / 5)
    c = butterworth_2D(c, cutoff=cf, sr=1 / 5)

    '''figure'''
    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="17p,5", MAP_TITLE_OFFSET=-0.5, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5',
                 FONT_LABEL='15p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')

    ps = 'Q6c'
    region = [min(lon), max(lon), min(lat), max(lat)]
    region2 = [min(lon) - 0.5, max(lon) + 0.5, min(lat) - 0.5, max(lat) + 0.5]
    lon, lat = np.meshgrid(lon, lat)

    if basin == 'Danube':
        gdf = gpd.read_file((FP.SHP + '/DRB_3_shapefiles'))
    else:
        gdf = gpd.read_file(
            (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    OB_list = [b, c, 0]

    for ob_ind, OB in enumerate(OB_list):
        if ob_ind == 2:
            pygmt.makecpt(cmap='polar+h0', series=[-0.5, 1.5], background='o')
            nse = _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['nse_DA'][:] - _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['nse_model'][:]
            kge = _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['kge_DA'][:] - _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['kge_model'][:]
        else:
            pygmt.makecpt(cmap='wysiwyg', series=[-1, 1], background='o')
            nse = _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['nse_model' if ob_ind == 0 else 'nse_DA'][:]
            kge = _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['kge_model' if ob_ind == 0 else 'kge_DA'][:]
        dataset = ['Model', 'DA', 'DA - Model']
        index = [nse, kge]
        label_index = ['NSE', 'KGE']

        for ind, value in enumerate(index):
            i += 1

            h = value[~np.isnan(value)]
            po = len(h[h > 0]) / len(h)
            print(fig_label[i], po)
            print(fig_label[i], np.mean(h[h > 0]), np.nanmean(h[h < 0]))

            ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=value.flatten(),
                               spacing=0.5, region=region)

            fig.grdimage(
                grid=ee,
                cmap=True,
                frame=['+t%s %s: %s' % (fig_label[i], label_index[ind], dataset[ob_ind])],
                # dpi=100,
                projection=ps,
                region=region2,
                # interpolation='b',
                # nan_transparent=True
            )
            # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
            fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)

            if ind == 1:
                # fig.colorbar(frame='a0.5f0.1')
                fig.shift_origin(yshift='7c', xshift='7c')
            else:
                fig.shift_origin(yshift='-7c')

    """=============================================Danube==============================================="""
    fig.shift_origin(xshift='-21c', yshift='-11c')

    # basin = 'Brahmaputra'
    basin = 'Danube'

    tt2 = truth(region=basin)
    b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    his_mask, lat, lon = tt2.shrink_mask()
    a = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
    a[:, his_mask] = b

    tt3 = None
    b = np.zeros_like(a)

    tt4 = None
    c = np.zeros_like(a)

    '''de-mean'''
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)
    c = c - np.mean(c, axis=0)

    a = butterworth_2D(a, cutoff=cf, sr=1 / 5)
    b = butterworth_2D(b, cutoff=cf, sr=1 / 5)
    c = butterworth_2D(c, cutoff=cf, sr=1 / 5)

    region = [min(lon), max(lon), min(lat), max(lat)]
    region2 = [min(lon) - 0.5, max(lon) + 0.5, min(lat) - 0.5, max(lat) + 0.5]
    lon, lat = np.meshgrid(lon, lat)

    if basin == 'Danube':
        gdf = gpd.read_file((FP.SHP + '/DRB_3_shapefiles'))
    else:
        gdf = gpd.read_file(
            (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    OB_list = [b, c, 0]

    for ob_ind, OB in enumerate(OB_list):
        if ob_ind == 2:
            pygmt.makecpt(cmap='polar+h0', series=[-0.5, 1.5], background='o')
            nse = _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['nse_DA'][:] - _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['nse_model'][:]
            kge = _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['kge_DA'][:] - _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['kge_model'][:]
        else:
            pygmt.makecpt(cmap='wysiwyg', series=[-1, 1], background='o')
            nse = _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['nse_model' if ob_ind == 0 else 'nse_DA'][:]
            kge = _H5['%s_%s_%s_%s_1' % (basin, mm.name, ff.name, gg.name)]['kge_model' if ob_ind == 0 else 'kge_DA'][:]
        dataset = ['Model', 'DA', 'DA - Model']
        index = [nse, kge]
        label_index = ['NSE', 'KGE']

        for ind, value in enumerate(index):
            i += 1

            h = value[~np.isnan(value)]
            po = len(h[h > 0]) / len(h)
            print(fig_label[i], po)
            print(fig_label[i], np.mean(h[h > 0]), np.nanmean(h[h < 0]))

            ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=value.flatten(),
                               spacing=0.5, region=region)

            fig.grdimage(
                grid=ee,
                cmap=True,
                frame=['+t%s %s: %s' % (fig_label[i], label_index[ind], dataset[ob_ind])],
                # dpi=100,
                projection=ps,
                region=region2,
                # interpolation='b',
                # nan_transparent=True
            )
            # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
            fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)

            if ind == 1:
                fig.colorbar(frame='xa0.5f0.1+lDimensionless', position="JBC+w5c/0.35c+h+o0c/0.5c")
                fig.shift_origin(yshift='4c', xshift='7c')
            else:
                fig.shift_origin(yshift='-4c')

    fig.savefig(_OUT[0])

    pass


def contribution_computation():
    from spatial_analysis import truth, obs, OL_DA
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data

    h5 = h5py.File(name='./res/metrics.h5', mode='w')

    basin = 'Brahmaputra'
    # basin = 'Danube'
    mm = mission.MAGIC
    ff = filter.F1E17
    gg = grid.G1
    isHPF = False

    for basin in ['Brahmaputra', 'Danube']:
        for mm in mission:
            for ff in filter:
                for gg in grid:
                    for isHPF in [False, True]:
                        cf = 60
                        tt2 = truth(region=basin)
                        b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
                        his_mask, lat, lon = tt2.shrink_mask()
                        a = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
                        a[:, his_mask] = b

                        tt3 = OL_DA(dd=data.OL, mm=mm, ff=ff, gg=gg, region=basin)
                        b, OL_mask = tt3.get_upscaled_value()

                        tt4 = OL_DA(dd=data.DA, mm=mm, ff=ff, gg=gg, region=basin)
                        c, DA_mask = tt4.get_upscaled_value()

                        '''de-mean'''
                        a = a - np.mean(a, axis=0)
                        b = b - np.mean(b, axis=0)
                        c = c - np.mean(c, axis=0)

                        '''HPF'''
                        if isHPF:
                            a = butterworth_2D(a, cutoff=cf, sr=1 / 5)
                            b = butterworth_2D(b, cutoff=cf, sr=1 / 5)
                            c = butterworth_2D(c, cutoff=cf, sr=1 / 5)

                        '''metric'''
                        nse_model = NSE_2D(a, b)
                        kge_model = KGE_2D(a, b)
                        rmse_model = RMSE_2D(a, b)
                        nse_DA = NSE_2D(a, c)
                        kge_DA = KGE_2D(a, c)
                        rmse_DA = RMSE_2D(a, c)

                        name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))
                        print(name)
                        sg = h5.create_group(name=name)
                        sg.create_dataset(name='nse_model', data=nse_model)
                        sg.create_dataset(name='kge_model', data=kge_model)
                        sg.create_dataset(name='rmse_model', data=rmse_model)
                        sg.create_dataset(name='nse_DA', data=nse_DA)
                        sg.create_dataset(name='kge_DA', data=kge_DA)
                        sg.create_dataset(name='rmse_DA', data=rmse_DA)

    pass


def contribution_computation_for_cov_residual():
    from spatial_analysis import truth, obs
    from spatial_analysis import OL_DA_cov_residual as OL_DA
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data

    h5 = h5py.File(name='./res/metrics_cov_residual.h5', mode='w')

    basin = 'Brahmaputra'
    # basin = 'Danube'
    mm = mission.MAGIC
    ff = filter.F1E17
    gg = grid.G1
    isHPF = False

    for basin in ['Brahmaputra', 'Danube']:
        for mm in mission:
            for ff in filter:
                for gg in [grid.G1]:
                    for isHPF in [False]:
                        cf = 60
                        tt2 = truth(region=basin)
                        b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
                        his_mask, lat, lon = tt2.shrink_mask()
                        a = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
                        a[:, his_mask] = b

                        tt3 = OL_DA(dd=data.OL, mm=mm, ff=ff, gg=gg, region=basin)
                        b, OL_mask = tt3.get_upscaled_value()

                        tt4 = OL_DA(dd=data.DA, mm=mm, ff=ff, gg=gg, region=basin)
                        c, DA_mask = tt4.get_upscaled_value()

                        '''de-mean'''
                        a = a - np.mean(a, axis=0)
                        b = b - np.mean(b, axis=0)
                        c = c - np.mean(c, axis=0)

                        '''HPF'''
                        if isHPF:
                            a = butterworth_2D(a, cutoff=cf, sr=1 / 5)
                            b = butterworth_2D(b, cutoff=cf, sr=1 / 5)
                            c = butterworth_2D(c, cutoff=cf, sr=1 / 5)

                        '''metric'''
                        nse_model = NSE_2D(a, b)
                        kge_model = KGE_2D(a, b)
                        rmse_model = RMSE_2D(a, b)
                        nse_DA = NSE_2D(a, c)
                        kge_DA = KGE_2D(a, c)
                        rmse_DA = RMSE_2D(a, c)

                        name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))
                        print(name)
                        sg = h5.create_group(name=name)
                        sg.create_dataset(name='nse_model', data=nse_model)
                        sg.create_dataset(name='kge_model', data=kge_model)
                        sg.create_dataset(name='rmse_model', data=rmse_model)
                        sg.create_dataset(name='nse_DA', data=nse_DA)
                        sg.create_dataset(name='kge_DA', data=kge_DA)
                        sg.create_dataset(name='rmse_DA', data=rmse_DA)

    pass


def contribution_fig():
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data
    from spatial_analysis import truth

    h5 = h5py.File(name=_H5_PATH, mode='r')
    isHPF = False
    basins = ['Danube']
    # basins = ['Brahmaputra']
    grids = [grid.G1]
    ffs = [filter.F1E17]

    NSE = {}
    KGE = {}

    for basin in basins:
        for mm in mission:
            for ff in ffs:
                for gg in grids:
                    name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))
                    NSE[mm] = h5[name]['nse_DA'][:] - h5[name]['nse_model'][:]
                    KGE[mm] = h5[name]['kge_DA'][:] - h5[name]['kge_model'][:]

    '''figure'''
    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="13p,5", MAP_TITLE_OFFSET=-0.5, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5',
                 FONT_LABEL='15p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar+h0', series=[-0.3, 0.3], background='o')

    tt2 = truth(region=basin)
    b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    his_mask, lat, lon = tt2.shrink_mask()

    ps = 'Q6c'
    region = [min(lon), max(lon), min(lat), max(lat)]
    region2 = [min(lon) - 0.5, max(lon) + 0.5, min(lat) - 0.5, max(lat) + 0.5]
    lon, lat = np.meshgrid(lon, lat)

    if basin == 'Danube':
        gdf = gpd.read_file((FP.SHP + '/DRB_3_shapefiles'))
    else:
        gdf = gpd.read_file(
            (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    vv_NSE = [NSE[mission.NGGM] - NSE[mission.GRACEC], NSE[mission.MAGIC] - NSE[mission.GRACEC]]
    vv_KGE = [KGE[mission.NGGM] - KGE[mission.GRACEC], KGE[mission.MAGIC] - KGE[mission.GRACEC]]
    vv = [vv_NSE, vv_KGE]
    mission_label = ['NGGM - GRACEC', 'MAGIC - GRACEC']
    label_index = ['NSE', 'KGE']

    for ind0, sds in enumerate(vv):
        for ind, value in enumerate(sds):
            if ind0 == 1 and ind == 0:
                fig.shift_origin(xshift='-14c', yshift='-7c')
            ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=value.flatten(),
                               spacing=0.5, region=region)

            fig.grdimage(
                grid=ee,
                cmap=True,
                frame=['+t%s: %s' % (label_index[ind0], mission_label[ind])],
                # dpi=100,
                projection=ps,
                region=region2,
                # interpolation='b',
                # nan_transparent=True
            )
            # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
            fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)
            fig.shift_origin(xshift='7c')

    fig.colorbar(frame='xa0.1f0.1+lDimensionless', position="JBC+w8c/0.35c+h+o-10c/0.5c")
    fig.savefig(_OUT[0])
    pass


def contribution_fig_combination():
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data
    from spatial_analysis import truth

    h5 = h5py.File(name=_H5_PATH, mode='r')
    # h5 = h5py.File(name=_H5_COV_PATH, mode='r')
    isHPF = False
    # basins = ['Danube']
    basins = ['Brahmaputra']
    grids = [grid.G1]
    ffs = [filter.F1E18]

    NSE = {}
    KGE = {}
    RMSE ={}

    for basin in basins:
        for mm in mission:
            for ff in ffs:
                for gg in grids:
                    name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))
                    NSE[mm] = h5[name]['nse_DA'][:] - h5[name]['nse_model'][:]
                    KGE[mm] = h5[name]['kge_DA'][:] - h5[name]['kge_model'][:]
                    RMSE[mm] = h5[name]['rmse_DA'][:] - h5[name]['rmse_model'][:]

    '''figure'''
    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="16p,5", MAP_TITLE_OFFSET=-0.5, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='17p,5',
                 FONT_LABEL='17p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar+h0', series=[-0.3, 0.3], background='o')

    tt2 = truth(region=basin)
    b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    his_mask, lat, lon = tt2.shrink_mask()

    ps = 'Q6c'
    region = [min(lon), max(lon), min(lat), max(lat)]
    region2 = [min(lon) - 0.5, max(lon) + 0.5, min(lat) - 0.5, max(lat) + 0.5]
    lon, lat = np.meshgrid(lon, lat)

    if basin == 'Danube':
        gdf = gpd.read_file((FP.SHP + '/DRB_3_shapefiles'))
    else:
        gdf = gpd.read_file(
            (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    vv_NSE = [NSE[mission.NGGM] - NSE[mission.GRACEC], NSE[mission.MAGIC] - NSE[mission.GRACEC]]
    vv_KGE = [KGE[mission.NGGM] - KGE[mission.GRACEC], KGE[mission.MAGIC] - KGE[mission.GRACEC]]
    vv_RMSE = [RMSE[mission.NGGM] - RMSE[mission.GRACEC], RMSE[mission.MAGIC] - RMSE[mission.GRACEC]]
    vv = [vv_NSE, vv_KGE, vv_RMSE]
    mission_label = ['NGGM - GRACE-C', 'MAGIC - GRACE-C']
    label_index = ['NSE', 'KGE', 'RMSE']

    for ind0, sds in enumerate(vv[0:2]):
        for ind, value in enumerate(sds):

            nwV= value[~np.isnan(value)]
            print(len(nwV[nwV>0])/len(nwV), np.mean(nwV[nwV>0]), np.mean(nwV[nwV<0]))

            if ind0 !=0 and ind == 0:
                fig.shift_origin(xshift='-14c', yshift='-7c')
            ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=value.flatten(),
                               spacing=0.5, region=region)

            fig.grdimage(
                grid=ee,
                cmap=True,
                frame=['+t%s: %s' % (label_index[ind0], mission_label[ind])],
                # dpi=100,
                projection=ps,
                region=region2,
                # interpolation='b',
                # nan_transparent=True
            )
            # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
            fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)
            fig.shift_origin(xshift='7c')

    fig.colorbar(frame='xa0.1f0.1+lDimensionless', position="JBC+w16c/0.5c+h+o0c/0.5c")

    pygmt.makecpt(cmap='polar+h0', series=[-20, 20], background='o')
    fig.shift_origin(yshift='-9c',xshift='-14c')

    for ind, value in enumerate(vv[2]):
        nwV = value[~np.isnan(value)]
        print(len(nwV[nwV < 0]) / len(nwV), np.mean(nwV[nwV < 0]), np.mean(nwV[nwV > 0]))
        ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=value.flatten(),
                           spacing=0.5, region=region)

        fig.grdimage(
            grid=ee,
            cmap=True,
            frame=['+t%s: %s' % (label_index[2], mission_label[ind])],
            # dpi=100,
            projection=ps,
            region=region2,
            # interpolation='b',
            # nan_transparent=True
        )
        # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
        fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)
        fig.shift_origin(xshift='7c')

    fig.colorbar(frame='xa10f5+lEWH [mm]', position="JBC+w16c/0.5c+h+o0c/0.5c")

    "=================================================================="
    pygmt.makecpt(cmap='polar+h0', series=[-0.3, 0.3], background='o')
    fig.shift_origin(xshift='1c', yshift='18c')
    basins = ['Danube']
    # basins = ['Brahmaputra']
    grids = [grid.G1]
    ffs = [filter.F1E17]

    NSE = {}
    KGE = {}
    RMSE = {}

    for basin in basins:
        for mm in mission:
            for ff in ffs:
                for gg in grids:
                    name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))
                    NSE[mm] = h5[name]['nse_DA'][:] - h5[name]['nse_model'][:]
                    KGE[mm] = h5[name]['kge_DA'][:] - h5[name]['kge_model'][:]
                    RMSE[mm] = h5[name]['rmse_DA'][:] - h5[name]['rmse_model'][:]
                    if mm==mission.GRACEC:
                        name = '%s_%s_%s_%s_%s' % (basin, mm.name, filter.F1E19.name, grid.G4.name, int(isHPF))
                        NSE[mm] = h5[name]['nse_DA'][:] - h5[name]['nse_model'][:]
                        KGE[mm] = h5[name]['kge_DA'][:] - h5[name]['kge_model'][:]
                        RMSE[mm] = h5[name]['rmse_DA'][:] - h5[name]['rmse_model'][:]

    '''figure'''
    tt2 = truth(region=basin)
    b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    his_mask, lat, lon = tt2.shrink_mask()

    ps = 'Q9.5c'
    region = [min(lon), max(lon), min(lat), max(lat)]
    region2 = [min(lon) - 0.5, max(lon) + 0.5, min(lat) - 0.5, max(lat) + 0.5]
    lon, lat = np.meshgrid(lon, lat)

    if basin == 'Danube':
        gdf = gpd.read_file((FP.SHP + '/DRB_3_shapefiles'))
    else:
        gdf = gpd.read_file(
            (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    vv_NSE = [NSE[mission.NGGM] - NSE[mission.GRACEC], NSE[mission.MAGIC] - NSE[mission.GRACEC]]
    vv_KGE = [KGE[mission.NGGM] - KGE[mission.GRACEC], KGE[mission.MAGIC] - KGE[mission.GRACEC]]
    vv_RMSE = [RMSE[mission.NGGM] - RMSE[mission.GRACEC], RMSE[mission.MAGIC] - RMSE[mission.GRACEC]]
    vv = [vv_NSE, vv_KGE, vv_RMSE]
    mission_label = ['NGGM - GRACE-C', 'MAGIC - GRACE-C']
    label_index = ['NSE', 'KGE', 'RMSE']

    for ind0, sds in enumerate(vv[0:2]):
        for ind, value in enumerate(sds):
            nwV = value[~np.isnan(value)]
            print(len(nwV[nwV > 0]) / len(nwV), np.mean(nwV[nwV > 0]), np.mean(nwV[nwV < 0]))

            if ind0 !=0 and ind == 0:
                fig.shift_origin(xshift='-19c', yshift='-7.c')
            ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=value.flatten(),
                               spacing=0.5, region=region)

            fig.grdimage(
                grid=ee,
                cmap=True,
                frame=['+t%s: %s' % (label_index[ind0], mission_label[ind])],
                # dpi=100,
                projection=ps,
                region=region2,
                # interpolation='b',
                # nan_transparent=True
            )
            # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
            fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)
            fig.shift_origin(xshift='9.5c')

    pygmt.makecpt(cmap='polar+h0', series=[-20, 20], background='o')
    fig.shift_origin(yshift='-9.5c',xshift='-19c')

    for ind, value in enumerate(vv[2]):
        nwV = value[~np.isnan(value)]
        print(len(nwV[nwV < 0]) / len(nwV), np.mean(nwV[nwV < 0]), np.mean(nwV[nwV > 0]))
        ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=value.flatten(),
                           spacing=0.5, region=region)

        fig.grdimage(
            grid=ee,
            cmap=True,
            frame=['+t%s: %s' % (label_index[2], mission_label[ind])],
            # dpi=100,
            projection=ps,
            region=region2,
            # interpolation='b',
            # nan_transparent=True
        )
        # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
        fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)
        fig.shift_origin(xshift='9.5c')

    fig.savefig(_OUT[0])
    pass


def scatter_plot_Brahmaputra():
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data
    from spatial_analysis import truth

    h5 = h5py.File(name=_H5_PATH, mode='r')
    isHPF = False
    # basins = ['Danube']
    basins = ['Brahmaputra']
    grids = [grid.G1]
    ffs = [filter.F1E19]

    NSE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    KGE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    RMSE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    model_NSE = 0
    model_KGE = 0
    model_RMSE = 0
    i = 0
    for basin in basins:
        for mm in mission:
            for ff in filter:
                for gg in grid:
                    name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))

                    nse = h5[name]['nse_DA'][:]
                    nse[nse < -1] = -1
                    NSE[mm].append(np.nanmean(nse))
                    kge = h5[name]['kge_DA'][:]
                    kge[kge < -1] = -1
                    KGE[mm].append(np.nanmean(kge))
                    rmse = h5[name]['rmse_DA'][:]
                    RMSE[mm].append(np.nanmean(rmse))

                    if i == 0:
                        hg = h5[name]['nse_model'][:]
                        hg[hg < -1] = -1
                        model_NSE = np.nanmean(hg)
                        hg = h5[name]['kge_model'][:]
                        hg[hg < -1] = -1
                        model_KGE = np.nanmean(hg)
                        hg = h5[name]['rmse_model'][:]
                        model_RMSE = np.nanmean(hg)
                    i += 1

                    if i == 21:
                        print(name)
                    pass

    x = np.arange(1, i / 3 + 1)
    '''figure'''
    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="13p,5", MAP_TITLE_OFFSET=-0.5, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5',
                 FONT_LABEL='15p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar+h0', series=[-0.3, 0.3], background='o')

    y0 = [NSE, KGE, RMSE]
    y1 = [model_NSE, model_KGE, model_RMSE]

    QI = 0
    z = y0[QI]
    z1 = y1[QI]
    label = ['NSE', 'KGE']

    fig.basemap(frame=["Wsne", "xa2f1g1", "ya0.2f0.1g0.1+l%s" % label[QI]],
                region=[0, 31, 0.2, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/6c")

    data = [[2.99, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 5, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 10, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 15, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 20, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 25, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    mm = mission.MAGIC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,gold",
    )
    fig.plot(x=x, y=z[mm], style="t0.4c", pen="1p", fill="gold")

    mm = mission.NGGM
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,dodgerblue@20",
    )
    fig.plot(x=x, y=z[mm], style="c0.25c", pen="1p", fill="dodgerblue@30")

    mm = mission.GRACEC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,seagreen",
    )
    fig.plot(x=x, y=z[mm], style="s0.3c", pen="1p", fill="seagreen@30")

    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=[-2, 32],
        y=[z1, z1],
        cmap=True,
        pen="2p,red3,--",
    )

    data = [[21, 0.75, 0.42, 1.8]]
    fig.plot(data=data, style="r", fill="lightgray@100", pen='1.5p,purple')
    fig.text(x=25, y=0.85, text='Best scenarios', font='11p,7')
    style = "v0.3c+e+a40+gpurple+h0+p2p,purple"
    fig.plot(x=21.5, y=0.82, style=style, pen="2p,purple", direction=[10, 0.7])
    fig.text(x=15, y=0.37, text='Model performance', font='12p,7')

    vecstyle = "v0.5c+b+e+h0.5"
    fig.plot(
        x=1, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0, y=0.94, text='Filtering Strength', font='9p,7')
    fig.text(x=3.0, y=0.90, text='1E14', font='9p,7')

    fig.plot(
        x=1 + 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5, y=0.93, text='1E15', font='9p,7')

    fig.plot(
        x=1 + 5 * 2, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 2, y=0.93, text='1E16', font='9p,7')

    fig.plot(
        x=1 + 5 * 3, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 3, y=0.93, text='1E17', font='9p,7')

    fig.plot(
        x=1 + 5 * 4, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 4, y=0.93, text='1E18', font='9p,7')

    fig.plot(
        x=1 + 5 * 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 5, y=0.93, text='1E19', font='9p,7')
    fig.text(x=1, y=0.24, text='(a)')

    fig.shift_origin(yshift='6.5c')
    fig.basemap(frame=["WSne+glavender"],
                region=[0, 31, 0, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/0.7c")
    fig.plot(
        x=[1, 4], y=[0.5, 0.5], pen="2p,gold", no_clip=True
    )
    fig.plot(
        x=2.5, y=0.5, style="t0.4c", pen="1p", fill="gold"
    )
    fig.text(x=6, y=0.5, text='MAGIC')

    fig.plot(
        x=[9, 12], y=[0.5, 0.5], pen="2p,dodgerblue", no_clip=True
    )
    fig.plot(
        x=10.5, y=0.5, style="c0.4c", pen="1p", fill="dodgerblue"
    )
    fig.text(x=14, y=0.5, text='NGGM')

    fig.plot(
        x=[16.5, 19.5], y=[0.5, 0.5], pen="2p,seagreen", no_clip=True
    )
    fig.plot(
        x=18, y=0.5, style="s0.4c", pen="1p", fill="seagreen"
    )
    fig.text(x=22, y=0.5, text='GRACE-C')

    """====================================KGE=========================================="""
    fig.shift_origin(yshift='-13.5c')

    QI = 1
    z = y0[QI]
    z1 = y1[QI]
    label = ['NSE', 'KGE']

    fig.basemap(frame=["Wsne", "xa2f1g1", "ya0.2f0.1g0.1+l%s" % label[QI]],
                region=[0, 31, 0.4, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/6c")

    data = [[2.99, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 5, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 10, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 15, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 20, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 25, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    mm = mission.MAGIC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,gold",
    )
    fig.plot(x=x, y=z[mm], style="t0.4c", pen="1p", fill="gold")

    mm = mission.NGGM
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,dodgerblue@20",
    )
    fig.plot(x=x, y=z[mm], style="c0.25c", pen="1p", fill="dodgerblue@30")

    mm = mission.GRACEC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,seagreen",
    )
    fig.plot(x=x, y=z[mm], style="s0.3c", pen="1p", fill="seagreen@30")

    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=[-2, 32],
        y=[z1, z1],
        cmap=True,
        pen="2p,red3,--",
    )

    data = [[21, 0.78, 0.42, 1.6]]
    fig.plot(data=data, style="r", fill="lightgray@100", pen='1.5p,purple')
    fig.text(x=25, y=0.85, text='Best scenarios', font='11p,7')
    style = "v0.3c+e+a40+gpurple+h0+p2p,purple"
    fig.plot(x=21.5, y=0.82, style=style, pen="2p,purple", direction=[10, 0.7])
    fig.text(x=5, y=0.47, text='Model performance', font='12p,7')

    vecstyle = "v0.5c+b+e+h0.5"
    fig.plot(
        x=1, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0, y=0.94, text='Filtering Strength', font='9p,7')
    fig.text(x=3.0, y=0.90, text='1E14', font='9p,7')

    fig.plot(
        x=1 + 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5, y=0.93, text='1E15', font='9p,7')

    fig.plot(
        x=1 + 5 * 2, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 2, y=0.93, text='1E16', font='9p,7')

    fig.plot(
        x=1 + 5 * 3, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 3, y=0.93, text='1E17', font='9p,7')

    fig.plot(
        x=1 + 5 * 4, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 4, y=0.93, text='1E18', font='9p,7')

    fig.plot(
        x=1 + 5 * 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 5, y=0.93, text='1E19', font='9p,7')
    fig.text(x=1, y=0.43, text='(b)')

    """====================================RMSE=========================================="""
    fig.shift_origin(yshift='-7c')

    QI = 2
    z = y0[QI]
    z1 = y1[QI]
    label = ['NSE', 'KGE', 'RMSE [mm]']

    fig.basemap(frame=["Wsne", "xa2f1g1+lScenario No.", "ya10f5g5+l%s" % label[QI]],
                region=[0, 31, 50, 120],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/6c")

    data = [[2.99, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 5, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 10, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 15, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 20, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 25, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    mm = mission.MAGIC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,gold",
    )
    fig.plot(x=x, y=z[mm], style="t0.4c", pen="1p", fill="gold")

    mm = mission.NGGM
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,dodgerblue@20",
    )
    fig.plot(x=x, y=z[mm], style="c0.25c", pen="1p", fill="dodgerblue@30")

    mm = mission.GRACEC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,seagreen",
    )
    fig.plot(x=x, y=z[mm], style="s0.3c", pen="1p", fill="seagreen@30")

    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=[-2, 32],
        y=[z1, z1],
        cmap=True,
        pen="2p,red3,--",
    )

    data = [[21, 67.5, 0.42, 1.8]]
    fig.plot(data=data, style="r", fill="lightgray@100", pen='1.5p,purple')
    fig.text(x=24.5, y=55, text='Best scenarios', font='11p,7')
    style = "v0.3c+e+a40+gpurple+h0+p2p,purple"
    fig.plot(x=21.5, y=60.5, style=style, pen="2p,purple", direction=[-15, 0.7])
    fig.text(x=15, y=113, text='Model performance', font='12p,7')

    vecstyle = "v0.5c+b+e+h0.5"
    fig.plot(
        x=1, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0, y=0.94, text='Filtering Strength', font='9p,7')
    fig.text(x=3.0, y=0.90, text='1E14', font='9p,7')

    fig.plot(
        x=1 + 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5, y=0.93, text='1E15', font='9p,7')

    fig.plot(
        x=1 + 5 * 2, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 2, y=0.93, text='1E16', font='9p,7')

    fig.plot(
        x=1 + 5 * 3, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 3, y=0.93, text='1E17', font='9p,7')

    fig.plot(
        x=1 + 5 * 4, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 4, y=0.93, text='1E18', font='9p,7')

    fig.plot(
        x=1 + 5 * 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 5, y=0.93, text='1E19', font='9p,7')
    fig.text(x=1, y=53.5, text='(c)')

    fig.shift_origin(yshift='-1.1c')
    fig.basemap(frame=["WSne",'x+lScenario No.'],
                region=[0, 31, 0, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/0.7c")

    fig.text(x=1, y=0.5, text='1\\260', font='11p')
    fig.text(x=2, y=0.5, text='2\\260', font='11p')
    fig.text(x=3, y=0.5, text='3\\260', font='11p')
    fig.text(x=4, y=0.5, text='4\\260', font='11p')
    fig.text(x=5, y=0.5, text='Bw', font='11p')
    fig.text(x=6, y=0.5, text='1\\260', font='11p')
    fig.text(x=7, y=0.5, text='2\\260', font='11p')
    fig.text(x=8, y=0.5, text='3\\260', font='11p')
    fig.text(x=9, y=0.5, text='4\\260', font='11p')
    fig.text(x=10, y=0.5, text='Bw', font='11p')
    fig.text(x=11, y=0.5, text='1\\260', font='11p')
    fig.text(x=12, y=0.5, text='2\\260', font='11p')
    fig.text(x=13, y=0.5, text='3\\260', font='11p')
    fig.text(x=14, y=0.5, text='4\\260', font='11p')
    fig.text(x=15, y=0.5, text='Bw', font='11p')
    fig.text(x=16, y=0.5, text='1\\260', font='11p')
    fig.text(x=17, y=0.5, text='2\\260', font='11p')
    fig.text(x=18, y=0.5, text='3\\260', font='11p')
    fig.text(x=19, y=0.5, text='4\\260', font='11p')
    fig.text(x=20, y=0.5, text='Bw', font='11p')
    fig.text(x=21, y=0.5, text='1\\260', font='11p')
    fig.text(x=22, y=0.5, text='2\\260', font='11p')
    fig.text(x=23, y=0.5, text='3\\260', font='11p')
    fig.text(x=24, y=0.5, text='4\\260', font='11p')
    fig.text(x=25, y=0.5, text='Bw', font='11p')
    fig.text(x=26, y=0.5, text='1\\260', font='11p')
    fig.text(x=27, y=0.5, text='2\\260', font='11p')
    fig.text(x=28, y=0.5, text='3\\260', font='11p')
    fig.text(x=29, y=0.5, text='4\\260', font='11p')
    fig.text(x=30, y=0.5, text='Bw', font='11p')

    fig.savefig(_OUT[0])

    pass


def scatter_plot_Danube():
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data
    from spatial_analysis import truth

    h5 = h5py.File(name=_H5_PATH, mode='r')
    isHPF = False
    basins = ['Danube']
    # basins = ['Brahmaputra']
    grids = [grid.G1]
    ffs = [filter.F1E19]

    NSE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    KGE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    RMSE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    model_NSE = 0
    model_KGE = 0
    model_RMSE = 0
    i = 0
    for basin in basins:
        for mm in mission:
            for ff in filter:
                for gg in grid:
                    name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))

                    nse = h5[name]['nse_DA'][:]
                    nse[nse < -1] = -1
                    NSE[mm].append(np.nanmean(nse))
                    kge = h5[name]['kge_DA'][:]
                    kge[kge < -1] = -1
                    KGE[mm].append(np.nanmean(kge))
                    rmse = h5[name]['rmse_DA'][:]
                    RMSE[mm].append(np.nanmean(rmse))

                    if i == 0:
                        hg = h5[name]['nse_model'][:]
                        hg[hg < -1] = -1
                        model_NSE = np.nanmean(hg)
                        hg = h5[name]['kge_model'][:]
                        hg[hg < -1] = -1
                        model_KGE = np.nanmean(hg)
                        hg = h5[name]['rmse_model'][:]
                        model_RMSE = np.nanmean(hg)
                    i += 1

                    if i == 21:
                        print(name)
                    pass

    x = np.arange(1, i / 3 + 1)
    '''figure'''
    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="13p,5", MAP_TITLE_OFFSET=-0.5, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5',
                 FONT_LABEL='15p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar+h0', series=[-0.3, 0.3], background='o')

    y0 = [NSE, KGE, RMSE]
    y1 = [model_NSE, model_KGE, model_RMSE]

    QI = 0
    z = y0[QI]
    z1 = y1[QI]
    label = ['NSE', 'KGE']

    fig.basemap(frame=["WSne", "xf1g1", "ya0.2f0.1g0.1+l%s" % label[QI]],
                region=[0, 31, -1, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/6c")

    data = [[2.99, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 5, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 10, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 15, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 20, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 25, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    dfd = list(np.arange(30))
    dfd.remove(1)
    dfd.remove(6)
    dfd.remove(11)
    dfd.remove(16)
    dfd.remove(21)
    dfd.remove(26)

    mm = mission.MAGIC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,gold",
    )
    fig.plot(x=x, y=z[mm], style="t0.4c", pen="1p", fill="gold")

    mm = mission.NGGM
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,dodgerblue@20",
    )
    fig.plot(x=x, y=z[mm], style="c0.25c", pen="1p", fill="dodgerblue@30")

    mm = mission.GRACEC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,seagreen",
    )
    fig.plot(x=x, y=z[mm], style="s0.3c", pen="1p", fill="seagreen@30")

    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=[-2, 32],
        y=[z1, z1],
        cmap=True,
        pen="2p,red3,--",
    )

    data = [[21, 0.78, 0.42, 1.4]]
    fig.plot(data=data, style="r", fill="lightgray@100", pen='1.5p,purple')
    fig.text(x=23, y=0.93, text='Best scenarios', font='11p,7')
    style = "v0.3c+e+a40+gpurple+h0+p2p,purple"
    fig.plot(x=21.5, y=0.87, style=style, pen="2p,purple", direction=[10, 0.7])
    fig.plot(x=11.5, y=0.72, style=style, pen="2p,purple", direction=[6, 5.5])
    fig.text(x=4.5, y=0.32, text='Model performance', font='12p,7')

    # vecstyle="v0.5c+b+e+h0.5"
    # fig.plot(
    #     x=1, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0,y=0.94,text='Filtering Strength', font='9p,7')
    # fig.text(x=3.0, y=0.90, text='1E14', font='9p,7')
    #
    # fig.plot(
    #     x=1+5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5, y=0.93, text='1E15', font='9p,7')
    #
    # fig.plot(
    #     x=1 + 5*2, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0 + 5*2, y=0.93, text='1E16', font='9p,7')
    #
    # fig.plot(
    #     x=1 + 5*3, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0 + 5*3, y=0.93, text='1E17', font='9p,7')
    #
    # fig.plot(
    #     x=1 + 5*4, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0 + 5*4, y=0.93, text='1E18', font='9p,7')
    #
    # fig.plot(
    #     x=1 + 5*5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0 + 5*5, y=0.93, text='1E19', font='9p,7')
    fig.text(x=1, y=-0.9, text='(a)')

    fig.shift_origin(yshift='6.5c')
    fig.basemap(frame=["Wsne+glavender"],
                region=[0, 31, 0, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/0.7c")
    fig.plot(
        x=[1, 4], y=[0.5, 0.5], pen="2p,gold", no_clip=True
    )
    fig.plot(
        x=2.5, y=0.5, style="t0.4c", pen="1p", fill="gold"
    )
    fig.text(x=6, y=0.5, text='MAGIC')

    fig.plot(
        x=[9, 12], y=[0.5, 0.5], pen="2p,dodgerblue", no_clip=True
    )
    fig.plot(
        x=10.5, y=0.5, style="c0.4c", pen="1p", fill="dodgerblue"
    )
    fig.text(x=14, y=0.5, text='NGGM')

    fig.plot(
        x=[16.5, 19.5], y=[0.5, 0.5], pen="2p,seagreen", no_clip=True
    )
    fig.plot(
        x=18, y=0.5, style="s0.4c", pen="1p", fill="seagreen"
    )
    fig.text(x=22, y=0.5, text='GRACE-C')

    """====================================KGE=========================================="""
    fig.shift_origin(yshift='-13.2c')

    QI = 1
    z = y0[QI]
    z1 = y1[QI]
    label = ['NSE', 'KGE']

    fig.basemap(frame=["Wsne", "xf1g1", "ya0.2f0.1g0.1+l%s" % label[QI]],
                region=[0, 31, -0.3, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/6c")

    data = [[2.99, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 5, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 10, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 15, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 20, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 25, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    mm = mission.MAGIC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,gold",
    )
    fig.plot(x=x, y=z[mm], style="t0.4c", pen="1p", fill="gold")

    mm = mission.NGGM
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,dodgerblue@20",
    )
    fig.plot(x=x, y=z[mm], style="c0.25c", pen="1p", fill="dodgerblue@30")

    mm = mission.GRACEC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,seagreen",
    )
    fig.plot(x=x, y=z[mm], style="s0.3c", pen="1p", fill="seagreen@30")

    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=[-2, 32],
        y=[z1, z1],
        cmap=True,
        pen="2p,red3,--",
    )

    data = [[21, 0.78, 0.42, 1.4]]
    fig.plot(data=data, style="r", fill="lightgray@100", pen='1.5p,purple')
    fig.text(x=25, y=0.90, text='Best scenarios', font='11p,7')
    style = "v0.3c+e+a40+gpurple+h0+p2p,purple"
    fig.plot(x=21.5, y=0.87, style=style, pen="2p,purple", direction=[10, 0.7])
    fig.text(x=4.5, y=0.34, text='Model performance', font='12p,7')

    # vecstyle="v0.5c+b+e+h0.5"
    # fig.plot(
    #     x=1, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0,y=-0.18,text='Filtering Strength', font='9p,7')
    # fig.text(x=3.0, y=-0.13, text='1E14', font='9p,7')
    #
    # fig.plot(
    #     x=1+5, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5, y=-0.18, text='1E15', font='9p,7')
    #
    # fig.plot(
    #     x=1+5*2, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5*2, y=-0.18, text='1E16', font='9p,7')
    #
    # fig.plot(
    #     x=1+5*3, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5*3, y=-0.18, text='1E17', font='9p,7')
    #
    # fig.plot(
    #     x=1+5*4, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5*4, y=-0.18, text='1E18', font='9p,7')
    #
    # fig.plot(
    #     x=1+5*5, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5*5, y=-0.18, text='1E15', font='9p,7')
    fig.text(x=1, y=-0.22, text='(b)')

    """====================================RMSE=========================================="""
    fig.shift_origin(yshift='-6.7c')

    QI = 2
    z = y0[QI]
    z1 = y1[QI]
    label = ['NSE', 'KGE', 'RMSE [mm]']

    fig.basemap(frame=["Wsne", "xa2f1g1", "ya10f5g5+l%s" % label[QI]],
                region=[0, 31, 15, 65],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/6c")

    data = [[2.99, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 5, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 10, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 15, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 20, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 25, 50, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    dfd = list(np.arange(30))
    dfd.remove(1)
    dfd.remove(6)
    dfd.remove(11)
    dfd.remove(16)
    dfd.remove(21)
    dfd.remove(26)

    mm = mission.MAGIC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,gold",
    )
    fig.plot(x=x, y=z[mm], style="t0.4c", pen="1p", fill="gold")

    mm = mission.NGGM
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,dodgerblue@20",
    )
    fig.plot(x=x, y=z[mm], style="c0.25c", pen="1p", fill="dodgerblue@30")

    mm = mission.GRACEC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,seagreen",
    )
    fig.plot(x=x, y=z[mm], style="s0.3c", pen="1p", fill="seagreen@30")

    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=[-2, 32],
        y=[z1, z1],
        cmap=True,
        pen="2p,red3,--",
    )

    data = [[16, 25, 0.42, 0.8]]
    fig.plot(data=data, style="r", fill="lightgray@100", pen='1.5p,purple')
    fig.text(x=22, y=20, text='Best scenarios', font='11p,7')
    style = "v0.3c+e+a40+gpurple+h0+p2p,purple"
    fig.plot(x=16.5, y=25, style=style, pen="2p,purple", direction=[-20, 1.8])
    fig.plot(x=24, y=31, style=style, pen="2p,purple", direction=[-128, 1.4])
    fig.text(x=4.5, y=52, text='Model performance', font='12p,7')

    vecstyle = "v0.5c+b+e+h0.5"
    fig.plot(
        x=1, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0, y=0.94, text='Filtering Strength', font='9p,7')
    fig.text(x=3.0, y=0.90, text='1E14', font='9p,7')

    fig.plot(
        x=1 + 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5, y=0.93, text='1E15', font='9p,7')

    fig.plot(
        x=1 + 5 * 2, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 2, y=0.93, text='1E16', font='9p,7')

    fig.plot(
        x=1 + 5 * 3, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 3, y=0.93, text='1E17', font='9p,7')

    fig.plot(
        x=1 + 5 * 4, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 4, y=0.93, text='1E18', font='9p,7')

    fig.plot(
        x=1 + 5 * 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 5, y=0.93, text='1E19', font='9p,7')
    fig.text(x=1, y=18, text='(c)')

    fig.shift_origin(yshift='-1.1c')
    fig.basemap(frame=["WSne",'x+lScenario No.'],
                region=[0, 31, 0, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/0.7c")

    fig.text(x=1, y=0.5, text='1\\260', font='11p')
    fig.text(x=2, y=0.5, text='2\\260', font='11p')
    fig.text(x=3, y=0.5, text='3\\260', font='11p')
    fig.text(x=4, y=0.5, text='4\\260', font='11p')
    fig.text(x=5, y=0.5, text='Bw', font='11p')
    fig.text(x=6, y=0.5, text='1\\260', font='11p')
    fig.text(x=7, y=0.5, text='2\\260', font='11p')
    fig.text(x=8, y=0.5, text='3\\260', font='11p')
    fig.text(x=9, y=0.5, text='4\\260', font='11p')
    fig.text(x=10, y=0.5, text='Bw', font='11p')
    fig.text(x=11, y=0.5, text='1\\260', font='11p')
    fig.text(x=12, y=0.5, text='2\\260', font='11p')
    fig.text(x=13, y=0.5, text='3\\260', font='11p')
    fig.text(x=14, y=0.5, text='4\\260', font='11p')
    fig.text(x=15, y=0.5, text='Bw', font='11p')
    fig.text(x=16, y=0.5, text='1\\260', font='11p')
    fig.text(x=17, y=0.5, text='2\\260', font='11p')
    fig.text(x=18, y=0.5, text='3\\260', font='11p')
    fig.text(x=19, y=0.5, text='4\\260', font='11p')
    fig.text(x=20, y=0.5, text='Bw', font='11p')
    fig.text(x=21, y=0.5, text='1\\260', font='11p')
    fig.text(x=22, y=0.5, text='2\\260', font='11p')
    fig.text(x=23, y=0.5, text='3\\260', font='11p')
    fig.text(x=24, y=0.5, text='4\\260', font='11p')
    fig.text(x=25, y=0.5, text='Bw', font='11p')
    fig.text(x=26, y=0.5, text='1\\260', font='11p')
    fig.text(x=27, y=0.5, text='2\\260', font='11p')
    fig.text(x=28, y=0.5, text='3\\260', font='11p')
    fig.text(x=29, y=0.5, text='4\\260', font='11p')
    fig.text(x=30, y=0.5, text='Bw', font='11p')

    fig.savefig(_OUT[0])

    pass


def scatter_plot_Brahmaputra_HPF():
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data
    from spatial_analysis import truth

    h5 = h5py.File(name=_H5_PATH, mode='r')
    isHPF = True
    # basins = ['Danube']
    basins = ['Brahmaputra']
    grids = [grid.G1]
    ffs = [filter.F1E19]

    NSE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    KGE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    RMSE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    model_NSE = 0
    model_KGE = 0
    model_RMSE = 0
    i = 0
    for basin in basins:
        for mm in mission:
            for ff in filter:
                for gg in grid:
                    name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))

                    nse = h5[name]['nse_DA'][:]
                    nse[nse < -1] = -1
                    NSE[mm].append(np.nanmean(nse))
                    kge = h5[name]['kge_DA'][:]
                    kge[kge < -1] = -1
                    KGE[mm].append(np.nanmean(kge))
                    rmse = h5[name]['rmse_DA'][:]
                    RMSE[mm].append(np.nanmean(rmse))

                    if i == 0:
                        hg = h5[name]['nse_model'][:]
                        hg[hg < -1] = -1
                        model_NSE = np.nanmean(hg)
                        hg = h5[name]['kge_model'][:]
                        hg[hg < -1] = -1
                        model_KGE = np.nanmean(hg)
                        hg = h5[name]['rmse_model'][:]
                        model_RMSE = np.nanmean(hg)
                    i += 1

                    if i == 21:
                        print(name)
                    pass

    x = np.arange(1, i / 3 + 1)
    '''figure'''
    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="13p,5", MAP_TITLE_OFFSET=-0.5, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='13p,5',
                 FONT_LABEL='15p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar+h0', series=[-0.3, 0.3], background='o')

    y0 = [NSE, KGE, RMSE]
    y1 = [model_NSE, model_KGE, model_RMSE]

    QI = 0
    z = y0[QI]
    z1 = y1[QI]
    label = ['NSE', 'KGE']

    fig.basemap(frame=["WSne", "xf1g1", "ya0.2f0.1g0.1+l%s" % label[QI]],
                region=[0, 31, -0.5, 0.6],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/6c")

    data = [[2.99, 0.5, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 5, 0.5, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 10, 0.5, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 15, 0.5, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 20, 0.5, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 25, 0.5, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    dfd = list(np.arange(30))
    # dfd.remove(1)
    # dfd.remove(6)
    # dfd.remove(11)
    # dfd.remove(16)
    # dfd.remove(21)
    # dfd.remove(26)

    mm = mission.MAGIC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,gold",
    )
    fig.plot(x=x, y=z[mm], style="t0.4c", pen="1p", fill="gold")

    mm = mission.NGGM
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,dodgerblue@20",
    )
    fig.plot(x=x, y=z[mm], style="c0.25c", pen="1p", fill="dodgerblue@30")

    mm = mission.GRACEC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,seagreen",
    )
    fig.plot(x=x, y=z[mm], style="s0.3c", pen="1p", fill="seagreen@30")

    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=[-2, 32],
        y=[z1, z1],
        cmap=True,
        pen="2p,red3,--",
    )

    data = [[16, 0.82, 0.42, 0.7]]
    fig.plot(data=data, style="r", fill="lightgray@100", pen='1.5p,purple')
    fig.text(x=23, y=0.93, text='Best scenarios', font='11p,7')
    style = "v0.3c+e+a40+gpurple+h0+p2p,purple"
    fig.plot(x=16.5, y=0.87, style=style, pen="2p,purple", direction=[5, 2.2])
    fig.plot(x=29, y=0.73, style=style, pen="2p,purple", direction=[165, 2.0])
    fig.text(x=15, y=0.19, text='Model performance', font='12p,7')

    # vecstyle="v0.5c+b+e+h0.5"
    # fig.plot(
    #     x=1, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0,y=0.94,text='Filtering Strength', font='9p,7')
    # fig.text(x=3.0, y=0.90, text='1E14', font='9p,7')
    #
    # fig.plot(
    #     x=1+5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5, y=0.93, text='1E15', font='9p,7')
    #
    # fig.plot(
    #     x=1 + 5*2, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0 + 5*2, y=0.93, text='1E16', font='9p,7')
    #
    # fig.plot(
    #     x=1 + 5*3, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0 + 5*3, y=0.93, text='1E17', font='9p,7')
    #
    # fig.plot(
    #     x=1 + 5*4, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0 + 5*4, y=0.93, text='1E18', font='9p,7')
    #
    # fig.plot(
    #     x=1 + 5*5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0 + 5*5, y=0.93, text='1E19', font='9p,7')
    fig.text(x=1, y=-0.44, text='(a)')

    fig.shift_origin(yshift='6.5c')
    fig.basemap(frame=["WSne+glavender"],
                region=[0, 31, 0, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/0.7c")
    fig.plot(
        x=[1, 4], y=[0.5, 0.5], pen="2p,gold", no_clip=True
    )
    fig.plot(
        x=2.5, y=0.5, style="t0.4c", pen="1p", fill="gold"
    )
    fig.text(x=6, y=0.5, text='MAGIC')

    fig.plot(
        x=[9, 12], y=[0.5, 0.5], pen="2p,dodgerblue", no_clip=True
    )
    fig.plot(
        x=10.5, y=0.5, style="c0.4c", pen="1p", fill="dodgerblue"
    )
    fig.text(x=14, y=0.5, text='NGGM')

    fig.plot(
        x=[16.5, 19.5], y=[0.5, 0.5], pen="2p,seagreen", no_clip=True
    )
    fig.plot(
        x=18, y=0.5, style="s0.4c", pen="1p", fill="seagreen"
    )
    fig.text(x=22, y=0.5, text='GRACE-C')

    """====================================KGE=========================================="""
    fig.shift_origin(yshift='-13.5c')

    QI = 1
    z = y0[QI]
    z1 = y1[QI]
    label = ['NSE', 'KGE']

    fig.basemap(frame=["Wsne", "xa2f1g1", "ya0.2f0.1g0.1+l%s" % label[QI]],
                region=[0, 31, 0.1, 0.7],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/6c")

    data = [[2.99, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 5, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 10, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 15, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 20, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 25, 0.5, 2.05, 10]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    mm = mission.MAGIC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,gold",
    )
    fig.plot(x=x, y=z[mm], style="t0.4c", pen="1p", fill="gold")

    mm = mission.NGGM
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,dodgerblue@20",
    )
    fig.plot(x=x, y=z[mm], style="c0.25c", pen="1p", fill="dodgerblue@30")

    mm = mission.GRACEC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x[dfd],
        y=np.array(z[mm])[dfd],
        cmap=True,
        pen="2p,seagreen",
    )
    fig.plot(x=x, y=z[mm], style="s0.3c", pen="1p", fill="seagreen@30")

    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=[-2, 32],
        y=[z1, z1],
        cmap=True,
        pen="2p,red3,--",
    )

    data = [[21, 0.78, 0.42, 1.4]]
    fig.plot(data=data, style="r", fill="lightgray@100", pen='1.5p,purple')
    fig.text(x=25, y=0.90, text='Best scenarios', font='11p,7')
    style = "v0.3c+e+a40+gpurple+h0+p2p,purple"
    fig.plot(x=21.5, y=0.87, style=style, pen="2p,purple", direction=[10, 0.7])
    fig.text(x=15, y=0.36, text='Model performance', font='12p,7')

    # vecstyle="v0.5c+b+e+h0.5"
    # fig.plot(
    #     x=1, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0,y=-0.18,text='Filtering Strength', font='9p,7')
    # fig.text(x=3.0, y=-0.13, text='1E14', font='9p,7')
    #
    # fig.plot(
    #     x=1+5, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5, y=-0.18, text='1E15', font='9p,7')
    #
    # fig.plot(
    #     x=1+5*2, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5*2, y=-0.18, text='1E16', font='9p,7')
    #
    # fig.plot(
    #     x=1+5*3, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5*3, y=-0.18, text='1E17', font='9p,7')
    #
    # fig.plot(
    #     x=1+5*4, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5*4, y=-0.18, text='1E18', font='9p,7')
    #
    # fig.plot(
    #     x=1+5*5, y=-0.25, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    # )
    # fig.text(x=3.0+5*5, y=-0.18, text='1E15', font='9p,7')
    fig.text(x=1, y=0.14, text='(b)')

    """====================================RMSE=========================================="""
    fig.shift_origin(yshift='-7c')

    QI = 2
    z = y0[QI]
    z1 = y1[QI]
    label = ['NSE', 'KGE', 'RMSE [mm]']

    fig.basemap(frame=["Wsne", "xa2f1g1", "ya5f5g5+l%s" % label[QI]],
                region=[0, 31, 10, 25],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/6c")

    data = [[2.99, 10, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 5, 10, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 10, 10, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 15, 10, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 20, 10, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    data = [[2.99 + 25, 10, 2.05, 100]]
    fig.plot(data=data, style="r", fill="lightgray@40")

    mm = mission.MAGIC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,gold",
    )
    fig.plot(x=x, y=z[mm], style="t0.4c", pen="1p", fill="gold")

    mm = mission.NGGM
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,dodgerblue@20",
    )
    fig.plot(x=x, y=z[mm], style="c0.25c", pen="1p", fill="dodgerblue@30")

    mm = mission.GRACEC
    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=x,
        y=z[mm],
        cmap=True,
        pen="2p,seagreen",
    )
    fig.plot(x=x, y=z[mm], style="s0.3c", pen="1p", fill="seagreen@30")

    fig.plot(
        # Set the figures frame and color as well as
        # annotations, ticks, and gridlines
        x=[-2, 32],
        y=[z1+1, z1+1],
        cmap=True,
        pen="2p,red3,--",
    )

    data = [[21, 0.78, 0.42, 1.6]]
    fig.plot(data=data, style="r", fill="lightgray@100", pen='1.5p,purple')
    fig.text(x=25, y=0.85, text='Best scenarios', font='11p,7')
    style = "v0.3c+e+a40+gpurple+h0+p2p,purple"
    fig.plot(x=21.5, y=0.82, style=style, pen="2p,purple", direction=[10, 0.7])
    fig.text(x=15, y=17.5, text='Model performance', font='12p,7')

    vecstyle = "v0.5c+b+e+h0.5"
    fig.plot(
        x=1, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0, y=0.94, text='Filtering Strength', font='9p,7')
    fig.text(x=3.0, y=0.90, text='1E14', font='9p,7')

    fig.plot(
        x=1 + 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5, y=0.93, text='1E15', font='9p,7')

    fig.plot(
        x=1 + 5 * 2, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 2, y=0.93, text='1E16', font='9p,7')

    fig.plot(
        x=1 + 5 * 3, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 3, y=0.93, text='1E17', font='9p,7')

    fig.plot(
        x=1 + 5 * 4, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 4, y=0.93, text='1E18', font='9p,7')

    fig.plot(
        x=1 + 5 * 5, y=0.97, style=vecstyle, direction=(0, 2.1), pen="2p", fill="red3"
    )
    fig.text(x=3.0 + 5 * 5, y=0.93, text='1E19', font='9p,7')
    fig.text(x=1, y=11, text='(c)')

    fig.shift_origin(yshift='-1.1c')
    fig.basemap(frame=["WSne",'x+lScenario No.'],
                region=[0, 31, 0, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X16c/0.7c")

    fig.text(x=1, y=0.5, text='1\\260', font='11p')
    fig.text(x=2, y=0.5, text='2\\260', font='11p')
    fig.text(x=3, y=0.5, text='3\\260', font='11p')
    fig.text(x=4, y=0.5, text='4\\260', font='11p')
    fig.text(x=5, y=0.5, text='Bw', font='11p')
    fig.text(x=6, y=0.5, text='1\\260', font='11p')
    fig.text(x=7, y=0.5, text='2\\260', font='11p')
    fig.text(x=8, y=0.5, text='3\\260', font='11p')
    fig.text(x=9, y=0.5, text='4\\260', font='11p')
    fig.text(x=10, y=0.5, text='Bw', font='11p')
    fig.text(x=11, y=0.5, text='1\\260', font='11p')
    fig.text(x=12, y=0.5, text='2\\260', font='11p')
    fig.text(x=13, y=0.5, text='3\\260', font='11p')
    fig.text(x=14, y=0.5, text='4\\260', font='11p')
    fig.text(x=15, y=0.5, text='Bw', font='11p')
    fig.text(x=16, y=0.5, text='1\\260', font='11p')
    fig.text(x=17, y=0.5, text='2\\260', font='11p')
    fig.text(x=18, y=0.5, text='3\\260', font='11p')
    fig.text(x=19, y=0.5, text='4\\260', font='11p')
    fig.text(x=20, y=0.5, text='Bw', font='11p')
    fig.text(x=21, y=0.5, text='1\\260', font='11p')
    fig.text(x=22, y=0.5, text='2\\260', font='11p')
    fig.text(x=23, y=0.5, text='3\\260', font='11p')
    fig.text(x=24, y=0.5, text='4\\260', font='11p')
    fig.text(x=25, y=0.5, text='Bw', font='11p')
    fig.text(x=26, y=0.5, text='1\\260', font='11p')
    fig.text(x=27, y=0.5, text='2\\260', font='11p')
    fig.text(x=28, y=0.5, text='3\\260', font='11p')
    fig.text(x=29, y=0.5, text='4\\260', font='11p')
    fig.text(x=30, y=0.5, text='Bw', font='11p')

    fig.savefig(_OUT[0])

    pass


def scatter_plot_cov_comparison_1(basins=['Brahmaputra']):
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data
    from spatial_analysis import truth

    h5 = h5py.File(name=_H5_PATH, mode='r')
    isHPF = False

    basins = basins
    NSE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    KGE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    RMSE = {mission.GRACEC: [],
           mission.NGGM: [],
           mission.MAGIC: []}
    model_NSE = 0
    model_KGE = 0
    i = 0
    for basin in basins:
        for mm in mission:
            for ff in filter:
                for gg in [grid.G1]:
                    name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))

                    nse = h5[name]['nse_DA'][:]
                    nse[nse < -1] = -1
                    NSE[mm].append(np.nanmean(nse))
                    kge = h5[name]['kge_DA'][:]
                    kge[kge < -1] = -1
                    KGE[mm].append(np.nanmean(kge))
                    rmse = h5[name]['rmse_DA'][:]
                    RMSE[mm].append(np.nanmean(rmse))

                    if i == 0:
                        hg = h5[name]['nse_model'][:]
                        hg[hg < -1] = -1
                        model_NSE = np.nanmean(hg)
                        hg = h5[name]['kge_model'][:]
                        hg[hg < -1] = -1
                        model_KGE = np.nanmean(hg)
                    i += 1

                    if i == 21:
                        print(name)
                    pass

    """========================================================="""
    h5 = h5py.File(name=_H5_COV_PATH, mode='r')

    NSE_2 = {mission.GRACEC: [],
             mission.NGGM: [],
             mission.MAGIC: []}
    KGE_2 = {mission.GRACEC: [],
             mission.NGGM: [],
             mission.MAGIC: []}
    RMSE_2 = {mission.GRACEC: [],
             mission.NGGM: [],
             mission.MAGIC: []}
    model_NSE = 0
    model_KGE = 0
    i = 0
    for basin in basins:
        for mm in mission:
            for ff in filter:
                for gg in [grid.G1]:
                    name = '%s_%s_%s_%s_%s' % (basin, mm.name, ff.name, gg.name, int(isHPF))

                    nse = h5[name]['nse_DA'][:]
                    nse[nse < -1] = -1
                    NSE_2[mm].append(np.nanmean(nse))
                    kge = h5[name]['kge_DA'][:]
                    kge[kge < -1] = -1
                    KGE_2[mm].append(np.nanmean(kge))
                    rmse = h5[name]['rmse_DA'][:]
                    RMSE_2[mm].append(np.nanmean(rmse))

                    if i == 0:
                        hg = h5[name]['nse_model'][:]
                        hg[hg < -1] = -1
                        model_NSE = np.nanmean(hg)
                        hg = h5[name]['kge_model'][:]
                        hg[hg < -1] = -1
                        model_KGE = np.nanmean(hg)
                    i += 1

                    if i == 21:
                        print(name)
                    pass

    return NSE, KGE, RMSE, NSE_2, KGE_2, RMSE_2


def scatter_plot_cov_comparison_Brahmaputra():
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data
    metric_Brahmaputra = {
        'NSE': {},
        'KGE': {},
        'RMSE':{},
        'NSE_2': {},
        'KGE_2': {},
        'RMSE_2': {}
    }

    metric_Danube = {
        'NSE': {},
        'KGE': {},
        'RMSE': {},
        'NSE_2': {},
        'KGE_2': {},
        'RMSE_2': {}
    }

    a, b, c, d, e, f = scatter_plot_cov_comparison_1(basins=['Brahmaputra'])
    metric_Brahmaputra['NSE'] = a
    metric_Brahmaputra['KGE'] = b
    metric_Brahmaputra['RMSE'] = c
    metric_Brahmaputra['NSE_2'] = d
    metric_Brahmaputra['KGE_2'] = e
    metric_Brahmaputra['RMSE_2'] = f

    g, h, i, j,k,l = scatter_plot_cov_comparison_1(basins=['Danube'])
    metric_Danube['NSE'] = g
    metric_Danube['KGE'] = h
    metric_Danube['RMSE'] = i
    metric_Danube['NSE_2'] = j
    metric_Danube['KGE_2'] = k
    metric_Danube['RMSE_2'] = l

    """===========================Start ploting================================"""
    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="19p,5", MAP_TITLE_OFFSET=0.2, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='16p,5',
                 FONT_LABEL='17p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar+h0', series=[-0.3, 0.3], background='o')

    label = ['NSE', 'KGE', 'RMSE']
    i = 0

    # for basin in ['Brahmaputra', 'Danube']:
    for basin in ['Brahmaputra']:
        for QI in [0, 1, 2]:
            for mm in [mission.GRACEC, mission.NGGM, mission.MAGIC]:

                if QI==2:
                    fig.basemap(frame=["WSne", "xf1g1a1", "yfg"],
                                region=[0, 7, 40, 110],
                                # Set a logarithmic transformation on the x-axis
                                projection="X6c/6c")
                elif QI==1:
                    fig.basemap(frame=["WSne", "xf1g1", "yf0.1g0.1"],
                                region=[0, 7, 0.2, 1],
                                # Set a logarithmic transformation on the x-axis
                                projection="X6c/6c")
                else:
                    # fig.basemap(frame=["WSne+t%s" % mm.name, "xf1g1", "yf0.1g0.1"],
                    #             region=[0, 7, 0.2, 1],
                    #             # Set a logarithmic transformation on the x-axis
                    #             projection="X6c/6c")

                    if mm.name == 'GRACEC':
                        fig.basemap(frame=["WSne+t%s" % 'GRACE-C', "xf1g1", "yf0.1g0.1"],
                                    region=[0, 7, 0.2, 1],
                                    # Set a logarithmic transformation on the x-axis
                                    projection="X6c/6c")
                    else:
                        fig.basemap(frame=["WSne+t%s" % mm.name, "xf1g1", "yf0.1g0.1"],
                                    region=[0, 7, 0.2, 1],
                                    # Set a logarithmic transformation on the x-axis
                                    projection="X6c/6c")

                if i%3==0:
                    if QI==2:
                        fig.basemap(frame=["WSne", "xf1g1a1", "yafg+lRMSE [mm]"],
                                    region=[0, 7, 40, 110],
                                    # Set a logarithmic transformation on the x-axis
                                    projection="X6c/6c")
                    elif QI==1:
                        fig.basemap(frame=["WSne", "xf1g1", "ya0.2f0.1g0.1+lKGE"],
                                    region=[0, 7, 0.2, 1],
                                    # Set a logarithmic transformation on the x-axis
                                    projection="X6c/6c")
                    else:
                        if mm.name == 'GRACEC':
                            fig.basemap(frame=["WSne+t%s"% 'GRACE-C', "xf1g1", "ya0.2f0.1g0.1+lNSE"],
                                        region=[0, 7, 0.2, 1],
                                        # Set a logarithmic transformation on the x-axis
                                        projection="X6c/6c")
                        else:
                            fig.basemap(frame=["WSne+t%s" % mm.name, "xf1g1", "ya0.2f0.1g0.1+lNSE"],
                                        region=[0, 7, 0.2, 1],
                                        # Set a logarithmic transformation on the x-axis
                                        projection="X6c/6c")

                if i==7:
                    fig.basemap(frame=["WSne", "xf1g1a1+lFiltering strength from weak to strong", "yfg"],
                                region=[0, 7, 40, 110],
                                # Set a logarithmic transformation on the x-axis
                                projection="X6c/6c")

                fig.plot(
                    # Set the figures frame and color as well as
                    # annotations, ticks, and gridlines
                    x=[5, 5],
                    y=[-1000, 1000],
                    cmap=True,
                    pen="4p,red3,--",
                )

                fig.plot(
                    # Set the figures frame and color as well as
                    # annotations, ticks, and gridlines
                    x=[4, 4],
                    y=[-1000, 1000],
                    cmap=True,
                    pen="4p,gold,--",
                )

                if QI==2:
                    data = [[2.5, 40, 2.6, 100]]
                else:
                    data = [[2.5, 0.5, 2.6, 100]]
                fig.plot(data=data, style="r", fill="lightgray@40")

                # data = [[2.99 + 5, 0.5, 2.05, 10]]
                # fig.plot(data=data, style="r", fill="lightgray@40")
                #
                # data = [[2.99 + 10, 0.5, 2.05, 10]]
                # fig.plot(data=data, style="r", fill="lightgray@40")

                x = [1, 2, 3, 4, 5, 6]
                if basin == 'Brahmaputra':
                    metric = metric_Brahmaputra
                else:
                    metric = metric_Danube

                y = metric['%s' % label[QI]][mm]
                fig.plot(
                    # Set the figures frame and color as well as
                    # annotations, ticks, and gridlines
                    x=x,
                    y=y,
                    cmap=True,
                    pen="2p,seagreen",
                )
                fig.plot(x=x, y=y, style="s0.5c", pen="1p", fill="seagreen@30")

                y = metric['%s_2' % label[QI]][mm]
                fig.plot(
                    # Set the figures frame and color as well as
                    # annotations, ticks, and gridlines
                    x=x,
                    y=y,
                    cmap=True,
                    pen="2p,dodgerblue",
                )
                fig.plot(x=x, y=y, style="c0.4c", pen="1p", fill="dodgerblue@30")

                if i % 3 != 2:
                    fig.shift_origin(xshift='7c')
                else:
                    fig.shift_origin(xshift='-14c', yshift='-7.2c')
                i += 1

    fig.shift_origin(yshift='29.5c')
    fig.basemap(frame=["WSne+glavender"],
                region=[0, 40, 0, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X19c/1.2c")

    fig.plot(
        x=[1, 4], y=[0.5, 0.5], pen="2p,dodgerblue", no_clip=True
    )
    fig.plot(
        x=2.5, y=0.5, style="c0.6c", pen="1p", fill="dodgerblue"
    )
    fig.text(x=12, y=0.5, text='DA with realistic covariance')

    fig.plot(
        x=[20.5, 23.5], y=[0.5, 0.5], pen="2p,seagreen", no_clip=True
    )
    fig.plot(
        x=22, y=0.5, style="s0.8c", pen="1p", fill="seagreen"
    )
    fig.text(x=32, y=0.5, text='DA with nominal covariance')

    fig.savefig(_OUT[0])

    pass


def scatter_plot_cov_comparison_Danube():
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data
    metric_Brahmaputra = {
        'NSE': {},
        'KGE': {},
        'RMSE':{},
        'NSE_2': {},
        'KGE_2': {},
        'RMSE_2': {}
    }

    metric_Danube = {
        'NSE': {},
        'KGE': {},
        'RMSE': {},
        'NSE_2': {},
        'KGE_2': {},
        'RMSE_2': {}
    }

    a, b, c, d, e, f = scatter_plot_cov_comparison_1(basins=['Brahmaputra'])
    metric_Brahmaputra['NSE'] = a
    metric_Brahmaputra['KGE'] = b
    metric_Brahmaputra['RMSE'] = c
    metric_Brahmaputra['NSE_2'] = d
    metric_Brahmaputra['KGE_2'] = e
    metric_Brahmaputra['RMSE_2'] = f

    g, h, i, j,k,l = scatter_plot_cov_comparison_1(basins=['Danube'])
    metric_Danube['NSE'] = g
    metric_Danube['KGE'] = h
    metric_Danube['RMSE'] = i
    metric_Danube['NSE_2'] = j
    metric_Danube['KGE_2'] = k
    metric_Danube['RMSE_2'] = l

    """===========================Start ploting================================"""
    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="19p,5", MAP_TITLE_OFFSET=0.2, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='16p,5',
                 FONT_LABEL='17p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar+h0', series=[-0.3, 0.3], background='o')

    label = ['NSE', 'KGE', 'RMSE']
    i = 0

    # for basin in ['Brahmaputra', 'Danube']:
    for basin in ['Danube']:
        for QI in [0, 1, 2]:
            for mm in [mission.GRACEC, mission.NGGM, mission.MAGIC]:

                if QI==2:
                    fig.basemap(frame=["WSne", "xf1g1a1", "yfg"],
                                region=[0, 7, 0, 70],
                                # Set a logarithmic transformation on the x-axis
                                projection="X6c/6c")
                elif QI==1:
                    fig.basemap(frame=["WSne", "xf1g1", "yf0.1g0.1"],
                                region=[0, 7, 0.2, 1],
                                # Set a logarithmic transformation on the x-axis
                                projection="X6c/6c")
                else:
                    if mm.name=='GRACEC':
                        fig.basemap(frame=["WSne+t%s"% 'GRACE-C', "xf1g1", "yf0.1g0.1"],
                                    region=[0, 7, 0.2, 1],
                                    # Set a logarithmic transformation on the x-axis
                                    projection="X6c/6c")
                    else:
                        fig.basemap(frame=["WSne+t%s" % mm.name, "xf1g1", "yf0.1g0.1"],
                                    region=[0, 7, 0.2, 1],
                                    # Set a logarithmic transformation on the x-axis
                                    projection="X6c/6c")

                if i%3==0:
                    if QI==2:
                        fig.basemap(frame=["WSne", "xf1g1a1", "yafg+lRMSE [mm]"],
                                    region=[0, 7, 0, 70],
                                    # Set a logarithmic transformation on the x-axis
                                    projection="X6c/6c")
                    elif QI==1:
                        fig.basemap(frame=["WSne", "xf1g1", "ya0.2f0.1g0.1+lKGE"],
                                    region=[0, 7, 0.2, 1],
                                    # Set a logarithmic transformation on the x-axis
                                    projection="X6c/6c")
                    else:
                        if mm.name=='GRACEC':
                            fig.basemap(frame=["WSne+t%s"% 'GRACE-C', "xf1g1", "ya0.2f0.1g0.1+lNSE"],
                                        region=[0, 7, 0.2, 1],
                                        # Set a logarithmic transformation on the x-axis
                                        projection="X6c/6c")
                        else:
                            fig.basemap(frame=["WSne+t%s"% mm.name, "xf1g1", "ya0.2f0.1g0.1+lNSE"],
                                        region=[0, 7, 0.2, 1],
                                        # Set a logarithmic transformation on the x-axis
                                        projection="X6c/6c")

                if i==7:
                    fig.basemap(frame=["WSne", "xf1g1a1+lFiltering strength from weak to strong", "yfg"],
                                region=[0, 7, 0, 70],
                                # Set a logarithmic transformation on the x-axis
                                projection="X6c/6c")



                # fig.plot(
                #     # Set the figures frame and color as well as
                #     # annotations, ticks, and gridlines
                #     x=[4, 4],
                #     y=[-1000, 1000],
                #     cmap=True,
                #     pen="4p,gold,--",
                # )

                # if QI==2:
                #     data = [[2.5, 40, 2.6, 100]]
                # else:
                #     data = [[2.5, 0.5, 2.6, 100]]
                # fig.plot(data=data, style="r", fill="lightgray@40")

                fig.plot(
                    # Set the figures frame and color as well as
                    # annotations, ticks, and gridlines
                    x=[4, 4],
                    y=[-1000, 1000],
                    cmap=True,
                    pen="4p,red3,--",
                )

                # data = [[2.99 + 5, 0.5, 2.05, 10]]
                # fig.plot(data=data, style="r", fill="lightgray@40")
                #
                # data = [[2.99 + 10, 0.5, 2.05, 10]]
                # fig.plot(data=data, style="r", fill="lightgray@40")

                x = [1, 2, 3, 4, 5, 6]
                if basin == 'Brahmaputra':
                    metric = metric_Brahmaputra
                else:
                    metric = metric_Danube

                y = metric['%s' % label[QI]][mm]
                fig.plot(
                    # Set the figures frame and color as well as
                    # annotations, ticks, and gridlines
                    x=x,
                    y=y,
                    cmap=True,
                    pen="2p,seagreen",
                )
                fig.plot(x=x, y=y, style="s0.5c", pen="1p", fill="seagreen@30")

                y = metric['%s_2' % label[QI]][mm]
                fig.plot(
                    # Set the figures frame and color as well as
                    # annotations, ticks, and gridlines
                    x=x,
                    y=y,
                    cmap=True,
                    pen="2p,dodgerblue",
                )
                fig.plot(x=x, y=y, style="c0.4c", pen="1p", fill="dodgerblue@30")

                if i % 3 != 2:
                    fig.shift_origin(xshift='7c')
                else:
                    fig.shift_origin(xshift='-14c', yshift='-7.2c')
                i += 1

    fig.shift_origin(yshift='29.5c')
    fig.basemap(frame=["WSne+glavender"],
                region=[0, 40, 0, 1],
                # Set a logarithmic transformation on the x-axis
                projection="X19c/1.2c")

    fig.plot(
        x=[1, 4], y=[0.5, 0.5], pen="2p,dodgerblue", no_clip=True
    )
    fig.plot(
        x=2.5, y=0.5, style="c0.6c", pen="1p", fill="dodgerblue"
    )
    fig.text(x=12, y=0.5, text='DA with realistic covariance')

    fig.plot(
        x=[20.5, 23.5], y=[0.5, 0.5], pen="2p,seagreen", no_clip=True
    )
    fig.plot(
        x=22, y=0.5, style="s0.8c", pen="1p", fill="seagreen"
    )
    fig.text(x=32, y=0.5, text='DA with nominal covariance')

    fig.savefig(_OUT[0])

    pass

def RMSE_2D(a: np.ndarray, b: np.ndarray):
    """

    :param a: three dimension [N,M,K]. N--time, M--lat, K--lon
    :param b: same as b
    :return: [M,K]
    """

    rmse = np.std(a - b, axis=0)
    return rmse


def CC_2D(a: np.ndarray, b: np.ndarray):
    """

    :param a: three dimension [N,M,K]. N--time, M--lat, K--lon
    :param b: same as b
    :return: [M,K]
    """
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)
    ff = np.sum(a * b, axis=0)
    d1 = np.sqrt(np.sum(a ** 2, axis=0))
    d2 = np.sqrt(np.sum(b ** 2, axis=0))

    r = ff / d1 / d2
    return r


def NSE_2D(a: np.ndarray, b: np.ndarray):
    """
    a: model, b: obs
    :param a: three dimension [N,M,K]. N--time, M--lat, K--lon
    :param b: same as b
    :return: [M,K]
    """
    return 1 - np.sum((a - b) ** 2, axis=0) / (np.sum(b ** 2, axis=0))


def KGE_2D(a: np.ndarray, b: np.ndarray):
    """

    :param a: three dimension [N,M,K]. N--time, M--lat, K--lon
    :param b: same as b
    :return: [M,K]
    """

    r = CC_2D(a, b)
    alpha = np.std(a, axis=0) / np.std(b, axis=0)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2)


def butterworth_2D(data_in, cutoff=30, sr=1 / 5, btype='high'):
    from scipy import signal
    '''filter'''
    # cy # days
    order = 3
    # cutoff = 365 * cy  # day
    # sr = 1 / 5  # day
    cutoff = 1 / (cutoff)
    wn = 2 * cutoff / sr
    assert wn <= 1
    b, a = signal.butter(order, wn, btype)
    # b, a = signal.butter(order, wn, 'lowpass')
    dd = data_in

    new_data = np.full_like(data_in, fill_value=np.nan)

    for n in range(new_data.shape[1]):
        for m in range(new_data.shape[2]):
            if np.isnan(dd[0, n, m]):
                continue
            data_out = signal.filtfilt(b, a, dd[:, n, m])
            new_data[:, n, m] = data_out
    return new_data


def RMS_spatial_map():
    from paper_RSE.PCA import PCA_EOF
    from spatial_analysis import truth, obs, OL_DA
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data

    RMS = []

    i = -1
    fig_label = ['(a)', '(d)', '(b)', '(e)', '(c)', '(f)', '(g)', '(j)', '(h)', '(k)', '(i)', '(l)']

    basin = 'Brahmaputra'
    # basin = 'Danube'
    mm = mission.MAGIC
    ff = filter.F1E17
    gg = grid.G1
    dd = data.OL


    if basin == 'Danube':
        gdf = gpd.read_file((FP.SHP + '/DRB_3_shapefiles'))
    else:
        gdf = gpd.read_file(
            (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    tt2 = truth(region=basin)
    b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    his_mask, lat, lon = tt2.shrink_mask()
    a = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
    a[:, his_mask] = b

    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="17p,5", MAP_TITLE_OFFSET=-0.5, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='27p,5',
                 FONT_LABEL='25p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='jet', series=[20, 50], background='o')

    ps = 'Q6c'
    region = [min(lon), max(lon), min(lat), max(lat)]
    region2 = [min(lon) - 0.5, max(lon) + 0.5, min(lat) - 0.5, max(lat) + 0.5]
    lon, lat = np.meshgrid(lon, lat)

    for mm in [mission.GRACEC, mission.NGGM, mission.MAGIC]:
        for ff in filter:


            tt3 = obs(region=basin)
            b = tt3.get_value_by_index(begin_index=634, end_ndex=634 + 242, mm=mm, ff=ff)
            his_mask, lat, lon = tt2.shrink_mask()
            a2 = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
            a2[:, his_mask] = b

            RMS2d = np.std(a-a2, axis=0)

            RMS.append(RMS2d)

            '''figure'''
            lon, lat = np.meshgrid(lon, lat)
            ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=RMS2d.flatten(),
                               spacing=0.5, region=region)

            fig.grdimage(
                grid=ee,
                cmap=True,
                frame=['+t%s' % (ff.name)],
                # dpi=100,
                projection=ps,
                region=region2,
                # interpolation='b',
                # nan_transparent=True
            )
            # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
            fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)

            if ff==filter.F1E17 and mm.name=='MAGIC':
                fig.colorbar(frame='xa10f5+lRMSE [mm]', position="JBC+w25c/0.35c+h+o-5c/2c")
            fig.shift_origin(xshift='7c')
            if ff==filter.F1E19:
                fig.shift_origin(xshift='-42c',yshift='-8c')



    fig.savefig('%s_filter_RMSE.png'%basin)
    fig.savefig(_OUT[0])

    pass


def RMS_spatial_map_Danube():
    from paper_RSE.PCA import PCA_EOF
    from spatial_analysis import truth, obs, OL_DA
    from paper_RSE.Analysis2 import basin_wise_TWS_DA_OL, mission, grid, filter, data

    RMS = []

    i = -1
    fig_label = ['(a)', '(d)', '(b)', '(e)', '(c)', '(f)', '(g)', '(j)', '(h)', '(k)', '(i)', '(l)']

    # basin = 'Brahmaputra'
    basin = 'Danube'
    mm = mission.MAGIC
    ff = filter.F1E17
    gg = grid.G1
    dd = data.OL


    if basin == 'Danube':
        gdf = gpd.read_file((FP.SHP + '/DRB_3_shapefiles'))
    else:
        gdf = gpd.read_file(
            (FP.SHP + '/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp'))

    tt2 = truth(region=basin)
    b = tt2.get_value_by_index(begin_index=634, end_ndex=634 + 242)
    his_mask, lat, lon = tt2.shrink_mask()
    a = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
    a[:, his_mask] = b

    fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL='20p', FONT_ANNOT_PRIMARY='9p', FONT_TITLE='14p', MAP_TITLE_OFFSET=0.1)
    pygmt.config(FONT_TITLE="17p,5", MAP_TITLE_OFFSET=-0.5, MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='27p,5',
                 FONT_LABEL='25p,5', MAP_TICK_LENGTH='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='jet', series=[10, 40], background='o')

    ps = 'Q6c'
    region = [min(lon), max(lon), min(lat), max(lat)]
    region2 = [min(lon) - 0.5, max(lon) + 0.5, min(lat) - 0.5, max(lat) + 0.5]
    lon, lat = np.meshgrid(lon, lat)

    for mm in [mission.GRACEC, mission.NGGM, mission.MAGIC]:
        for ff in filter:


            tt3 = obs(region=basin)
            b = tt3.get_value_by_index(begin_index=634, end_ndex=634 + 242, mm=mm, ff=ff)
            his_mask, lat, lon = tt2.shrink_mask()
            a2 = np.full(shape=(b.shape[0], his_mask.shape[0], his_mask.shape[1]), fill_value=np.nan)
            a2[:, his_mask] = b

            RMS2d = np.std(a-a2, axis=0)

            RMS.append(RMS2d)

            '''figure'''
            lon, lat = np.meshgrid(lon, lat)
            ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=RMS2d.flatten(),
                               spacing=0.5, region=region)

            fig.grdimage(
                grid=ee,
                cmap=True,
                frame=['+t%s' % (ff.name)],
                # dpi=100,
                projection=ps,
                region=region2,
                # interpolation='b',
                # nan_transparent=True
            )
            # fig.coast(shorelines="1/0.2p", region=region2, projection=ps)
            fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="1p,black", region=region2, projection=ps)

            if ff==filter.F1E17 and mm.name=='MAGIC':
                fig.colorbar(frame='xa10f5+lRMSE [mm]', position="JBC+w25c/0.35c+h+o-5c/2c")
            fig.shift_origin(xshift='7c')
            if ff==filter.F1E19:
                fig.shift_origin(xshift='-42c',yshift='-5c')



    fig.savefig('%s_filter_RMSE.png'%basin)
    fig.savefig(_OUT[0])

    pass


