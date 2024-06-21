import numpy as np
from pathlib import Path
import pygmt
import h5py
from src_hydro.GeoMathKit import GeoMathKit


def model_component_comparison_to_Leire():
    import scipy
    import mat73
    """
    :return:
    """

    statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf']

    '''load my result: MDB'''
    hf = h5py.File('../temp/basin_ts.h5', 'r')
    fan = hf['basin_0']
    date_begin = '2000-01-01'
    date_end = '2022-12-31'
    daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)
    day_first = daylist[0].year + (daylist[0].month - 1) / 12 + daylist[0].day / 365.25
    fan_time = day_first + np.arange(len(daylist)) / 365.25

    '''load lerie's result'''
    w3ra_mat = mat73.loadmat(
        '/media/user/My Book/Fan/W3/exp/comparison_GRACE_W3_Aus/COMP_OL_2000_2015/COMP_OL_2000_2015.mat')
    leire_time = 2000 + np.arange(5846) / 365.25
    Leire = {}
    for state in statesnn:
        Leire[state] = np.mean(np.mean(w3ra_mat['%s_subb' % state], 2), 0)

    '''comparison'''
    fig = pygmt.Figure()
    for state in ['S0', 'Ss', 'Sd']:
        if state == 'S0':
            fig.basemap(region=[2000 - 0.5, 2023.5, -10, 80], projection='X12c/3c',
                        frame=["WSne", "xa2f1", 'ya20f10+lwater [mm]'])
        elif state == 'Ss':
            fig.basemap(region=[2000 - 0.5, 2023.5, -5, 35], projection='X12c/3c',
                        frame=["WSne", "xa2f1", 'ya10f5+lwater [mm]'])
        else:
            fig.basemap(region=[2000 - 0.5, 2023.5, 120, 220], projection='X12c/3c',
                        frame=["WSne", "xa2f1", 'ya20f10+lwater [mm]'])

        fig.plot(x=fan_time, y=fan[state][:], pen="0.5p,black", label='Fan %s' % (state))
        fig.plot(x=leire_time, y=Leire[state], pen="0.5p,red", label='Leire %s' % (state))
        fig.legend(position='jTR', box='+gwhite+p0.5p')
        fig.shift_origin(yshift='-4c')

    fig.shift_origin(yshift='12c', xshift='14c')
    for state in ['Sr', 'Sg', 'Mleaf']:
        if state == 'Sg':
            fig.basemap(region=[2000 - 0.5, 2023.5, -5, 35], projection='X12c/3c',
                        frame=["WSne", "xa2f1", 'ya10f5+lwater [mm]'])
        elif state == 'Sr':
            fig.basemap(region=[2000 - 0.5, 2023.5, -2, 6], projection='X12c/3c',
                        frame=["WSne", "xa2f1", 'ya2f1+lwater [mm]'])
        else:
            fig.basemap(region=[2000 - 0.5, 2023.5, -2, 6], projection='X12c/3c',
                        frame=["WSne", "xa2f1", 'ya2f1+lwater [mm]'])

        fig.plot(x=fan_time, y=fan[state][:], pen="0.5p,black", label='Fan %s' % (state))
        fig.plot(x=leire_time, y=Leire[state], pen="0.5p,red", label='Leire %s' % (state))
        fig.legend(position='jTR', box='+gwhite+p0.5p')
        fig.shift_origin(yshift='-4c')

    # fig.savefig('../temp/FanVSLeire.pdf')
    fig.show()

    pass


def compariosn_to_GRACE():
    statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf']

    '''load my result: MDB'''
    date_begin = '2000-01-01'
    date_end = '2022-12-31'
    daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)
    day_first = daylist[0].year + (daylist[0].month - 1) / 12 + daylist[0].day / 365.25
    fan_time = day_first + np.arange(len(daylist)) / 365.25

    hf = h5py.File('../temp/basin_ts.h5', 'r')
    tws_dict = {}
    for basin in hf.keys():
        tws = []
        fan = hf[basin]
        for state in statesnn:
            a = fan[state][:]
            tws.append(a)

        tws_dict[basin] = np.sum(np.array(tws), axis=0)

    hf.close()

    '''load src_GRACE'''
    data_path = Path('/media/user/My Book/Fan/W3/exp/comparison_GRACE_W3_Aus')
    grace = h5py.File(data_path / 'src_GRACE.hdf5', 'r')
    grace_time = grace['year_fraction'][:]

    '''figure'''
    basins = ['MDB', 'MDB_sub_1', 'MDB_sub_2', 'MDB_sub_3', 'MDB_sub_4']
    fig = pygmt.Figure()
    for id in [1, 2]:
        No = '%s' % (id)
        grace_tws = grace['MDB']['ewha'][No][:]
        grace_scale_factor = np.array(grace['MDB']['scale_factor'][No]).min()
        # grace_scale_factor = 1
        basin = basins[id]
        fig.basemap(region=[2002, 2023.5, -150, 200], projection='X12c/3c',
                    frame=["WSne+t%s" % basin, "xa2f1", 'ya100f20+lwater storage [mm]'])

        w3ra = tws_dict['basin_%s' % id]
        fig.plot(x=fan_time, y=w3ra - np.mean(w3ra), pen="0.5p,black", label='W3RA')

        fig.plot(x=grace_time, y=grace_tws * 1000 / grace_scale_factor, pen="0.5p,blue", label='src_GRACE')
        fig.legend(position='jBR', box='+gwhite+p0.5p')

        fig.shift_origin(yshift='-4.5c')

    fig.shift_origin(yshift='9c', xshift='14c')
    for id in [3, 4, 0]:
        No = '%s' % (id)
        grace_tws = grace['MDB']['ewha'][No][:]
        grace_scale_factor = np.array(grace['MDB']['scale_factor'][No]).min()
        # grace_scale_factor = 1
        basin = basins[id]
        fig.basemap(region=[2002, 2023.5, -150, 200], projection='X12c/3c',
                    frame=["WSne+t%s" % basin, "xa2f1", 'ya100f20+lwater storage [mm]'])

        w3ra = tws_dict['basin_%s' % id]
        fig.plot(x=fan_time, y=w3ra - np.mean(w3ra), pen="0.5p,black", label='W3RA')

        fig.plot(x=grace_time, y=grace_tws * 1000 / grace_scale_factor, pen="0.5p,blue", label='src_GRACE')
        fig.legend(position='jBR', box='+gwhite+p0.5p')

        fig.shift_origin(yshift='-4.5c')

    # fig.savefig('../temp/W3RAvsGRACE.pdf')
    # fig.savefig('../temp/W3RAvsGRACE.png')
    fig.show()

    pass


def monthly_update_daily_update():
    from src_DA.Analysis import Postprocessing_basin
    import pygmt
    import h5py
    from src_hydro.GeoMathKit import GeoMathKit
    import json
    from pathlib import Path
    import numpy as np
    import scipy.signal as signal

    exp_prefix = 'exp_new'

    '''load OL result'''
    ens = 30
    case = 'Sg'
    basin = 'DRB'

    pp = Postprocessing_basin(ens=ens, case=case, basin=basin)
    states_traditional = pp.load_states(load_dir='/home/user/Desktop/res', prefix='exp_traditional_%s' % basin)
    states_new = pp.load_states(load_dir='/home/user/Desktop/res', prefix='exp_new_%s' % basin)
    GRACE = pp.load_GRACE(load_dir='/home/user/Desktop/res', prefix='exp_new_%s' % basin)
    states_OL = pp.load_states(load_dir='/home/user/Desktop/res', prefix='OL_%s' % basin)

    OL_time = states_OL['time']
    DA_time = states_new['time']
    GR_time = GRACE['time']

    basin_num = len(list(states_OL.keys())) - 1

    '''plot figure'''
    fig = pygmt.Figure()

    i = 0
    offset = 3
    height = 6
    signal = 'TWS'
    basin_id = 'basin_0'

    GRACE = GRACE['original'][basin_id]
    OL = states_OL[basin_id][signal][0]
    DA_tradional = np.mean(np.array(list(states_traditional[basin_id][signal].values()))[1:, ], axis=0)
    DA_new = np.mean(np.array(list(states_new[basin_id][signal].values()))[1:, ], axis=0)

    values = [GRACE, OL, DA_tradional, DA_new]
    vvmin = []
    vvmax = []
    for vv in values:
        vvmin.append(np.min(vv[5:]))
        vvmax.append(np.max(vv[5:]))

    vmin, vmax = min(vvmin), min(vvmax)
    dmin = vmin - (vmax - vmin) * 0.1
    dmax = vmax + (vmax - vmin) * 0.1
    sp_1 = int(np.round((vmax - vmin) / 10))
    if sp_1 == 0:
        sp_1 = 0.5
    sp_2 = sp_1 * 2

    fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X18c/5c',
                frame=["WSne+t%s" % (basin + '_' + signal + '_' + basin_id), "xa2f1",
                       'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

    fig.plot(x=OL_time, y=OL, pen="0.5p,grey,-", label='%s' % ('OL'), transparency=30)
    fig.plot(x=GR_time, y=GRACE, style="c.1c", fill="black", label='%s' % ('GRACE'), transparency=30)
    fig.plot(x=DA_time, y=DA_tradional, pen="0.5p,green", label='%s' % ('DA_old'), transparency=30)
    fig.plot(x=DA_time, y=DA_new, pen="0.5p,red", label='%s' % ('DA_new'), transparency=30)

    # fig.legend(position='jTR', box='+gwhite+p0.5p')
    fig.legend(position='jBL')

    fig.shift_origin(yshift='-%sc' % height)

    dmin = -8
    dmax = 8
    sp_2 = 2
    sp_1 = 1

    fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X18c/5c',
                frame=["WSne", "xa2f1",
                       'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

    fig.plot(x=DA_time, y=DA_new - DA_tradional, pen="1p,black", label='%s' % ('DA_new minus DA_old'))
    fig.legend(position='jBL')

    fig.savefig('/home/user/Desktop/res/test.png')

    fig.show()

    pass


def map2D_comparison():
    from src_auxiliary.ts import ts
    from src_auxiliary.upscaling import upscaling
    from datetime import datetime
    from src_hydro.GeoMathKit import GeoMathKit
    import geopandas as gpd

    signal = 'trend'

    '''load DA'''
    hf = h5py.File('/home/user/Desktop/res/monthly_mean_TWS_DRB_DA.h5', 'r')

    time = []
    vv = []

    for x, v in hf.items():
        m = x.split('-')
        n = datetime(year=int(m[0]), month=int(m[1]), day=15)
        time.append(GeoMathKit.year_fraction(n))
        vv.append(v[:])
        pass

    tsd = ts().set_period(time=np.array(time)).setDecomposition()
    res_DA = tsd.getSignal(obs=vv)

    '''load OL'''
    hf = h5py.File('/home/user/Desktop/res/monthly_mean_TWS_DRB_uOL.h5', 'r')

    time = []
    vv = []

    for x, v in hf.items():
        m = x.split('-')
        n = datetime(year=int(m[0]), month=int(m[1]), day=15)
        time.append(GeoMathKit.year_fraction(n))
        vv.append(v[:])
        pass

    tsd = ts().set_period(time=np.array(time)).setDecomposition()
    res_OL = tsd.getSignal(obs=vv)

    '''load GRACE'''
    hf = h5py.File('/home/user/Desktop/res/DRB_gridded_signal.hdf5', 'r')
    gr_time = hf['time_epoch'][:].astype(str)
    gr_data = hf['tws'][:]
    time = []

    for i in range(len(gr_time)):
        m = gr_time[i].split('-')
        n = datetime(year=int(m[0]), month=int(m[1]), day=int(m[2]))
        time.append(GeoMathKit.year_fraction(n))

    tsd = ts().set_period(time=np.array(time)).setDecomposition()
    res_GRACE = tsd.getSignal(obs=gr_data)
    xx = res_GRACE[signal]
    if signal != 'trend':
        xx = xx[0]

    '''1d --> 2d map'''
    us = upscaling(basin='DRB')
    us.configure_model_state()
    us.configure_GRACE()
    # us.downscale_model(model_state=1, GRACEres=0.5, modelres=0.1)
    res_GRACE = us.get2D_GRACE(GRACE_obs=xx)

    xx = res_DA[signal]
    if signal != 'trend':
        xx = xx[0]
    res_DA = us.get2D_model_state(model_state=xx)
    '''mask'''
    res_DA[us.bshp_mask == False] = np.nan

    xx = res_OL[signal]
    if signal != 'trend':
        xx = xx[0]
    res_OL = us.get2D_model_state(model_state=xx)
    '''mask'''
    res_OL[us.bshp_mask == False] = np.nan

    '''plot figure'''
    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')
    # pygmt.makecpt(cmap='jet', series=[0, 70, 5], background='o')
    pygmt.makecpt(cmap='polar+h0', series=[-5, 2, 0.5], background='o')

    res = 0.5
    err = res / 10
    lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
    region = [min(lon), max(lon), min(lat), max(lat)]
    lon, lat = np.meshgrid(lon, lat)

    DA = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_DA.flatten(),
                       spacing=(res, res), region=region)

    OL = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_OL.flatten(),
                       spacing=(res, res), region=region)

    GRACE = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_GRACE.flatten(),
                          spacing=(res, res), region=region)

    region = [7, 32, 40, 52]
    fig.grdimage(
        grid=OL,
        cmap=True,
        frame=['xa5f5g5', 'ya5f5g5'] + ['+tOL'],
        dpi=100,
        projection='Q12c',
        region=region,
        interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection="Q12c")

    gdf = gpd.read_file('../data/basin/shp/DRB_3_shapefiles')
    fig.plot(data=gdf.boundary, pen="1p,black")

    # if signal != 'trend':
    #     fig.colorbar(frame='af+l%s amplitude [mm]' % signal, position="JBC+w13c/0.45c+h+o7c/1.2c")
    # else:
    #     fig.colorbar(frame='af+l%s [mm/yr]' % signal)

    fig.shift_origin(xshift='14c')
    fig.grdimage(
        grid=DA,
        cmap=True,
        frame=['xa5f5g5', 'ya5f5g5'] + ['+tDA'],
        dpi=100,
        projection='Q12c',
        region=region,
        interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection="Q12c")

    gdf = gpd.read_file('../data/basin/shp/DRB_3_shapefiles')
    fig.plot(data=gdf.boundary, pen="1p,black")
    # fig.colorbar()

    fig.shift_origin(xshift='14c')
    fig.grdimage(
        grid=GRACE,
        cmap=True,
        frame=['xa5f5g5', 'ya5f5g5'] + ['+tGRACE'],
        dpi=100,
        projection='Q12c',
        region=region,
        interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection="Q12c")

    gdf = gpd.read_file('../data/basin/shp/DRB_3_shapefiles')
    fig.plot(data=gdf.boundary, pen="1p,black")

    '''===================================================='''
    '''filtered'''
    fig.shift_origin(xshift='-28c', yshift='-8c')

    fig.grdimage(
        grid=OL,
        cmap=True,
        frame=['xa5f5g5', 'ya5f5g5'] + ['+tOL smoothed'],
        dpi=100,
        projection='Q12c',
        region=region,
        # interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection="Q12c")

    gdf = gpd.read_file('../data/basin/shp/DRB_3_shapefiles')
    fig.plot(data=gdf.boundary, pen="1p,black")

    if signal != 'trend':
        fig.colorbar(frame='a10f10+l%s amplitude [mm]' % signal, position="JBC+w13c/0.45c+h+o14c/1.2c")
    else:
        fig.colorbar(frame='af+l%s [mm/yr]' % signal, position="JBC+w13c/0.45c+h+o14c/1.2c")
        pass

    fig.shift_origin(xshift='14c')
    fig.grdimage(
        grid=DA,
        cmap=True,
        frame=['xa5f5g5', 'ya5f5g5'] + ['+tDA, smoothed'],
        dpi=100,
        projection='Q12c',
        region=region,
        # interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection="Q12c")

    gdf = gpd.read_file('../data/basin/shp/DRB_3_shapefiles')
    fig.plot(data=gdf.boundary, pen="1p,black")
    # fig.colorbar()

    fig.shift_origin(xshift='14c')
    fig.grdimage(
        grid=GRACE,
        cmap=True,
        frame=['xa5f5g5', 'ya5f5g5'] + ['+tGRACE, smoothed'],
        dpi=100,
        projection='Q12c',
        region=region,
        # interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection="Q12c")

    gdf = gpd.read_file('../data/basin/shp/DRB_3_shapefiles')
    fig.plot(data=gdf.boundary, pen="1p,black")

    fig.show()

    pass


def tile_GRACE_1():
    import pygmt
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from src_DA.shp2mask import basin_shp_process

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='7p', COLOR_NAN='white')
    pygmt.makecpt(cmap='jet', series=[-200, 200], background='o')

    region = [-90, 20, -30, 20]

    fig.coast(shorelines="1/0.2p", region=region, projection="Q9c", frame=['xa30f15', 'ya30f15'])

    '''new'''
    gdf_list = []
    invalid_basins = []
    tile = '80'
    gdf = gpd.read_file(filename='../res/global_shp_new/Tile%s_subbasins.shp' % tile)

    GRACE = h5py.File('/media/user/My Book/Fan/GRACE/output/Tile%s_signal.hdf5' % tile, 'r')

    res = 3
    err = res / 10
    lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
    region = [min(lon), max(lon), min(lat), max(lat)]
    lon, lat = np.meshgrid(lon, lat)

    fshp = '../res/global_shp_new/Tile%s_subbasins.shp' % tile
    mask = basin_shp_process(res=res, basin_name='Tile%s' % tile).shp_to_mask(shp_path=str(fshp), issave=False).mask

    x = np.full(shape=mask[0].shape, fill_value=np.nan)

    for i in range(1, 1 + len(gdf)):
        x[mask[i]] = GRACE['sub_basin_%s' % i][0]
        pass

    GR = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=x.flatten(),
                       spacing=(res, res), region=region)
    region = [-90, 20, -30, 20]
    fig.grdimage(
        grid=GR,
        cmap=True,
        frame=['xa5f5', 'ya5f5'] + ['+tOL'],
        dpi=100,
        projection='Q9c',
        region=region,
        interpolation='n'
    )

    fig.colorbar()
    fig.plot(data=gdf.boundary, pen="0.2p,black")
    fig.coast(shorelines="1/0.2p", region=region, projection="Q9c")
    fig.show()
    pass


def tile_GRACE_2():
    import geopandas as gpd
    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='7p', COLOR_NAN='white')
    pygmt.makecpt(cmap='jet', series=[-200, 200], background='o')

    region = [-90, 20, -30, 20]

    fig.coast(shorelines="1/0.2p", region=region, projection="Q9c", frame=['xa30f15', 'ya30f15'])

    res = 0.5
    err = res / 10
    lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
    region = [min(lon), max(lon), min(lat), max(lat)]
    lon, lat = np.meshgrid(lon, lat)

    a = h5py.File('/media/user/My Book/Fan/GRACE/ewh/2002-04-17.hdf5', 'r')
    x = np.flipud(a['data'][:])

    GR = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=x.flatten(),
                       spacing=(res, res), region=region)
    region = [-90, 20, -30, 20]
    fig.grdimage(
        grid=GR,
        cmap=True,
        frame=['xa5f5', 'ya5f5'] + ['+tOL'],
        dpi=100,
        projection='Q9c',
        region=region,
        # interpolation='n'
    )
    fig.colorbar()

    tile = '80'
    gdf = gpd.read_file(filename='../res/global_shp_new/Tile%s_subbasins.shp' % tile)
    fig.plot(data=gdf.boundary, pen="0.2p,black")
    fig.coast(shorelines="1/0.2p", region=region, projection="Q9c")
    fig.show()
    pass


def Danube_exp():
    import pandas as pd
    import geopandas as gpd

    region = [5, 30, 40, 55]

    grid = pygmt.datasets.load_earth_relief(resolution="10m", region=region)
    fig = pygmt.Figure()
    fig.grdimage(grid=grid, projection="Q10c", frame="a5", cmap="geo")
    fig.coast(shorelines="1/0.2p", projection="Q10c", borders=["1/0.5p,black", "2/0.5p,red", "3/0.5p,blue"])
    fig.colorbar(position="jTR+o0.4c/0.4c+h+w5c/0.3c+ml",
                 # Add a box around the colobar with a fill ("+g") in "white" color and
                 # a transparency ("@") of 30 % and with a 0.8-points thick black
                 # outline ("+p")
                 frame=["a1500", "x", "y+lm"])

    geo_df = gpd.read_file('../data/basin/shp/DRB_3_shapefiles')
    fig.plot(data=geo_df.geometry, pen="1.5p,blue")
    fig.text(x=13, y=48, text='Upper Basin', fill='white@40')
    fig.text(x=20, y=46, text='Middle Basin', fill='white@40')
    fig.text(x=26, y=45, text='Lower Basin', fill='white@40')

    # fig.show()

    # fig.shift_origin(yshift='-10c')
    # # pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    # # pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')
    # # pygmt.makecpt(cmap='polar', series=[1, 11, 1], background='o')
    #
    #
    #
    # fig.coast(shorelines="1/0.2p", region=region, projection="Q8c")

    #
    fig.show()

    pass


def CorrelationGDRB():
    import h5py
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib
    import matplotlib as mpl

    hm = h5py.File('/media/user/My Book/Fan/GRACE/output/GDRB_cov.hdf5', 'r')
    i = 0
    cov = hm['data'][i, :, :]

    def heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw=None, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current Axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if ax is None:
            ax = plt.gca()

        if cbar_kw is None:
            cbar_kw = {}

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=25)
        font_size = 18  # Adjust as appropriate.
        cbar.ax.tick_params(labelsize=font_size)

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor", fontsize=20)
        plt.setp(ax.get_yticklabels(), fontsize=20)

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    def correlation_from_covariance(covariance):
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        return correlation

    fig, ax = plt.subplots()

    vegetables = list(range(1, 20))
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    # harvest = cov

    # corr_matrix = np.corrcoef(harvest)
    corr_matrix =correlation_from_covariance(cov)
    im, _ = heatmap(corr_matrix, vegetables, vegetables,
                    cmap="PuOr", vmin=-1, vmax=1,
                    cbarlabel="correlation coeff.")

    def func(x, pos):
        return f"{x:.2f}".replace("0.", ".").replace("1.00", "")

    annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=10)

    fig.tight_layout()
    plt.show()

    pass


def show_component_ensemble():
    import pygmt
    import h5py
    from src_hydro.GeoMathKit import GeoMathKit
    from src_DA.Analysis import Postprocessing_basin

    ens= 30
    case= 'RDA_DRB_BasinScale'
    basin = 'DRB'
    file_postfix = 'DA'

    '''collect basin average results'''
    pp = Postprocessing_basin(ens=ens, case=case, basin=basin,
                              date_begin='2002-03-31',
                              date_end='2010-01-31')

    # states = pp.get_states(post_fix=file_postfix, dir=self._outdir2)
    data_dir = '/home/user/Desktop/res1/RDA_DRB_BasinScale'
    states = pp.load_states(prefix=(file_postfix + '_' + basin), load_dir=data_dir)

    fan_time = states['time']

    '''plot figure'''
    fig = pygmt.Figure()
    statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'TWS']
    i = 0
    for state in statesnn:
        i += 1

        vv = states['basin_0'][state][0]

        vmin, vmax = np.min(vv[20:]), np.max(vv[20:])
        dmin = vmin - (vmax - vmin) * 0.5
        dmax = vmax + (vmax - vmin) * 0.5
        sp_1 = int(np.round((vmax - vmin) / 10))
        if sp_1 == 0:
            sp_1 = 0.5
        sp_2 = sp_1 * 2

        if i == 4:
            fig.shift_origin(yshift='12c', xshift='14c')
            pass

        fig.basemap(region=[fan_time[0] - 0.2, fan_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                    frame=["WSne", "xa2f1", 'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

        mm=np.zeros(len(states['basin_0'][state][0]))
        for es in reversed(range(1, ens + 1)):
            vv = states['basin_0'][state][es]

            if es == 0:
                fig.plot(x=fan_time, y=vv, pen="1p,blue", label='%s' % (state), transparency=30)
            else:
                fig.plot(x=fan_time, y=vv, pen="0.3p,grey")

            mm+=vv
        mm/=ens
        fig.plot(x=fan_time, y=mm, pen="1p,blue", label='%s' % (state), transparency=30)
        fig.legend(position='jTR', box='+gwhite+p0.5p')
        fig.shift_origin(yshift='-4c')

    # fig.savefig('/home/user/Desktop/DRB.png')
    pass


def show_subbasin():
    from src_DA.Analysis import Postprocessing_basin
    import pygmt
    import h5py
    from src_hydro.GeoMathKit import GeoMathKit
    import json
    from pathlib import Path
    import numpy as np
    from src_hydro.EnumType import init_mode
    from src_FlowControl.DA_GRACE import DA_GRACE

    signal = 'TWS'

    ens=30
    case = 'RDA_GDRB_BasinScale'
    basin = 'GDRB'
    '''for open-loop'''
    begin_time = '2002-01-01'
    end_time = '2010-01-31'
    data_dir = '/home/user/Desktop/res1/RDA_GDRB_BasinScale'

    '''load OL result'''
    file_postfix = 'OL'
    pp = Postprocessing_basin(ens=ens, case=case, basin=basin,
                              date_begin=begin_time,
                              date_end=end_time)
    # states_OL = pp.get_states(post_fix=file_postfix, dir=demo._outdir2)
    states_OL = pp.load_states(prefix=(file_postfix + '_' + basin), load_dir=data_dir)
    file_postfix = 'DA'
    states_DA = pp.load_states(prefix=(file_postfix + '_' + basin), load_dir=data_dir)

    '''load GRACE'''
    # dp_dir = Path(setting_dir) / 'DA_setting.json'
    # dp4 = json.load(open(dp_dir, 'r'))
    # GRACE = pp.get_GRACE(obs_dir=dp4['obs']['dir'])
    GRACE = pp.load_GRACE(prefix=(file_postfix + '_' + basin), load_dir=data_dir)

    OL_time = states_OL['time']
    DA_time = states_DA['time']
    GR_time = GRACE['time']

    basin_num = len(list(states_DA.keys())) - 1

    '''plot figure'''
    fig = pygmt.Figure()

    i = 0
    offset = 12
    height = 3.3

    keys = list(GRACE['ens_mean'].keys())
    keys

    for j in range(len(keys)):
        i += 1
        if j>0:break
        basin_id = 'basin_%s' % j
        GRACE_ens_mean = GRACE['ens_mean'][basin_id]
        GRACE_original = GRACE['original'][basin_id]
        OL = states_OL[basin_id][signal][0]
        OL_ens_mean = np.mean(np.array(list(states_OL[basin_id][signal].values()))[1:, ], axis=0)
        DA_ens_mean = np.mean(np.array(list(states_DA[basin_id][signal].values()))[1:, ], axis=0)

        values = [GRACE_ens_mean, OL, OL_ens_mean, DA_ens_mean]
        vvmin = []
        vvmax = []
        for vv in values:
            vvmin.append(np.min(vv[5:]))
            vvmax.append(np.max(vv[5:]))

        vmin, vmax = min(vvmin), min(vvmax)
        dmin = vmin - (vmax - vmin) * 0.1
        dmax = vmax + (vmax - vmin) * 0.6
        sp_1 = int(np.round((vmax - vmin) / 5))
        if sp_1 == 0:
            sp_1 = 0.5
        sp_2 = sp_1 * 2

        if j==3:
            ff = 'WSne'
        else:
            ff = 'Wsne'
        fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X12c/5c',
                    frame=[ff, "xa1f1",
                           'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

        # fig.plot(x=OL_time, y=OL, pen="0.5p,blue,-", label='%s' % ('OL_unperturbed'), transparency=30)
        fig.plot(x=OL_time, y=OL, pen="1p,grey", label='%s' % ('OL'), transparency=30)
        # fig.plot(x=OL_time, y=OL_ens_mean, pen="0.5p,red", label='%s' % ('OL_ens_mean'), transparency=30)
        fig.plot(x=DA_time, y=DA_ens_mean, pen="1p,green", label='%s' % ('DA'), transparency=30)
        # fig.plot(x=GR_time, y=GRACE_ens_mean, pen="0.5p,black", label='%s' % ('GRACE_ens_mean'), transparency=30)
        # fig.plot(x=GR_time, y=GRACE_original, pen="0.5p,purple,--.", label='%s' % ('GRACE_original'),
        #          transparency=30)
        fig.plot(x=GR_time, y=GRACE_original, style="c.15c", fill="black", label='%s' % ('GRACE'), transparency=30)
        if j==0:
            text = ''
        elif j==1:
            text = '(b) DRB-UB'
        elif j == 2:
            text = '(c) DRB-MB'
        elif j == 3:
            text = '(d) DRB-LB'

        fig.text(text=text, font="12p,Helvetica,black", fill='lightblue', justify='TR', position='TR',
                 pen='0.25p')
        # fig.legend(position='jTR', box='+gwhite+p0.5p')
        fig.legend(position='jBL')

        sf = '%sc' % ((offset - 1) * height)
        if i % offset == 0:
            fig.shift_origin(yshift=sf, xshift='14c')
            continue

        fig.shift_origin(yshift='-%sc' % height)

        pass

    # fig_postfix = '0'
    # fig.savefig(str(Path(figure_output) / ('DA_%s_%s_%s.pdf' % (basin, signal, fig_postfix))))
    # fig.savefig(str(Path(figure_output) / ('DA_%s_%s_%s.png' % (basin, signal, fig_postfix))))

    fig.show()

    pass


def show_trend_annual():
    from src_auxiliary.ts import ts
    from src_auxiliary.upscaling import upscaling
    from datetime import datetime
    from src_hydro.GeoMathKit import GeoMathKit
    import geopandas as gpd
    import h5py
    import numpy as np
    import json
    import os
    import pygmt

    signal='trend'
    # signal = 'annual'

    ens=30
    case = 'RDA_DRB_BasinScale'
    basin = 'DRB'
    '''for open-loop'''
    begin_time = '2002-01-01'
    end_time = '2010-01-31'
    res_output = '/home/user/Desktop/res1/RDA_DRB_BasinScale'
    shp_path = '../data/basin/shp/DRB_3_shapefiles/DRB_subbasins.shp'
    box = [50.5, 42, 8.5, 29.5]

    '''load shp'''
    gdf = gpd.read_file(shp_path)


    '''load DA'''
    hf = h5py.File(Path(res_output) / ('monthly_mean_TWS_%s_DA.h5' % basin), 'r')

    time = []
    vv = []

    for x, v in hf.items():
        m = x.split('-')
        n = datetime(year=int(m[0]), month=int(m[1]), day=15)
        time.append(GeoMathKit.year_fraction(n))
        vv.append(v[:])
        pass

    tsd = ts().set_period(time=np.array(time)).setDecomposition()
    res_DA = tsd.getSignal(obs=vv)

    '''load OL'''
    hf = h5py.File(Path(res_output) / ('monthly_mean_TWS_%s_uOL.h5' % basin), 'r')

    time = []
    vv = []

    for x, v in hf.items():
        m = x.split('-')
        n = datetime(year=int(m[0]), month=int(m[1]), day=15)
        time.append(GeoMathKit.year_fraction(n))
        vv.append(v[:])
        pass

    tsd = ts().set_period(time=np.array(time)).setDecomposition()
    res_OL = tsd.getSignal(obs=vv)

    '''load GRACE'''
    hf = h5py.File(('/media/user/My Book/Fan/GRACE/output/%s_gridded_signal.hdf5' % basin), 'r')
    gr_time = hf['time_epoch'][:].astype(str)
    gr_data = hf['tws'][:]
    time = []

    for i in range(len(gr_time)):
        m = gr_time[i].split('-')
        if len(m) == 2: m.append('15')
        n = datetime(year=int(m[0]), month=int(m[1]), day=int(m[2]))
        time.append(GeoMathKit.year_fraction(n))

    tsd = ts().set_period(time=np.array(time)).setDecomposition()
    res_GRACE = tsd.getSignal(obs=gr_data)
    xx = res_GRACE[signal]
    if signal != 'trend':
        xx = xx[0]

    '''1d --> 2d map'''
    us = upscaling(basin=basin)
    us.configure_model_state(globalmask_fn=Path(res_output) / 'mask_global.h5')
    us.configure_GRACE()
    # us.downscale_model(model_state=1, GRACEres=0.5, modelres=0.1)
    res_GRACE = us.get2D_GRACE(GRACE_obs=xx)

    xx = res_DA[signal]
    if signal != 'trend':
        xx = xx[0]
    res_DA = us.get2D_model_state(model_state=xx)
    '''mask'''
    res_DA[us.bshp_mask == False] = np.nan

    xx = res_OL[signal]
    if signal != 'trend':
        xx = xx[0]
    res_OL = us.get2D_model_state(model_state=xx)
    '''mask'''
    res_OL[us.bshp_mask == False] = np.nan

    '''plot figure'''
    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')
    #
    nan_mask = (1 - np.isnan(res_GRACE)).astype(bool)
    # vmax = np.nanmax(res_DA[nan_mask]) * 0.8
    # vmin = np.nanmin(res_DA[nan_mask]) * 0.8
    vmax = 5
    vmin = -10

    if signal == 'trend':
        pygmt.makecpt(cmap='polar+h0', series=[vmin, vmax], background='o')
    else:
        pygmt.makecpt(cmap='jet', series=[0, vmax], background='o')

    res = 0.5
    err = res / 10
    lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
    region = [min(lon), max(lon), min(lat), max(lat)]
    lon, lat = np.meshgrid(lon, lat)

    DA = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_DA.flatten(),
                       spacing=(res, res), region=region)

    OL = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_OL.flatten(),
                       spacing=(res, res), region=region)

    GRACE = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=res_GRACE.flatten(),
                          spacing=(res, res), region=region)

    region = [box[2] - 2, box[3] + 2, box[1] - 2, box[0] + 2]
    ps = 'Q8c'
    offset = '-6c'
    fig.grdimage(
        grid=OL,
        cmap=True,
        frame=['xa5f5', 'ya5f5'] + ['+t(a) OL'],
        dpi=100,
        projection=ps,
        region=region,
        interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)

    fig.plot(data=gdf.boundary, pen="1p,black")

    fig.shift_origin(xshift='10c')
    fig.grdimage(
        grid=DA,
        cmap=True,
        frame=['xa5f5', 'ya5f5'] + ['+t(b) DA'],
        dpi=100,
        projection=ps,
        region=region,
        interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)

    fig.plot(data=gdf.boundary, pen="1p,black")
    # fig.colorbar()

    fig.shift_origin(xshift='10c')
    fig.grdimage(
        grid=GRACE,
        cmap=True,
        frame=['xa5f5', 'ya5f5'] + ['+t(c) GRACE'],
        dpi=100,
        projection=ps,
        region=region,
        interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)

    fig.plot(data=gdf.boundary, pen="1p,black")

    '''===================================================='''
    '''filtered'''
    fig.shift_origin(xshift='-20c', yshift=offset)

    fig.grdimage(
        grid=OL,
        cmap=True,
        frame=['xa5f5', 'ya5f5'] + ['+t(d) OL smoothed'],
        dpi=100,
        projection=ps,
        region=region,
        # interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)

    fig.plot(data=gdf.boundary, pen="1p,black")

    if signal != 'trend':
        fig.colorbar(frame='a20f10+l%s amplitude [mm]' % signal, position="JBC+w13c/0.45c+h+o10c/1.2c")
    else:
        fig.colorbar(frame='af+l%s [mm/yr]' % signal, position="JBC+w13c/0.45c+h+o10c/1.2c")
        pass

    fig.shift_origin(xshift='10c')
    fig.grdimage(
        grid=DA,
        cmap=True,
        frame=['xa5f5', 'ya5f5'] + ['+t(e) DA, smoothed'],
        dpi=100,
        projection=ps,
        region=region,
        # interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)

    fig.plot(data=gdf.boundary, pen="1p,black")
    # fig.colorbar()

    fig.shift_origin(xshift='10c')
    fig.grdimage(
        grid=GRACE,
        cmap=True,
        frame=['xa5f5', 'ya5f5'] + ['+t(f) GRACE, smoothed'],
        dpi=100,
        projection=ps,
        region=region,
        # interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)

    fig.plot(data=gdf.boundary, pen="1p,black")
    fig.show()
    pass


if __name__ == '__main__':
    # model_component_comparison_to_Leire()
    # compariosn_to_GRACE()
    # monthly_update_daily_update()
    # map2D_comparison()
    # tile_GRACE_1()
    # tile_GRACE_2()
    # Danube_exp()
    # CorrelationGDRB()
    # show_component_ensemble()
    # show_subbasin()
    show_trend_annual()
