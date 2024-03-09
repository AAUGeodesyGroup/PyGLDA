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
    gr_time= hf['time_epoch'][:].astype(str)
    gr_data = hf['tws'][:]
    time=[]

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


if __name__ == '__main__':
    # model_component_comparison_to_Leire()
    # compariosn_to_GRACE()
    # monthly_update_daily_update()
    map2D_comparison()
