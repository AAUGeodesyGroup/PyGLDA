import numpy as np
from pathlib import Path
import pygmt
import h5py
from src.GeoMathKit import GeoMathKit


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
    tws_dict={}
    for basin in hf.keys():
        tws = []
        fan = hf[basin]
        for state in statesnn:
            a = fan[state][:]
            tws.append(a)

        tws_dict[basin] = np.sum(np.array(tws), axis=0)

    hf.close()

    '''load GRACE'''
    data_path = Path('/media/user/My Book/Fan/W3/exp/comparison_GRACE_W3_Aus')
    grace = h5py.File(data_path / 'GRACE.hdf5', 'r')
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

        w3ra = tws_dict['basin_%s'%id]
        fig.plot(x=fan_time, y=w3ra-np.mean(w3ra), pen="0.5p,black", label='W3RA')

        fig.plot(x=grace_time, y=grace_tws * 1000/grace_scale_factor, pen="0.5p,blue", label='GRACE')
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

        w3ra = tws_dict['basin_%s'%id]
        fig.plot(x=fan_time, y=w3ra-np.mean(w3ra), pen="0.5p,black", label='W3RA')

        fig.plot(x=grace_time, y=grace_tws * 1000/grace_scale_factor, pen="0.5p,blue", label='GRACE')
        fig.legend(position='jBR', box='+gwhite+p0.5p')

        fig.shift_origin(yshift='-4.5c')

    # fig.savefig('../temp/W3RAvsGRACE.pdf')
    # fig.savefig('../temp/W3RAvsGRACE.png')
    fig.show()

    pass


if __name__ == '__main__':
    # model_component_comparison_to_Leire()
    compariosn_to_GRACE()
