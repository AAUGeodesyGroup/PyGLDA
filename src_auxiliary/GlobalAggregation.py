import sys

import h5py
import numpy as np

sys.path.append('../')


def TWS_trend_annual_phase():
    from src_auxiliary.ts import ts
    from src_auxiliary.upscaling import upscaling
    from datetime import datetime
    from src_hydro.GeoMathKit import GeoMathKit
    import geopandas as gpd
    import h5py
    import numpy as np
    import json
    import os
    from pathlib import Path
    from demo.global_DA import GDA
    import warnings
    warnings.filterwarnings("ignore")

    OL_map = {}
    DA_map = {}
    GR_map = {}

    dp_dir = Path(GDA.setting_dir) / 'setting.json'
    dp4 = json.load(open(dp_dir, 'r'))
    sh = dp4['input']['mask_fn']

    ''''''

    for tile_ID in range(1, 300):

        try:
            GDA.set_tile(tile_ID=tile_ID, create_folder=False)

            '''load DA'''
            hf = h5py.File(Path(GDA.res_output) / ('monthly_mean_TWS_%s_DA.h5' % GDA.basin), 'r')

            us = upscaling(basin=GDA.basin)
            us.configure_model_state(globalmask_fn=Path(sh) / GDA.case / 'mask/mask_global.h5')
            us.configure_GRACE()
            us.extra_mask()

        except Exception:
            # print(Path(GDA.res_output) / ('monthly_mean_TWS_%s_DA.h5' % GDA.basin))
            continue

        print('Tile_ID: %s is available.' % tile_ID)

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
        hf = h5py.File(Path(GDA.res_output) / ('monthly_mean_TWS_%s_OL.h5' % GDA.basin), 'r')
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
        dp_dir = Path(GDA.setting_dir) / 'DA_setting.json'
        dp4 = json.load(open(dp_dir, 'r'))
        hf = h5py.File(Path(dp4['obs']["GRACE"]['preprocess_res']) / ('%s_gridded_signal.hdf5' % GDA.basin), 'r')
        gr_time = hf['time_epoch'][:].astype(str)
        gr_data = hf['tws'][:]
        time = []

        for i in range(len(gr_time)):
            m = gr_time[i].split('-')
            n = datetime(year=int(m[0]), month=int(m[1]), day=int(m[2]))
            time.append(GeoMathKit.year_fraction(n))

        tsd = ts().set_period(time=np.array(time)).setDecomposition()
        res_GRACE = tsd.getSignal(obs=gr_data)

        '''1d --> 2d map'''
        for key in ['trend', 'annual']:
            if key == 'trend':
                if 'trend' not in GR_map.keys():
                    GR_map[key] = us.get2D_GRACE(res_GRACE[key])

                    temp = us.get2D_model_state(res_OL[key])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map[key] = temp

                    temp = us.get2D_model_state(res_DA[key])
                    # temp[us.bshp_mask == False] = np.nan
                    DA_map[key] = temp
                else:
                    temp = us.get2D_GRACE(res_GRACE[key])
                    GR_map[key][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_model_state(res_OL[key])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map[key][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_model_state(res_DA[key])
                    # temp[us.bshp_mask == False] = np.nan
                    DA_map[key][~np.isnan(temp)] = temp[~np.isnan(temp)]


            else:
                if 'amplitude' not in GR_map.keys():
                    GR_map['amplitude'] = us.get2D_GRACE(res_GRACE[key][0])
                    GR_map['phase'] = us.get2D_GRACE(res_GRACE[key][1])

                    temp = us.get2D_model_state(res_OL[key][0])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map['amplitude'] = temp

                    temp = us.get2D_model_state(res_OL[key][1])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map['phase'] = temp

                    temp = us.get2D_model_state(res_DA[key][0])
                    # temp[us.bshp_mask == False] = np.nan
                    DA_map['amplitude'] = temp

                    temp = us.get2D_model_state(res_DA[key][1])
                    # temp[us.bshp_mask == False] = np.nan
                    DA_map['phase'] = us.get2D_model_state(res_DA[key][1])
                else:
                    temp = us.get2D_GRACE(res_GRACE[key][0])
                    GR_map['amplitude'][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_GRACE(res_GRACE[key][1])
                    GR_map['phase'][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_model_state(res_OL[key][0])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map['amplitude'][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_model_state(res_OL[key][1])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map['phase'][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_model_state(res_DA[key][0])
                    # temp[us.bshp_mask == False] = np.nan
                    DA_map['amplitude'][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_model_state(res_DA[key][1])
                    # temp[us.bshp_mask == False] = np.nan
                    DA_map['phase'][~np.isnan(temp)] = temp[~np.isnan(temp)]

    '''save the global map for visualization'''
    fn = h5py.File('/work/data_for_w3/w3ra/temp/GDA_DA_map.h5', 'w')
    fn.create_dataset(name='trend', data=DA_map['trend'])
    fn.create_dataset(name='amplitude', data=DA_map['amplitude'])
    fn.create_dataset(name='phase', data=DA_map['phase'])
    fn.close()

    fn = h5py.File('/work/data_for_w3/w3ra/temp/GDA_GR_map.h5', 'w')
    fn.create_dataset(name='trend', data=GR_map['trend'])
    fn.create_dataset(name='amplitude', data=GR_map['amplitude'])
    fn.create_dataset(name='phase', data=GR_map['phase'])
    fn.close()

    fn = h5py.File('/work/data_for_w3/w3ra/temp/GDA_OL_map.h5', 'w')
    fn.create_dataset(name='trend', data=OL_map['trend'])
    fn.create_dataset(name='amplitude', data=OL_map['amplitude'])
    fn.create_dataset(name='phase', data=OL_map['phase'])
    fn.close()

    pass


class tile_aggregation:

    def __init__(self):
        self.__AG_map_h = None
        self.__AG_mask_h = None
        self.AG_map = None
        self.tile_list = np.array([1, 16, 31, 46, 61, 76, 91, 106, 121, 136])
        self.__previous_tile = 1
        self.ratio = 0.5
        pass

    def __tile_change(self, tile_ID=0):
        if np.where(self.tile_list > tile_ID)[0][0] != np.where(self.tile_list > self.__previous_tile)[0][0]:
            self.__previous_tile = tile_ID
            return True
        else:
            return False

    def aggregate(self, temp=None, tile_ID=0, H_inside_mask=None):
        if self.__tile_change(tile_ID=tile_ID):
            print('SSSS')
            if self.AG_map is None:
                self.AG_map = self.__AG_map_h
            else:
                print('v')
                self.AG_map = self.__vertical_aggregation(self.AG_map, self.__AG_map_h, self.__AG_mask_h)
            self.__AG_map_h = None
            self.__AG_mask_h = None

        if self.__AG_map_h is None:
            self.__AG_map_h = temp
            self.__AG_mask_h = H_inside_mask
            return

        print('h')
        self.__AG_map_h = self.__horizontal_aggregation(self.__AG_map_h, temp, H_inside_mask)
        self.__AG_mask_h += H_inside_mask

        pass

    def finish(self):
        if self.__AG_map_h is not None:
            if self.AG_map is None:
                self.AG_map = self.__AG_map_h
            else:
                self.AG_map = self.__vertical_aggregation(self.AG_map, self.__AG_map_h, self.__AG_mask_h)
        pass

    def __horizontal_aggregation(self, DA_map, temp, inside_mask):

        overlap = (~np.isnan(temp)) * (~np.isnan(DA_map))
        o1 = overlap * (~inside_mask)
        overlap = overlap.astype(int)
        o1 = o1.astype(int)

        if np.any(o1):
            o1x = np.where(o1 > 0.5)[1]
            o2 = overlap - o1
            o2x = np.where(o2 > 0.5)[1]

            if len(o2x) == 0:
                overlap = overlap.astype(bool)
                DA_map[(~np.isnan(temp)) * (~overlap)] = temp[(~np.isnan(temp)) * (~overlap)]
                return DA_map

            o1y = np.ones(o1.shape)
            o1y[:, o1x.min():(o2x.min())] = np.linspace(start=1, stop=self.ratio, num=o2x.min() - o1x.min())
            o1z = np.ones(o1.shape)
            o1z[:, o1x.min():(o2x.min())] = np.linspace(start=0, stop=1 - self.ratio,
                                                        num=o2x.min() - o1x.min())

            overlap = overlap.astype(bool)
            DA_map[overlap] *= o1y[overlap]
            temp[overlap] *= o1z[overlap]

            o2y = np.ones(o2.shape)
            o2y[:, o2x.min():(o2x.max() + 1)] = np.linspace(start=self.ratio, stop=1, num=o2x.max() - o2x.min() + 1)
            o2z = np.ones(o2.shape)
            o2z[:, o2x.min():(o2x.max() + 1)] = np.linspace(start=1 - self.ratio, stop=0,
                                                            num=o2x.max() - o2x.min() + 1)
            temp[overlap] *= o2y[overlap]
            DA_map[overlap] *= o2z[overlap]

            DA_map[overlap] += temp[overlap]

            DA_map[(~np.isnan(temp)) * (~overlap)] = temp[(~np.isnan(temp)) * (~overlap)]


        else:
            DA_map[~np.isnan(temp)] = temp[~np.isnan(temp)]

        return DA_map

    def __vertical_aggregation(self, DA_map, temp, inside_mask):

        overlap = (~np.isnan(temp)) * (~np.isnan(DA_map))
        o1 = overlap * (~inside_mask)
        overlap = overlap.astype(int)
        o1 = o1.astype(int)

        if np.any(o1):
            o1x = np.where(o1 > 0.5)[0]
            o2 = overlap - o1
            o2x = np.where(o2 > 0.5)[0]

            if len(o2x) == 0:
                overlap = overlap.astype(bool)
                DA_map[(~np.isnan(temp)) * (~overlap)] = temp[(~np.isnan(temp)) * (~overlap)]
                return DA_map

            o1y = np.ones(o1.shape)
            o1y[o1x.min():(o2x.min()), :] = np.linspace(start=1, stop=self.ratio, num=o2x.min() - o1x.min())[:, None]
            o1z = np.ones(o1.shape)
            o1z[o1x.min():(o2x.min()), :] = np.linspace(start=0, stop=1 - self.ratio,
                                                        num=o2x.min() - o1x.min())[:, None]

            overlap = overlap.astype(bool)
            DA_map[overlap] *= o1y[overlap]
            temp[overlap] *= o1z[overlap]

            o2y = np.ones(o2.shape)
            o2y[o2x.min():(o2x.max() + 1), :] = np.linspace(start=self.ratio, stop=1, num=o2x.max() - o2x.min() + 1)[:,
                                                None]
            o2z = np.ones(o2.shape)
            o2z[o2x.min():(o2x.max() + 1), :] = np.linspace(start=1 - self.ratio, stop=0,
                                                            num=o2x.max() - o2x.min() + 1)[:, None]
            temp[overlap] *= o2y[overlap]
            DA_map[overlap] *= o2z[overlap]

            DA_map[overlap] += temp[overlap]

            DA_map[(~np.isnan(temp)) * (~overlap)] = temp[(~np.isnan(temp)) * (~overlap)]


        else:
            DA_map[~np.isnan(temp)] = temp[~np.isnan(temp)]

        return DA_map


def TWS_trend_annual_phase_overlap_average():
    from src_auxiliary.ts import ts
    from src_auxiliary.upscaling import upscaling
    from datetime import datetime
    from src_hydro.GeoMathKit import GeoMathKit
    import geopandas as gpd
    import h5py
    import numpy as np
    import json
    import os
    from pathlib import Path
    from demo.global_DA import GDA
    import warnings
    warnings.filterwarnings("ignore")

    OL_map = {}
    DA_map = {}
    GR_map = {}

    dp_dir = Path(GDA.setting_dir) / 'setting.json'
    dp4 = json.load(open(dp_dir, 'r'))
    sh = dp4['input']['mask_fn']

    DA_trendmap = tile_aggregation()
    DA_ampmap = tile_aggregation()
    DA_phasemap = tile_aggregation()

    ''''''

    for tile_ID in range(1, 300):
    # for tile_ID in [68, 69, 83]:

        try:
            GDA.set_tile(tile_ID=tile_ID, create_folder=False)

            '''load DA'''
            hf = h5py.File(Path(GDA.res_output) / ('monthly_mean_TWS_%s_DA.h5' % GDA.basin), 'r')

            us = upscaling(basin=GDA.basin)
            us.configure_model_state(globalmask_fn=Path(sh) / GDA.case / 'mask/mask_global.h5')
            us.configure_GRACE()
            # us.extra_mask()

        except Exception:
            # print(Path(GDA.res_output) / ('monthly_mean_TWS_%s_DA.h5' % GDA.basin))
            continue

        print('Tile_ID: %s is available.' % tile_ID)

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
        hf = h5py.File(Path(GDA.res_output) / ('monthly_mean_TWS_%s_OL.h5' % GDA.basin), 'r')
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
        dp_dir = Path(GDA.setting_dir) / 'DA_setting.json'
        dp4 = json.load(open(dp_dir, 'r'))
        hf = h5py.File(Path(dp4['obs']["GRACE"]['preprocess_res']) / ('%s_gridded_signal.hdf5' % GDA.basin), 'r')
        gr_time = hf['time_epoch'][:].astype(str)
        gr_data = hf['tws'][:]
        time = []

        for i in range(len(gr_time)):
            m = gr_time[i].split('-')
            n = datetime(year=int(m[0]), month=int(m[1]), day=int(m[2]))
            time.append(GeoMathKit.year_fraction(n))

        tsd = ts().set_period(time=np.array(time)).setDecomposition()
        res_GRACE = tsd.getSignal(obs=gr_data)

        '''1d --> 2d map'''
        hmask = us.in_mask()

        temp = us.get2D_model_state(res_DA['trend'])
        DA_trendmap.aggregate(temp=temp, tile_ID=tile_ID, H_inside_mask=hmask)

        temp = us.get2D_model_state(res_DA['annual'][0])
        DA_ampmap.aggregate(temp=temp, tile_ID=tile_ID, H_inside_mask=hmask)

        temp = us.get2D_model_state(res_DA['annual'][1])
        DA_phasemap.aggregate(temp=temp, tile_ID=tile_ID, H_inside_mask=hmask)


        for key in ['trend', 'annual']:
            if key == 'trend':

                if 'trend' not in GR_map.keys():
                    GR_map[key] = us.get2D_GRACE(res_GRACE[key])

                    temp = us.get2D_model_state(res_OL[key])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map[key] = temp

                else:
                    temp = us.get2D_GRACE(res_GRACE[key])
                    GR_map[key][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_model_state(res_OL[key])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map[key][~np.isnan(temp)] = temp[~np.isnan(temp)]

            else:
                if 'amplitude' not in GR_map.keys():
                    GR_map['amplitude'] = us.get2D_GRACE(res_GRACE[key][0])
                    GR_map['phase'] = us.get2D_GRACE(res_GRACE[key][1])

                    temp = us.get2D_model_state(res_OL[key][0])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map['amplitude'] = temp

                    temp = us.get2D_model_state(res_OL[key][1])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map['phase'] = temp

                else:
                    temp = us.get2D_GRACE(res_GRACE[key][0])
                    GR_map['amplitude'][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_GRACE(res_GRACE[key][1])
                    GR_map['phase'][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_model_state(res_OL[key][0])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map['amplitude'][~np.isnan(temp)] = temp[~np.isnan(temp)]

                    temp = us.get2D_model_state(res_OL[key][1])
                    # temp[us.bshp_mask == False] = np.nan
                    OL_map['phase'][~np.isnan(temp)] = temp[~np.isnan(temp)]



    '''save the global map for visualization'''
    fn = h5py.File('/work/data_for_w3/w3ra/temp/GDA_DA_map.h5', 'w')
    # fn.create_dataset(name='trend', data=DA_map['trend'])
    DA_trendmap.finish()
    DA_ampmap.finish()
    DA_phasemap.finish()

    fn.create_dataset(name='trend', data=DA_trendmap.AG_map)
    fn.create_dataset(name='amplitude', data=DA_ampmap.AG_map)
    fn.create_dataset(name='phase', data=DA_phasemap.AG_map)
    fn.close()

    fn = h5py.File('/work/data_for_w3/w3ra/temp/GDA_GR_map.h5', 'w')
    fn.create_dataset(name='trend', data=GR_map['trend'])
    fn.create_dataset(name='amplitude', data=GR_map['amplitude'])
    fn.create_dataset(name='phase', data=GR_map['phase'])
    fn.close()

    fn = h5py.File('/work/data_for_w3/w3ra/temp/GDA_OL_map.h5', 'w')
    fn.create_dataset(name='trend', data=OL_map['trend'])
    fn.create_dataset(name='amplitude', data=OL_map['amplitude'])
    fn.create_dataset(name='phase', data=OL_map['phase'])
    fn.close()

    pass


def plot_TWS_trend_annual_phase():
    import pygmt
    import numpy as np

    # fn = h5py.File('/work/data_for_w3/w3ra/temp/GDA_OL_map.h5', 'r')
    fn = h5py.File('/home/user/Desktop/temp/GDA_OL_map.h5', 'r')
    OL_map = {}
    for key, item in fn.items():
        OL_map[key] = item[:]

    # fn = h5py.File('/work/data_for_w3/w3ra/temp/GDA_DA_map.h5', 'r')
    fn = h5py.File('/home/user/Desktop/temp/GDA_DA_map.h5', 'r')
    DA_map = {}
    for key, item in fn.items():
        DA_map[key] = item[:]

    # fn = h5py.File('/work/data_for_w3/w3ra/temp/GDA_GR_map.h5', 'r')
    fn = h5py.File('/home/user/Desktop/temp/GDA_GR_map.h5', 'r')
    GR_map = {}
    for key, item in fn.items():
        GR_map[key] = item[:]

    '''visualization'''
    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')

    res = 0.5
    err = res / 10
    lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
    region = [min(lon), max(lon), min(lat), max(lat)]
    lon, lat = np.meshgrid(lon, lat)

    DA_trend = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=DA_map['trend'].flatten(),
                             spacing=(res, res), region=region)

    OL_trend = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=OL_map['trend'].flatten(),
                             spacing=(res, res), region=region)

    GRACE_trend = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=GR_map['trend'].flatten(),
                                spacing=(res, res), region=region)

    DA_amplitude = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=DA_map['amplitude'].flatten(),
                                 spacing=(res, res), region=region)

    OL_amplitude = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=OL_map['amplitude'].flatten(),
                                 spacing=(res, res), region=region)

    GRACE_amplitude = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=GR_map['amplitude'].flatten(),
                                    spacing=(res, res), region=region)

    DA_phase = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=DA_map['phase'].flatten() / 360 * 12,
                             spacing=(res, res), region=region)

    OL_phase = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=OL_map['phase'].flatten() / 360 * 12,
                             spacing=(res, res), region=region)

    GRACE_phase = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=GR_map['phase'].flatten() / 360 * 12,
                                spacing=(res, res), region=region)

    ps = 'Q12c'
    offset = '-8.5c'

    '''plot trend'''
    pygmt.makecpt(cmap='vik', reverse=True, series=[-15, 15], background='o')

    region = [-180, 180, -60, 90]

    fig.grdimage(
        grid=OL_trend,
        cmap=True,
        dpi=100,
        frame=['xf15', 'yf15', '+tW3RA'],
        projection=ps,
        region=region,
        # interpolation='n',
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)
    # fig.coast(shorelines=True)

    fig.shift_origin(yshift='-6.2c')

    fig.grdimage(
        grid=GRACE_trend,
        cmap=True,
        dpi=100,
        frame=['xf15', 'yf15', '+tGRACE'],
        projection=ps,
        region=region,
        # interpolation='n',
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)
    # fig.coast(shorelines=True)

    fig.shift_origin(yshift='-6.2c')

    fig.grdimage(
        grid=DA_trend,
        cmap=True,
        dpi=100,
        frame=['xf15', 'yf15', '+tDA'],
        projection=ps,
        region=region,
        interpolation='n',
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)
    # fig.coast(shorelines=True)

    fig.colorbar(frame='a10f5+lTrend [mm/yr]', position="JBC+w10c/0.45c+h+o0c/1.2c")

    '''Amp'''
    pygmt.makecpt(cmap='inferno', reverse=True, series=[0, 300], background='o')
    fig.shift_origin(yshift='12.4c', xshift='13c')
    fig.grdimage(
        grid=OL_amplitude,
        cmap=True,
        dpi=100,
        frame=['xf15', 'yf15', '+tW3RA'],
        projection=ps,
        region=region,
        # interpolation='n',
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)
    # fig.coast(shorelines=True)

    fig.shift_origin(yshift='-6.2c')

    fig.grdimage(
        grid=GRACE_amplitude,
        cmap=True,
        dpi=100,
        frame=['xf15', 'yf15', '+tGRACE'],
        projection=ps,
        region=region,
        # interpolation='n',
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)
    # fig.coast(shorelines=True)
    fig.shift_origin(yshift='-6.2c')

    fig.grdimage(
        grid=DA_amplitude,
        cmap=True,
        dpi=100,
        frame=['xf15', 'yf15', '+tDA'],
        projection=ps,
        region=region,
        # interpolation='n',
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)
    # fig.coast(shorelines=True)

    fig.colorbar(frame='a50f25+lAnnual amplitude [mm]', position="JBC+w10c/0.45c+h+o0c/1.2c")

    '''Phase'''
    pygmt.makecpt(cmap='dem4', series=[0, 12, 1], reverse=True, no_bg=True, cyclic=True)
    fig.shift_origin(yshift='12.4c', xshift='13c')
    fig.grdimage(
        grid=OL_phase,
        cmap=True,
        dpi=100,
        frame=['xf15', 'yf15', '+tW3RA'],
        projection=ps,
        region=region,
        # interpolation='n',
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)
    # fig.coast(shorelines=True)

    fig.shift_origin(yshift='-6.2c')

    fig.grdimage(
        grid=GRACE_phase,
        cmap=True,
        dpi=100,
        frame=['xf15', 'yf15', '+tGRACE'],
        projection=ps,
        region=region,
        # interpolation='n',
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)
    # fig.coast(shorelines=True)

    fig.shift_origin(yshift='-6.2c')

    fig.grdimage(
        grid=DA_phase,
        cmap=True,
        dpi=100,
        frame=['xf15', 'yf15', '+tDA'],
        projection=ps,
        region=region,
        # interpolation='n',
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=ps)
    # fig.coast(shorelines=True)

    fig.colorbar(frame='a1f1+lAnnual phase [months]', position="JBC+w10c/0.45c+h+o0c/1.2c")

    fig.show()

    #

    pass


def demo1():
    TWS_trend_annual_phase_overlap_average()
    # TWS_trend_annual_phase()
    pass


def demo2():
    plot_TWS_trend_annual_phase()
    pass


if __name__ == '__main__':
    demo1()
    # demo2()
