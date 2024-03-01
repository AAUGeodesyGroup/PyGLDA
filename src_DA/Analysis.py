import sys

sys.path.append('../')

import numpy as np
from pathlib import Path
import h5py
from src_DA.shp2mask import basin_shp_process
from datetime import datetime
from src_hydro.GeoMathKit import GeoMathKit


class BasinSignalAnalysis:

    def __init__(self, basin_mask: basin_shp_process, state_dir='', par_dir=''):
        self.__bm = basin_mask
        self.__state_dir = Path(state_dir)

        '''load parameters to calculate the TWS: fraction of each HRU'''
        hf = h5py.File(str(Path(par_dir) / 'par.h5'), 'r')
        self.__Fhru = hf['data']['Fhru'][:]
        hf.close()
        pass

    def configure_mask2D(self, **kwargs):

        mask_2D, mask_1D, lat, lon, res = self.__bm.collect_mask
        coordinate = {}

        err = res / 10
        lats = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lons = np.arange(-180 + res / 2, 180 - res / 2 + err, res)

        # for k in mask_2D.keys():
        #     '''crop the 2D map to reduce the size as well as the memory'''
        #     lats_1D = lat[mask_2D[k]]
        #     lons_1D = lon[mask_2D[k]]
        #     lat_min, lat_max, lon_min, lon_max = np.min(lats_1D), np.max(lats_1D), np.min(lons_1D), np.max(lons_1D)
        #     lati = [np.argmin(np.fabs(lats - lat_max)),
        #             np.argmin(np.fabs(lats - lat_min))]
        #     loni = [np.argmin(np.fabs(lons - lon_min)),
        #             np.argmin(np.fabs(lons - lon_max))]
        #     boundary = [lati[0], lati[1] + 1, loni[0], loni[1] + 1]
        #
        #     mask_2D[k] = mask_2D[k][boundary[0]:boundary[1], boundary[2]:boundary[3]]
        #     coordinate[k] = {
        #         'lat': lat[boundary[0]:boundary[1], boundary[2]:boundary[3]],
        #         'lon': lon[boundary[0]:boundary[1], boundary[2]:boundary[3]]
        #     }

        '''crop the 2D map to reduce the size as well as the memory'''
        k = 'basin_0'
        lats_1D = lat[mask_2D[k]]
        lons_1D = lon[mask_2D[k]]
        lat_min, lat_max, lon_min, lon_max = np.min(lats_1D), np.max(lats_1D), np.min(lons_1D), np.max(lons_1D)
        lati = [np.argmin(np.fabs(lats - lat_max)),
                np.argmin(np.fabs(lats - lat_min))]
        loni = [np.argmin(np.fabs(lons - lon_min)),
                np.argmin(np.fabs(lons - lon_max))]
        boundary = [lati[0], lati[1] + 1, loni[0], loni[1] + 1]

        for k in mask_2D.keys():
            mask_2D[k] = mask_2D[k][boundary[0]:boundary[1], boundary[2]:boundary[3]]
            coordinate[k] = {
                'lat': lat[boundary[0]:boundary[1], boundary[2]:boundary[3]],
                'lon': lon[boundary[0]:boundary[1], boundary[2]:boundary[3]]
            }

        self.__mask_2D = mask_2D
        self.coordinate = coordinate
        return self

    def get_2D_map(self, this_day: datetime, save_dir='../temp', save=False):

        map_2D = {}

        mask_1D = self.__bm.collect_mask[1]

        mask_2D = self.__mask_2D

        fn = self.__state_dir / ('state.%s.h5' % this_day.strftime('%Y%m%d'))

        hf = h5py.File(str(fn), 'r')

        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']

        for basin, bm in mask_2D.items():

            map_2D[basin] = {}

            tws = np.zeros(np.shape(mask_2D[basin]))
            for key in statesnn:

                vv = np.sum(hf[key][:] * self.__Fhru, axis=0)

                dd = np.zeros(np.shape(mask_2D[basin]))
                dd[mask_2D[basin]] = vv[mask_1D[basin]]
                if key == 'Mleaf':
                    dd *= 4
                map_2D[basin][key] = dd
                tws += dd
                pass
            map_2D[basin]['TWS'] = tws
            pass
        hf.close()

        if save:
            hf = h5py.File(str(Path(save_dir) / ('map2D_%s.h5' % (this_day.strftime('%Y%m%d')))), 'w')
            for gr, bb in map_2D.items():
                dict_group = hf.create_group(gr)
                for key, vv in bb.items():
                    dict_group.create_dataset(key, data=vv)
            hf.close()
        pass

    def get_basin_average(self, date_begin='2002-04-01', date_end='2002-04-02',
                          save_dir='../temp', save=False, post_fix=None):

        mask_2D, mask_1D, lat, lon, res = self.__bm.collect_mask

        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        if post_fix is None:
            pf = ''
        else:
            pf = '_'+post_fix

        '''initialize the dict'''
        map_average = {}
        lat_basin = {}
        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf']
        for basin, bm in mask_1D.items():
            map_average[basin] = {}
            lat_basin[basin] = np.cos(np.deg2rad(lat[mask_2D[basin]]))
            for key in statesnn:
                map_average[basin][key] = []
                pass
            pass

        '''generate nanmask because of forcing field'''
        nan_mask = {}
        day = daylist[1]
        fn = self.__state_dir / ('state.%s.h5' % day.strftime('%Y%m%d'))
        hf = h5py.File(str(fn), 'r')
        for basin, bm in mask_1D.items():
            vv = np.sum(hf['FreeWater'][:] * self.__Fhru, axis=0)
            nan_mask[basin] = (1 - np.isnan(vv[bm])).astype(bool)
            pass

        '''start recording data'''
        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf']
        for day in daylist:
            print(day.strftime('%Y-%m-%d'))
            fn = self.__state_dir / ('state.%s.h5' % day.strftime('%Y%m%d'))
            hf = h5py.File(str(fn), 'r')
            for key in statesnn:
                vv = np.sum(hf[key][:] * self.__Fhru, axis=0)
                if key == 'Mleaf':
                    vv *= 4
                for basin, bm in mask_1D.items():
                    basin_average = np.sum(vv[bm][nan_mask[basin]] * lat_basin[basin][nan_mask[basin]]) / \
                                    np.sum(lat_basin[basin][nan_mask[basin]])
                    # basin_average = np.sum(vv[bm] * lat_basin[basin]) / np.sum(lat_basin[basin])
                    map_average[basin][key].append(basin_average)
                    pass
            hf.close()

        # '''test TWS'''
        # tws = {}
        # for basin in mask_1D.keys():
        #     tws[basin] =0
        #
        # for basin in mask_1D.keys():
        #     for state in statesnn:
        #         if state != 'Mleaf':
        #             tws[basin] += map_average[basin][state][0]
        #         else:
        #             tws[basin] += map_average[basin][state][0]

        if save:
            hf = h5py.File(str(Path(save_dir) / ('basin_ts%s.h5' % pf)), 'w')
            for gr, bb in map_average.items():
                dict_group = hf.create_group(gr)
                for key, vv in bb.items():
                    dict_group.create_dataset(key, data=np.array(vv))
            hf.close()

        pass


class Postprocessing:
    def __init__(self, configureDA):
        pass


def demo1():
    bs = basin_shp_process(res=0.1, basin_name='MDB').shp_to_mask(
        shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp')

    bs.mask_to_vec(model_mask_global='/media/user/My Book/Fan/W3RA_data/crop_input/single_run_test/mask/mask_global.h5')
    # bs.mask_nan()

    an = BasinSignalAnalysis(basin_mask=bs, state_dir='/media/user/My Book/Fan/W3RA_data/output/state_single_run_test',
                             par_dir="/media/user/My Book/Fan/W3RA_data/crop_input/single_run_test/par")

    # an.configure_mask2D().get_2D_map(this_day=datetime.strptime('2001-01-07', '%Y-%m-%d'), save=True)

    an.get_basin_average(save=True, date_begin='2002-04-01', date_end='2002-04-02')

    pass


if __name__ == '__main__':
    demo1()
