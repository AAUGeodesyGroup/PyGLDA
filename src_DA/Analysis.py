import sys

sys.path.append('../')

import numpy as np
from pathlib import Path
import h5py
from src_DA.shp2mask import basin_shp_process
from datetime import datetime, timedelta
from src_GHM.GeoMathKit import GeoMathKit


class BasinSignalAnalysis:
    """
    This script is for real time analysis to extract basin time series signal for each ensemble member
    """

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
            pf = '_' + post_fix

        '''initialize the dict'''
        map_average = {}
        lat_basin = {}
        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']
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
        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']
        for day in daylist:
            # print(day.strftime('%Y-%m-%d'))
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


class Postprocessing_basin:
    """
    This is also for the basin signal analysis. However, it is different from the above one at:
    1. this is a post-processing step to collect all the ensemble results together for further analysis
    2. in addition to the states, this script also deals with GRACE data for a collective analysis.
    """

    def __init__(self, ens, case: str, basin: str, date_begin='2002-04-01', date_end='2002-04-02'):
        self.ens = ens
        self.case = case
        self.basin = basin

        """deal with the time tag"""
        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)
        day_first = daylist[0].year + (daylist[0].month - 1) / 12 + daylist[0].day / 365.25
        self.time_tag = day_first + np.arange(len(daylist)) / 365.25

        self.__states = None
        self.__GRACE = None
        pass

    def get_states(self, post_fix: str, dir: str):
        if post_fix != '':
            post_fix = '_' + post_fix

        temp_dict = {}
        for ens in range(self.ens + 1):
            fn = h5py.File(Path(dir) / ('output_%s_ensemble_%s' % (self.case, ens)) / ('basin_ts%s.h5' % post_fix), 'r')
            ens_dict = {}
            for basin, item in fn.items():
                state_dict = {}
                for state, item2 in item.items():
                    state_dict[state] = item2[:]

                state_dict['TWS'] = np.sum(np.array(list(state_dict.values())), axis=0)

                ens_dict[basin] = state_dict

            temp_dict[ens] = ens_dict

        '''reformulate for easier analysis'''
        basins = fn.keys()
        state = list(item.keys()) + ['TWS']

        ff = {}
        for basin in basins:
            dd = {}
            for st in state:
                ss = {}
                for ens in range(self.ens + 1):
                    ss[ens] = temp_dict[ens][basin][st]
                dd[st] = ss
            ff[basin] = dd

        assert len(self.time_tag) == len(ss[ens]), 'Time tag is incorrect!'

        ff['time'] = self.time_tag
        self.__states = ff
        return ff

    def get_GRACE(self, obs_dir: str):
        kk = {}

        gr = h5py.File(Path(obs_dir) / ('%s_obs_GRACE.hdf5' % self.basin), 'r')

        str_time = list(gr['time_epoch'][:].astype(str))
        fraction_time = []
        for tt in str_time:
            if len(tt.split('-')) == 2:
                tt = tt + '-15'
            da = datetime.strptime(tt, '%Y-%m-%d')
            fraction_time.append(da.year + (da.month - 1) / 12 + da.day / 365.25)

        kk['time'] = np.array(fraction_time)
        kk['original'] = {}
        kk['ens_mean'] = {}

        a = gr['ens_1'][:]

        '''calculate how many ensemble member it has'''
        ens_num = -1  # because ens= 0 is included
        for dd in gr.keys():
            if 'ens_' in dd:
                ens_num += 1
        print(ens_num)
        try:
            basin_num = np.shape(gr['cov'][0])[0]
        except Exception:
            basin_num = 1

        for ens_id in range(2, ens_num + 1):
            a += gr['ens_%s' % ens_id]

        ens_mean = a / ens_num
        original = gr['ens_0'][:]

        for basin in range(1, 1 + basin_num):
            kk['original']['basin_%s' % basin] = original[:, basin - 1]
            kk['ens_mean']['basin_%s' % basin] = ens_mean[:, basin - 1]

        # kk['original']['basin_0'] = np.mean(original, axis=1)
        # kk['ens_mean']['basin_0'] = np.mean(ens_mean, axis=1)

        '''Area-weighted'''
        sub_basin_area = gr['sub_basin_area'][:]
        kk['original']['basin_0'] = np.sum(original * sub_basin_area[None, :], axis=1) / np.sum(sub_basin_area)
        kk['ens_mean']['basin_0'] = np.sum(ens_mean * sub_basin_area[None, :], axis=1) / np.sum(sub_basin_area)

        self.__GRACE = kk

        return kk

    def save_states(self, save_dir='../temp', prefix='1'):
        ff = self.__states

        hf = h5py.File(Path(save_dir) / ('Res_%s.h5df' % prefix), 'w')

        for ii, jj in ff.items():
            if ii == 'time':
                hf.create_dataset(name=ii, data=jj)
                continue
            dict_group1 = hf.create_group(ii)
            for kk, ll in jj.items():
                dict_group2 = dict_group1.create_group(kk)
                for mm, nn in ll.items():
                    dict_group2.create_dataset(name=str(mm), data=nn)
                    pass

        hf.close()

        pass

    def save_GRACE(self, save_dir='../temp', prefix='1'):

        ff = self.__GRACE

        hf = h5py.File(Path(save_dir) / ('GRACE_%s.h5df' % prefix), 'w')

        for ii, jj in ff.items():
            if ii == 'time':
                hf.create_dataset(name=ii, data=jj)
                continue
            dict_group1 = hf.create_group(ii)
            for kk, ll in jj.items():
                dict_group1.create_dataset(name=kk, data=ll)
                pass

        hf.close()
        pass

    def load_states(self, load_dir='../temp', prefix='1'):
        """
        This is simply for loading states from pre-saved results. No calculation has been done.
        It is mostly used for plotting a figure
        """
        ff = {}

        hf = h5py.File(Path(load_dir) / ('Res_%s.h5df' % prefix), 'r')

        for ii, jj in hf.items():
            if ii == 'time':
                ff[ii] = jj[:]
                continue
            dict_group1 = {}
            for kk, ll in jj.items():
                dict_group2 = {}
                for mm, nn in ll.items():
                    dict_group2[int(mm)] = nn[:]
                    pass
                dict_group1[kk] = dict_group2
            ff[ii] = dict_group1
        return ff

    def load_GRACE(self, load_dir='../temp', prefix='1'):
        """
        This is simply for loading GRACE from pre-saved results. No calculation has been done.
        It is mostly used for plotting a figure
        """
        ff = {}

        hf = h5py.File(Path(load_dir) / ('GRACE_%s.h5df' % prefix), 'r')

        for ii, jj in hf.items():
            if ii == 'time':
                ff[ii] = jj[:]
                continue
            dict_group1 = {}
            for kk, ll in jj.items():
                dict_group1[kk] = ll[:]

            ff[ii] = dict_group1

        return ff


class Postprocessing_grid_first:
    """
    This is also a post-processing script. Unlike the previous one, this one targets the spatial 2D map (state) instead
    of the basin average.
    And for the sake of data storage, the spatial data of state has been monthly averaged.

    """

    def __init__(self, state_dir='', par_dir=''):
        self.__state_dir = Path(state_dir)

        '''load parameters to calculate the TWS: fraction of each HRU'''
        hf = h5py.File(str(Path(par_dir) / 'par.h5'), 'r')
        self.__Fhru = hf['data']['Fhru'][:]
        hf.close()
        pass

    def monthlymean(self, state='TWS', date_begin='2002-04-01', date_end='2002-04-02'):
        days = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        if state == 'TWS':
            statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']
        else:
            assert state in ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']
            statesnn = [state]

        tws_monthly_mean = {}
        TWS = []

        for this_day in days:
            fn = self.__state_dir / ('state.%s.h5' % this_day.strftime('%Y%m%d'))
            hf = h5py.File(str(fn), 'r')
            tws = None
            for key in statesnn:
                vv = np.sum(hf[key][:] * self.__Fhru, axis=0)
                if key == 'Mleaf':
                    vv *= 4
                if tws is None:
                    tws = vv
                else:
                    tws += vv
            hf.close()

            TWS.append(tws)

            year, month = this_day.year, this_day.month
            if month == 12:
                last_day = datetime(year, month, 31)
            else:
                last_day = datetime(year, month + 1, 1) + timedelta(days=-1)

            if this_day == last_day:
                '''save monthly mean'''
                print(last_day)
                tws_monthly_mean[this_day.strftime('%Y-%m')] = np.mean(np.array(TWS), axis=0)
                TWS = []
                pass

        return tws_monthly_mean


class Postprocessing_grid_second:

    def __init__(self, ens, case, basin):
        self.ens = ens
        self.case = case
        self.basin = basin
        pass

    def ensemble_mean(self, state='TWS', postfix='', fdir=''):

        temp_dict = {}
        temp_dict_2 = {}
        for ens in range(self.ens + 1):
            fn = h5py.File(Path(fdir) / ('output_%s_ensemble_%s' % (self.case, ens)) / (
                    'monthly_mean_%s_%s.h5' % (state, postfix)), 'r')
            for time, vv in fn.items():
                if ens == 0:
                    '''error-free result'''
                    temp_dict[time] = vv[:]
                else:
                    '''ensemble result'''
                    if ens == 1:
                        temp_dict_2[time] = vv[:]
                    else:
                        temp_dict_2[time] += vv[:]

                if ens == self.ens:
                    temp_dict_2[time] /= self.ens

        return temp_dict, temp_dict_2


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
