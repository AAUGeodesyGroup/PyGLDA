import sys

sys.path.append('../')

import numpy as np
import h5py
import netCDF4 as nc
from src_OBS.prepare_GRACE import GRACE_preparation
from src_GHM.GeoMathKit import GeoMathKit
import os
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum

from src_OBS.obs_auxiliary import aux_ESAsing_5daily, aux_ESM3_5daily


class ESA_SING_5daily(GRACE_preparation):
    """
    loading data from ESA simulated 5-daily solutions
    """

    def __init__(self, basin_name='Brahmaputra_ExpZero',
                 shp_path='../data/basin/shp/ESA_SING/subbasins_3/Brahmaputra_3_subbasins.shp'):
        super().__init__(basin_name=basin_name, shp_path=shp_path)
        self._filter = None
        self._mission = None
        pass

    def set_extra_info(self, filter='1e+16', mission='GRACE-C-like', grid_def='subbasins_3'):
        self._filter = filter
        self._mission = mission
        self._grid_def = grid_def

        '''get a complete time list for all ESA simulation dataset'''
        self._aux = aux_ESAsing_5daily().setTimeReference(day_begin='1990-04-01', day_end='2020-05-01',
                                                          dir_in='/media/user/My Book/Fan/ESA_SING/ESA_5daily')

        return self

    def basin_TWS(self, day_begin='2002-04-01', day_end='2002-05-09',
                  dir_in='/media/user/My Book/Fan/ESA_SING/Brahmaputra',
                  dir_out='/media/user/My Book/Fan/ESA_SING/TestRes'):
        """
        This is unique for our dataset; Users must change for their own purpose
        The data: -89.75--->89.75, -179.75--->179.75
        The mask: lon: -180-->180, lat: 90--->-90
        """

        print()
        print('Start to pre-process GRACE to obtain basin-wise TWS over places of interest...')

        '''load basin mask'''
        res = 0.5
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')
        basins_num = len(list(mf.keys())) - 1

        '''load EWH'''
        name = Path(dir_in) / 'gravity_signal_cov' / ('%s_%s_Danube_v5.hdf5' % (
            self._mission, self._filter))
        # name = Path(dir_in) / 'gravity_signal_cov' / ('%s_%s_Brahmaputra_v5.hdf5' % (
        #     self._mission, self._filter))

        hh = h5py.File(name=name, mode='r')

        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)
        assert len(hh['year_fraction'][:]) == len(self._aux.getTimeReference()['duration'])

        TWS = hh[self._grid_def][self._mission]['EWH_alpha_%s' % self._filter][:, a:b]
        """m ==> mm"""
        TWS = TWS * 1000

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_signal.hdf5' % self.basin_name), 'w')
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            hm.create_dataset(key, data=TWS[i - 1])

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)
        hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass

    def basin_COV(self, day_begin='2002-04-01', day_end='2002-05-09',
                  dir_in='/media/user/My Book/Fan/ESA_SING/Brahmaputra',
                  dir_out='/media/user/My Book/Fan/ESA_SING/TestRes', isDiagonal=False):
        """
        This is unique for our dataset; Users must change for their own purpose.

        The data: -89.75--->89.75, -179.75--->179.75
        The mask: lon: -180-->180, lat: 90--->-90
        """
        print()
        print('Start to pre-process GRACE to obtain covariance over places of interest...')

        '''load basin mask'''
        res = 0.5
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')

        '''load EWH'''
        name = Path(dir_in) / 'gravity_signal_cov' / ('%s_%s_Danube_v5.hdf5' % (
            self._mission, self._filter))
        # name = Path(dir_in) / 'gravity_signal_cov' / ('%s_%s_Brahmaputra_v5.hdf5' % (
        #     self._mission, self._filter))

        hh = h5py.File(name=name, mode='r')

        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)
        assert len(hh['year_fraction'][:]) == len(self._aux.getTimeReference()['duration'])

        COV = hh[self._grid_def][self._mission]['VCM_alpha_%s' % self._filter][:, :]

        """m ==> mm"""
        COV = COV * 1e6

        '''Optional: remove off-diagonal matrix'''
        if isDiagonal:
            COV = np.diag(np.diag(COV))

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_cov.hdf5' % self.basin_name), 'w')
        hm.create_dataset('data', data=np.array(COV))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)

        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass

    def grid_TWS(self, day_begin='2002-04-01', day_end='2002-05-09',
                 dir_in='/media/user/My Book/Fan/ESA_SING/globe',
                 dir_out='/media/user/My Book/Fan/ESA_SING/TestRes'):
        """
        This is unique for our dataset; Users must change for their own purpose
        The data: -89.75--->89.75, -179.75--->179.75
        The mask: lon: -180-->180, lat: 90--->-90
        """

        '''load mask'''
        res = 0.5
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')
        mask = mf['basin'][:] * self._05deg_mask

        '''load EWH'''
        name = Path(dir_in) / ('Global_EWH_05_%s_%s.hdf5' % (self._mission, self._filter))
        hh = h5py.File(name=name, mode='r')

        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)
        assert len(hh['time'][0, :]) == len(self._aux.getTimeReference()['duration'])
        TWS = np.flip(hh['EWHA'][a:b, :], axis=-2)[:, mask.astype(bool)]
        """m ==> mm"""
        TWS = TWS * 1000

        print()
        print('Start to pre-process GRACE to obtain gridded TWS over places of interest...')

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_gridded_signal.hdf5' % self.basin_name), 'w')

        hm.create_dataset(name='tws', data=np.array(TWS))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)

        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass


class ESA_SING_temp_resolution(Enum):
    fivedays = 0
    monthly = 1


class ESA_SING_mission(Enum):
    GRACE_C_Like = 0
    NGGM = 1
    MAGIC = 2


class ESA_SING_ESM2(GRACE_preparation):
    """
    loading L3b (filtered) gridded data (1-degree) from ESA-SING simulated 5-daily solutions or monthly solution.
    """

    def __init__(self, basin_name='Brahmaputra_ExpZero',
                 shp_path='../data/basin/shp/ESA_SING/subbasins_3/Brahmaputra_3_subbasins.shp'):
        super().__init__(basin_name=basin_name, shp_path=shp_path)
        self._mission = None
        pass

    def set_extra_info(self, mission=ESA_SING_mission.GRACE_C_Like):
        self._mission = mission

        '''get a complete time list for all ESA simulation dataset'''
        '''This specification is for ESM2 and 5daily solution'''
        self._aux = aux_ESAsing_5daily().setTimeReference(day_begin='1990-04-01', day_end='2020-05-01',
                                                          dir_in='/media/user/My Book/Fan/ESA_SING/ESA_5daily')
        self._file_name = 'L3b_fivedaily_product_1995_2006_V2.4.nc'

        return self

    def basin_TWS(self, day_begin='2002-04-01', day_end='2002-05-09',
                  dir_in='/media/user/My Book/Fan/ESA_SING/ESA_SING_L3b',
                  dir_out='/media/user/My Book/Fan/ESA_SING/TestRes'):
        """
        Both tws gridded data and mask data follows lon: -180-->180, lat: 90--->-90.
        The spatial resolution is 1 degree in ESA-SING's simulation.
        """

        print()
        print('Start to pre-process GRACE to obtain basin-wise TWS over places of interest...')

        '''load basin mask'''
        res = 1
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')
        basins_num = len(list(mf.keys())) - 1

        '''load EWH'''
        name = Path(dir_in) / self._file_name

        hh = nc.Dataset(name)

        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)

        '''reference: truth'''
        ref = hh['Reference'][a:b, :, :]

        kw = 'L3b_' + self._mission.name + '_VADER_filtered'
        tws = hh[kw][a:b, :, :]  # unit is mm

        mask = {}
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            mask[key] = mf[key][:] * self._1deg_mask
            assert np.max(mask[key]) > 0.1, 'Basin shape file is incompatible with GRACE-1deg land mask, please revise!'

        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)

        TWS = {}
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            a = tws[:, mask[key].astype(bool)]
            b = lat[mask[key].astype(bool)]
            b = np.cos(np.deg2rad(b))
            TWS[key] = np.sum(a * b, axis=1) / np.sum(b)
            # TWS[key].append(np.mean(a))
            pass

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_signal.hdf5' % self.basin_name), 'w')
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            hm.create_dataset(key, data=TWS[key])

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)
        hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass

    def basin_COV(self, day_begin='2002-04-01', day_end='2002-05-09',
                  dir_in='/media/user/My Book/Fan/ESA_SING/ESA_SING_L3b',
                  dir_out='/media/user/My Book/Fan/ESA_SING/TestRes', isDiagonal=False):
        """
        Both tws gridded data and mask data follows lon: -180-->180, lat: 90--->-90.
        The spatial resolution is 1 degree in ESA-SING's simulation.
        For ESA-SING simulation, the COV is stattic. In this function, COV is obtained from residuals.
        """
        print()
        print('Start to pre-process GRACE to obtain covariance over places of interest...')

        '''load basin mask'''
        res = 1
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')
        basins_num = len(list(mf.keys())) - 1

        '''load EWH'''
        name = Path(dir_in) / self._file_name

        hh = nc.Dataset(name)

        '''reference: truth'''
        ref = hh['Reference'][:, :, :]

        kw = 'L3b_' + self._mission.name + '_VADER_filtered'
        tws = hh[kw][:, :, :]  # unit is mm

        mask = {}
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            mask[key] = mf[key][:] * self._1deg_mask
            assert np.max(mask[key]) > 0.1, 'Basin shape file is incompatible with GRACE-1deg land mask, please revise!'

        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)

        TWS_residual_sample = []
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            a1 = tws[:, mask[key].astype(bool)]
            a2 = ref[:, mask[key].astype(bool)]
            b = lat[mask[key].astype(bool)]
            b = np.cos(np.deg2rad(b))
            TWS_residual_sample.append(np.sum(a1 * b, axis=1) / np.sum(b) - np.sum(a2 * b, axis=1) / np.sum(b))
            # TWS[key].append(np.mean(a))
            pass

        COV = np.cov(np.array(TWS_residual_sample))

        '''Optional: remove off-diagonal matrix'''
        if isDiagonal:
            COV = np.diag(np.diag(COV))

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_cov.hdf5' % self.basin_name), 'w')
        hm.create_dataset('data', data=np.array(COV))

        dt = h5py.special_dtype(vlen=str)
        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)
        hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)

        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass

    def grid_TWS(self, day_begin='2002-04-01', day_end='2002-05-09',
                 dir_in='/media/user/My Book/Fan/ESA_SING/ESA_SING_L3b',
                 dir_out='/media/user/My Book/Fan/ESA_SING/TestRes'):
        """
        Both tws gridded data and mask data follows lon: -180-->180, lat: 90--->-90.
        The spatial resolution is 1 degree in ESA-SING's simulation.
        For ESA-SING simulation, the COV is stattic. In this function, COV is obtained from residuals.
        """

        '''load basin mask'''
        res = 1
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')
        mask = mf['basin'][:] * self._1deg_mask

        '''load EWH'''
        name = Path(dir_in) / self._file_name

        hh = nc.Dataset(name)

        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)

        '''reference: truth'''
        ref = hh['Reference'][a:b, :, :]

        kw = 'L3b_' + self._mission.name + '_VADER_filtered'
        TWS = hh[kw][a:b][:, mask.astype(bool)]  # unit is mm

        print('\nStart to pre-process GRACE to obtain gridded TWS over places of interest...')

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_gridded_signal.hdf5' % self.basin_name), 'w')

        hm.create_dataset(name='tws', data=np.array(TWS))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)

        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass


class ESA_SING_ESM3(ESA_SING_ESM2):
    """
    loading L3b (filtered) gridded data (1-degree) from ESA-SING simulated 5-daily solutions or monthly solution.
    """

    def __init__(self, basin_name='Brahmaputra_ExpZero',
                 shp_path='../data/basin/shp/ESA_SING/subbasins_3/Brahmaputra_3_subbasins.shp'):
        super().__init__(basin_name=basin_name, shp_path=shp_path)
        pass

    def set_extra_info(self, mission=ESA_SING_mission.GRACE_C_Like):

        self._mission = mission

        '''get a complete time list for all ESA simulation dataset'''
        '''This specification is for ESM2 and 5daily solution'''
        self._aux = aux_ESM3_5daily().setTimeReference(day_begin='1990-04-01', day_end='2030-05-01',
                                                          dir_in='/media/user/My Book/Fan/ESA_SING/ESM3.0/5daily')

        self._file_name = 'L3_5day_product_2007_2020_recommended.nc'

        return self


def demo1():
    es = ESA_SING_5daily(basin_name='Brahmaputra_ExpZero',
                         shp_path='../data/basin/shp/ESA_SING/subbasins_3/Brahmaputra_3_subbasins.shp')

    # es.set_extra_info(filter='1e+16', mission='GRACE-C-like')
    # es.set_extra_info(filter='1e+16', mission='MAGIC')
    es.set_extra_info(filter='1e+16', mission='NGGM')

    es.generate_mask()

    es.basin_TWS(day_begin='2003-09-01', day_end='2006-12-31')

    es.basin_COV(day_begin='2003-09-01', day_end='2006-12-31')

    es.grid_TWS(day_begin='2003-09-01', day_end='2006-12-31')
    pass


def demo2():
    es = ESA_SING_ESM3(basin_name='Europe',
                       shp_path='../data/basin/shp/Europe/Grid_3/Europe_subbasins.shp')

    # es.set_extra_info(mission=ESA_SING_mission.GRACE_C_Like)
    # es.set_extra_info(mission=ESA_SING_mission.NGGM)
    es.set_extra_info(mission=ESA_SING_mission.MAGIC)

    es.generate_mask(res_05=False)

    es.basin_TWS(day_begin='2012-03-01', day_end='2018-12-31')

    es.basin_COV(day_begin='2012-03-01', day_end='2018-12-31', isDiagonal=True)

    es.grid_TWS(day_begin='2012-03-01', day_end='2018-12-31')
    pass


if __name__ == '__main__':
    demo2()
