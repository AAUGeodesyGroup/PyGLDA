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

from src_OBS.obs_auxiliary import aux_ESAsing_5daily


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
        name = Path(dir_in) / 'gravity_signal_cov' / ('%s_%s_Brahmaputra_v5.hdf5' % (
            self._mission, self._filter))

        hh = h5py.File(name=name, mode='r')

        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)
        assert len(hh['year_fraction'][:]) == len(self._aux.getTimeReference()['duration'])

        TWS = hh[self._grid_def][self._mission]['EWH_alpha_%s' % self._filter][:, a:b]
        """m ==> mm"""
        TWS = TWS*1000

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_signal.hdf5' % self.basin_name), 'w')
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            hm.create_dataset(key, data=TWS[i - 1])

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)

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
        name = Path(dir_in) / 'gravity_signal_cov' / ('%s_%s_Brahmaputra_v5.hdf5' % (
            self._mission, self._filter))

        hh = h5py.File(name=name, mode='r')

        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)
        assert len(hh['year_fraction'][:]) == len(self._aux.getTimeReference()['duration'])

        COV = hh[self._grid_def][self._mission]['VCM_alpha_%s' % self._filter][:, :]

        """m ==> mm"""
        COV = COV*1e6

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
        name = Path(dir_in) / ('Global_EWH_05_%s.hdf5' % self._filter)
        hh = h5py.File(name=name, mode='r')

        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)
        assert len(hh['time'][0, :]) == len(self._aux.getTimeReference()['duration'])
        TWS = np.flip(hh['global_EWH'][self._mission][a:b, :], axis=-2)[:, mask.astype(bool)]
        """m ==> mm"""
        TWS = TWS*1000

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


if __name__ == '__main__':
    demo1()
