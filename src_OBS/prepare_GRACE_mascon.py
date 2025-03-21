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


class GRACE_CSR_mascon(GRACE_preparation):

    def __init__(self, basin_name='MDB', shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp'):
        super().__init__(basin_name=basin_name, shp_path=shp_path)
        pass

    def basin_TWS(self, month_begin='2002-04', month_end='2002-04',
                  dir_in='/media/user/Backup Plus/GRACE/Mascon/CSR',
                  dir_out='/media/user/Backup Plus/GRACE/Mascon/CSR/test'):
        """

        """

        '''===============handling data mask===================='''
        res = 0.5
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')

        basins_num = len(list(mf.keys())) - 1

        mask = {}
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            mask[key] = mf[key][:]*self._05deg_mask

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        '''be aware that Mascon has a latitude from -89.875 to 89.875, and a longitude from 0.125 to 359.875'''

        '''===============handling mascon data========================'''
        filename = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
        csr = nc.Dataset(Path(dir_in) / filename)

        initial_day = datetime.strptime(csr.time_epoch.split('T')[0], '%Y-%m-%d')
        timelist = np.array(csr['time'][:].astype(int))

        available_month = {}
        index = 0
        for dd in timelist:
            this_month = initial_day + timedelta(days=int(dd))
            # print(this_month.strftime('%Y-%m'))
            available_month[this_month.strftime('%Y-%m')] = index
            index += 1

        '''particular correction is applied to the mascon monthlist'''
        available_month['2011-10'] -= 1
        available_month['2011-11'] = available_month['2011-10'] + 1
        available_month['2015-04'] -= 1
        # available_month['2015-05'] = available_month['2015-04'] + 1

        mascon_data = np.array(csr['lwe_thickness'])[:, 0::2, 0::2]  # the unit is centimeter

        '''===============computation========================'''
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)
        TWS = {}
        print('Start to pre-process GRACE to obtain basin-wise TWS over places of interest...')
        for i in range(1, basins_num + 1):
            TWS['sub_basin_%d' % i] = []
            pass

        time_epoch = []
        for month in monthlist:
            this_month_shortname = month.strftime('%Y-%m')
            # print(this_month_shortname)
            #
            if this_month_shortname not in available_month.keys():
                continue

            print(this_month_shortname)
            time_epoch.append(this_month_shortname + '-15')  # TODO: assumed to be the mid day of the month but should be checked later

            data = mascon_data[available_month[this_month_shortname]].copy()
            '''convert to be compatible with the mask grid definition'''
            data = np.flipud(data)
            tws_one_month = np.roll(data, 360)

            for i in range(1, basins_num + 1):
                key = 'sub_basin_%d' % i
                a = tws_one_month[mask[key].astype(bool)]
                b = lat[mask[key].astype(bool)]
                b = np.cos(np.deg2rad(b))
                '''convert unit from cm to mm'''
                TWS[key].append(np.sum(a * b) / np.sum(b)*10)
                # TWS[key].append(np.mean(a))
                pass

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_signal.hdf5' % self.basin_name), 'w')
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            hm.create_dataset(key, data=np.array(TWS[key]))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)

        hm.close()
        print('Finished: %s'% datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 

        pass


    def grid_TWS(self, month_begin='2002-04', month_end='2002-04',
                  dir_in='/media/user/Backup Plus/GRACE/Mascon/CSR',
                  dir_out='/media/user/Backup Plus/GRACE/Mascon/CSR/test'):
        """

        """

        '''===============handling data mask===================='''
        res = 0.5
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')

        basins_num = len(list(mf.keys())) - 1

        mask = mf['basin'][:]*self._05deg_mask

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        '''be aware that Mascon has a latitude from -89.875 to 89.875, and a longitude from 0.125 to 359.875'''

        '''===============handling mascon data========================'''
        filename = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
        csr = nc.Dataset(Path(dir_in) / filename)

        initial_day = datetime.strptime(csr.time_epoch.split('T')[0], '%Y-%m-%d')
        timelist = np.array(csr['time'][:].astype(int))

        available_month = {}
        index = 0
        for dd in timelist:
            this_month = initial_day + timedelta(days=int(dd))
            # print(this_month.strftime('%Y-%m'))
            available_month[this_month.strftime('%Y-%m')] = index
            index += 1

        '''particular correction is applied to the mascon monthlist'''
        available_month['2011-10'] -= 1
        available_month['2011-11'] = available_month['2011-10'] + 1
        available_month['2015-04'] -= 1
        # available_month['2015-05'] = available_month['2015-04'] + 1

        mascon_data = np.array(csr['lwe_thickness'])[:, 0::2, 0::2]  # the unit is centimeter

        '''===============computation========================'''
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)
        TWS = []
        print('Start to pre-process GRACE to obtain grid-wise TWS over places of interest...')
        time_epoch = []
        for month in monthlist:
            this_month_shortname = month.strftime('%Y-%m')
            # print(this_month_shortname)
            #
            if this_month_shortname not in available_month.keys():
                continue

            print(this_month_shortname)
            time_epoch.append(this_month_shortname + '-15')  # TODO: assumed to be the mid day of the month but should be checked later

            data = mascon_data[available_month[this_month_shortname]].copy()
            '''convert to be compatible with the mask grid definition, convert unit from cm to mm'''
            data = np.flipud(data)*10
            tws_one_month = np.roll(data, 360)

            a = tws_one_month[mask.astype(bool)]
            TWS.append(a)

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_gridded_signal.hdf5' % self.basin_name), 'w')

        hm.create_dataset(name='tws', data=np.array(TWS))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
        hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
        hm.close()
        print('Finished: %s'% datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 

        pass


def demo1():
    dp = GRACE_CSR_mascon()
    # dp.basin_TWS(month_end='2012-04')
    dp.grid_TWS(month_end='2012-04')
    # dp.basin_COV()
    pass

def demo2():

    # GR = GRACE_CSR_mascon(basin_name='US',
    #                        shp_path='/media/user/Backup Plus/GRACE/shapefiles/USgrid/UnitedStatesGrid.shp')

    GR = GRACE_CSR_mascon(basin_name='Africa',
                           shp_path='/media/user/Backup Plus/GRACE/shapefiles/Africa/Africa_subbasins.shp')

    GR.generate_mask()

    GR.basin_TWS(month_begin='2002-04', month_end='2023-05',
                  dir_in='/media/user/Backup Plus/GRACE/Mascon/CSR',
                  dir_out='/media/user/My Book/Fan/GRACE/output')

    GR.grid_TWS(month_begin='2002-04', month_end='2023-05',
                  dir_in='/media/user/Backup Plus/GRACE/Mascon/CSR',
                  dir_out='/media/user/My Book/Fan/GRACE/output')

    GR.basin_COV(month_begin='2002-04', month_end='2023-05', dir_in='/media/user/My Book/Fan/GRACE_spatial_resolution_study/degree_60/sample/DDK3/',
                  dir_out='/media/user/My Book/Fan/GRACE/output')

    pass

if __name__ == '__main__':
    demo1()
