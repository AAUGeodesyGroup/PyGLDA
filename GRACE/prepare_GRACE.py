import sys

sys.path.append('../')

import numpy as np
import h5py
from DA.shp2mask import basin_shp_process
from src.GeoMathKit import GeoMathKit
import os
from pathlib import Path


class GRACE_preparation:

    def __init__(self, basin_name='MDB', shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp'):
        self.basin_name = basin_name
        self.__shp_path = shp_path
        pass

    def generate_mask(self):
        """
        generate and save global mask for later use
        """

        '''for signal: 0.5 degree'''
        bs1 = basin_shp_process(res=0.5, basin_name=self.basin_name).shp_to_mask(
            shp_path=self.__shp_path, issave=True)

        '''for cov: 1 degree'''
        bs2 = basin_shp_process(res=1, basin_name=self.basin_name).shp_to_mask(
            shp_path=self.__shp_path, issave=True)

        pass

    def basin_TWS(self, month_begin='2002-04', month_end='2002-04',
                  dir_in='/media/user/My Book/Fan/GRACE/ewh', dir_out='/media/user/My Book/Fan/GRACE/output'):
        """
        basin averaged TWS, monthly time-series, [mm]
        Be careful to deal with mask and TWS.
        mask: lon: -180-->180, lat: 90--->-90
        tws: lon: -180-->180, lat: -90--->90
        """

        '''load mask'''
        res = 0.5
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')

        basins_num = len(list(mf.keys())) - 1

        mask = {}
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            '''flip upside down to be compatible with the TWS dataset'''
            mask[key] = np.flipud(mf[key][:])

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        '''flip upside down to be compatible with the TWS dataset'''
        lat = np.flipud(lat)

        '''calculate the basin averaged TWS, monthly'''
        directory = Path(dir_in)
        fns = os.listdir(directory)
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        TWS = {}
        for i in range(1, basins_num + 1):
            TWS['sub_basin_%d' % i] = []
            pass

        time_epoch = []
        for month in monthlist:

            '''search for the file'''
            fn = month.strftime('%Y-%m')
            tn = None
            for filename in fns:
                if fn in filename:
                    tn = filename

            if tn is None:
                continue

            print(tn)
            fn = directory / tn
            time_epoch.append(tn.split('.')[0])

            tws_one_month = h5py.File(fn, 'r')['data'][:]

            for i in range(1, basins_num + 1):
                key = 'sub_basin_%d' % i
                a = tws_one_month[mask[key].astype(bool)]
                b = lat[mask[key].astype(bool)]
                b = np.cos(np.deg2rad(b))
                TWS[key].append(np.sum(a * b) / np.sum(b))
                # TWS[key].append(np.mean(a))
                pass

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
        pass

    def basin_COV(self, month_begin='2002-04', month_end='2002-04',
                  dir_in='/media/user/My Book/Fan/GRACE/DDK3_timeseries',
                  dir_out='/media/user/My Book/Fan/GRACE/output'):
        """
        COV of subbasins, monthly time-series.
        Be careful to deal with mask and TWS.
        mask: lon: -180-->180, lat: 90--->-90
        tws: lon: -180-->180, lat: -90--->90
        """

        '''load mask'''
        res = 1
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')

        basins_num = len(list(mf.keys())) - 1

        mask = {}
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            '''flip upside down to be compatible with the TWS dataset'''
            mask[key] = np.flipud(mf[key][:])

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        '''flip upside down to be compatible with the TWS dataset'''
        lat = np.flipud(lat)

        '''calculate the COV, monthly'''
        directory = Path(dir_in)
        fns = os.listdir(directory)
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        COV = []
        time_epoch = []
        for month in monthlist:

            '''search for the file'''
            fn = month.strftime('%Y-%m')
            tn = None
            for filename in fns:
                if fn in filename:
                    tn = filename

            if tn is None:
                continue

            print(tn)
            time_epoch.append(fn)
            fn = directory / tn

            tws_one_month = np.load(fn)

            vv = []
            for i in range(1, basins_num + 1):
                key = 'sub_basin_%d' % i
                a = tws_one_month[:, mask[key].astype(bool)]
                b = lat[mask[key].astype(bool)]
                b = np.cos(np.deg2rad(b))
                vv.append(np.sum(a * b, 1) / np.sum(b))
                # TWS[key].append(np.mean(a))
                pass

            '''transform into mm'''
            cov = np.cov(np.array(vv) * 1000)
            COV.append(cov)
            pass

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_cov.hdf5' % self.basin_name), 'w')
        hm.create_dataset('data', data=np.array(COV))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)

        hm.close()
        pass

        pass


def demo1():
    GR = GRACE_preparation(basin_name='Brahmaputra',
                           shp_path='../data/basin/shp/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp')
    # GR = GRACE_preparation(basin_name='MDB',
    #                        shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp')
    # GR.basin_TWS(month_begin='2002-04', month_end='2023-06')
    GR.basin_TWS(month_begin='2002-04', month_end='2023-06', dir_in='/media/user/Backup Plus/GRACE/ewh',
                 dir_out='/home/user/test/output')
    # GR.generate_mask()
    # GR.basin_COV(month_begin='2002-04', month_end='2023-06')
    pass


if __name__ == '__main__':
    demo1()
