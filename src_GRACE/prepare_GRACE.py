import sys

sys.path.append('../')

import numpy as np
import h5py
from src_DA.shp2mask import basin_shp_process
from src_hydro.GeoMathKit import GeoMathKit
import os
from pathlib import Path


class GRACE_preparation:

    def __init__(self, basin_name='MDB', shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp'):
        self.basin_name = basin_name
        self.__shp_path = shp_path
        pass

    def generate_mask(self, box_mask=None):
        """
        generate and save global mask for later use
        """

        '''for signal: 0.5 degree'''
        bs1 = basin_shp_process(res=0.5, basin_name=self.basin_name).configureBox(box_mask).shp_to_mask(
            shp_path=self.__shp_path, issave=True)

        '''for cov: 1 degree'''
        bs2 = basin_shp_process(res=1, basin_name=self.basin_name).configureBox(box_mask).shp_to_mask(
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
        print()
        print('Start to pre-process GRACE to obtain signal over places of interest...')
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

            date = tn.split('.')[0]
            if len(date.split('-')) == 2:
                time_epoch.append(date + '-15')  #TODO: assumed to be the mid day of the month but should be checked later
            else:
                time_epoch.append(date)

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
        print('Finished')
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

        print()
        print('Start to pre-process GRACE to obtain COV over places of interest...')
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

            tws_one_month = np.load(str(fn))

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
        print('Finished')

        pass

    def grid_TWS(self, month_begin='2002-04', month_end='2002-04',
                 dir_in='/media/user/My Book/Fan/GRACE/ewh', dir_out='/media/user/My Book/Fan/GRACE/output'):
        """
        gridded TWS, monthly time-series, [mm]
        Be careful to deal with mask and TWS.
        mask: lon: -180-->180, lat: 90--->-90
        tws: lon: -180-->180, lat: -90--->90
        """

        '''load mask'''
        res = 0.5
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.basin_name, res), 'r')

        basins_num = len(list(mf.keys())) - 1

        mask = mf['basin'][:]

        '''calculate the basin averaged TWS, monthly'''
        directory = Path(dir_in)
        fns = os.listdir(directory)
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        TWS = []
        print()
        print('Start to pre-process GRACE to obtain signal over places of interest...')

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

            '''take care: GRACE is -90 90; model TWS and mask is 90 -90'''
            '''flip upside down to be compatible with the mask'''
            tws_one_month = np.flipud(tws_one_month)
            a = tws_one_month[mask.astype(bool)]

            TWS.append(a)
            pass

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_gridded_signal.hdf5' % self.basin_name), 'w')

        hm.create_dataset(name='tws', data=np.array(TWS))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)

        hm.close()
        print('Finished')
        pass


class GRACE_global_preparation:

    def __init__(self):
        pass

    def configure_global_mask(self, fn='/media/user/My Book/Fan/W3RA_data/basin_selection/GlobalLandMaskForGRACE.hdf5'):
        self._1deg_mask = np.flipud(h5py.File(fn, 'r')['resolution_1']['mask'][:])
        self._05deg_mask = np.flipud(h5py.File(fn, 'r')['resolution_05']['mask'][:])
        return self

    def configure_global_shp(self, fn='../data/basin/shp/global_shp_new'):
        valid_basin_mask_05 = {}
        valid_basin_mask_10 = {}

        for i in range(1, 300):
            fshp = Path(fn) / ('Tile%s_subbasins.shp' % i)
            if not fshp.is_file():
                continue
            print(fshp)
            valid_basin_mask_05['Tile%s' % i] = basin_shp_process(res=0.5, basin_name='Tile%s' % i).shp_to_mask(
                shp_path=str(fshp), issave=False).mask
            valid_basin_mask_10['Tile%s' % i] = basin_shp_process(res=1, basin_name='Tile%s' % i).shp_to_mask(
                shp_path=str(fshp), issave=False).mask

            for nn in valid_basin_mask_10['Tile%s' % i].keys():
                valid_basin_mask_05['Tile%s' % i][nn] = (
                        valid_basin_mask_05['Tile%s' % i][nn] * self._05deg_mask).astype(bool)
                valid_basin_mask_10['Tile%s' % i][nn] = (
                        valid_basin_mask_10['Tile%s' % i][nn] * self._1deg_mask).astype(bool)
            pass

        self.valid_mask = {
            '05': valid_basin_mask_05,
            '1': valid_basin_mask_10
        }

        return self

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
        mask = self.valid_mask['05']

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        area = np.cos(np.deg2rad(lat))

        '''calculate the basin averaged TWS, monthly'''
        directory = Path(dir_in)
        fns = os.listdir(directory)
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        print()
        print('Start to pre-process GRACE to obtain signal over places of interest...')

        TWS = {}
        for tile, tile_mask in mask.items():
            TWS[tile] = {}

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
            date = tn.split('.')[0]
            if len(date.split('-')) == 2:
                time_epoch.append(date + '-15')  #TODO: assumed to be the mid day of the month but should be checked later
            else:
                time_epoch.append(date)

            '''flip the data to be compatible with the mask'''
            tws_one_month = np.flipud(h5py.File(fn, 'r')['data'][:])

            '''loop for the tile'''
            for tile, tile_mask in mask.items():
                basins_num = len(list(tile_mask.keys())) - 1
                '''loop for the subbasins'''
                for i in range(1, basins_num + 1):
                    a = tws_one_month[tile_mask[i]]
                    b = area[tile_mask[i]]
                    c = np.sum(a * b) / np.sum(b)
                    if 'sub_basin_%d' % i in TWS[tile].keys():
                        TWS[tile]['sub_basin_%d' % i].append(c)
                    else:
                        TWS[tile]['sub_basin_%d' % i] = [c]
            pass

        '''save it into h5'''
        for tile, vv in TWS.items():
            out_dir = Path(dir_out)
            hm = h5py.File(out_dir / ('%s_signal.hdf5' % tile), 'w')
            for key, item in vv.items():
                hm.create_dataset(key, data=np.array(item))

            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
            hm.close()

        print('Finished')
        pass

    def basin_COV(self, month_begin='2002-04', month_end='2002-04',
                  dir_in='/media/user/My Book/Fan/GRACE/DDK3_timeseries',
                  dir_out='/media/user/My Book/Fan/GRACE/output'):

        '''load mask'''
        res = 1
        mask = self.valid_mask['1']

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        area = np.cos(np.deg2rad(lat))

        '''calculate the basin averaged TWS, monthly'''
        directory = Path(dir_in)
        fns = os.listdir(directory)
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        print()
        print('Start to pre-process GRACE to obtain signal over places of interest...')

        COV = {}
        for tile, tile_mask in mask.items():
            COV[tile] = []

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

            '''flip the data to be compatible with the mask'''
            tws_one_month = np.flipud(np.load(str(fn)))

            '''loop for the tile'''
            for tile, tile_mask in mask.items():
                basins_num = len(list(tile_mask.keys())) - 1
                '''loop for the subbasins'''
                vv = []
                for i in range(1, basins_num + 1):
                    a = tws_one_month[:, tile_mask[i]]
                    b = area[tile_mask[i]]
                    vv.append(np.sum(a * b, 1) / np.sum(b))

                '''transform into mm'''
                cov = np.cov(np.array(vv) * 1000)

                COV[tile].append(cov)

            pass

        '''save it into h5'''
        for tile, vv in COV.items():
            out_dir = Path(dir_out)
            hm = h5py.File(out_dir / ('%s_cov.hdf5' % tile), 'w')
            hm.create_dataset('data', data=np.array(vv))
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
            hm.close()

        print('Finished')
        pass

    def grid_TWS(self, month_begin='2002-04', month_end='2002-04',
                 dir_in='/media/user/My Book/Fan/GRACE/ewh', dir_out='/media/user/My Book/Fan/GRACE/output'):
        """
        basin averaged TWS, monthly time-series, [mm]
        Be careful to deal with mask and TWS.
        mask: lon: -180-->180, lat: 90--->-90
        tws: lon: -180-->180, lat: -90--->90
        """

        '''load mask'''
        res = 0.5
        mask = self.valid_mask['05']

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        area = np.cos(np.deg2rad(lat))

        '''calculate the basin averaged TWS, monthly'''
        directory = Path(dir_in)
        fns = os.listdir(directory)
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        print()
        print('Start to pre-process GRACE to obtain signal over places of interest...')

        TWS = {}
        for tile, tile_mask in mask.items():
            TWS[tile] = []

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
            date = tn.split('.')[0]
            if len(date.split('-')) == 2:
                time_epoch.append(date + '-15')  # assumed to be the mid day of the month but should be checked later
            else:
                time_epoch.append(date)

            '''flip the data to be compatible with the mask'''
            tws_one_month = np.flipud(h5py.File(fn, 'r')['data'][:])

            '''loop for the tile'''
            for tile, tile_mask in mask.items():
                a = tws_one_month[tile_mask[0]]
                TWS[tile].append(a)

            pass

        '''save it into h5'''
        for tile, vv in TWS.items():
            out_dir = Path(dir_out)
            hm = h5py.File(out_dir / ('%s_gridded_signal.hdf5' % tile), 'w')

            hm.create_dataset(name='tws', data=np.array(vv))

            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
            hm.close()

        print('Finished')
        pass


def demo1():
    # GR = GRACE_preparation(basin_name='Brahmaputra',
    #                        shp_path='../data/basin/shp/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp')
    # GR = GRACE_preparation(basin_name='MDB',
    #                        shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp')
    # GR.basin_TWS(month_begin='2002-04', month_end='2023-06')
    GR = GRACE_preparation(basin_name='GDRB',
                           shp_path='../data/basin/shp/GDRB_shapefiles/GDRB_subbasins.shp')

    GR.generate_mask()
    GR.basin_TWS(month_begin='2002-04', month_end='2010-01')
    GR.grid_TWS(month_begin='2002-04', month_end='2010-01')
    GR.basin_COV(month_begin='2002-04', month_end='2010-01')

    pass


def demo2():
    # gr = GRACE_global_preparation().configure_global_mask(fn='/media/user/Backup Plus/GRACE/GlobalLandMask.hdf5')
    # gr.configure_global_shp()
    # gr.basin_TWS(month_begin='2002-04', month_end='2023-05', dir_in='/media/user/Backup Plus/GRACE/ewh',
    #              dir_out='/media/user/Backup Plus/GRACE/output')
    # gr.grid_TWS(month_begin='2002-04', month_end='2023-05', dir_in='/media/user/Backup Plus/GRACE/ewh',
    #             dir_out='/media/user/Backup Plus/GRACE/output')
    # gr.basin_COV(month_begin='2002-04', month_end='2023-05')

    begin = '2002-04'
    end = '2023-05'
    gr = GRACE_global_preparation().configure_global_mask(fn='/media/user/My Book/Fan/GRACE/basin_selection/GlobalLandMask.hdf5')
    gr.configure_global_shp()
    gr.basin_TWS(month_begin=begin, month_end=end, dir_in='/media/user/My Book/Fan/GRACE/ewh',
                 dir_out='/media/user/My Book/Fan/GRACE/output')
    gr.grid_TWS(month_begin=begin, month_end=end, dir_in='/media/user/My Book/Fan/GRACE/ewh',
                dir_out='/media/user/My Book/Fan/GRACE/output')
    gr.basin_COV(month_begin=begin, month_end=end)
    pass


if __name__ == '__main__':
    demo1()
    # demo2()
