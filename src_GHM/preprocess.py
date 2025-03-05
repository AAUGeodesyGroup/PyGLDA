import numpy as np
from src_GHM.config_settings import config_settings
import h5py
import json
from pathlib import Path
import metview as mv
from datetime import datetime
from src_GHM.GeoMathKit import GeoMathKit


class preprocess_base:

    def __init__(self, dp1: str):
        self._settings_1 = json.load(open(dp1, 'r'))
        pass

    def process(self, dp2: str):
        settings_2 = json.load(open(dp2, 'r'))

        self._set_mask(global_mask_dir=settings_2['mask_dir'], out_dir=settings_2['out_dir'])

        self._crop_climatelogies(data_path=settings_2['clim_dir'])

        self._crop_parameters(data_path=settings_2['par_dir'])

        self._crop_forcing_field(data_path=settings_2['forcing_dir'],
                                 date_begin=settings_2['date_begin'],
                                 date_end=settings_2['date_end'])
        pass

    def _set_mask(self, global_mask_dir: str, out_dir: str, *args):

        fn = Path(global_mask_dir) / 'mask.h5'
        file = h5py.File(str(fn), 'r')
        mask = file['mask'][:].astype(bool)
        file.close()

        '''to find common part of all the masks'''
        for ext in args:
            mask *= ext

        '''to select mask within a given tile'''
        settings_1 = self._settings_1
        '''Define Resolution'''
        res = settings_1['run']['res']
        err = res / 10
        lats = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lons = np.arange(-180 + res / 2, 180 - res / 2 + err, res)

        bounds = settings_1['bounds']
        lati = [np.argmin(np.fabs(lats - max(bounds['lat']))),
                np.argmin(np.fabs(lats - min(bounds['lat'])))]
        loni = [np.argmin(np.fabs(lons - min(bounds['lon']))),
                np.argmin(np.fabs(lons - max(bounds['lon'])))]

        self._RoI = [lati[0], lati[1] + 1, loni[0], loni[1] + 1]
        # self.RoI[res] = [lati[0], lati[1], loni[0], loni[1] + 1]
        self._CoorRoI = [lats[self._RoI[0]], lats[self._RoI[1] - 1],
                         lons[self._RoI[2]], lons[self._RoI[3] - 1]]
        dl = self._RoI
        tmp = np.zeros(np.shape(mask))
        tmp[dl[0]:dl[1], dl[2]:dl[3]] = 1
        mask *= tmp.astype(bool)
        self._local_mask = mask[dl[0]:dl[1], dl[2]:dl[3]]

        '''save final mask'''
        self._out_dir = Path(out_dir) / settings_1['bounds']['prefix']
        if not self._out_dir.exists():
            self._out_dir.mkdir()

        out_dir = self._out_dir / 'mask'
        if not out_dir.exists():
            out_dir.mkdir()

        fn = out_dir / 'mask_global.h5'
        hf = h5py.File(str(fn), 'w')
        hf.create_dataset('mask', data=mask[0:len(lats), 0:len(lons)].astype(int))
        hf.close()

        fn = out_dir / 'mask.h5'
        hf = h5py.File(str(fn), 'w')
        hf.create_dataset('mask', data=self._local_mask.astype(int))
        hf.close()
        pass

    def _crop_forcing_field(self, data_path: str, date_begin='', date_end=''):

        # print('*************forcing field******************')

        dl = self._RoI
        lat0, lat1, lon0, lon1 = self._CoorRoI

        out_dir = Path(self._out_dir) / 'forcing'
        if not out_dir.exists():
            out_dir.mkdir()

        months = GeoMathKit.monthListByMonth(begin=date_begin, end=date_end)

        for date in months:
            print('%s'%date.strftime('%Y-%m'))
            fn = Path(data_path) / ('W3_ERA5_daily_%04d%02d.grib' % (date.year, date.month))
            pars = ['ssrd', '2t', 'tp']

            fn_out = out_dir / ('%s.h5' % date.strftime('%Y-%m'))
            file = h5py.File(str(fn_out), 'w')
            dict_group = file.create_group('data')
            for par in pars:
                grib = mv.read(source=str(fn), area=[lat0, lon0, lat1, lon1], param=par)
                '''A potential risk here for cropping the area'''
                a = grib.values().reshape(  tuple([-1] + list(np.shape(self._local_mask)))  )[:, self._local_mask]
                dict_group[par] = a
                pass
            file.close()

        pass

    def _crop_climatelogies(self, data_path: str):


        dl = self._RoI

        out_dir = Path(self._out_dir) / 'clim'
        if not out_dir.exists():
            out_dir.mkdir()

        pp = Path(data_path)
        dict_new = {}
        for i in range(1, 13):
            fn = pp / ('clim_%02d.h5' % i)
            file = h5py.File(str(fn), 'r')
            dict_group_load = file['data']
            dict_group_keys = dict_group_load.keys()
            for k in dict_group_keys:
                dict_new[k] = dict_group_load[k][dl[0]:dl[1], dl[2]:dl[3]][self._local_mask]
            file.close()

            fn = out_dir / ('clim_%02d.h5' % i)
            file = h5py.File(str(fn), 'w')
            dict_group = file.create_group('data')
            for k, v in dict_new.items():
                dict_group[k] = v
                pass
            file.close()

        pass

    def _crop_parameters(self, data_path: str):


        dl = self._RoI

        fn = Path(data_path) / 'par.h5'
        dict_new = {}
        file = h5py.File(str(fn), 'r')
        dict_group_load = file['data']
        dict_group_keys = dict_group_load.keys()
        for k in dict_group_keys:
            tmp = dict_group_load[k]
            if np.shape(tmp) == (1, 1):
                dict_new[k] = tmp[:]
            else:
                dict_new[k] = tmp[:, dl[0]:dl[1], dl[2]:dl[3]][:, self._local_mask]
            pass

        out_dir = Path(self._out_dir) / 'par'
        if not out_dir.exists():
            out_dir.mkdir()

        fn = out_dir / 'par.h5'
        hf = h5py.File(str(fn), 'w')
        dict_group = hf.create_group('data')
        for k, v in dict_new.items():
            dict_group[k] = v
            pass
        hf.close()

        pass

    @staticmethod
    def save_default_json():
        dp = {}
        dp['date_begin'] = '2000-01'
        dp['date_end'] = '2000-02'
        dp['out_dir'] = '/media/user/My Book/Fan/W3RA_data/crop_input'

        dp['mask_dir'] = '/media/user/My Book/Fan/W3RA_data/global/pyW3RA/mask'
        dp['forcing_dir'] = '/media/user/My Book/Fan/W3RA_data/global/pyW3RA/forcing'
        dp['clim_dir'] = '/media/user/My Book/Fan/W3RA_data/global/pyW3RA/clim'
        dp['par_dir'] = '/media/user/My Book/Fan/W3RA_data/global/pyW3RA/par'

        default_path = Path.cwd().parent / 'settings' / 'pre_process.json'
        with open(default_path, 'w') as f:
            json.dump(dp, f, indent=4)

        return default_path


def demo1():
    dp1 = '../settings/setting_2.json'
    pre = preprocess_base(dp1=dp1)

    dp2 = '../settings/pre_process.json'
    pre.process(dp2=dp2)

    pass


if __name__ == '__main__':
    demo1()
