import sys

sys.path.append('../')

import numpy as np
import h5py
import netCDF4 as nc
from src_GHM.GeoMathKit import GeoMathKit
from pathlib import Path
from datetime import datetime, timedelta
from src_GHM.EnumType import states_var
import calendar
import os


class obs_auxiliary:
    """
    1. provide a list of observation's time reference: which days the obs stand for.
    2. set layer attribution
    3. more..
    """

    def __init__(self):
        self._layer = None
        self._TimeRef = {
            'duration': None,
            'time_epoch': None
        }
        pass

    def setLayers(self, layer: dict = None):
        if layer is None:
            layer = {
                states_var.S0: True,
                states_var.Ss: True,
                states_var.Sd: True,
                states_var.Sr: True,
                states_var.Sg: True,
                states_var.Mleaf: True,
                states_var.FreeWater: True,
                states_var.DrySnow: True
            }

        self._layer = []
        for key, value in layer.items():
            if value: self._layer.append(key.name)

        # print(self._layer)
        return self

    def getLayers(self):
        return self._layer

    def getTimeReference(self):
        return self._TimeRef

    def setTimeReference(self):
        return self

    def selectSubTimePeriod(self, day_begin='2003-09-01', day_end='2006-12-31'):
        timeRef = self.getTimeReference()
        ss = np.array(timeRef['duration'])
        assert day_end > ss[0] and day_begin < ss[-1]

        a = np.where(ss > day_begin)[0][0]
        b = np.where(ss < day_end)[0][-1]

        if day_end >= timeRef['duration'][b].split('_')[1]:
            b += 1

        x = {}
        x['time_epoch'] = timeRef['time_epoch'][a:b].copy()
        x['duration'] = timeRef['duration'][a:b].copy()
        return x, a, b

    def save_H5(self, basin_name: str, dir_out='/media/user/Backup Plus/GRACE/Mascon/CSR/test'):
        """save it into h5"""
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_aux.hdf5' % basin_name), 'w')
        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=self._TimeRef['time_epoch'], dtype=dt)
        hm.create_dataset('duration', data=self._TimeRef['duration'], dtype=dt)
        hm.create_dataset('layer', data=self._layer, dtype=dt)
        hm.close()
        pass


class aux_GRACE_mascon_monthly(obs_auxiliary):

    def __init__(self):
        super().__init__()
        pass

    def setTimeReference(self, month_begin='2002-04', month_end='2023-05',
                         dir_in='/media/user/Backup Plus/GRACE/Mascon/CSR'):

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

        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        time_epoch = []
        duration = []
        for month in monthlist:
            this_month_shortname = month.strftime('%Y-%m')
            # print(this_month_shortname)
            #
            if this_month_shortname not in available_month.keys():
                continue

            time_epoch.append(this_month_shortname + '-15')

            _, days_of_the_month = calendar.monthrange(month.year, month.month)
            dt = this_month_shortname + '-01' + '_' + this_month_shortname + '-%s' % days_of_the_month
            duration.append(dt)

        self._TimeRef['duration'] = duration
        self._TimeRef['time_epoch'] = time_epoch

        return self


class aux_GRACE_SH_monthly(obs_auxiliary):

    def __init__(self):
        super().__init__()
        pass

    def setTimeReference(self, month_begin='2002-04', month_end='2022-04',
                         dir_in='/media/user/My Book/Fan/GRACE/ewh'):

        directory = Path(dir_in)
        fns = os.listdir(directory)
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        time_epoch = []
        duration = []
        for month in monthlist:

            '''search for the file'''
            fn = month.strftime('%Y-%m')
            tn = None
            for filename in fns:
                if fn in filename:
                    tn = filename

            if tn is None:
                continue

            # print(tn)
            fn = directory / tn

            date = tn.split('.')[0]
            if len(date.split('-')) == 2:
                time_epoch.append(
                    date + '-15')  # TODO: assumed to be the mid day of the month but should be checked later
            else:
                time_epoch.append(date)

            _, days_of_the_month = calendar.monthrange(month.year, month.month)
            ss = month.strftime('%Y-%m')
            dt = ss + '-01' + '_' + ss + '-%s' % days_of_the_month
            duration.append(dt)

        self._TimeRef['duration'] = duration
        self._TimeRef['time_epoch'] = time_epoch

        return self


class aux_ESAsing_5daily(obs_auxiliary):

    def __init__(self):
        super().__init__()
        pass

    def setTimeReference(self, day_begin='1990-04-01', day_end='2008-05-01',
                         dir_in='/media/user/My Book/Fan/ESA_SING/ESA_5daily'):

        day_begin = datetime.strptime(day_begin, '%Y-%m-%d')
        day_end = datetime.strptime(day_end, '%Y-%m-%d')
        year_limit = np.arange(1995, 2007)
        time_epoch = []
        duration = []
        for year in year_limit:
            directory = Path(dir_in) / ('%s' % year)
            fns = os.listdir(directory)
            fns.sort()

            for ff in fns:
                x = ff.split('_')[4]
                a, b = x.split('-')
                ss1 = datetime.strptime(a, '%Y%m%d')
                ss2 = datetime.strptime(b, '%Y%m%d')
                ss2 = ss2 - timedelta(days=int(1))
                ss3 = ss1 + timedelta(days=int(2))

                if ss1 >= day_begin and ss2 <= day_end:
                    duration.append(ss1.strftime('%Y-%m-%d') + '_' + ss2.strftime('%Y-%m-%d'))
                    time_epoch.append(ss3.strftime('%Y-%m-%d'))

            pass

        self._TimeRef['duration'] = duration
        self._TimeRef['time_epoch'] = time_epoch

        return self

class aux_ESM3_5daily(obs_auxiliary):

    def __init__(self):
        super().__init__()
        pass

    def setTimeReference(self, day_begin='1990-04-01', day_end='2008-05-01',
                         dir_in='/media/user/My Book/Fan/ESA_SING/ESM3.0/5daily'):

        day_begin = datetime.strptime(day_begin, '%Y-%m-%d')
        day_end = datetime.strptime(day_end, '%Y-%m-%d')
        year_limit = np.arange(2007, 2021)
        time_epoch = []
        duration = []
        for year in year_limit:
            directory = Path(dir_in) / ('%s' % year)
            fns = os.listdir(directory)
            fns.sort()

            for ff in fns:
                x = ff.split('.')[0].split('_')
                a, b = x[2], x[3]
                ss1 = datetime.strptime(a, '%Y%m%d')
                ss2 = datetime.strptime(b, '%Y%m%d')
                ss2 = ss2 - timedelta(days=int(1))
                ss3 = ss1 + timedelta(days=int(2))

                if ss1 >= day_begin and ss2 <= day_end:
                    duration.append(ss1.strftime('%Y-%m-%d') + '_' + ss2.strftime('%Y-%m-%d'))
                    time_epoch.append(ss3.strftime('%Y-%m-%d'))

            pass

        self._TimeRef['duration'] = duration
        self._TimeRef['time_epoch'] = time_epoch

        return self


def demo1():
    # aux = aux_ESAsing_5daily()
    # aux = aux_GRACE_SH_monthly()
    aux = aux_GRACE_mascon_monthly()
    aux.setTimeReference(dir_in='/media/user/My Book/Fan/SummerSchool/External Data/GRACE/SaGEA/signal_Mascon')

    pass


if __name__ == '__main__':
    demo1()
