import numpy as np
from src_hydro.config_settings import config_settings
import h5py
from pathlib import Path
from src_hydro.GeoMathKit import GeoMathKit
from calendar import monthrange
from Enumtype import data_var
import shutil


class dataManager_single_run:
    """to store the necessary data and remove all temporal data to save space for the hard drive """

    def __init__(self):
        self._outputOtherVariables_dir = None
        self._variable = None
        self._setting = None
        self._Fhru = None
        self._out_dir = None
        self._mask = None
        self._statesnn = None
        self._state_dir = None

    def configure(self, setting_fn, out_dir, variable=data_var.TotalWater):
        setting = config_settings.loadjson(setting_fn).process()
        self._setting = setting
        self._mask = setting.mask
        self._out_dir = Path(out_dir) / setting.bounds.prefix
        self._state_dir = Path(setting.statedir)
        self._outputOtherVariables_dir = Path(setting.outdir)

        hf = h5py.File(str(Path(setting.input.pars) / 'par.h5'), 'r')
        self._Fhru = hf['data']['Fhru'][:]

        if not self._out_dir.exists():
            self._out_dir.mkdir()

        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']
        self._variable = variable
        # TODO: discharge data to be preserved
        if variable == data_var.TotalWater:
            self._statesnn = statesnn
        elif variable == data_var.GroundWater:
            self._statesnn = ['Sg']
        elif variable == data_var.SoilWater:
            self._statesnn = ['S0', 'Ss', 'Sd']
        elif variable == data_var.SurfaceWater:
            self._statesnn = ['Sr']

        return self

    def aggregation(self, date_begin='2002-04-01', date_end='2002-04-02'):

        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        '''start recording data'''

        mm = -1
        for day in daylist:
            if day.month != mm:
                mm = day.month

                fn_w = self._out_dir / ('%s.%s.h5' % (self._variable.name, day.strftime('%Y%m')))
                print(day.strftime('%Y%m'))
                h5_w = h5py.File(fn_w, 'w')

            fn_r = self._state_dir / ('state.%s.h5' % day.strftime('%Y%m%d'))
            try:
                hf = h5py.File(str(fn_r), 'r')
            except Exception:
                if day != daylist[-1]:
                    print('Gap found for this day: %s' % day.strftime('%Y%m%d'))
                continue

            ta = []
            for key in self._statesnn:
                vv = np.sum(hf[key][:] * self._Fhru, axis=0)
                if key == 'Mleaf':
                    vv *= 4
                ta.append(vv)
            ta_sum = np.sum(np.array(ta), axis=0)

            ta_sum_2D = np.full(np.shape(self._mask), np.nan)

            ta_sum_2D[self._mask] = ta_sum

            h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_2D)

        pass

    def aggregation_other(self, date_begin='2002-04-01', date_end='2002-04-02', variable= 'tws'):
        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        '''start recording data'''

        mm = -1
        for day in daylist:
            if day.year != mm:
                mm = day.year

                fn_w = self._out_dir / ('%s.%s.h5' % (variable, day.strftime('%Y')))
                print(day.strftime('%Y'))
                h5_w = h5py.File(fn_w, 'w')

            fn_r = self._outputOtherVariables_dir / ('output.%s.h5' % day.strftime('%Y%m%d'))
            try:
                hf = h5py.File(str(fn_r), 'r')
            except Exception:
                if day != daylist[-1]:
                    print('Gap found for this day: %s' % day.strftime('%Y%m%d'))
                continue

            ta_sum_2D = np.full(np.shape(self._mask), np.nan)
            ta_sum_2D[self._mask] = hf['Stot'][0,:]

            h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_2D)

        pass

    def cleanup(self):
        """
        To clean up the disk space by removing temporal daily output.
        """

        settings = self._setting
        '''step-1: clean the forcing field that has a large storage'''
        try:
            shutil.rmtree(settings.input.meteo)
        except Exception as inst:
            # print(inst)
            pass

        '''step-2: clean the output'''
        try:
            shutil.rmtree(settings.outdir)
        except Exception as inst:
            # print(inst)
            pass

        '''step-3: clean the state'''
        try:
            shutil.rmtree(settings.statedir)
        except Exception as inst:
            # print(inst)
            pass

        pass


class dataManager_ensemble(dataManager_single_run):

    def __init__(self, ens=2):
        super().__init__()
        self._ens = ens
        pass

    def configure(self, setting_fn, out_dir, variable=data_var.TotalWater):
        setting = config_settings.loadjson(setting_fn).process()
        self._setting = setting
        self._mask = setting.mask
        self._out_dir = Path(out_dir) / setting.bounds.prefix

        self._state_dir = {}
        for ens in range(self._ens + 1):
            self._state_dir[ens] = Path(setting.output.dir) / ('state_' + setting.bounds.prefix + '_ensemble_%s' % ens)

        hf = h5py.File(str(Path(setting.input.pars) / 'par.h5'), 'r')
        self._Fhru = hf['data']['Fhru'][:]

        if not self._out_dir.exists():
            self._out_dir.mkdir()

        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']
        self._variable = variable
        # TODO: discharge data to be preserved
        if variable == data_var.TotalWater:
            self._statesnn = statesnn
        elif variable == data_var.GroundWater:
            self._statesnn = ['Sg']
        elif variable == data_var.SoilWater:
            self._statesnn = ['S0', 'Ss', 'Sd']
        elif variable == data_var.SurfaceWater:
            self._statesnn = ['Sr']

        return self

    def aggregation(self, date_begin='2002-04-01', date_end='2002-04-02'):

        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        '''start recording data'''

        mm = -1
        for day in daylist:
            if day.month != mm:
                mm = day.month
                fn_w = self._out_dir / ('%s.%s.h5' % (self._variable.name, day.strftime('%Y%m')))
                print(day.strftime('%Y%m'))
                h5_w = h5py.File(fn_w, 'w')

            fn_r = self._state_dir[0] / ('state.%s.h5' % day.strftime('%Y%m%d'))
            try:
                hf = h5py.File(str(fn_r), 'r')
            except Exception:
                if day != daylist[-1]:
                    print('Gap found for this day: %s' % day.strftime('%Y%m%d'))
                continue

            ta_sum_ens = []
            for ens in range(1, self._ens + 1):
                fn_r = self._state_dir[ens] / ('state.%s.h5' % day.strftime('%Y%m%d'))
                hf = h5py.File(str(fn_r), 'r')

                ta = []
                for key in self._statesnn:
                    vv = np.sum(hf[key][:] * self._Fhru, axis=0)
                    if key == 'Mleaf':
                        vv *= 4
                    ta.append(vv)

                ta_sum = np.sum(np.array(ta), axis=0)
                ta_sum_ens.append(ta_sum)

            ta_sum = np.mean(ta_sum_ens, axis=0)

            ta_sum_2D = np.full(np.shape(self._mask), np.nan)

            ta_sum_2D[self._mask] = ta_sum

            h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_2D)

        pass

    def cleanup(self):
        """
        To clean up the disk space by removing temporal daily output.
        """

        settings = self._setting
        '''step-1: clean the forcing field that has a large storage'''

        try:
            dir_ens = settings.input.meteo / 'ens_forcing'
            shutil.rmtree(dir_ens)
        except Exception as inst:
            # print(inst)
            pass

        try:
            dir_ens = settings.input.pars / 'ens_par'
            shutil.rmtree(dir_ens)
        except Exception as inst:
            # print(inst)
            pass

        for ens in range(-1, self._ens + 1):

            '''step-2: clean the output'''
            dir_ens = Path(settings.output.dir) / ('state_' + settings.bounds.prefix + '_ensemble_%s' % ens)
            if ens < 0:
                dir_ens = Path(settings.output.dir) / ('state_' + settings.bounds.prefix)
                pass
            try:
                shutil.rmtree(dir_ens)
            except Exception as inst:
                # print(inst)
                pass

            '''step-3: clean the state'''
            dir_ens = Path(settings.output.dir) / ('output_' + settings.bounds.prefix + '_ensemble_%s' % ens)
            if ens < 0:
                dir_ens = Path(settings.output.dir) / ('output_' + settings.bounds.prefix)
            try:
                shutil.rmtree(dir_ens)
            except Exception as inst:
                # print(inst)
                pass

        pass


def demo1():
    dsr = dataManager_single_run().configure(setting_fn='../settings/single_run_Anna/setting.json',
                                             out_dir='/media/user/My Book/Fan/W3RA_data/save_data/SR',
                                             variable=data_var.TotalWater)

    # dsr.aggregation(date_begin='1999-12-31', date_end='2001-01-31')
    # dsr.cleanup()
    dsr.aggregation_other(date_begin='1995-01-01', date_end='2022-12-31')
    pass


def demo2():
    dsr = dataManager_ensemble().configure(setting_fn='../settings/OL_run/setting.json',
                                           out_dir='/media/user/My Book/Fan/W3RA_data/save_data/DA',
                                           variable=data_var.TotalWater)

    dsr.aggregation(date_begin='1999-12-31', date_end='2001-01-31')
    dsr.cleanup()
    pass


if __name__ == '__main__':
    demo1()
