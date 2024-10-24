import sys

sys.path.append('../')

import numpy as np
from src_hydro.config_settings import config_settings
import h5py
from pathlib import Path
from src_hydro.GeoMathKit import GeoMathKit
from calendar import monthrange
from src_postprocessing.Enumtype import data_var, data_dim
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

    def configure(self, setting_fn, out_dir, variable=data_var.TotalWater, dims=data_dim.one_dimension):
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
        elif variable == data_var.DeepSoilWater:
            self._statesnn = ['Sd']

        self._dims = dims


        print()
        print('=================Collecting data for %s================='%variable.name)

        return self

    def reduce_datasize(self, global_basin_mask, save_mask=None):
        """this is to keep only points that participate in the data assimilation"""
        basin_mask = global_basin_mask
        ss = self._setting
        dl = ss.RoI[ss.run.res]
        basin_mask = basin_mask[dl[0]: dl[1], dl[2]: dl[3]]
        basin_mask_1D = basin_mask[self._mask.astype(bool)].astype(bool)
        self._mask *= basin_mask
        self._basin_mask_1D = basin_mask_1D

        if save_mask is not None:
            res = ss.run.res
            err = res / 10
            lats = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
            lons = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
            lats = lats[dl[0]: dl[1]]
            lons = lons[dl[2]: dl[3]]

            fn_w = Path(save_mask) / ('%s.h5' % ss.bounds.prefix)
            h5_w = h5py.File(fn_w, 'w')
            h5_w.create_dataset(name='mask', data=self._mask.astype(int))
            h5_w.create_dataset(name='lat', data= lats)
            h5_w.create_dataset(name='lon', data=lons)
            h5_w.close()

        return self

    def change_variable(self, variable= data_var.TotalWater):
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
        elif variable == data_var.DeepSoilWater:
            self._statesnn = ['Sd']

        print()
        print('=================Collecting data for %s================='%variable.name)
        return self


    def aggregation_daily(self, date_begin='2002-04-01', date_end='2002-04-02'):
        """
        save daily states, but collected on yearly basis
        """

        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        '''start recording data'''

        YY = -1
        for day in daylist:
            if day.year != YY:
                YY = day.year

                fn_w = self._out_dir / ('%s.%s.h5' % (self._variable.name, day.strftime('%Y')))
                print(day.strftime('%Y'))
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
            ta_sum = ta_sum[self._basin_mask_1D]

            if self._dims == data_dim.one_dimension:
                '''save 1-D data for smaller space'''
                h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum)
            else:
                '''save 2D data for easy access'''
                ta_sum_2D = np.full(np.shape(self._mask), np.nan)
                ta_sum_2D[self._mask] = ta_sum
                h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_2D)

        pass

    def aggregation_monthly(self, date_begin='2002-04-01', date_end='2002-04-02'):

        """
        save monthly mean states, but collected on yearly basis.
        """

        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        '''start recording data'''

        mm = -1
        yy = -1
        mres = []
        for day in daylist:
            if day.month != mm or day == daylist[-1]:
                if len(mres) > 0:
                    h5_w.create_dataset(name='%s%02d' % (yy, mm), data=np.mean(np.array(mres), axis=0))

                mm = day.month
                if day != daylist[-1]:
                    print(day.strftime('%Y%m'))

                if day.year != yy:
                    fn_w = self._out_dir / ('%s.%s.h5' % (self._variable.name, day.strftime('%Y')))
                    h5_w = h5py.File(fn_w, 'w')
                    yy = day.year
                mres = []

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
            ta_sum = ta_sum[self._basin_mask_1D]

            if self._dims == data_dim.one_dimension:
                '''save 1-D data for smaller space'''
                mres.append(ta_sum)
            else:
                '''save 2D data for easy access'''
                ta_sum_2D = np.full(np.shape(self._mask), np.nan)
                ta_sum_2D[self._mask] = ta_sum
                mres.append(ta_sum_2D)

        pass

    def aggregation_other(self, date_begin='2002-04-01', date_end='2002-04-02', variable='tws'):
        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)
        """
        daily output of variables other than the states, but collected on yearly basis
        """

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

            '''to set by users!!'''
            ta_sum = hf['Stot'][0, :]
            ta_sum = ta_sum[self._basin_mask_1D]

            if self._dims == data_dim.one_dimension:
                '''save 1-D data for smaller space'''
                h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum)
            else:
                '''save 2D data for easy access'''
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


class dataManager_ensemble_member(dataManager_single_run):
    """
    Instead of the sing run mode, this class targets to extracting data from each ensemble member
    """

    def __init__(self, ens=2):
        super().__init__()
        self._ens = ens
        pass

    def configure(self, setting_fn, out_dir, variable=data_var.TotalWater, dims=data_dim.one_dimension):
        super().configure(setting_fn=setting_fn, out_dir=out_dir, variable=variable, dims=dims)
        self._state_dir = Path(self._setting.output.dir) / (
                'state_' + self._setting.bounds.prefix + '_ensemble_%s' % self._ens)
        self._outputOtherVariables_dir = Path(self._setting.output.dir) / (
                'output_' + self._setting.bounds.prefix + '_ensemble_%s' % self._ens)
        return self


class dataManager_ensemble_statistic(dataManager_single_run):

    def __init__(self, ens_size=2):
        super().__init__()
        self._ens_size = ens_size
        pass

    def configure(self, setting_fn, out_dir_mean, out_dir_std, variable=data_var.TotalWater,
                  dims=data_dim.one_dimension):
        super().configure(setting_fn=setting_fn, out_dir=out_dir_mean, variable=variable, dims=dims)
        self._state_dir = {}
        self._outputOtherVariables_dir = {}
        for ens in range(self._ens_size + 1):
            self._state_dir[ens] = Path(self._setting.output.dir) / (
                    'state_' + self._setting.bounds.prefix + '_ensemble_%s' % ens)
            self._outputOtherVariables_dir[ens] = Path(self._setting.output.dir) / (
                    'output_' + self._setting.bounds.prefix + '_ensemble_%s' % ens)

        self._out_dir_std = Path(out_dir_std) / self._setting.bounds.prefix
        if not self._out_dir_std.exists():
            self._out_dir_std.mkdir()

        return self

    def aggregation_daily(self, date_begin='2002-04-01', date_end='2002-04-02'):

        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        '''start recording data'''

        YY = -1
        for day in daylist:
            if day.year != YY:
                YY = day.year
                fn_w = self._out_dir / ('%s.%s.h5' % (self._variable.name, day.strftime('%Y')))
                fn_std_w = self._out_dir_std / ('%s.%s.h5' % (self._variable.name, day.strftime('%Y')))
                print(day.strftime('%Y%m'))
                h5_w = h5py.File(fn_w, 'w')
                h5_std_w = h5py.File(fn_std_w, 'w')

            fn_r = self._state_dir[0] / ('state.%s.h5' % day.strftime('%Y%m%d'))
            try:
                hf = h5py.File(str(fn_r), 'r')
            except Exception:
                if day != daylist[-1]:
                    print('Gap found for this day: %s' % day.strftime('%Y%m%d'))
                continue

            ta_sum_ens = []
            for ens in range(1, self._ens_size + 1):
                fn_r = self._state_dir[ens] / ('state.%s.h5' % day.strftime('%Y%m%d'))
                hf = h5py.File(str(fn_r), 'r')

                ta = []
                for key in self._statesnn:
                    vv = np.sum(hf[key][:] * self._Fhru, axis=0)
                    if key == 'Mleaf':
                        vv *= 4
                    ta.append(vv)

                ta_sum = np.sum(np.array(ta), axis=0)
                ta_sum = ta_sum[self._basin_mask_1D]
                ta_sum_ens.append(ta_sum)

            ta_sum_mean = np.mean(ta_sum_ens, axis=0)
            ta_sum_std = np.std(ta_sum_ens, axis=0)

            if self._dims == data_dim.one_dimension:
                h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_mean)
                h5_std_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_std)
            else:

                ta_sum_mean_2D = np.full(np.shape(self._mask), np.nan)
                ta_sum_std_2D = np.full(np.shape(self._mask), np.nan)
                ta_sum_mean_2D[self._mask] = ta_sum_mean
                ta_sum_std_2D[self._mask] = ta_sum_std
                h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_mean_2D)
                h5_std_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_std_2D)

        pass

    def aggregation_monthly(self, date_begin='2002-04-01', date_end='2002-04-02'):
        """
        save monthly mean states, but collected on yearly basis.
        """

        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        '''start recording data'''

        mm = -1
        yy = -1
        mres = []
        for day in daylist:
            if day.month != mm or day == daylist[-1]:
                if len(mres) > 0:
                    monthly_mean = np.mean(np.array(mres), axis=0)
                    ensemble_mean = np.mean(monthly_mean, axis=0)
                    ensemble_std = np.std(monthly_mean, axis=0)

                    if self._dims == data_dim.one_dimension:
                        h5_w.create_dataset(name='%s%02d' % (yy, mm), data=ensemble_mean)
                        h5_std_w.create_dataset(name='%s%02d' % (yy, mm), data=ensemble_std)
                    else:
                        ensemble_mean_2D = np.full(np.shape(self._mask), np.nan)
                        ensemble_std_2D = np.full(np.shape(self._mask), np.nan)
                        ensemble_mean_2D[self._mask] = ensemble_mean
                        ensemble_std_2D[self._mask] = ensemble_std
                        h5_w.create_dataset(name='%s%02d' % (yy, mm), data=ensemble_mean_2D)
                        h5_std_w.create_dataset(name='%s%02d' % (yy, mm), data=ensemble_std_2D)

                mm = day.month
                if day != daylist[-1]:
                    print(day.strftime('%Y%m'))

                if day.year != yy:
                    fn_w = self._out_dir / ('%s.%s.h5' % (self._variable.name, day.strftime('%Y')))
                    h5_w = h5py.File(fn_w, 'w')
                    fn_std_w = self._out_dir_std / ('%s.%s.h5' % (self._variable.name, day.strftime('%Y')))
                    h5_std_w = h5py.File(fn_std_w, 'w')
                    yy = day.year
                mres = []

            try:
                h5py.File(str(self._state_dir[0] / ('state.%s.h5' % day.strftime('%Y%m%d'))), 'r')
            except Exception:
                if day != daylist[-1]:
                    print('Gap found for this day: %s' % day.strftime('%Y%m%d'))
                continue

            ta_sum_ens = []
            for ens in range(1, self._ens_size + 1):
                fn_r = self._state_dir[ens] / ('state.%s.h5' % day.strftime('%Y%m%d'))
                hf = h5py.File(fn_r, 'r')

                ta = []
                for key in self._statesnn:
                    vv = np.sum(hf[key][:] * self._Fhru, axis=0)
                    if key == 'Mleaf':
                        vv *= 4
                    ta.append(vv)
                ta_sum = np.sum(np.array(ta), axis=0)
                ta_sum = ta_sum[self._basin_mask_1D]
                ta_sum_ens.append(ta_sum)

            ta_sum_ens = np.array(ta_sum_ens)
            mres.append(ta_sum_ens)

        pass

    def aggregation_others(self, date_begin='2002-04-01', date_end='2002-04-02', variable='tws'):

        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        '''start recording data'''

        YY = -1
        for day in daylist:
            if day.year != YY:
                YY = day.year

                fn_w = self._out_dir / ('%s.%s.h5' % (variable, day.strftime('%Y')))
                # fn_std_w = self._out_dir_std / ('%s.%s.h5' % (variable, day.strftime('%Y')))
                print(day.strftime('%Y%m'))
                h5_w = h5py.File(fn_w, 'w')
                # h5_std_w = h5py.File(fn_std_w, 'w')

            fn_r = self._outputOtherVariables_dir[0] / ('output.%s.h5' % day.strftime('%Y%m%d'))
            try:
                hf = h5py.File(str(fn_r), 'r')
            except Exception:
                if day != daylist[-1]:
                    print('Gap found for this day: %s' % day.strftime('%Y%m%d'))
                continue

            ta_sum_ens = []
            for ens in range(1, self._ens_size + 1):
                fn_r = self._outputOtherVariables_dir[ens] / ('output.%s.h5' % day.strftime('%Y%m%d'))
                hf = h5py.File(str(fn_r), 'r')

                '''to set by users!!'''
                ta_sum = hf['Stot'][0, :]
                ta_sum = ta_sum[self._basin_mask_1D]

                ta_sum_ens.append(ta_sum)

            ta_sum_mean = np.mean(ta_sum_ens, axis=0)
            ta_sum_std = np.std(ta_sum_ens, axis=0)

            if self._dims == data_dim.one_dimension:
                h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_mean)
                # h5_std_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_std)
            else:

                ta_sum_mean_2D = np.full(np.shape(self._mask), np.nan)
                # ta_sum_std_2D = np.full(np.shape(self._mask), np.nan)
                ta_sum_mean_2D[self._mask] = ta_sum_mean
                # ta_sum_std_2D[self._mask] = ta_sum_std
                h5_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_mean_2D)
                # h5_std_w.create_dataset(name=day.strftime('%Y%m%d'), data=ta_sum_std_2D)

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

        for ens in range(-1, self._ens_size + 1):

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
    from src_DA.shp2mask import basin_shp_process

    basin = 'US'
    shp_path = '../data/basin/shp/USgrid/US_subbasins.shp'
    global_basin_mask = basin_shp_process(res=0.1, basin_name=basin, save_dir='../data/basin/mask').shp_to_mask(
        shp_path=shp_path).mask[0]
    dsr = dataManager_ensemble_statistic().configure(setting_fn='../settings/Ucloud_DA/setting.json',
                                                     out_dir_mean='/work/data_for_w3/w3ra/save_data/DA_monthly_mean',
                                                     out_dir_std='/work/data_for_w3/w3ra/save_data/DA_monthly_std',
                                                     variable=data_var.TotalWater, dims=data_dim.one_dimension)
    dsr.reduce_datasize(global_basin_mask=global_basin_mask, save_mask='/work/data_for_w3/w3ra/save_data/Mask')

    '''====================================================================='''
    dsr.aggregation_monthly(date_begin='2002-01-01', date_end='2023-03-31')

    dsr.change_variable(variable=data_var.GroundWater)
    dsr.aggregation_monthly(date_begin='2002-01-01', date_end='2023-03-31')

    dsr.change_variable(variable=data_var.DeepSoilWater)
    dsr.aggregation_monthly(date_begin='2002-01-01', date_end='2023-03-31')

    dsr.change_variable(variable=data_var.SoilWater)
    dsr.aggregation_monthly(date_begin='2002-01-01', date_end='2023-03-31')

    dsr.change_variable(variable=data_var.SurfaceWater)
    dsr.aggregation_monthly(date_begin='2002-01-01', date_end='2023-03-31')
    pass


if __name__ == '__main__':
    demo2()
