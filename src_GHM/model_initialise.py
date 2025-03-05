import h5py

from src_GHM.config_settings import config_settings
from src_GHM.config_parameters import config_parameters
from src_GHM.EnumType import init_mode, states_var
from src_GHM.GeoMathKit import GeoMathKit
from datetime import datetime, timedelta
from pathlib import Path
import netCDF4 as nc
import numpy as np


class model_initialise:

    def __init__(self, settings: config_settings, par: config_parameters):

        self.states_hotrun = {}

        self.__settings = settings
        self.__par = par

        pass

    def configure_InitialStates(self):
        print('\nInitializing model states: %s' % self.__settings.init.mode.name)

        settings, par = self.__settings, self.__par

        if self.__settings.init.mode == init_mode.resume:

            today = datetime.strptime(settings.run.fromdate, '%Y-%m-%d')
            yesterday = today - timedelta(days=1)

            states_file = 'state.' + yesterday.strftime("%Y%m%d.h5")
            fn = Path(settings.statedir)/states_file
            assert fn.is_file(), 'Initialization failed: file %s not found' % fn
            dd = h5py.File(fn, 'r')
            for key in dd.keys():
                self.states_hotrun[states_var[key]] = dd[key][:]
            pass

        elif self.__settings.init.mode == init_mode.warm:
            today = datetime.strptime(settings.run.fromdate, '%Y-%m-%d')
            yesterday = datetime.strptime(settings.init.date, '%Y-%m-%d')
            assert today == yesterday+timedelta(days=1), 'Mismatch between the initial date and running date!'

            if len(self.__settings.init.spec) == 0:
                states_file = 'state.' + yesterday.strftime("%Y%m%d.h5")
            else:
                states_file = self.__settings.init.spec + '_state.' + yesterday.strftime("%Y%m%d.h5")

            fn = Path(settings.init.dir) / states_file
            assert fn.is_file(), 'Initialization failed: file %s not found' % fn
            dd = h5py.File(fn, 'r')
            for key in dd.keys():
                self.states_hotrun[states_var[key]] = dd[key][:]
            pass

        elif self.__settings.init.mode == init_mode.cold:
            Ncells_1 = (1, len(settings.mask[settings.mask]))
            Ncells_2 = (par.Nhru, len(settings.mask[settings.mask]))

            print('Guess cold states.')

            self.states_hotrun[states_var.S0] = 0.5 * par.S0FC* np.ones(Ncells_2)
            self.states_hotrun[states_var.Ss] = 0.5 * par.SsFC* np.ones(Ncells_2)
            self.states_hotrun[states_var.Sd] = 0.5 * par.SdFC* np.ones(Ncells_2)
            self.states_hotrun[states_var.Sr] = np.zeros(Ncells_1)

            #todo: set a suitable value!
            self.states_hotrun[states_var.Sg] = 100* np.ones(Ncells_1)
            # self.states_hotrun[states_var.Sg] = 1000 * par.porosity * 0.5
            # above: initial state can be chosen  to fasten spinup
            self.states_hotrun[states_var.Mleaf] = 2.0 / par.SLA * np.ones(Ncells_2)
            self.states_hotrun[states_var.DrySnow] = np.zeros(Ncells_2)
            self.states_hotrun[states_var.FreeWater] = np.zeros(Ncells_2)
            pass


        self.__spin_up_ready()

        '''masked array converts into array'''
        for key in self.states_hotrun.keys():
            self.states_hotrun[key] = np.array(self.states_hotrun[key])
        pass

        return self

    def __spin_up_ready(self):

        settings = self.__settings

        if self.__settings.init.mode == init_mode.cold:
            '''Run - note that this first run is for warm up and will be overwritten by final run.'''
            settings.run.fromdate = str(settings.init.spinup[0])
            settings.run.todate = str(settings.init.spinup[1])

            daylist = GeoMathKit.dayListByDay(settings.run.fromdate, settings.run.todate)
            Ndays = len(daylist)

            print('Spin up run for %s to %s (%d days)' % (settings.run.fromdate, settings.run.todate, Ndays))

            # '''estimate net ice flow rate'''
            # self.__par.NetIceFlow = ((self.__par.PermIce > 0).astype(np.int)) * self.states_hotrun[
            #     states_var.snow_dry] / Ndays
            # np.save('NetIceFlow.npy', self.__par.NetIceFlow)
            # '''reset to 10 years worth of net glacier movement.'''
            # self.states_hotrun[states_var.snow_dry] = (10 * 365) * self.__par.NetIceFlow
            pass
        elif self.__settings.init.mode == init_mode.warm:
            # '''set to 10 years worth of net glacier movement.'''
            # self.states_hotrun[states_var.snow_dry] = (10 * 365) * self.__par.NetIceFlow
            pass
        elif self.__settings.init.mode == init_mode.resume:
            pass

        return self


def demo1():
    dp = '../settings/setting_2.json'
    settings = config_settings.loadjson(dp).process()
    par = config_parameters(settings)
    model_init = model_initialise(settings=settings, par=par).configure_InitialStates()
    pass


if __name__ == '__main__':
    demo1()
