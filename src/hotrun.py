import sys

sys.path.append('../')

from src.config_settings import config_settings
from src.config_parameters import config_parameters
from src.EnumType import states_var
from src.GeoMathKit import GeoMathKit
from pathlib import Path
import numpy as np
from src.W3RA_timestep_model import timestep_model
from src.ext_adapter import ext_adapter
from src.model_initialise import model_initialise
import h5py
from datetime import datetime


class model_run:

    def __init__(self, settings: config_settings, model_init: model_initialise, par: config_parameters,
                 ext: ext_adapter):
        self.__settings = settings
        self.__states_init = model_init.states_hotrun
        self.__par = par
        self.__ext = ext
        self.__timestepModel = timestep_model(settings=settings, par=par, ext=ext)

        pass

    def execute(self):
        settings = self.__settings
        states = self.__states_init
        par = self.__par
        ext_adapter = self.__ext
        timestepModel = self.__timestepModel

        mm = -1

        daylist = GeoMathKit.dayListByDay(settings.run.fromdate, settings.run.todate)

        print('\n===== RUN for %s to %s =====' % (settings.run.fromdate, settings.run.todate))

        for day in daylist:

            if day.month != mm:
                print('\nDoing year/month %04d%02d' % (day.year, day.month))
                # '''update the par '''
                # climate.update_par(par=par, date=day)
                # '''update the adapter'''
                # ext_adapter.updatePar(par=par)
                # '''update the time-step W3 model'''
                # timestepModel.updatePar(par=par)
                mm = day.month

                pass
            '''prepare external forcing'''
            print('.', end='')
            ext_adapter.update(date=day)

            '''main entrance to the W3 update'''
            out = timestepModel.updateState(state=states)

            '''save the state of the final epoch or save everyday'''
            if (day is daylist[-1]) or settings.run.save_states_every_day:
                fn = Path(settings.statedir) / ('state.%04d%02d%02d.h5' % (day.year, day.month, day.day))
                f_w = h5py.File(fn, 'w')
                statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']
                for key in statesnn:
                    f_w.create_dataset(data=states[states_var[key]], name=key)
                f_w.close()

            '''save the output everyday'''
            if settings.run.save_output_every_day:
                out_nn = []
                for x, y in settings.output.var.items():
                    if y:
                        out_nn.append(x)
                fn = Path(settings.outdir) / ('output.%04d%02d%02d.h5' % (day.year, day.month, day.day))
                f_w = h5py.File(fn, 'w')
                for key in out_nn:
                    f_w.create_dataset(data=out[key], name=key)
                f_w.close()

        pass


class model_run_daily:

    def __init__(self, settings: config_settings, model_init: model_initialise, par: config_parameters,
                 ext: ext_adapter):
        self.__settings = settings
        self.__states_init = model_init.states_hotrun
        self.__par = par
        self.__ext = ext
        self.__timestepModel = timestep_model(settings=settings, par=par, ext=ext)

        pass

    def update(self, is_first_day=False, previous_states=None, day='2002-02-04'):

        day = datetime.strptime(day, '%Y-%m-%d')
        settings = self.__settings

        par = self.__par
        ext_adapter = self.__ext
        timestepModel = self.__timestepModel

        if is_first_day:
            states = self.__states_init
        else:
            states = previous_states

        '''prepare external forcing'''
        print(day.strftime('%Y-%m-%d'))
        ext_adapter.update(date=day)

        '''main entrance to the W3 update'''
        out = timestepModel.updateState(state=states)

        '''save the state of the final epoch or save everyday'''
        if settings.run.save_states_every_day:
            fn = Path(settings.statedir) / ('state.%04d%02d%02d.h5' % (day.year, day.month, day.day))
            f_w = h5py.File(fn, 'w')
            statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'Mleaf', 'FreeWater', 'DrySnow']
            for key in statesnn:
                f_w.create_dataset(data=states[states_var[key]], name=key)
            f_w.close()

        '''save the output everyday'''
        if settings.run.save_output_every_day:
            out_nn = []
            for x, y in settings.output.var.items():
                if y:
                    out_nn.append(x)
            fn = Path(settings.outdir) / ('output.%04d%02d%02d.h5' % (day.year, day.month, day.day))
            f_w = h5py.File(fn, 'w')
            for key in out_nn:
                f_w.create_dataset(data=out[key], name=key)
            f_w.close()

        return states


def demo1():
    dp = '../settings/setting_2.json'
    settings = config_settings.loadjson(dp).process()
    par = config_parameters(settings)
    model_init = model_initialise(settings=settings, par=par).configure_InitialStates()
    ext = ext_adapter(par=par, settings=settings)
    model_instance = model_run(settings=settings, par=par, model_init=model_init, ext=ext)
    model_instance.execute()
    pass


if __name__ == '__main__':
    demo1()
