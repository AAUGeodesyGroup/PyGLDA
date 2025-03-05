from src_GHM.config_settings import config_settings
from src_GHM.config_parameters import config_parameters
from src_GHM.EnumType import states_var, forcing
from src_GHM.GeoMathKit import GeoMathKit
from datetime import datetime
from pathlib import Path
import numpy as np
import h5py


class climatologies:

    def __init__(self, settings: config_settings):
        self.__date = datetime.strptime('10000101', '%Y%m%d')
        self.__settings = settings
        self.__firsttime = True
        self.clim = {}
        pass

    def update_clim(self, date: datetime):

        if not self.__firsttime:
            if date.month == self.__date.month:
                return self
        else:
            self.__firsttime = False
            pass

        '''Otherwise update the parameters with climatological data'''
        # print('load climatologies', end='')

        settings = self.__settings

        fn = Path(settings.input.clim) / ('clim_%02d.h5' % date.month)
        file = h5py.File(fn, 'r')
        dict_group_load = file['data']
        dict_group_keys = dict_group_load.keys()
        for k in dict_group_keys:
            self.clim[k] = dict_group_load[k][:][None, :]
            pass

        return self


def demo1():
    dp = '../settings/setting_2.json'
    settings = config_settings.loadjson(dp).process()
    par = config_parameters(settings)
    clim = climatologies(settings).update_clim(date=datetime.strptime('20030101', '%Y%m%d'))
    pass


if __name__ == '__main__':
    demo1()
