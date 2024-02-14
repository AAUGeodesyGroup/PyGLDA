
import json
from pathlib import Path
import time
import numpy as np
import h5py
from src.config_settings import config_settings


class config_parameters:

    def __init__(self, settings: config_settings):
        self.settings = settings
        par = self.__configure1()

        for var in par.keys():
            # self.__setattr__(var, np.array(temp[:][self.settings.mask], float))
            self.__setattr__(var, par[var])

        self.Nhru = int(self.Nhru)
        pass



    def __configure1(self):
        """
        Set parameters
        """
        '''H5 convert to dict'''
        par = {}
        file = h5py.File(Path(self.settings.input.pars) / 'par.h5', 'r')
        key_word = None
        if self.settings.parallel_id <0:
            key_word = 'data'
        else:
            key_word = 'ens_%s'%self.settings.parallel_id

        dict_group_load = file[key_word]
        dict_group_keys = dict_group_load.keys()
        for k in dict_group_keys:
            par[k] = dict_group_load[k][:]

            if np.shape(par[k]) == (1,1):
                par[k] = par[k][0,0]

        return par

    def __configure2(self):
        pass


class perturbing_parameters:
    """
    prepare for later use
    """

    def __init__(self, par: config_parameters):
        self.par = par
        self.methods = {1: self.__method1,
                        2: self.__method2,
                        3: self.__method3}
        pass

    def pertubate(self, method: int, *args):
        return self.methods[method](*args)

    def __method1(self, *args):
        return self.par

    def __method2(self, *args):
        return self.par

    def __method3(self, *args):
        return self.par


def demo1():
    # dp = config_settings.save_default()
    dp = '../settings/setting_2.json'
    settings = config_settings.loadjson(dp).process()
    par = config_parameters(settings).par
    pass


if __name__ == '__main__':
    demo1()
