from src_GHM.config_settings import config_settings
from src_GHM.config_parameters import config_parameters
from src_GHM.climatologies import climatologies
from src_GHM.ext_forcing import load_forcing
from src_GHM.EnumType import states_var, forcing
from src_GHM.GeoMathKit import GeoMathKit
from datetime import datetime
from pathlib import Path
import numpy as np
import h5py


class ext_adapter:

    def __init__(self, par: config_parameters, settings: config_settings):
        self.__date = datetime.strptime('10000101', '%Y%m%d')
        self.__par = par

        self.__forcing = load_forcing(par=par, settings=settings)
        self.__clim = climatologies(settings=settings)

        self.Pg = None
        self.Rg = None
        self.Ta = None
        self.T24 = None
        self.pe = None
        self.u2 = None
        self.pair = None
        self.ns_alb = None
        self.fday = None
        pass

    def update(self, date: datetime):
        par = self.__par
        forcing = self.__forcing.update_Forcing(date=date)
        clim = self.__clim.update_clim(date=date).clim

        '''Daylength'''
        num = (date - datetime.strptime('%04d-01-01' % date.year, '%Y-%m-%d')).days
        doy = num + 1
        m = 1 - np.tan(par.latitude * np.pi / 180) * np.tan(
            (23.439 * np.pi / 180) * np.cos(2. * np.pi * (doy + 9) / 365.25))
        '''%fraction daylength'''
        fday = np.fmin(np.fmax(0.02, np.arccos(1 - np.fmin(np.fmax(0, m), 2)) / np.pi), 1)

        '''Assign forcing and estimate effective meteorological variables'''
        Pg = forcing['prcp'] * (24 * 60 * 60)  # % from kg m-2 s-1 to mm d-1
        Rg = np.fmax(forcing['dswrf'], 0.01)  # %  already in W m-2 s-1; set minimum of 0.01 to avoid numerical problems
        TMIN, TMAX = forcing['tmin'], forcing['tmax']
        Ta = (TMIN + 0.75 * (TMAX - TMIN)) - 273.15  # from K to degC
        T24 = (TMIN + 0.5 * (TMAX - TMIN)) - 273.15  # from K to degC
        pe = 610.8 * np.exp(17.27 * (TMIN - 273.15) / (237.3 + TMIN - 273.15))

        '''% rescale factor because windspeed climatology is at 50m'''
        WindFactor = 0.59904
        u2 = WindFactor * clim['windspeed'] * (1 - (1 - fday) * 0.25) / fday
        pair = clim['pres']  # % already in Pa
        ns_alb = clim['albedo']

        self.Pg = Pg
        self.Rg = Rg
        self.Ta = Ta
        self.T24 = T24
        self.pe = pe
        self.u2 = u2
        self.pair = pair
        self.ns_alb = ns_alb
        self.fday = fday

        return self


def demo1():
    dp = '../settings/setting_2.json'
    settings = config_settings.loadjson(dp).process()
    par = config_parameters(settings)
    day = datetime.strptime('20030101', '%Y%m%d')
    input = ext_adapter(par=par, settings=settings)
    input.update(date=day)
    pass


if __name__ == '__main__':
    demo1()
