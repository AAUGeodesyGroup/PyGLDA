import h5py
import numpy as np

from src_hydro.config_settings import config_settings
from src_hydro.config_parameters import config_parameters
from src_hydro.EnumType import states_var, forcing, forcingSource, perturbation_choice_forcing_field
from src_hydro.GeoMathKit import GeoMathKit
from datetime import datetime, timedelta
from pathlib import Path


class load_forcing:

    def __init__(self, par: config_parameters, settings: config_settings):
        self.__settings = settings
        self.__par = par
        self.__fieldtype = forcingSource[settings.run.exttype]
        self.__date = datetime(1000, 1, 1)

        pass


    def update_Forcing(self, date: datetime):
        ext = None
        if self.__fieldtype == forcingSource.ERA5:
            ext = self.__getForcingFromERA5(date)
        elif self.__fieldtype == forcingSource.E20WFDEI:
            ext = self.__getForcingFromE20WFDEI(date)

        return ext

    def __getForcingFromERA5(self, date: datetime):
        """
        :param date:
        :return:
        """
        ext = {}

        settings = self.__settings
        lat0, lat1, lon0, lon1 = settings.CoorRoI[settings.run.res]

        fn = Path(settings.input.meteo) / ('%s.h5' % (date.strftime('%Y-%m')))
        ff=h5py.File(str(fn), 'r')

        if settings.parallel_id <0:
            key_word = 'data'
        else:
            key_word = 'ens_%s'%settings.parallel_id

        dict_group_load = ff[key_word]
        ext['dswrf'] = dict_group_load['ssrd'][int(date.day)-1]/86400  #(J -> W)
        ext['prcp'] = dict_group_load['tp'][int(date.day)-1]*1000/86400    #(m/day -> mm/s)
        ext['tmax'] = dict_group_load['2t'][int(date.day)-1]
        ext['tmin'] = dict_group_load['2t'][int(date.day)-1]

        return ext

    def __getForcingFromE20WFDEI(self, date: datetime):
        import scipy.io as scio
        """
        This code is demonstrating a comparison between the matlab code and python code. Not sure if we use it again.
        The data is stored in a yearly basis for each variable. Reading mat files (after being interpolated) on daily basis
        could slow down the process.
        :param date:
        :return:
        """
        settings = self.__settings
        par = self.__par

        varn = ['Precip', 'LWdown', 'PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind', 'SWdown']
        forcefield, ext = {}, {}
        num = (date - datetime.strptime('%04d-01-01' % date.year, '%Y-%m-%d')).days
        dl = settings.RoI[settings.run.res]

        for var in varn:
            if var == 'Precip':
                data = scio.loadmat(Path(settings.input.rain) / ('%s_%04d_%03d.mat' % (var, date.year, num + 1)))
                forcefield[var] = (data[var].T)[dl[0]:dl[1], dl[2]:dl[3]]
                pass
            else:
                data = scio.loadmat(Path(settings.input.meteo) / ('%s_%04d_%03d.mat' % (var, date.year, num + 1)))
                forcefield[var] = (data['data1'])[dl[0]:dl[1], dl[2]:dl[3]]
                pass

        self.__date = date

        '''% Assign forcing and estimate effective meteorological variables'''
        MSWEP, Rainf, Snowf, Tair, Wind, PSurf, SWdown, Qair, LWdown = forcefield['Precip'], forcefield['Rainf'], \
            forcefield[
                'Snowf'], \
            forcefield['Tair'], forcefield['Wind'], forcefield['PSurf'], forcefield['SWdown'], forcefield['Qair'], \
            forcefield['LWdown']

        Pg = MSWEP * (24 * 60 * 60)
        Pg_E20 = (Rainf + Snowf) * (24 * 60 * 60)
        Pg[Pg == np.nan] = Pg_E20[Pg == np.nan]

        '''other variables'''
        SnowFrac = np.fmax(0, Snowf / (Rainf + Snowf))
        '''already in W m-2 s-1; set minimum of 0.01 to avoid numerical problems   '''
        Rg = np.fmax(SWdown, 0.01)
        '''from K to degC'''
        Ta = Tair - 273.15
        T24 = Tair.copy() - 273.15
        '''already in Pa'''
        pair = PSurf
        '''in principle approximate but should be OK (see Monteith & Unsworth) '''
        pe = (Qair * pair) / 0.622
        '''Adjusted from u2 to u1; wind as at 1m above surface, apparently'''
        u1 = Wind

        '''Assign in correct matrix dimensions, select only the required data points, reduce to single precision'''
        HRU_fields = ['Pg', 'Rg', 'Ta', 'pe', 'T24', 'u1', 'pair', 'LWdown', 'SnowFrac']
        for key in HRU_fields:
            var = eval(key)
            ext[forcing[key]] = var[settings.mask]


        # '''Apply adjustment layers'''
        # '''multiply with rainfall gain factor'''
        # ext[forcing.Pg] *= par.PgainF
        # ext[forcing.Ta] += par.T_offset
        # ext[forcing.T24] += par.T_offset
        #
        # '''Other forcing'''
        # doy = num + 1
        # m = 1 - np.tan(par.lat * np.pi / 180) * np.tan((23.439 * np.pi / 180) * np.cos(2. * np.pi * (doy + 9) / 365.25))
        # '''%fraction daylength'''
        # fday = np.fmin(np.fmax(0.02, np.arccos(1 - np.fmin(np.fmax(0, m), 2)) / np.pi), 1)
        # ext[forcing.fday] = fday
        # '''ppm'''
        # ext[forcing.CO2] = (1.20575 * 1e-8 * date.year ** 2 - 4.641 * 1e-5 * date.year + 0.04496)

        return ext


def demo1():
    dp = '../settings/setting_2.json'
    settings = config_settings.loadjson(dp).process()
    par = config_parameters(settings)
    ext_da = load_forcing(par=par, settings=settings)
    ext=ext_da.update_Forcing(date=datetime.strptime('2003-01-02', '%Y-%m-%d'))
    pass


if __name__ == '__main__':
    demo1()
