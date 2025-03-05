import json
from pathlib import Path
import time
import numpy as np
from src_GHM.EnumType import init_mode, forcingSource
import netCDF4 as nc
import h5py


class config_settings:

    def __init__(self):
        self.bounds = self.config_bounds().__dict__
        self.init = self.config_initialise().__dict__
        self.run = self.config_running().__dict__
        self.input = self.config_inputpath().__dict__
        self.output = self.config_output().__dict__

        pass

    @staticmethod
    def save_default():
        obj1 = config_settings()
        tc_dic = obj1.__dict__
        default_path = Path.cwd().parent / 'settings' / 'setting.json'
        with open(default_path, 'w') as f:
            json.dump(tc_dic, f, indent=4)
        return default_path

    @staticmethod
    def loadjson(js):
        Obj = config_settings()

        dict1 = json.load(open(js, 'r'))
        Obj.__dict__ = dict1
        return Obj

    def convert1Dto2D_py(self, val):
        """
        input should be python 1-d array
        """
        # map2d= np.zeros(self.mask.shape)
        map2d = np.full(self.mask.shape, np.nan)
        map2d[self.mask] = val
        return map2d

    def convertToMatlab1D(self, val):

        return self.convert1Dto2D_py(val).T[self.mask.T]

    def convert1Dto2D_mat(self, val):
        """
         input should be matlab 1-d array
         """
        # map2d= np.zeros(self.mask.shape)
        map2d = np.full(self.mask.shape, np.nan).T
        map2d[self.mask.T] = val
        return map2d.T

    def process(self, Parallel_ID=-1):
        """
        process the settings in depth
        :return:
        """
        self.parallel_id = Parallel_ID

        '''assign the settings into class attributes'''
        bounds = self.bounds
        self.bounds = self.config_bounds()
        self.bounds.__dict__.update(bounds)

        run = self.run
        self.run = self.config_running()
        self.run.__dict__.update(run)

        init = self.init
        self.init = self.config_initialise()
        self.init.__dict__.update(init)
        self.init.mode = init_mode[self.init.mode]

        input = self.input
        self.input = self.config_inputpath()
        self.input.__dict__.update(input)
        self.input.clim = Path(self.input.clim)/self.bounds.prefix/'clim'
        self.input.mask_fn = Path(self.input.mask_fn)/self.bounds.prefix/'mask'
        self.input.pars = Path(self.input.pars) / self.bounds.prefix
        self.input.meteo = Path(self.input.meteo) / self.bounds.prefix

        if Parallel_ID <0:
            self.input.pars = self.input.pars/'par'
            self.input.meteo = self.input.meteo/'forcing'
        else:
            self.input.pars = self.input.pars / 'ens_par'
            self.input.meteo = self.input.meteo/'ens_forcing'


        output = self.output
        self.output = self.config_output()
        self.output.__dict__.update(output)

        if Parallel_ID < 0:
            self.outdir = Path(self.output.dir) / ('output_' + self.bounds.prefix)
            self.statedir = Path(self.output.dir) / ('state_' + self.bounds.prefix)
        else:
            self.outdir = Path(self.output.dir) / ('output_' + self.bounds.prefix + '_ensemble_%s' % Parallel_ID)
            self.statedir = Path(self.output.dir) / ('state_' + self.bounds.prefix + '_ensemble_%s' % Parallel_ID)

        # print('Create the desired directory structure')
        if not self.outdir.exists():
            self.outdir.mkdir()

        if not self.statedir.exists():
            self.statedir.mkdir()
        # print('Done.')

        '''screen output'''
        print('\nConfiguring W3v2 run')
        print('Spatial resolution: %s'%self.run.res)
        print('Area: lat- %s, lon- %s' % (self.bounds.lat, self.bounds.lon))
        print('Forcing field: %s'%self.run.exttype)
        '''Define Resolution'''
        self.RoI = {}
        self.CoorRoI = {}
        # Get grid address for input data
        GridRes = [0.05, 0.10, 0.25, 0.50]
        for res in GridRes:
            err = res / 10
            lats = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
            lons = np.arange(-180 + res / 2, 180 - res / 2 + err, res)

            lati = [np.argmin(np.fabs(lats - max(self.bounds.lat))),
                    np.argmin(np.fabs(lats - min(self.bounds.lat)))]
            loni = [np.argmin(np.fabs(lons - min(self.bounds.lon))),
                    np.argmin(np.fabs(lons - max(self.bounds.lon)))]
            # self.RoI[res] = {'lat': lati,
            #                  'lon': loni}
            # Todo: fix the bug in matlab
            self.RoI[res] = [lati[0], lati[1] + 1, loni[0], loni[1] + 1]
            # self.RoI[res] = [lati[0], lati[1], loni[0], loni[1] + 1]
            self.CoorRoI[res] = [lats[self.RoI[res][0]], lats[self.RoI[res][1] - 1],
                                 lons[self.RoI[res][2]], lons[self.RoI[res][3] - 1]]
            pass

        '''Create a 'mask' for area to be sampled at the resolution required. 
        # It returns the location of the cells in the at-resolution grids (transposed grids) to be sampled.
        # Load area of interest mask'''
        ff = h5py.File(self.input.mask_fn/'mask.h5', 'r')
        mask = ff['mask'][:]
        self.mask = mask == 1

        if not self.mask.any():
            raise Exception('Abort: no land cells')

        return self

    class config_bounds:
        def __init__(self):
            # self.lat = [89.975, -89.975]
            # self.lon = [-179.975, 179.975]
            self.lat = [69.975, 20.975]
            self.lon = [-79.975, 79.975]
            self.prefix = 'FanPy'
            pass

    class config_running:
        def __init__(self):
            self.fromdate = '2000-01-02'
            self.todate = '2000-01-31'
            self.res = 0.05
            self.exttype = forcingSource.ERA5.name
            self.save_states_every_day = False
            self.save_output_every_day = False
            pass

    class config_initialise:
        def __init__(self):
            self.mode = init_mode.warm.name
            self.date = '2000-01-01'
            self.spinup = ['2000-01-01', '2010-01-01']
            self.dir = str(Path.cwd().parent.parent / 'Run_W3_V2_Albert' / 'DATA' / 'init')
            self.spec = ''
            pass

    class config_inputpath:
        def __init__(self):
            """
            """
            '''Specify the forcing '''
            # self.rain = Path.cwd().parent.parent / 'Run_W3_V2_Albert' / 'DATA' / 'Forcing' / 'meteo_ecmwf' / 'highRes'
            # self.meteo = Path.cwd().parent.parent / 'Run_W3_V2_Albert' / 'DATA' / 'Forcing' / 'meteo_ecmwf' / 'highRes'
            # self.rain = '/media/user/Backup Plus/ERA5/Raw_W3'
            self.meteo = '/media/user/Backup Plus/ERA5/Raw_W3'

            '''Specify the climatology data'''
            self.clim = Path.cwd().parent.parent / 'Run_W3_V2_Albert' / 'DATA' / 'climatologies'

            '''Specify the parameters'''
            self.pars = Path.cwd().parent.parent / 'Run_W3_V2_Albert' / 'DATA' / 'parameters'

            '''Specify the land mask'''
            self.mask_fn = self.pars / 'mask.h5'

            self.meteo = str(self.meteo)
            self.clim = str(self.clim)
            self.pars = str(self.pars)
            self.mask_fn = str(self.mask_fn)

            pass

    class config_output:
        def __init__(self):
            self.dir = str(Path.cwd().parent.parent / 'Run_W3_V2_Albert')
            self.var = {
                'ETtot': False,
                'Qtot': False,
                'Qnet': False,
                'E0': False,
                'Ei': False,
                'Et': False,
                'Es': False,
                'Er': False,
                'RSn': False,
                'RLn': False,
                'LE': False,
                'H': False,
                'albedo': False,
                'LAI': False,
                'Ssnow': False,
                'z_g': False,
                'S0': False,
                'Sroot': False,
                'Ssoil': False,
                'Sg': False,
                'Ts': False,
                'fsnow': False,
                'Sr': False
            }


if __name__ == '__main__':
    # demo2()
    dp = config_settings.save_default()
    a = config_settings.loadjson(dp).process()
    pass
