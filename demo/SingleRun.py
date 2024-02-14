from datetime import datetime
from pathlib import Path
import numpy as np


class demo_single_run:

    def __init__(self, case='single_run_test', setting_dir='../settings/single_run'):
        self.case = case
        self.setting_dir = Path(setting_dir)

        print('Welcome to PyW3RA, the user case is: %s' % case)
        pass

    def configure_time(self, begin_time='2000-01-01', end_time='2000-01-31'):
        self.time = [datetime.strptime(begin_time, '%Y-%m-%d'), datetime.strptime(end_time, '%Y-%m-%d')]
        return self

    def configure_area(self, box=[-9.9, -43.8, 112.4, 154.3], basin='MDB'):
        self.box = box
        self.basin = basin
        return self

    def generate_settings(self):
        import json
        '''modify dp2'''
        dp_dir = self.setting_dir / 'pre_process.json'
        dp2 = json.load(open(dp_dir, 'r'))
        dp2['date_begin'], dp2['date_end'] = [self.time[0].strftime('%Y-%m'), self.time[1].strftime('%Y-%m')]
        with open(dp_dir, 'w') as f:
            json.dump(dp2, f, indent=4)

        '''modify dp1'''
        dp_dir = self.setting_dir / 'setting.json'
        dp1 = json.load(open(dp_dir, 'r'))
        dp1['init']['spinup'] = [self.time[0].strftime('%Y-%m-%d'), self.time[1].strftime('%Y-%m-%d')]
        dp1['init']['mode'] = 'cold'
        dp1['bounds']['lat'] = self.box[:2]
        dp1['bounds']['lon'] = self.box[2:4]
        dp1['bounds']['prefix'] = self.case
        dp1['input']["meteo"] = dp2['out_dir']
        dp1['input']["pars"] = dp2['out_dir']
        dp1['input']["clim"] = dp2['out_dir']
        dp1['input']["mask_fn"] = dp2['out_dir']

        with open(dp_dir, 'w') as f:
            json.dump(dp1, f, indent=4)

        self.__outdir = dp2['out_dir']
        self.__outdir2 = dp1['output']['dir']
        return self

    def preprocess(self):
        from src.preprocess import preprocess_base
        print()
        print('Data preparation...')
        '''preprocess'''
        dp1 = self.setting_dir / 'setting.json'
        pre = preprocess_base(dp1=str(dp1))

        dp2 = self.setting_dir / 'pre_process.json'
        pre.process(dp2=str(dp2))
        print('Finished')
        pass

    def model_run(self, arg= 'resume'):
        from src.config_settings import config_settings
        from src.config_parameters import config_parameters
        from src.model_initialise import model_initialise
        from src.ext_adapter import ext_adapter
        from src.hotrun import model_run

        print()
        print('Model is going to start running...')
        dp = self.setting_dir / 'setting.json'
        settings = config_settings.loadjson(dp).process()
        par = config_parameters(settings)
        model_init = model_initialise(settings=settings, par=par).configure_InitialStates()
        ext = ext_adapter(par=par, settings=settings)
        model_instance = model_run(settings=settings, par=par, model_init=model_init, ext=ext)
        model_instance.execute()
        print('\nFinished')
        pass

    def extract_signal(self):
        from DA.shp2mask import basin_shp_process
        from DA.Analysis import BasinSignalAnalysis
        import os

        print()

        print('Result analysis for %s' % self.basin)

        '''search for the shape file'''
        pp = Path('../data/basin/shp/')
        target = []
        for root, dirs, files in os.walk(pp):
            for file_name in files:
                if (self.basin in file_name) and file_name.endswith('.shp') and ('subbasins' in file_name):
                    target.append(os.path.join(root, file_name))
        assert len(target) == 1

        '''generate mask and vec files for given basin'''
        outdir = Path(self.__outdir)
        bs = basin_shp_process(res=0.1, basin_name=self.basin).shp_to_mask(
            shp_path=str(target[0]))
        statedir = Path(self.__outdir2)
        bs.mask_to_vec(model_mask_global=str(outdir / self.case / 'mask' / 'mask_global.h5'))

        '''basin analysis'''
        an = BasinSignalAnalysis(basin_mask=bs, state_dir=str(statedir / ('state_%s' % self.case)),
                                 par_dir=str(outdir / self.case / 'par'))

        # an.configure_mask2D().get_2D_map(this_day=datetime.strptime('2001-01-07', '%Y-%m-%d'), save=True)

        an.get_basin_average(save=True, date_begin=self.time[0].strftime('%Y-%m-%d'),
                             date_end=self.time[1].strftime('%Y-%m-%d'),
                             save_dir=str(statedir / ('output_%s' % self.case)))
        print('Finished')
        pass

    def visualize_signal(self):
        import pygmt
        import h5py
        from src.GeoMathKit import GeoMathKit

        '''basin average time-series'''

        '''load my result: MDB'''
        statedir = Path(self.__outdir2)
        hf = h5py.File(statedir / ('output_%s' % self.case) / 'basin_ts.h5', 'r')
        fan = hf['basin_0']
        date_begin = self.time[0].strftime('%Y-%m-%d')
        date_end = self.time[1].strftime('%Y-%m-%d')
        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)
        day_first = daylist[0].year + (daylist[0].month - 1) / 12 + daylist[0].day / 365.25
        fan_time = day_first + np.arange(len(daylist)) / 365.25

        a = []
        for key in fan.keys():
            a.append(fan[key][:])
            pass
        tws = np.sum(np.array(a), axis=0)

        '''plot figure'''
        fig = pygmt.Figure()
        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'TWS']
        i =0
        for state in statesnn:
            i+=1
            if state == 'TWS':
                vv = tws
            else:
                vv = fan[state][:]

            vmin, vmax= np.min(vv[20:]), np.max(vv[20:])
            dmin = vmin-(vmax-vmin)*0.1
            dmax = vmax + (vmax - vmin) * 0.1
            sp_1 = int(np.round((vmax-vmin)/10))
            if sp_1==0:
                sp_1 = 0.5
            sp_2 = sp_1*2

            if i==4:
                fig.shift_origin(yshift='12c', xshift='14c')
                pass

            fig.basemap(region=[fan_time[0] - 0.2, fan_time[-1]+0.2, dmin, dmax], projection='X12c/3c',
                    frame=["WSne", "xa2f1", 'ya%df%d+lwater [mm]'%(sp_2,sp_1)])

            fig.plot(x=fan_time, y=vv, pen="0.8p,blue", label='%s' % (state))
            fig.legend(position='jTR', box='+gwhite+p0.5p')
            fig.shift_origin(yshift='-4c')

        fig.savefig(statedir / ('output_%s' % self.case) / 'result.pdf')
        fig.savefig(statedir / ('output_%s' % self.case) / 'result.png')
        fig.show()
        pass


def demo1():
    dd = demo_single_run()
    dd.configure_time(begin_time='2000-01-01', end_time='2005-01-31')
    dd.configure_area()
    dd.generate_settings()
    dd.preprocess()
    dd.model_run()
    dd.extract_signal()
    # dd.visualize_signal()

    pass


if __name__ == '__main__':
    demo1()
