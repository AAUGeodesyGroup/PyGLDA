from datetime import datetime
from pathlib import Path
import numpy as np
from src_hydro.EnumType import init_mode


class SingleModel:
    """
    this is designed for running the model alone
    """

    def __init__(self, case='single_run_test', setting_dir='../settings/single_run'):
        self.case = case
        self.setting_dir = Path(setting_dir)

        print('Welcome to PyW3RA, the user case is: %s' % case)
        pass

    def configure_time(self, begin_time='2000-01-01', end_time='2000-01-31'):
        self.period = [datetime.strptime(begin_time, '%Y-%m-%d'), datetime.strptime(end_time, '%Y-%m-%d')]
        return self

    def configure_area(self, box=[-9.9, -43.8, 112.4, 154.3], basin='MDB'):
        self.box = box
        self.basin = basin
        return self

    def generate_settings(self, mode: init_mode):
        import json
        from datetime import datetime, timedelta
        '''modify dp2'''
        dp_dir = self.setting_dir / 'pre_process.json'
        dp2 = json.load(open(dp_dir, 'r'))
        dp2['date_begin'], dp2['date_end'] = [self.period[0].strftime('%Y-%m'), self.period[1].strftime('%Y-%m')]
        with open(dp_dir, 'w') as f:
            json.dump(dp2, f, indent=4)

        '''modify dp1'''
        dp_dir = self.setting_dir / 'setting.json'
        dp1 = json.load(open(dp_dir, 'r'))
        dp1['bounds']['lat'] = self.box[:2]
        dp1['bounds']['lon'] = self.box[2:4]
        dp1['bounds']['prefix'] = self.case
        dp1['input']["meteo"] = dp2['out_dir']
        dp1['input']["pars"] = dp2['out_dir']
        dp1['input']["clim"] = dp2['out_dir']
        dp1['input']["mask_fn"] = dp2['out_dir']

        dp1['init']['mode'] = mode.name
        if mode == init_mode.cold:
            dp1['init']['spinup'] = [self.period[0].strftime('%Y-%m-%d'), self.period[1].strftime('%Y-%m-%d')]
        else:
            dp1['run']['fromdate'] = self.period[0].strftime('%Y-%m-%d')
            dp1['run']['todate'] = self.period[1].strftime('%Y-%m-%d')
            dp1['init']['date'] = (self.period[0] - timedelta(days=1)).strftime('%Y-%m-%d')

        with open(dp_dir, 'w') as f:
            json.dump(dp1, f, indent=4)

        self._outdir = dp2['out_dir']
        self._outdir2 = dp1['output']['dir']
        return self

    def preprocess(self):
        from src_hydro.preprocess import preprocess_base
        print()
        print('Data preparation...')
        '''preprocess'''
        dp1 = self.setting_dir / 'setting.json'
        pre = preprocess_base(dp1=str(dp1))

        dp2 = self.setting_dir / 'pre_process.json'
        pre.process(dp2=str(dp2))
        print('Finished')
        pass

    def model_run(self, arg='resume'):
        from src_hydro.config_settings import config_settings
        from src_hydro.config_parameters import config_parameters
        from src_hydro.model_initialise import model_initialise
        from src_hydro.ext_adapter import ext_adapter
        from src_hydro.hotrun import model_run

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

    def create_ini_states(self, mode: init_mode, modifydate: str):
        import shutil
        from src_hydro.config_settings import config_settings
        import os
        """place the state of the last day into the ini folder"""
        dp = self.setting_dir / 'setting.json'
        settings = config_settings.loadjson(dp).process()

        if mode == init_mode.cold:
            the_last_day = settings.init.spinup[-1]
        else:
            the_last_day = settings.run.todate

        sd = Path(self._outdir2) / ('state_%s' % self.case)
        source_file = sd / ('state.%s.h5' % datetime.strptime(the_last_day, '%Y-%m-%d').strftime('%Y%m%d'))
        target_folder = Path(settings.init.dir)

        shutil.copy(source_file, target_folder)

        oldname = 'state.%s.h5' % datetime.strptime(the_last_day, '%Y-%m-%d').strftime('%Y%m%d')
        x = oldname.split('.')
        newname = x[0] + '.' + modifydate + '.' + x[2]
        os.rename(target_folder / oldname, target_folder / newname)
        pass

    def extract_signal(self):
        from src_DA.shp2mask import basin_shp_process
        from src_DA.Analysis import BasinSignalAnalysis
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
        outdir = Path(self._outdir)
        bs = basin_shp_process(res=0.1, basin_name=self.basin).shp_to_mask(
            shp_path=str(target[0]))
        statedir = Path(self._outdir2)
        bs.mask_to_vec(model_mask_global=str(outdir / self.case / 'mask' / 'mask_global.h5'))

        '''basin analysis'''
        an = BasinSignalAnalysis(basin_mask=bs, state_dir=str(statedir / ('state_%s' % self.case)),
                                 par_dir=str(outdir / self.case / 'par'))

        # an.configure_mask2D().get_2D_map(this_day=datetime.strptime('2001-01-07', '%Y-%m-%d'), save=True)

        an.get_basin_average(save=True, date_begin=self.period[0].strftime('%Y-%m-%d'),
                             date_end=self.period[1].strftime('%Y-%m-%d'),
                             save_dir=str(statedir / ('output_%s' % self.case)))
        print('Finished')
        pass

    def visualize_signal(self, fig_path: str, fig_postfix='0'):
        import pygmt
        import h5py
        from src_hydro.GeoMathKit import GeoMathKit

        '''basin average time-series'''

        '''load my result'''
        statedir = Path(self._outdir2)
        hf = h5py.File(statedir / ('output_%s' % self.case) / 'basin_ts.h5', 'r')
        fan = hf['basin_0']
        date_begin = self.period[0].strftime('%Y-%m-%d')
        date_end = self.period[1].strftime('%Y-%m-%d')
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
        i = 0
        for state in statesnn:
            i += 1
            if state == 'TWS':
                vv = tws
            else:
                vv = fan[state][:]

            vmin, vmax = np.min(vv[20:]), np.max(vv[20:])
            dmin = vmin - (vmax - vmin) * 0.1
            dmax = vmax + (vmax - vmin) * 0.1
            sp_1 = int(np.round((vmax - vmin) / 10))
            if sp_1 == 0:
                sp_1 = 0.5
            sp_2 = sp_1 * 2

            if i == 4:
                fig.shift_origin(yshift='12c', xshift='14c')
                pass

            fig.basemap(region=[fan_time[0] - 0.2, fan_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                        frame=["WSne", "xa2f1", 'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

            fig.plot(x=fan_time, y=vv, pen="0.8p,blue", label='%s' % (state))
            fig.legend(position='jTR', box='+gwhite+p0.5p')
            fig.shift_origin(yshift='-4c')

        fig.savefig(str(Path(fig_path) / ('%s_%s.pdf' % (self.case, fig_postfix))))
        fig.savefig(str(Path(fig_path) / ('%s_%s.png' % (self.case, fig_postfix))))
        # fig.show()
        pass

    def visualize_comparison_GRACE(self, fig_path: str, fig_postfix='0'):
        import pygmt
        import h5py
        from src_hydro.GeoMathKit import GeoMathKit
        import json

        '''basin average time-series'''

        '''load my result'''
        date_begin = self.period[0].strftime('%Y-%m-%d')
        date_end = self.period[1].strftime('%Y-%m-%d')
        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)
        day_first = daylist[0].year + (daylist[0].month - 1) / 12 + daylist[0].day / 365.25
        fan_time = day_first + np.arange(len(daylist)) / 365.25

        statedir = Path(self._outdir2)
        hf = h5py.File(statedir / ('output_%s' % self.case) / 'basin_ts.h5', 'r')

        tws_model = {}
        for x in range(1, len(list(hf.keys()))):
            fan = hf['basin_%s' % x]
            a = []
            for key in fan.keys():
                a.append(fan[key][:])
                pass
            tws = np.sum(np.array(a), axis=0)
            tws_model['sub_basin_%s' % x] = tws
            pass

        '''load GRACE'''
        dp_dir = self.setting_dir / 'DA_setting.json'
        dp2 = json.load(open(dp_dir, 'r'))
        fn = dp2['obs']['GRACE']['preprocess_res']
        gf = h5py.File(Path(fn) / ('%s_signal.hdf5' % self.basin), 'r')
        str_time = list(gf['time_epoch'][:].astype(str))
        fraction_time = []
        for tt in str_time:
            da = datetime.strptime(tt, '%Y-%m-%d')
            fraction_time.append(da.year + (da.month - 1) / 12 + da.day / 365.25)

        '''plot figure'''
        fig = pygmt.Figure()
        i = 0
        for basin in tws_model.keys():
            i += 1
            print(basin)
            vv = tws_model[basin]

            vmin, vmax = np.min(vv[20:]), np.max(vv[20:])
            dmin = vmin - (vmax - vmin) * 0.1
            dmax = vmax + (vmax - vmin) * 0.1
            sp_1 = int(np.round((vmax - vmin) / 10))
            if sp_1 == 0:
                sp_1 = 0.5
            sp_2 = sp_1 * 2

            fig.basemap(region=[fan_time[0] - 0.2, fan_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                        frame=["WSne+t%s" % basin, "xa2f1", 'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

            fig.plot(x=fan_time, y=vv, pen="0.8p,blue", label='model')
            fig.plot(x=fraction_time, y=gf[basin][:] - np.mean(gf[basin][:]) + np.mean(vv), pen="0.8p,red",
                     label='GRACE')
            fig.legend(position='jTR', box='+gwhite+p0.5p')

            if i % 12 == 0:
                fig.shift_origin(yshift='52.8c', xshift='14c')
                continue
            fig.shift_origin(yshift='-4.8c')

        fig.savefig(str(Path(fig_path) / ('compare_GRACE_%s_%s.pdf' % (self.case, fig_postfix))))
        fig.savefig(str(Path(fig_path) / ('compare_GRACE_%s_%s.png' % (self.case, fig_postfix))))
        # fig.show()

        pass
