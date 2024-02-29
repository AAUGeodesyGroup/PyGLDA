from datetime import datetime
from pathlib import Path
import numpy as np
import sys
from src_FlowControl.SingleModel import SingleModel


class OpenLoop(SingleModel):

    def __init__(self, case='test', setting_dir='../settings/OL_run', ens=3):
        super().__init__(case, setting_dir)
        self.ens = ens
        pass

    def generate_settings(self):
        import json
        '''modify dp2'''
        dp_dir = self.setting_dir / 'pre_process.json'
        dp2 = json.load(open(dp_dir, 'r'))
        dp2['date_begin'], dp2['date_end'] = [self.period[0].strftime('%Y-%m'), self.period[1].strftime('%Y-%m')]
        with open(dp_dir, 'w') as f:
            json.dump(dp2, f, indent=4)

        '''modify dp1'''
        dp_dir = self.setting_dir / 'setting.json'
        dp1 = json.load(open(dp_dir, 'r'))
        dp1['init']['spinup'] = [self.period[0].strftime('%Y-%m-%d'), self.period[1].strftime('%Y-%m-%d')]
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

        '''modify dp3'''
        dp_dir = self.setting_dir / 'perturbation.json'
        dp3 = json.load(open(dp_dir, 'r'))
        dp3['dir']["in"] = str(Path(dp2['out_dir']) / self.case)
        dp3['dir']["out"] = str(Path(dp2['out_dir']) / self.case)
        dp3['ensemble'] = self.ens

        with open(dp_dir, 'w') as f:
            json.dump(dp3, f, indent=4)

        self._outdir = dp2['out_dir']
        self._outdir2 = dp1['output']['dir']
        return self

    def perturbation(self):
        from src_DA.Perturbation import perturbation

        dp = self.setting_dir / 'perturbation.json'
        pp = perturbation(dp=dp).setDate(month_begin=self.period[0].strftime('%Y-%m'),
                                         month_end=self.period[1].strftime('%Y-%m'))
        print('')
        print('Start to generate ensembles with given perturbation...')
        # pp.perturbe_par()
        # pp.perturbe_forcing()
        # pp.perturbe_coherent_par(percentage=0.4)
        # pp.perturbe_coherent_forcing(percentage=0.3)
        pp.perturb_par_spatial_coherence()
        pp.perturb_forcing_spatial_coherence()
        print('Finished!')

        pass

    def model_run(self):
        from src_hydro.config_settings import config_settings
        from src_hydro.config_parameters import config_parameters
        from src_hydro.model_initialise import model_initialise
        from src_hydro.ext_adapter import ext_adapter
        from src_hydro.hotrun import model_run
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        f = open('../log/OL/log_%s.txt' % rank, 'w')
        sys.stdout = f

        print()
        print('Model is going to start running...')
        dp = self.setting_dir / 'setting.json'
        settings = config_settings.loadjson(dp).process(Parallel_ID=rank)
        par = config_parameters(settings)
        model_init = model_initialise(settings=settings, par=par).configure_InitialStates()
        ext = ext_adapter(par=par, settings=settings)
        model_instance = model_run(settings=settings, par=par, model_init=model_init, ext=ext)
        model_instance.execute()
        print('\nFinished')
        pass

    def extract_signal(self):
        from src_DA.shp2mask import basin_shp_process
        from src_DA.Analysis import BasinSignalAnalysis
        import os
        import json
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        f = open('../log/OL/signal_log_%s.txt' % rank, 'w')
        sys.stdout = f
        sys.stdout.flush()

        print()

        print('Result analysis for %s' % self.basin)

        dp_dir = self.setting_dir / 'pre_process.json'
        dp2 = json.load(open(dp_dir, 'r'))
        dp_dir = self.setting_dir / 'setting.json'
        dp1 = json.load(open(dp_dir, 'r'))
        self._outdir = dp2['out_dir']
        self._outdir2 = dp1['output']['dir']

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
            shp_path=str(target[0]), issave=False)
        statedir = Path(self._outdir2)
        bs.mask_to_vec(model_mask_global=str(outdir / self.case / 'mask' / 'mask_global.h5'))

        '''basin analysis'''

        statedir2 = str(statedir / ('state_%s_ensemble_%s' % (self.case, rank)))

        print(statedir2)
        an = BasinSignalAnalysis(basin_mask=bs, state_dir=statedir2,
                                 par_dir=str(outdir / self.case / 'par'))

        # an.configure_mask2D().get_2D_map(this_day=datetime.strptime('2001-01-07', '%Y-%m-%d'), save=True)

        save_dir = str(statedir / ('output_%s_ensemble_%s' % (self.case, rank)))
        an.get_basin_average(save=True, date_begin=self.period[0].strftime('%Y-%m-%d'),
                             date_end=self.period[1].strftime('%Y-%m-%d'),
                             save_dir=save_dir)
        print('Finished')
        pass

    def visualize_signal(self, fig_path: str, postfix='0'):
        import pygmt
        import h5py
        from src_hydro.GeoMathKit import GeoMathKit

        '''basin average time-series'''

        '''load my result: MDB'''
        statedir = Path(self._outdir2)
        ens_numbers = self.ens + 1

        fan_dict = {}
        for ens in range(ens_numbers):
            hf = h5py.File(statedir / ('output_%s_ensemble_%s' % (self.case, ens)) / 'basin_ts.h5', 'r')
            fan = hf['basin_0']
            fan_dict[ens] = fan
            # hf.close()

        date_begin = self.period[0].strftime('%Y-%m-%d')
        date_end = self.period[1].strftime('%Y-%m-%d')
        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)
        day_first = daylist[0].year + (daylist[0].month - 1) / 12 + daylist[0].day / 365.25
        fan_time = day_first + np.arange(len(daylist)) / 365.25

        tws = {}
        for ens in range(ens_numbers):
            a = []
            fan = fan_dict[ens]
            for key in fan.keys():
                a.append(fan[key][:])
                pass
            tws[ens] = np.sum(np.array(a), axis=0)

        '''plot figure'''
        fig = pygmt.Figure()
        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'TWS']
        i = 0
        for state in statesnn:
            i += 1

            if state == 'TWS':
                '''ens=0'''
                vv = tws[0]
            else:
                '''ens=0'''
                fan = fan_dict[0]
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

            for ens in reversed(range(ens_numbers)):
                if state == 'TWS':
                    vv = tws[ens]
                else:
                    fan = fan_dict[ens]
                    vv = fan[state][:]

                if ens == 0:
                    fig.plot(x=fan_time, y=vv, pen="1p,blue", label='%s' % (state), transparency=30)
                else:
                    fig.plot(x=fan_time, y=vv, pen="0.3p,grey")

            fig.legend(position='jTR', box='+gwhite+p0.5p')
            fig.shift_origin(yshift='-4c')

        fig.savefig(str(Path(fig_path) / ('%s_%s.pdf' % (self.case, postfix))))
        fig.savefig(str(Path(fig_path) / ('%s_%s.png' % (self.case, postfix))))
        # fig.show()
        pass
