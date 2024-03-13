from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import sys
from src_FlowControl.SingleModel import SingleModel
from src_hydro.EnumType import init_mode


class OpenLoop(SingleModel):

    def __init__(self, case='test', setting_dir='../settings/OL_run', ens=3):
        super().__init__(case, setting_dir)
        self.ens = ens
        pass

    def generate_settings(self, mode: init_mode):
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

        if rank != 0:
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

    def extract_signal(self, postfix=None):
        from src_DA.shp2mask import basin_shp_process
        from src_DA.Analysis import BasinSignalAnalysis, Postprocessing_grid_first
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
        save_dir = str(statedir / ('output_%s_ensemble_%s' % (self.case, rank)))
        an = BasinSignalAnalysis(basin_mask=bs, state_dir=statedir2,
                                 par_dir=str(outdir / self.case / 'par'))

        # an.configure_mask2D().get_2D_map(this_day=datetime.strptime('2001-01-07', '%Y-%m-%d'), save=True)

        an.get_basin_average(save=True, date_begin=self.period[0].strftime('%Y-%m-%d'),
                             date_end=self.period[1].strftime('%Y-%m-%d'),
                             save_dir=save_dir, post_fix=postfix)

        '''2D monthly mean for further analysis'''
        pg = Postprocessing_grid_first(state_dir=statedir2,
                                       par_dir=str(outdir / self.case / 'par'))

        state = 'TWS'
        mm = pg.monthlymean(state=state, date_begin=self.period[0].strftime('%Y-%m-%d'),
                            date_end=self.period[1].strftime('%Y-%m-%d'))

        '''save it'''
        hf = h5py.File(Path(save_dir) / ('monthly_mean_%s_%s.h5' % (state, postfix)), 'w')
        for key, vv in mm.items():
            hf.create_dataset(name=key, data=vv[:])
        print('Finished')
        pass

    def post_processing(self, file_postfix=None, save_dir = '../temp', isGRACE= False):
        import h5py
        from src_hydro.GeoMathKit import GeoMathKit
        from src_DA.Analysis import Postprocessing_basin, Postprocessing_grid_second
        import json

        '''collect monthly mean grid results'''
        pg = Postprocessing_grid_second(ens=self.ens, case=self.case, basin=self.basin)
        state = 'TWS'
        res1, res2 = pg.ensemble_mean(state=state, postfix=file_postfix, fdir=self._outdir2)
        '''save results'''
        # save_dir = '../temp'
        hf = h5py.File(Path(save_dir) / ('monthly_mean_%s_%s_uOL.h5' % (state, self.basin)), 'w')
        for key, vv in res1.items():
            hf.create_dataset(name=key, data=vv[:])

        hf = h5py.File(Path(save_dir) / ('monthly_mean_%s_%s_%s.h5' % (state, self.basin, file_postfix)), 'w')
        for key, vv in res2.items():
            hf.create_dataset(name=key, data=vv[:])

        '''collect basin average results'''
        pp = Postprocessing_basin(ens=self.ens, case=self.case, basin=self.basin,
                                  date_begin=self.period[0].strftime('%Y-%m-%d'),
                                  date_end=self.period[1].strftime('%Y-%m-%d'))

        states = pp.get_states(post_fix=file_postfix, dir=self._outdir2)
        pp.save_states(prefix=(file_postfix + '_' + self.basin), save_dir=save_dir)

        '''collect GRACE basin average results'''
        if isGRACE:
            dp_dir = Path(self.setting_dir) / 'DA_setting.json'
            dp4 = json.load(open(dp_dir, 'r'))
            pp.get_GRACE(obs_dir=dp4['obs']['dir'])
            pp.save_GRACE(prefix=(file_postfix + '_' + self.basin), save_dir=save_dir)

        pass

    def visualize_signal(self, fig_path: str, fig_postfix='0', file_postfix=None, data_dir=None):
        import pygmt
        import h5py
        from src_hydro.GeoMathKit import GeoMathKit
        from src_DA.Analysis import Postprocessing_basin

        '''collect basin average results'''
        pp = Postprocessing_basin(ens=self.ens, case=self.case, basin=self.basin,
                                  date_begin=self.period[0].strftime('%Y-%m-%d'),
                                  date_end=self.period[1].strftime('%Y-%m-%d'))

        # states = pp.get_states(post_fix=file_postfix, dir=self._outdir2)
        if data_dir is None:
            states = pp.load_states(prefix=(file_postfix + '_' + self.basin))
        else:
            states = pp.load_states(prefix=(file_postfix + '_' + self.basin), load_dir=data_dir)

        fan_time = states['time']

        '''plot figure'''
        fig = pygmt.Figure()
        statesnn = ['S0', 'Ss', 'Sd', 'Sr', 'Sg', 'TWS']
        i = 0
        for state in statesnn:
            i += 1

            vv = states['basin_0'][state][0]

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

            for ens in reversed(range(self.ens + 1)):
                vv = states['basin_0'][state][ens]

                if ens == 0:
                    fig.plot(x=fan_time, y=vv, pen="1p,blue", label='%s' % (state), transparency=30)
                else:
                    fig.plot(x=fan_time, y=vv, pen="0.3p,grey")

            fig.legend(position='jTR', box='+gwhite+p0.5p')
            fig.shift_origin(yshift='-4c')

        fig.savefig(str(Path(fig_path) / ('Component_%s_%s.pdf' % (self.case, fig_postfix))))
        fig.savefig(str(Path(fig_path) / ('Component_%s_%s.png' % (self.case, fig_postfix))))
        # fig.show()
        pass
