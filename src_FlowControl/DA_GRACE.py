from src_FlowControl import run_context as _rc
import os
from datetime import datetime, timedelta
from pathlib import Path
from src_FlowControl.OpenLoop import OpenLoop
from src_OBS.prepare_GRACE import GRACE_preparation
from src_DA.configure_DA import config_DA
from src_OBS.GRACE_perturbation_old import GRACE_perturbed_obs
from src_GHM.EnumType import states_var, init_mode
from src_DA.shp2mask import basin_shp_process
from src_DA.ObsDesignMatrix import DM_basin_average
import h5py
import numpy as np
import sys


class DA_GRACE(OpenLoop):

    def __init__(self, case='test', setting_dir='../settings/DA_local', ens=3):
        super().__init__(case, setting_dir, ens)

        pass

    def _dm_dir(self):
        """folder of the design matrix: units/<unit_set>/DM/<domain>__<ensemble> (layout) or ../temp (legacy)"""
        from src_FlowControl import run_context as rc
        return rc.dm_dir('../temp')

    def _log_file(self, kind, rank):
        from src_FlowControl import run_context as rc
        d = rc.log_dir()
        if d is None:
            return '../log/OL/%s_%s.txt' % (kind, rank)
        d.mkdir(parents=True, exist_ok=True)
        return str(d / ('DA_%s_rank%s.txt' % (kind, rank)))

    def generate_settings(self, mode: init_mode):
        """modify the json setting file to make each sub-setting consistent with each other"""
        import os
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
        from src_FlowControl import run_context as rc
        dp1['init']['spec'] = rc.init_name(self.case)          # the domain (layout) or the case (legacy)

        dp1['init']['mode'] = mode.name
        '''folder layout (src_FlowControl/run_context.py): domains -> <domain>/OL/<ensemble>/ens_kk, <case>/DA/ens_kk; legacy -> shared'''
        from src_FlowControl import run_context as rc
        dp1['output']['layout'] = 'domains' if rc.get_layout() is not None else 'legacy'
        dp1['output']['stage'] = {'warm': 'OL', 'resume': 'DA'}.get(mode.name, '')
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
        dp3['dir']["in"] = str(rc.crop_dir(dp2['out_dir'], self.case))           # <domain>/input or legacy
        dp3['dir']["out"] = str(rc.ens_input_dir(dp2['out_dir'], self.case))     # <domain>/ensembles/<ensemble>
        dp3['ensemble'] = self.ens

        with open(dp_dir, 'w') as f:
            json.dump(dp3, f, indent=4)

        self._outdir = dp2['out_dir']
        self._outdir2 = dp1['output']['dir']

        '''modify dp4'''
        dp_dir = self.setting_dir / 'DA_setting.json'
        dp4 = json.load(open(dp_dir, 'r'))
        dp4['basic']['ensemble'] = self.ens
        dp4['basic']['fromdate'] = self.period[0].strftime('%Y-%m-%d')
        dp4['basic']['todate'] = self.period[1].strftime('%Y-%m-%d')
        dp4['basic']['basin'] = self.basin

        '''the shape file: the case's own (RDA.shp_path) or the only one of this basin below <data_root>/basin/shp'''
        dp4['basic']['basin_shp'] = _rc.find_basin_shp(self.basin)

        with open(dp_dir, 'w') as f:
            json.dump(dp4, f, indent=4)

        return self

    def GRACE_obs_preprocess(self):
        import json

        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        basin = configDA.basic.basin
        shp_path = configDA.basic.basin_shp

        '''pre-process of src_OBS'''
        dp_dir = self.setting_dir / 'setting.json'
        dp1 = json.load(open(dp_dir, 'r'))
        boxmask = {'lat': dp1['bounds']['lat'],
                   'lon': dp1['bounds']['lon']}
        GR = GRACE_preparation(basin_name=basin,
                               shp_path=shp_path)
        GR.generate_mask(box_mask=boxmask)

        begin_day = datetime.strptime(configDA.basic.fromdate, '%Y-%m-%d')
        end_day = datetime.strptime(configDA.basic.todate, '%Y-%m-%d')

        GR.basin_TWS(month_begin=begin_day.strftime('%Y-%m'),
                     month_end=end_day.strftime('%Y-%m'),
                     dir_in=configDA.obs.GRACE['EWH_grid_dir'],
                     dir_out=configDA.obs.GRACE['preprocess_res'])

        GR.basin_COV(month_begin=begin_day.strftime('%Y-%m'),
                     month_end=end_day.strftime('%Y-%m'),
                     dir_in=configDA.obs.GRACE['cov_dir'],
                     dir_out=configDA.obs.GRACE['preprocess_res'])

        pass

    def gather_OLmean(self, post_fix='OL'):
        """gather the OL mean to be reduced from GRACE to acquire the TWS anomaly"""
        # TODO: this is obs-specific and one should take care how to compute the mean. from DA or OL?
        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        ens_TWS = []
        for ens_id in range(0, self.ens + 1):
            statedir = Path(self._outdir2)
            from src_FlowControl.run_context import find_ens_dir
            ens_dir = str(find_ens_dir(statedir, 'output', self.case, ens_id, stage=post_fix))
            mm = h5py.File(str(Path(ens_dir) / ('basin_ts_%s.h5' % post_fix)), 'r')

            subbasin_num = len(list(mm.keys())) - 1

            basin_tws = []
            for subbasin in range(1, subbasin_num + 1):

                nn = mm['basin_%s' % subbasin]

                tws_temporal_mean = 0
                for component in nn.keys():
                    tws_temporal_mean += np.mean(nn[component][:])

                basin_tws.append(tws_temporal_mean)

                pass

            ens_TWS.append(basin_tws)

        ens_TWS = np.array(ens_TWS)

        mean_0 = ens_TWS[0]
        mean_1 = np.mean(ens_TWS[1:], axis=0)

        from src_FlowControl import run_context as rc
        fn = rc.ol_mean_path(configDA.obs.GRACE['OL_mean'], self.case, self.basin)   # <case>/grace/ol_mean.hdf5
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        ww = h5py.File(fn, 'w')
        ww.create_dataset(data=mean_0, name='mean_unperturbed')
        ww.create_dataset(data=mean_1, name='mean_ensemble')
        ww.close()

        pass

    def generate_perturbed_GRACE_obs(self):

        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        begin_day = datetime.strptime(configDA.basic.fromdate, '%Y-%m-%d')
        end_day = datetime.strptime(configDA.basic.todate, '%Y-%m-%d')

        ob = GRACE_perturbed_obs(ens=configDA.basic.ensemble, basin_name=configDA.basic.basin)
        ob.configure_dir(input_dir=configDA.obs.GRACE['preprocess_res'],
                         obs_dir=configDA.obs.dir). \
            configure_time(month_begin=begin_day.strftime('%Y-%m'), month_end=end_day.strftime('%Y-%m'))

        ob.perturb_TWS().remove_temporal_mean()
        from src_FlowControl import run_context as rc
        fn = rc.ol_mean_path(configDA.obs.GRACE['OL_mean'], self.case, self.basin)   # <case>/grace/ol_mean.hdf5
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        ob.add_temporal_mean(fn=str(fn))
        ob.save()

        pass

    def prepare_design_matrix(self):
        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        basin = configDA.basic.basin
        shp_path = configDA.basic.basin_shp

        layer = {}
        for key, vv in configDA.basic.layer.items():
            layer[states_var[key]] = vv

        bs = basin_shp_process(res=0.1, basin_name=basin).shp_to_mask(shp_path=shp_path)

        print('\nPreparing the design matrix that projects the state to observation space...')
        from src_FlowControl import run_context as rc
        land_mask = str(rc.crop_dir(self._outdir, self.case) / 'mask' / 'mask_global.h5')
        bs.mask_to_vec(model_mask_global=land_mask)

        state_sample = rc.nan_sample_path(configDA.basic.NaNmaskStatesDir, self.case)
        bs.mask_nan(sample=state_sample)

        par_dir = str(rc.crop_dir(self._outdir, self.case) / 'par')
        dm_save_dir = self._dm_dir()
        dm = DM_basin_average(shp=bs, layer=layer, par_dir=par_dir)
        dm.vertical_aggregation(isVec=True).basin_average()
        dm.saveDM(out_path=dm_save_dir)

        pass

    def get_states_sample_for_mask(self):
        """
        this is to copy a state file to given directory for masking out the NaN values.
        """
        import shutil

        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        target_folder = Path(configDA.basic.NaNmaskStatesDir)

        from src_FlowControl import run_context as rc
        source_folder = rc.find_ens_dir(self._outdir2, 'state', self.case, 0, stage='OL')

        if rc.get_layout() is None:
            '''legacy: an arbitrary day (listdir order is not sorted, issue 11)'''
            ss = os.listdir(source_folder)
            assert 'h5' in ss[-1]
            taget_fn = ss[-1]
        else:
            '''the last day of the open loop, chosen by date (issue 11)'''
            import json
            ol_end = json.load(open(self.setting_dir / 'setting.json'))['run']['todate']
            ol_files = sorted(f for f in os.listdir(source_folder) if f.startswith('state.') and f.endswith('.h5'))
            want = 'state.%s.h5' % ol_end.replace('-', '')
            taget_fn = want if want in ol_files else ol_files[-1]

        dst = rc.nan_sample_path(target_folder, self.case)      # <domain>/OL/<ensemble>/nan_sample.h5 or legacy
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_folder / taget_fn, dst)

        pass

    def run_DA(self, rank: int):
        """The main entrance to the data assimilation experiment"""
        from src_GHM.config_settings import config_settings
        from src_GHM.config_parameters import config_parameters
        from src_GHM.model_initialise import model_initialise
        from src_GHM.ext_adapter import ext_adapter
        from src_GHM.hotrun import model_run_daily
        import json
        from src_DA.observations import GRACE_obs
        from src_DA.ExtractStates import EnsStates
        from src_DA.EnKF_fixed_time_old import DataAssimilation, DataAssimilation_monthly, \
            DataAssimilation_monthlymean_dailyupdate, DataAssimilation_monthly_diag

        if rank != 0:
            f = open('../log/OL/log_%s.txt' % rank, 'w')
            sys.stdout = f

        print('Configure DA experiments: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        '''configure DA'''
        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        '''configure model'''
        dp = self.setting_dir / 'setting.json'
        dp1 = json.load(open(dp, 'r'))
        settings = config_settings.loadjson(dp).process(Parallel_ID=rank)
        par = config_parameters(settings)
        model_init = model_initialise(settings=settings, par=par).configure_InitialStates()
        ext = ext_adapter(par=par, settings=settings)
        model_instance = model_run_daily(settings=settings, par=par, model_init=model_init, ext=ext)

        print('Configure shapefile: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        '''define the basin-shp file and derive the corresponding mask'''
        basin = configDA.basic.basin
        shp_path = configDA.basic.basin_shp
        bs = basin_shp_process(res=0.1, basin_name=basin).shp_to_mask(shp_path=shp_path)

        dp_dir = self.setting_dir / 'pre_process.json'
        dp2 = json.load(open(dp_dir, 'r'))
        lm = dp2['out_dir']
        from src_FlowControl import run_context as rc
        land_mask = str(rc.crop_dir(lm, self.case) / 'mask' / 'mask_global.h5')
        bs.mask_to_vec(model_mask_global=land_mask)
        state_sample = rc.nan_sample_path(configDA.basic.NaNmaskStatesDir, self.case)
        bs.mask_nan(sample=state_sample)

        print('Configure OBS and its design matrix: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        '''obtain the design matrix from pre-saved data'''
        layer = {}
        for key, vv in configDA.basic.layer.items():
            layer[states_var[key]] = vv
        dm_save_dir = '../temp'
        dm = DM_basin_average(shp=bs, layer=layer, LoadfromDisk=True, dir=dm_save_dir)
        # par_dir = str(Path(lm) / self.case / 'par')
        # dm_save_dir = '../temp'
        # dm = DM_basin_average(shp=bs, layer=layer, par_dir=par_dir)
        # dm.vertical_aggregation(isVec=True).basin_average()

        '''obtain the GRACE observation'''
        gr = GRACE_obs(basin=self.basin, dir_obs=configDA.obs.dir, ens_id=rank)

        '''states extract operator'''
        sv = EnsStates(DM=dm, Ens=configDA.basic.ensemble)
        sv.configure_dir(states_dir=str(Path(dp1['output']['dir']) / ('state_%s_ensemble_%s' % (self.case, rank))))

        '''DA experiment'''
        # da = DataAssimilation(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # da = DataAssimilation_monthly(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        da = DataAssimilation_monthly_diag(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # da = DataAssimilation_monthlymean_dailyupdate(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        da.configure_design_matrix(DM=dm)

        '''running with MPI parallelization'''
        print('User case: %s' % self.case)
        print()
        da.run_mpi()

        pass


class DA_GRACE_flexibile(DA_GRACE):
    def __init__(self, case='test', setting_dir='../settings/DA_local', ens=3):
        super().__init__(case, setting_dir, ens)

        pass

    def get_obs_aux(self, configDA, begin_day, end_day):
        """time reference (epochs, windows) of the observations between begin_day and end_day ('YYYY-MM-DD'), by
        obs.GRACE.kind; a missing key gives a clear error naming the key and the settings file"""
        from src_OBS.obs_auxiliary import aux_ESAsing_5daily, aux_GRACE_SH_monthly, aux_GRACE_mascon_monthly, \
            aux_ESM3_5daily
        for key in ('kind', 'aux_for_time_epochs'):
            if key not in configDA.obs.GRACE:
                raise KeyError('PyGLDA v1: obs.GRACE.%s is missing in %s (kind: SH_monthly, Mascon_monthly, ESA_SING or '
                               'ESA_SING_ESM3; aux_for_time_epochs: folder of the GRACE product)'
                               % (key, self.setting_dir / 'DA_setting.json'))
        kind = configDA.obs.GRACE['kind']
        if kind == 'SH_monthly':
            t1 = datetime.strptime(begin_day, '%Y-%m-%d').strftime('%Y-%m')
            t2 = datetime.strptime(end_day, '%Y-%m-%d').strftime('%Y-%m')
            obs_aux = aux_GRACE_SH_monthly().setTimeReference(month_begin=t1, month_end=t2,
                                                              dir_in=configDA.obs.GRACE['aux_for_time_epochs'])
        elif kind == 'ESA_SING':
            obs_aux = aux_ESAsing_5daily().setTimeReference(day_begin=begin_day, day_end=end_day,
                                                            dir_in=configDA.obs.GRACE['aux_for_time_epochs'])
        elif kind == 'ESA_SING_ESM3':
            obs_aux = aux_ESM3_5daily().setTimeReference(day_begin=begin_day, day_end=end_day,
                                                         dir_in=configDA.obs.GRACE['aux_for_time_epochs'])
        elif kind == 'Mascon_monthly':
            t1 = datetime.strptime(begin_day, '%Y-%m-%d').strftime('%Y-%m')
            t2 = datetime.strptime(end_day, '%Y-%m-%d').strftime('%Y-%m')
            obs_aux = aux_GRACE_mascon_monthly().setTimeReference(month_begin=t1, month_end=t2,
                                                                  dir_in=configDA.obs.GRACE['aux_for_time_epochs'])
        else:
            raise ValueError('PyGLDA v1: unknown obs.GRACE.kind %r in %s' % (kind, self.setting_dir / 'DA_setting.json'))
        return obs_aux

    def generate_perturbed_GRACE_obs(self):
        from src_OBS.GRACE_perturbation import GRACE_perturbed_obs

        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        begin_day = configDA.basic.fromdate
        end_day = configDA.basic.todate

        ob = GRACE_perturbed_obs(ens=configDA.basic.ensemble, basin_name=configDA.basic.basin)
        ob.configure_dir(input_dir=configDA.obs.GRACE['preprocess_res'], obs_dir=configDA.obs.dir)

        obs_aux = self.get_obs_aux(configDA=configDA, begin_day=begin_day, end_day=end_day)

        ob.configure_obs_aux(obs_aux=obs_aux)
        ob.perturb_TWS().remove_temporal_mean()
        from src_FlowControl import run_context as rc
        fn = rc.ol_mean_path(configDA.obs.GRACE['OL_mean'], self.case, self.basin)   # <case>/grace/ol_mean.hdf5
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        ob.add_temporal_mean(fn=str(fn))
        ob.save()

        pass

    def run_DA(self, rank: int):
        """The main entrance to the data assimilation experiment"""
        from src_GHM.config_settings import config_settings
        from src_GHM.config_parameters import config_parameters
        from src_GHM.model_initialise import model_initialise
        from src_GHM.ext_adapter import ext_adapter
        from src_GHM.hotrun import model_run_daily
        import json
        from src_DA.observations import GRACE_obs
        from src_DA.ExtractStates import EnsStates
        from src_DA.EnKF import EnKF, EnKF_localization_v1, EnKF_localization_v2, EnKF_domain_localization
        from src_DA.EnSQRA import EnSQRA, EnSQRA_V2

        if rank != 0:
            f = open(self._log_file('log', rank), 'w')
            sys.stdout = f

        print('\n================ Configure DA experiment ===================')

        '''configure DA'''
        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        '''configure model'''
        dp = self.setting_dir / 'setting.json'
        dp1 = json.load(open(dp, 'r'))
        settings = config_settings.loadjson(dp).process(Parallel_ID=rank)
        par = config_parameters(settings)
        model_init = model_initialise(settings=settings, par=par).configure_InitialStates()
        ext = ext_adapter(par=par, settings=settings)
        model_instance = model_run_daily(settings=settings, par=par, model_init=model_init, ext=ext)

        print('Configure shapefile: %s ... (may take a while)' % configDA.basic.basin)
        '''define the basin-shp file and derive the corresponding mask'''
        basin = configDA.basic.basin
        shp_path = configDA.basic.basin_shp
        bs = basin_shp_process(res=0.1, basin_name=basin).shp_to_mask(shp_path=shp_path)

        dp_dir = self.setting_dir / 'pre_process.json'
        dp2 = json.load(open(dp_dir, 'r'))
        lm = dp2['out_dir']
        from src_FlowControl import run_context as rc
        land_mask = str(rc.crop_dir(lm, self.case) / 'mask' / 'mask_global.h5')
        bs.mask_to_vec(model_mask_global=land_mask)
        state_sample = rc.nan_sample_path(configDA.basic.NaNmaskStatesDir, self.case)
        bs.mask_nan(sample=state_sample)

        print('Configure OBS and its design matrix: %s' % (configDA.obs.GRACE['kind']))
        '''obtain the design matrix from pre-saved data'''
        layer = {}
        for key, vv in configDA.basic.layer.items():
            layer[states_var[key]] = vv
        dm_save_dir = self._dm_dir()
        dm = DM_basin_average(shp=bs, layer=layer, LoadfromDisk=True, dir=dm_save_dir)
        # par_dir = str(Path(lm) / self.case / 'par')
        # dm_save_dir = '../temp'
        # dm = DM_basin_average(shp=bs, layer=layer, par_dir=par_dir)
        # dm.vertical_aggregation(isVec=True).basin_average()

        '''obtain the GRACE observation'''
        gr = GRACE_obs(basin=self.basin, dir_obs=configDA.obs.dir, ens_id=rank)

        '''states extract operator'''
        sv = EnsStates(DM=dm, Ens=configDA.basic.ensemble)
        '''the DA's own state folder (layout: <case>/DA/ens_kk/state; legacy: the shared folder)'''
        sv.configure_dir(states_dir=str(settings.statedir))

        '''DA experiment'''
        # da = DataAssimilation(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # da = DataAssimilation_monthly(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # da = EnKF(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # da = EnSQRA(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # da = EnSQRA_V2(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        '''filter by name (DA_setting.json method.fusion_method, src_DA/filter_registry.py) and its inflation'''
        from src_DA.filter_registry import get_filter
        da = get_filter(configDA.method.fusion_method)(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        da.configure_inflation_factor(inflation=configDA.method.inflation)
        print('Filter: %s (%s), inflation %s' % (configDA.method.fusion_method, type(da).__name__,
                                                 configDA.method.inflation))
        # da = EnKF_domain_localization(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # da = EnKF_localization_v2(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # da = DataAssimilation_monthlymean_dailyupdate(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        da.configure_design_matrix(DM=dm)
        '''per-window analysis time -> <case>/log/analysis_times.csv (written by the analysis rank)'''
        if rc.log_dir() is not None:
            da.analysis_log = str(rc.log_dir() / 'analysis_times.csv')

        '''running with MPI parallelization'''
        print('User case: %s' % self.case)
        print()
        da.run_mpi()

        pass
