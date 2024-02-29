from datetime import datetime
from pathlib import Path
from src_FlowControl.OpenLoop import OpenLoop
from src_GRACE.prepare_GRACE import GRACE_preparation
from src_DA.configure_DA import config_DA
from src_GRACE.GRACE_perturbation import GRACE_perturbed_obs
from src_hydro.EnumType import states_var
from src_DA.shp2mask import basin_shp_process
from src_DA.ObsDesignMatrix import DM_basin_average
import h5py
import numpy as np
import sys


class DA_GRACE(OpenLoop):

    def __init__(self, case='test', setting_dir='../settings/DA_local', ens=3):
        super().__init__(case, setting_dir, ens)

        pass

    def generate_settings(self):
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

        '''modify dp4'''
        dp_dir = self.setting_dir / 'DA_setting.json'
        dp4 = json.load(open(dp_dir, 'r'))
        dp4['basic']['ensemble'] = self.ens
        dp4['basic']['fromdate'] = self.period[0].strftime('%Y-%m-%d')
        dp4['basic']['todate'] = self.period[1].strftime('%Y-%m-%d')
        dp4['basic']['basin'] = self.basin

        '''search for the shape file'''
        pp = Path('../data/basin/shp/')
        target = []
        for root, dirs, files in os.walk(pp):
            for file_name in files:
                if (self.basin in file_name) and file_name.endswith('.shp') and ('subbasins' in file_name):
                    target.append(os.path.join(root, file_name))
        assert len(target) == 1

        dp4['basic']['basin_shp'] = str(target[0])

        with open(dp_dir, 'w') as f:
            json.dump(dp4, f, indent=4)
        return self

    def GRACE_obs_preprocess(self):

        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        basin = configDA.basic.basin
        shp_path = configDA.basic.basin_shp

        '''pre-process of src_GRACE'''
        GR = GRACE_preparation(basin_name=basin,
                               shp_path=shp_path)
        GR.generate_mask()

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

    def gather_OLmean(self):

        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        ens_TWS = []
        for ens_id in range(0, self.ens + 1):
            statedir = Path(self._outdir2)
            ens_dir = str(statedir / ('output_%s_ensemble_%s' % (self.case, ens_id)))
            mm = h5py.File(str(Path(ens_dir) / 'basin_ts.h5'), 'r')

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

        fn = Path(configDA.obs.GRACE['OL_mean']) / ('%s_%s.hdf5' % (self.case, self.basin))
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
        ob.add_temporal_mean()
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

        land_mask = str(self._outdir / self.case / 'mask' / 'mask_global.h5')
        bs.mask_to_vec(model_mask_global=land_mask)
        bs.mask_nan(sample='/media/user/My Book/Fan/W3RA_data/states_sample/state.h5')

        par_dir = str(self._outdir / self.case / 'par')
        dm_save_dir = '../temp'
        dm = DM_basin_average(shp=bs, layer=layer, par_dir=par_dir)
        dm.vertical_aggregation(isVec=True).basin_average()
        dm.saveDM(out_path=dm_save_dir)

        pass