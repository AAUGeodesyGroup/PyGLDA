from datetime import datetime
from pathlib import Path
from FlowControl.OpenLoop import OpenLoop
from GRACE.prepare_GRACE import GRACE_preparation
from DA.configure_DA import config_DA
from GRACE.GRACE_observation import GRACE_obs
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
        dp4['basic']['fromdate'] = self.time[0].strftime('%Y-%m-%d')
        dp4['basic']['todate'] = self.time[1].strftime('%Y-%m-%d')
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

        '''pre-process of GRACE'''
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

    def generate_perturbed_GRACE_obs(self):

        dp_dir = self.setting_dir / 'DA_setting.json'
        configDA = config_DA.loadjson(dp_dir).process()

        begin_day = datetime.strptime(configDA.basic.fromdate, '%Y-%m-%d')
        end_day = datetime.strptime(configDA.basic.todate, '%Y-%m-%d')

        ob = GRACE_obs(ens=configDA.basic.ensemble, basin_name=configDA.basic.basin)
        ob.configure_dir(input_dir=configDA.obs.GRACE['preprocess_res'],
                         obs_dir=configDA.obs.dir). \
            configure_time(month_begin=begin_day.strftime('%Y-%m'), month_end=end_day.strftime('%Y-%m'))

        ob.perturb_TWS().save()

        pass
