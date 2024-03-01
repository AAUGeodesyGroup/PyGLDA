import json
from pathlib import Path
import numpy as np
from src_DA.EnumDA import HydroModel, FusionMethod
from src_hydro.EnumType import states_var


class config_DA:

    def __init__(self):
        self.basic = self.config_basic().__dict__
        self.obs = self.config_obs().__dict__
        self.model = self.config_model().__dict__
        self.method = self.config_method().__dict__

        pass

    @staticmethod
    def save_default():
        obj1 = config_DA()
        tc_dic = obj1.__dict__
        default_path = Path.cwd().parent / 'settings' / 'DA_setting.json'
        with open(default_path, 'w') as f:
            json.dump(tc_dic, f, indent=4)
        return default_path

    @staticmethod
    def loadjson(js):
        Obj = config_DA()

        dict1 = json.load(open(js, 'r'))
        Obj.__dict__ = dict1
        return Obj

    def process(self):
        '''assign the settings into class attributes'''
        basic = self.basic
        self.basic = self.config_basic()
        self.basic.__dict__.update(basic)

        model = self.model
        self.model = self.config_model()
        self.model.__dict__.update(model)
        self.model.name = HydroModel[self.model.name]

        obs = self.obs
        self.obs = self.config_obs()
        self.obs.__dict__.update(obs)

        method = self.method
        self.method = self.config_method()
        self.method.__dict__.update(method)
        self.method.fusion_method = FusionMethod[self.method.fusion_method]

        return self

    class config_basic:
        def __init__(self):
            self.fromdate = '2000-01-02'
            self.todate = '2000-01-31'
            self.basin = 'MDB'
            self.basin_shp = '../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp'
            self.NaNmaskStatesDir = '/media/user/My Book/Fan/W3RA_data/states_sample'
            self.ensemble = 30
            self.layer = {
                states_var.S0.name: True,
                states_var.Ss.name: True,
                states_var.Sd.name: True,
                states_var.Sr.name: True,
                states_var.Sg.name: True,
                states_var.Mleaf.name: True,
                states_var.FreeWater.name: True,
                states_var.DrySnow.name: True
            }
            pass

    class config_model:

        def __init__(self):
            self.name = HydroModel.w3ra_v0.name

    class config_obs:

        def __init__(self):
            self.name = 'src_GRACE'
            self.dir = '/media/user/My Book/Fan/GRACE/obs'
            self.GRACE = {
                'EWH_grid_dir': '/media/user/My Book/Fan/src_GRACE/ewh',
                'cov_dir': '/media/user/My Book/Fan/src_GRACE/DDK3_timeseries',
                'preprocess_res': '/media/user/My Book/Fan/src_GRACE/output',
                'OL_mean': '/media/user/My Book/Fan/src_GRACE/OLmean'
            }

    class config_method:

        def __init__(self):
            self.fusion_method = FusionMethod.EnKF_v0.name


def demo1():
    dp = config_DA.save_default()
    a = config_DA.loadjson(dp).process()

    pass


if __name__ == '__main__':
    demo1()
