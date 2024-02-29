from src.EnumType import states_var
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from src.GeoMathKit import GeoMathKit
from DA.ObsDesignMatrix import DM_basin_average


class extract_states:

    def __init__(self, DM: DM_basin_average):
        self.DM = DM
        mask_2D, mask_1D, lat, lon, res = self.DM.shp.collect_mask
        self.mask_1D = mask_1D
        self.__basin_num = len(list(mask_1D.keys())) - 1
        self.__basin_valid_num = np.sum(mask_1D['basin_0'].astype(int))

        pass

    def configure_dir(self, states_dir: str):
        self.__states_dir = Path(states_dir)
        return self

    def get_states_from_date(self, date='2002-04-01'):
        fn = self.__states_dir / ('state.%s.h5' % datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d'))

        fn_h5 = h5py.File(fn, 'r')

        ss = np.zeros((0, self.__basin_valid_num))
        for key in self.DM.statesnn:
            if (key == states_var.Sg) or (key == states_var.Sr):
                k = 1
            else:
                k = 2
            vv = fn_h5[key.name][0:k, self.mask_1D['basin_0']]
            ss = np.vstack((ss, vv))

        return ss.T.flatten()

    def restore_states(self, old_states: dict, new_states):
        """
        update the old states
        """

        new_states = new_states.reshape((-1, len(self.DM.statesnn)))
        new_states = new_states.T

        m = 0
        for key in self.DM.statesnn:

            if (key == states_var.Sg) or (key == states_var.Sr):
                k = 1
            else:
                k = 2

            n = m + k
            old_states[key][0:k, self.mask_1D['basin_0']] = new_states[m:n]
            m = n

        return old_states

    def get_states_by_transfer(self, states: dict):

        ss = np.zeros((0, self.__basin_valid_num))
        for key in self.DM.statesnn:
            if (key == states_var.Sg) or (key == states_var.Sr):
                k = 1
            else:
                k = 2
            vv = states[key][0:k, self.mask_1D['basin_0']]
            ss = np.vstack((ss, vv))

        return ss.T.flatten()

    def get_states_TS(self, date_begin='2002-04-01', date_end='2002-04-01'):

        ts = []

        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)

        for day in daylist:
            ts.append(self.get_states_from_date(date=day))

        return ts


class EnsStates:
    """
    to extract the states of the ensembles
    """

    def __init__(self, DM: DM_basin_average, Ens=30):
        self.DM = DM
        mask_2D, mask_1D, lat, lon, res = self.DM.shp.collect_mask
        self.mask_1D = mask_1D
        self.__basin_num = len(list(mask_1D.keys())) - 1
        self.__basin_valid_num = np.sum(mask_1D['basin_0'].astype(int))
        self.Ens = Ens

        pass

    def configure_dir(self, states_dir: str):
        self.__states_dir = states_dir
        return self

    def get_states_from_date(self, date='2002-04-01'):

        states_assemble = []

        for i in range(1, self.Ens + 1):

            fn = Path(self.__states_dir + '_ensemble_%s' % i) / (
                    'state.%s.h5' % datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d'))

            fn_h5 = h5py.File(fn, 'r')

            ss = np.zeros((0, self.__basin_valid_num))
            for key in self.DM.statesnn:
                if (key == states_var.Sg) or (key == states_var.Sr):
                    k = 1
                else:
                    k = 2
                vv = fn_h5[key.name][0:k, self.mask_1D['basin_0']]
                ss = np.vstack((ss, vv))

            states_assemble.append(ss.T.flatten())

        states_assemble = np.array(states_assemble)

        return states_assemble.T

    def get_states_by_transfer(self, states_ens: list):

        states_assemble = []
        for i in range(len(states_ens)):

            states = states_ens[i]

            ss = np.zeros((0, self.__basin_valid_num))
            for key in self.DM.statesnn:
                if (key == states_var.Sg) or (key == states_var.Sr):
                    k = 1
                else:
                    k = 2
                vv = states[key][0:k, self.mask_1D['basin_0']]
                ss = np.vstack((ss, vv))

            states_assemble.append(ss.T.flatten())

        states_assemble = np.array(states_assemble)

        return states_assemble.T

    def restore_states(self, old_states: dict, new_states):
        """
        update the old states: for only one state
        """

        new_states = new_states.reshape((-1, len(self.DM.statesnn)))
        new_states = new_states.T

        m = 0
        for key in self.DM.statesnn:

            if (key == states_var.Sg) or (key == states_var.Sr):
                k = 1
            else:
                k = 2

            n = m + k
            old_states[key][0:k, self.mask_1D['basin_0']] = new_states[m:n]
            m = n

        return old_states

def demo1():
    from DA.Analysis import basin_shp_process
    bs = basin_shp_process(res=0.1, basin_name='MDB').shp_to_mask(
        shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp')

    bs.mask_to_vec(model_mask_global='/media/user/My Book/Fan/W3RA_data/crop_input/single_run_test/mask/mask_global.h5')

    bs.mask_nan()

    layer = {
        states_var.S0: True,
        states_var.Ss: True,
        states_var.Sd: True,
        states_var.Sr: True,
        states_var.Sg: True,
        states_var.Mleaf: True,
        states_var.FreeWater: True,
        states_var.DrySnow: True
    }

    dm = DM_basin_average(shp=bs, layer=layer,
                          par_dir="/media/user/My Book/Fan/W3RA_data/crop_input/single_run_test/par",
                          LoadfromDisk=True, dir='../temp')

    rs = extract_states(DM=dm).configure_dir(
        states_dir='/media/user/My Book/Fan/W3RA_data/output/state_single_run_test')

    states = rs.get_states_from_date(date='2002-04-01')

    tws = dm.getDM() @ states
    '''verified!'''

    pass


if __name__ == '__main__':
    demo1()
