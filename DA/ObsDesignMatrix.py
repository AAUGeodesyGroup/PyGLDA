from src.EnumType import states_var
from DA.shp2mask import basin_shp_process
import numpy as np
from scipy.linalg import block_diag
import h5py
from pathlib import Path
from datetime import datetime


class DM_basin_average:

    def __init__(self, shp: basin_shp_process, layer: dict = None, par_dir: str = '../temp', **kwargs):
        self._A = None
        self.layer = None
        self.shp = shp

        '''load parameters: Fraction of HRU'''
        hf = h5py.File(str(Path(par_dir) / 'par.h5'), 'r')
        self.__Fhru = hf['data']['Fhru'][:]

        '''to decide the dimension of vertical layers to be involved in the computation'''
        if layer is None:
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
        states_nn = []
        vertical_dim = 0
        for k, v in layer.items():
            if v:
                states_nn.append(k)
                if (k == states_var.Sg) or (k == states_var.Sr):
                    increment = 1
                else:
                    increment = 2
                vertical_dim += increment

        self.statesnn = sorted(states_nn, key=lambda x: x.value)
        self.__vertical_dim = vertical_dim

        '''load DM from pre-saved matrix'''
        if 'LoadfromDisk' in kwargs:
            if kwargs['LoadfromDisk'] is True:
                self._A = h5py.File(Path(kwargs['dir']) / 'DM.hdf5', 'r')['data'][:]

        pass

    def vertical_aggregation(self, isVec=True):
        mask_2D, mask_1D, lat, lon, res = self.shp.collect_mask

        '''to decide the amount of grid cells to be involved in the computation'''
        basin_valid_num = np.sum(mask_1D['basin_0'].astype(int))

        vertical_dim = self.__vertical_dim

        Fhru = self.__Fhru[:, mask_1D['basin_0']]

        '''create a matrix to represent the vertical aggregation operator: H'''
        H = []
        Fhru_index = []
        for k in self.statesnn:
            m = 1
            if k == states_var.Mleaf:
                m = 4
            if (k == states_var.Sg) or (k == states_var.Sr):
                H.append(m)
                Fhru_index.append(2)
            else:
                H.append(m)
                Fhru_index.append(0)
                H.append(m)
                Fhru_index.append(1)

        H = np.array(H)

        if isVec:
            '''To avoid the numerical issue because of huge matrix addressed next'''
            Fhru_new = []
            for x in Fhru_index:
                if x < 2:
                    Fhru_new.append(Fhru[x])
                else:
                    Fhru_new.append(np.ones(len(Fhru[0])))

            Fhru_new = np.array(Fhru_new)

            self._A = H[:, None] * Fhru_new
            return self

        '''A very large matrix will be generated. However, this should be always avoided.'''
        H_mat = None
        H_vec = []
        for i in range(basin_valid_num):
            print(i)
            vv = []
            for x in Fhru_index:
                if x < 2:
                    vv.append(Fhru[x, i])
                else:
                    vv.append(1)
            vv = np.array(vv)
            H_new = H * vv

            if H_mat is None:
                H_mat = H_new
            else:
                H_mat = block_diag(H_mat, H_new)
                pass

        self._A = H_mat

        return self

    def basin_average(self):
        mask_2D, mask_1D, lat, lon, res = self.shp.collect_mask

        '''to decide the amount of grid cells to be involved in the computation'''
        basin_valid_num = np.sum(mask_1D['basin_0'].astype(int))

        sub_basin_num = len(list(mask_1D.keys())) - 1

        B = []
        for i in range(1, sub_basin_num + 1):
            '''area computation for each cell'''
            lat_basin = np.cos(np.deg2rad(lat[mask_2D['basin_%s' % i]]))
            A = np.sum(lat_basin)

            '''basin-average for each basin'''
            x = np.zeros(basin_valid_num)
            x[mask_1D['basin_%s' % i][mask_1D['basin_0']]] = lat_basin / A

            B.append(x)

            pass

        B = np.array(B)

        if np.shape(self._A)[0] == np.shape(B)[1]:
            self._A = B @ self._A
            return self

        '''re-organize the matrix to speed up the computation'''
        A = self._A
        len_A = len(A)
        y0 = np.repeat(B, len_A, axis=1)

        y1 = A.T.flatten()
        y = y0 * y1[None, :]

        self._A = y
        return self

    def upscaling(self):
        return self

    def filtering(self):
        return self

    def getDM(self):
        return self._A.copy()

    def saveDM(self, out_path: str):

        fn = h5py.File(Path(out_path) / 'DM.hdf5', 'w')
        fn.create_dataset('data', data=self._A)
        fn.close()

        pass


class reorder_states:

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

    def get_states_by_transfer(self, states: dict):
        pass

    def get_states_TS(self, date_begin='2002-04-01', date_end='2002-04-01'):
        pass


def demo1():
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
                          LoadfromDisk=False, dir='../temp')
    dm.vertical_aggregation(isVec=True).basin_average()
    dm.saveDM(out_path='../temp')

    pass


def demo2():
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

    rs = reorder_states(DM=dm).configure_dir(states_dir= '/media/user/My Book/Fan/W3RA_data/output/state_single_run_test')

    states = rs.get_states_from_date(date='2002-04-01')

    '''verified!'''
    pass


if __name__ == '__main__':
    # demo1()
    demo2()
