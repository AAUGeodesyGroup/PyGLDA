import numpy as np
from src_DA.Analysis import basin_shp_process
import h5py


class upscaling:
    """A simple block mean of model states,to the same resolution as GRACE"""

    def __init__(self, basin: str):
        self.__basin = basin
        pass

    def configure_model_state(self, GRACEres=0.5, modelres=0.1,
                              globalmask_fn='/media/user/My Book/Fan/W3RA_data/crop_input/case_study_SR/mask/mask_global.h5'):
        """
        the input model state is 1-D array at a high resolution, the output is 2-D array at a coarse resolution in global scale
        notice: could be broadcast
        """
        gm = h5py.File(globalmask_fn, 'r')['mask'][:].astype(bool)

        assert np.shape(gm) == (int(180 / modelres), int(360 / modelres))

        self.__scale_factor = int(GRACEres / modelres)
        self.__gm = gm
        return self

    def get2D_model_state(self, model_state):
        n = self.__scale_factor
        sample = np.full(np.shape(self.__gm), np.nan)
        sample[self.__gm] = model_state
        rows, cols = sample.shape
        new = np.nanmin(sample.reshape(rows // n, n, cols // n, n), axis=(1, 3))

        return new

    def configure_GRACE(self, res=0.5):
        '''load mask'''
        mf = h5py.File('../data/basin/mask/%s_res_%s.h5' % (self.__basin, res), 'r')

        self.bshp_mask = mf['basin'][:].astype(bool)

        return self

    def get2D_GRACE(self, GRACE_obs):
        """
        the input is 1-d array, the output is 2-D array at global scale.
        notice: could be broadcas
        """
        sample = np.full(self.bshp_mask.shape, np.nan)
        sample[self.bshp_mask] = GRACE_obs
        return sample


def demo1():
    us = upscaling(basin='DRB')
    us.configure_model_state()
    # us.downscale_model(model_state=1, GRACEres=0.5, modelres=0.1)
    a = us.get2D_model_state(model_state=1)
    us.configure_GRACE(res=0.5)
    b = us.get2D_GRACE(GRACE_obs=1)
    pass


if __name__ == '__main__':
    demo1()
