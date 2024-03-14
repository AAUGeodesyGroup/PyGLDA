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
        from pathlib import Path
        import os

        '''search for the shape file'''
        pp = Path('../data/basin/shp/')
        target = []
        for root, dirs, files in os.walk(pp):
            for file_name in files:
                if (str(file_name).split('_')[0] == self.__basin) and file_name.endswith('.shp') and (
                        'subbasins' in file_name):
                # if (self.__basin in file_name) and file_name.endswith('.shp') and ('subbasins' in file_name):
                    target.append(os.path.join(root, file_name))
        assert len(target) == 1, target

        mf = basin_shp_process(res=res, basin_name=self.__basin).shp_to_mask(shp_path=str(target[0])).mask

        '''load land mask'''
        GRACE_05deg_land_mask = '../data/GRACE/GlobalLandMaskForGRACE.hdf5'
        lm = np.flipud(h5py.File(GRACE_05deg_land_mask, 'r')['resolution_05']['mask'][:])

        self.bshp_mask = (mf[0]*lm).astype(bool)

        return self

    def get2D_GRACE(self, GRACE_obs):
        """
        the input is 1-d array, the output is 2-D array at global scale.
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
