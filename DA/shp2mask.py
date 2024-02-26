import sys

sys.path.append('../')

import numpy as np
import geopandas as gpd
import shapely.vectorized
from pathlib import Path
import h5py


class basin_shp_process:

    def __init__(self, res, basin_name='MDB', save_dir='../data/basin/mask'):
        self._res = res
        self._basin_name = basin_name
        self._save_dir = Path(save_dir)

        self.collect_mask = None
        self.NaN_mask = None
        pass

    def shp_to_mask(self, shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp', issave=False):
        res = self._res
        basin_name = self._basin_name

        if issave:
            h5fn = str(self._save_dir / ('%s_res_%s.h5' % (basin_name, res)))
            hf = h5py.File(h5fn, 'w')

        mask_basin = {}
        gdf = gpd.read_file(shp_path)

        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)

        lon, lat = np.meshgrid(lon, lat)

        mask_all = np.zeros(np.shape(lat))
        for id in np.arange(gdf.ID.size) + 1:
            # if id !=3: continue
            bd1 = gdf[gdf.ID == id]
            mask1 = shapely.vectorized.touches(bd1.geometry.item(), lon, lat)
            mask2 = shapely.vectorized.contains(bd1.geometry.item(), lon, lat)
            mask3 = mask1 + mask2
            # mask3=mask2
            # print('Mask for %s' % bd1.ID.values[0])
            mask_basin[bd1.ID.values[0]] = mask3
            mask_all += mask3

            if issave:
                hf.create_dataset('sub_basin_%s' % id, data=mask3.astype(int))

        if issave:
            hf.create_dataset('basin', data=mask_all.astype(int))
            hf.close()

        mask_basin[0] = mask_all.astype(bool)

        self.mask = mask_basin
        return self

    def mask_to_vec(self, external_mask=None,
                    model_mask_global='/media/user/My Book/Fan/W3RA_data/crop_input/test/mask/mask_global.h5'):

        basin_name = self._basin_name
        res = self._res

        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)

        lon, lat = np.meshgrid(lon, lat)

        if external_mask is None:
            this_mask = self.mask
        else:
            this_mask = external_mask

        """
        get 1-d vector of the mask of interest
        """

        ff = h5py.File(model_mask_global, 'r')
        model_mask = ff['mask'][:]

        mask_2D = {}
        '''overlap the basin masks'''
        for k, v in this_mask.items():
            mask_2D['basin_%s' % k] = (v * model_mask).astype(bool)

        '''mask 1D inside the land'''
        mask_1D = {}
        # mask_1D['land'] = model_mask[model_mask.astype(bool)]
        for k, v in mask_2D.items():
            mask_1D[k] = (v.astype(int))[model_mask.astype(bool)]
            mask_1D[k] = mask_1D[k].astype(bool)

        self.collect_mask = (mask_2D, mask_1D, lat, lon, res)
        return self

    def mask_nan(self, sample='/media/user/My Book/Fan/W3RA_data/states_sample/state.h5'):
        """
        this is done for masking out the NaN values in states. And to do this work, one sample state file is necessary.
        NaN likely exists because of the mismatch between forcing field and model parameters/mask.
        """

        h5_fn = h5py.File(Path(sample), 'r')
        sample = h5_fn['FreeWater'][0, :]

        NaN_mask = (1 - np.isnan(sample)).astype(bool)

        mask_2D, mask_1D, lat, lon, res = self.collect_mask

        for k, v in mask_2D.items():
            mask_2D[k][mask_2D[k]] = NaN_mask[mask_1D[k]]
            pass

        for k, v in mask_1D.items():
            mask_1D[k] = mask_1D[k] * NaN_mask

        self.collect_mask = (mask_2D, mask_1D, lat, lon, res)

        return self


def demo1():
    bs = basin_shp_process(res=0.1, basin_name='MDB').shp_to_mask(
        shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp')

    x = bs.mask_to_vec()

    del bs
    pass


if __name__ == '__main__':
    demo1()
