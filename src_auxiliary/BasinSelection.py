import numpy as np
import h5py


class BasinSelection:
    """this is very import because some of the basins/subbasins shape files defined by the users might be invalid as
    they are over the oceans"""

    def __init__(self):
        pass

    def configure_mask(self, model_land_mask='/media/user/My Book/Fan/W3RA_data/basin_selection/model_land_mask.h5',
                       GRACE_1deg_land_mask='/media/user/My Book/Fan/W3RA_data/basin_selection/GlobalLandMaskForGRACE'
                                            '.hdf5',
                       Forcing_mask='/media/user/My Book/Fan/W3RA_data/basin_selection/forcing_mask.npy'):
        self.__model = h5py.File(model_land_mask, 'r')['mask'][:-1, :]
        self.__GRACE = np.flipud(h5py.File(GRACE_1deg_land_mask, 'r')['resolution_1']['mask'][:])
        self.__Forcing = np.load(Forcing_mask)
        return self

    def configure_cell(self, cell=(3, 3)):
        self.__cell = cell

        return self

    def getValidCells(self):
        sub_basin = self.__cell

        N = 180 // sub_basin[0]
        M = 360 // sub_basin[1]

        num_sub_basins = N * M
        id_basins = np.arange(num_sub_basins).reshape((N, M))

        lat_lu = np.array([90 - i * sub_basin[0] for i in range(N)])
        lon_lu = np.array([-180 + i * sub_basin[1] for i in range(M)])

        lat_ru = lat_lu.copy()
        lon_ru = lon_lu + sub_basin[1]

        lat_ld = lat_lu - sub_basin[0]
        lon_ld = lon_lu.copy()

        # lat_rd = lat_lu - sub_basin[0]
        # lon_rd = lon_lu + sub_basin[1]

        # lon_lu, lat_lu = np.meshgrid(lon_lu, lat_lu)
        lon_ru, lat_ru = np.meshgrid(lon_ru, lat_ru)
        lon_ld, lat_ld = np.meshgrid(lon_ld, lat_ld)
        # lon_rd, lat_rd = np.meshgrid(lon_rd, lat_rd)

        '''define 0.1 deg grid'''
        res = 0.1
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon_01, lat_01 = np.meshgrid(lon, lat)

        invalid_id_01_list = []
        for id in range(num_sub_basins):
            m = id_basins == id
            x = (lat_ld[m][:] <= lat_01) * (lat_01 <= lat_ru[m][:]) * (lon_ld[m][:] <= lon_01) * (
                    lon_01 <= lon_ru[m][:])
            a = np.sum(self.__model[x])
            b = np.sum(self.__Forcing[x])
            if a / 900 < 0.1 or b / 900 < 0.1:
                print('Subbasin %s is invalid!' % id)
                invalid_id_01_list.append(id)
            pass

        '''define 1 deg grid'''
        res = 1
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon_01, lat_01 = np.meshgrid(lon, lat)

        invalid_id_10_list = []
        for id in range(num_sub_basins):
            m = id_basins == id
            x = (lat_ld[m][:] <= lat_01) * (lat_01 <= lat_ru[m][:]) * (lon_ld[m][:] <= lon_01) * (
                    lon_01 <= lon_ru[m][:])
            a = np.sum(self.__GRACE[x])
            if a < 1:
                print('Subbasin %s is invalid!' % id)
                invalid_id_10_list.append(id)
            pass

        np.save('../res/invalid_subbasin.npy', np.array(list(set(invalid_id_01_list).union(set(invalid_id_10_list)))))

        pass


def generateForcingSample():
    import metview as mv
    from pathlib import Path

    fn = '/media/user/My Book/Fan/W3RA_data/basin_selection/W3_ERA5_daily_199912.grib'
    grib = mv.read(source=str(fn), area=[90, -180, -90, 180], param='tp')[0]

    mask = 1 - np.isnan(grib.values().reshape(1801, 3600))[:-1, :]

    np.save('/media/user/My Book/Fan/W3RA_data/basin_selection/forcing_mask.npy', mask)
    pass


def demo1():
    generateForcingSample()
    pass


def demo2():
    bs = BasinSelection()
    bs.configure_mask().configure_cell()
    bs.getValidCells()
    pass


if __name__ == '__main__':
    # demo1()
    demo2()
