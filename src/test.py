# import metview as mv
import netCDF4
import numpy as np
from datetime import datetime

# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
import numpy as np
# from scipy
import xarray as xr
import nctoolkit as nc


def dd():
    # nc=mv.read('/media/user/Backup Plus/ERA5/Temperature_W3RA/daily_mean_2m_temperature_1997_01.nc')
    # mv.setcurrent(nc, 't2m')

    target_grid = {'grid': [1, 1], 'area': [33.2, 21.2, 38, 28]}
    # fire_1x1_target_grid = mv.regrid(target_grid,
    #                                  data=nc)
    # tp = netCDF4.Dataset('/media/user/Backup Plus/ERA5/Temperature_W3RA/daily_mean_2m_temperature_1997_01.nc')

    # foo = xr.DataArray(tp[''])

    # tp = nc.open_data('/media/user/Backup Plus/ERA5/Temperature_W3RA/daily_mean_2m_temperature_1997_01.nc')
    #
    ds= xr.open_dataset('/media/user/Backup Plus/ERA5/Temperature_W3RA/daily_mean_2m_temperature_1997_01.nc')

    # print(x)

    pass


if __name__ == '__main__':
    dd()