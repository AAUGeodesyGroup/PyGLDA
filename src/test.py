# import metview as mv
import netCDF4
import numpy as np
import h5py


def dd():

    ff = h5py.File('/home/user/Downloads/par.h5', 'r')
    par = {}

    # key_word = 'ens_%s' % 1
    key_word = 'ens_0'

    dict_group_load = ff[key_word]
    dict_group_keys = dict_group_load.keys()
    for k in dict_group_keys:
        par[k] = dict_group_load[k][:]

        if np.shape(par[k]) == (1, 1):
            par[k] = par[k][0, 0]

    beta = par['beta']
    FdrainFC = par['FdrainFC']
    wz = np.ones(np.shape(FdrainFC))*0.5
    fD = (wz > 1) * np.fmax(FdrainFC, 1 - 1. / wz) + (wz <= 1) * FdrainFC * np.exp(beta * (wz - 1))

    pass

def dd2():

    ff = np.load('/home/user/Downloads/wz_26.npy')
    ff1 = np.load('/home/user/Downloads/wz2_26.npy')
    ff2 = np.load('/home/user/Downloads/wz3_26.npy')

    pass


if __name__ == '__main__':
    dd2()
    # dd()