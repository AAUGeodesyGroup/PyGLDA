import h5py
import scipy
import numpy as np


def exp_para():
    par = scipy.io.loadmat('/media/user/My Book/Fan/W3RA_matlab/CDA/init/Parameters/ens1/par_10k_Aussie.mat')
    name = list(par.keys())[-1]

    par_dict = {}
    for var in par[name].dtype.names:
        a = np.array(par[name][var][0])[0].astype(float)

        # if np.shape(a) == (1, 1):
        #     par_dict[var] = a[0, 0]
        # else:
        #     par_dict[var] = a

        par_dict[var] = a

    hf = h5py.File('../temp/%s.h5' % name, 'w')
    dict_group = hf.create_group('data')
    for k, v in par_dict.items():
        dict_group[k] = v
        pass
    hf.close()

    # '''H5 convert to dict'''
    # dict_new = {}
    # file = h5py.File('../temp/par.h5', 'r')
    # dict_group_load = file['data']
    # dict_group_keys = dict_group_load.keys()
    # for k in dict_group_keys:
    #     dict_new[k] = dict_group_load[k][:]

    pass


def exp_clim():
    clim = {}

    par = scipy.io.loadmat(
        '/media/user/My Book/Fan/W3RA_matlab/CDA/init_static/climatologies/albedo_clim_10k_Aussie.mat')
    name = list(par.keys())[-1]
    clim['albedo'] = par[name]

    par = scipy.io.loadmat(
        '/media/user/My Book/Fan/W3RA_matlab/CDA/init_static/climatologies/Prin_pres_CLIM_20002008_10k_Aussie.mat')
    name = list(par.keys())[-1]
    clim['pres'] = par[name]

    par = scipy.io.loadmat(
        '/media/user/My Book/Fan/W3RA_matlab/CDA/init_static/climatologies/Windspeed50m_clim_10k_Aussie.mat')
    name = list(par.keys())[-1]
    clim['windspeed'] = par[name]

    for i in range(1, 13):
        hf = h5py.File('../temp/clim/clim_%02d.h5' % i, 'w')
        dict_group = hf.create_group('data')
        for k, v in clim.items():
            dict_group[k] = v[:, :, i - 1]
            pass
        hf.close()

    # '''H5 convert to dict'''
    # dict_new = {}
    # file = h5py.File('../temp/clim/clim_1.h5', 'r')
    # dict_group_load = file['data']
    # dict_group_keys = dict_group_load.keys()
    # for k in dict_group_keys:
    #     dict_new[k] = dict_group_load[k][:]

    pass


def exp_landmask():
    # par = scipy.io.loadmat(
    #     '/media/user/My Book/Fan/W3RA_matlab/CDA/init_static/landmask_10k_Aussie.mat')
    # name = list(par.keys())[-1]
    # mask = par[name]
    #
    # hf = h5py.File('../temp/mask.h5', 'w')
    # hf.create_dataset('data', data=mask)
    # hf.close()

    '''H5 convert to dict'''
    dict_new = {}
    file = h5py.File('../temp/mask.h5', 'r')
    mask = file['data'][:]

    pass


def exp_states():
    par = scipy.io.loadmat('/media/user/My Book/Fan/W3RA_data/init/state_Aussie_20021231.mat')
    name = list(par.keys())[-1]

    par_dict = {}
    for var in par[name].dtype.names:
        a = np.array(par[name][var][0])[0].astype(float)

        # if np.shape(a) == (1, 1):
        #     par_dict[var] = a[0, 0]
        # else:
        #     par_dict[var] = a

        par_dict[var] = a

    hf = h5py.File('../temp/%s.h5' % name, 'w')
    for k, v in par_dict.items():
        hf.create_dataset(k, data=v)
        pass
    hf.close()

    # '''H5 convert to dict'''
    # dict_new = {}
    # dict_group_load = h5py.File('../temp/state.h5', 'r')
    # dict_group_keys = dict_group_load.keys()
    # for k in dict_group_keys:
    #     dict_new[k] = dict_group_load[k][:]

    pass


def mat2h5_global_2D():
    """
    mat2h5 in global 2-D map
    :return:
    """

    '''load land mask'''
    par = scipy.io.loadmat(
        '/media/user/My Book/Fan/W3RA_data/global/matlab/landmask_10k.mat')
    name = list(par.keys())[-1]
    mask = par[name].astype(bool)

    '''parameter'''
    par = scipy.io.loadmat('/media/user/My Book/Fan/W3RA_data/global/matlab/par_10k.mat')
    name = list(par.keys())[-1]

    par_dict = {}
    for var in par[name].dtype.names:
        a = np.array(par[name][var][0])[0].astype(float)
        par_dict[var] = a

    par_2D = {}
    g = np.ones(np.shape(mask))
    for k, v in par_dict.items():
        if np.shape(v) == (1, 1):
            par_2D[k] = v
        else:
            par_2D[k] = convert2Dto3D_mat(mask=mask, val=v)
            pass

        g = g * (1 - np.isnan(par_2D[k]))
        pass

    hf = h5py.File('../temp/%s.h5' % name, 'w')
    dict_group = hf.create_group('data')
    for k, v in par_2D.items():
        if np.shape(v) == (1, 1):
            dict_group[k] = v
        else:
            dict_group[k] = np.concatenate((v[:, :, 1800:], v[:, :, :1800]), axis=2)
            pass
    hf.close()

    '''climatlogies'''
    clim = {}
    par = scipy.io.loadmat(
        '/media/user/My Book/Fan/W3RA_data/global/matlab/climatologies/albedo_clim_10k.mat')
    name = list(par.keys())[-1]
    clim['albedo'] = par[name]

    par = scipy.io.loadmat(
        '/media/user/My Book/Fan/W3RA_data/global/matlab/climatologies/Prin_pres_CLIM_20002008_10k.mat')
    name = list(par.keys())[-1]
    clim['pres'] = par[name]

    par = scipy.io.loadmat(
        '/media/user/My Book/Fan/W3RA_data/global/matlab/climatologies/Windspeed50m_clim_10k.mat')
    name = list(par.keys())[-1]
    clim['windspeed'] = par[name]

    for i in range(1, 13):
        hf = h5py.File('../temp/clim/clim_%02d.h5' % i, 'w')
        dict_group = hf.create_group('data')
        for k, v in clim.items():
            m = v[:, :, i - 1]
            g = g * (1 - np.isnan(m))

            dict_group[k] = np.concatenate((m[:, 1800:], m[:, :1800]), axis=1)
            pass
        hf.close()

    mask= np.concatenate((mask[:, 1800:], mask[:, :1800]), axis=1)
    g=np.concatenate((g[:, :, 1800:], g[:, :, :1800]), axis=2)

    hf = h5py.File('../temp/mask_1.h5', 'w')
    hf.create_dataset('mask', data=mask.astype(int))
    hf.close()

    mask = g * mask
    mask2 = mask[0] * mask[1]

    hf = h5py.File('../temp/mask_2.h5', 'w')
    hf.create_dataset('mask', data=mask2.astype(int))
    hf.close()


    '''final output: lat: 90-->-90;  lon: -180-->180'''

    pass


def extract_mat2h5_global_2D():
    '''load mask first'''
    dict_new = {}
    file = h5py.File('../temp/mask_1.h5', 'r')
    mask = file['mask'][:].astype(bool)
    file.close()

    dict_new = {}
    file = h5py.File('../temp/mask_2.h5', 'r')
    mask2 = file['mask'][:].astype(bool)
    file.close()

    '''H5 convert to dict'''
    dict_new = {}
    file = h5py.File('../temp/par.h5', 'r')
    dict_group_load = file['data']
    dict_group_keys = dict_group_load.keys()
    for k in dict_group_keys:
        dict_new[k] = dict_group_load[k][:]

    pass


def convert1Dto2D_mat(mask, val):
    """
     input should be matlab 1-d array
     """
    map2d = np.full(mask.shape, np.nan).T
    map2d[mask.T] = val.flatten()
    return map2d.T


def convert2Dto3D_mat(mask, val):
    """
     input should be matlab 2-d array
     """
    dd = np.shape(val)[0]
    map2d = np.full([dd] + list(np.shape(mask)), np.nan).T
    map2d[mask.T, :] = val.T
    return map2d.T


if __name__ == '__main__':
    # exp_para()
    # exp_clim()
    # exp_landmask()
    # exp_states()
    mat2h5_global_2D()
    # extract_mat2h5_global_2D()
