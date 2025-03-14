import numpy as np
from datetime import datetime
from pathlib import Path
import h5py
from src_OBS.obs_auxiliary import obs_auxiliary


class GRACE_obs:

    def __init__(self, basin='MDB', dir_obs='/home/user/test/obs/', ens_id=0):
        self.__time_list = None

        '''load the pre-stored observations as well as the covariance matrix'''
        obs_fn = Path(dir_obs) / ('%s_obs_GRACE.hdf5' % basin)
        obs_h5 = h5py.File(obs_fn, 'r')

        self.__time_list = list(obs_h5['time_epoch'][:].astype('str'))

        if 'duration' in obs_h5.keys():
            self.__duration_list = list(obs_h5['duration'][:].astype('str'))

        self.__obs = obs_h5['ens_%s' % ens_id][:]

        self.__cov = obs_h5['cov'][:]

        if np.ndim(self.__cov[0]) < 2:
            self.__cov = self.__cov[:, None, None]

        obs_h5.close()
        pass

    def get_time_list(self):
        return self.__time_list.copy()

    def set_date(self, date='2002-01-01'):
        if date in self.__time_list:
            self.__date_index = self.__time_list.index(date)
        else:
            self.__date_index = None

        return self

    def set_month(self, month='2002-01'):

        for date in self.__time_list:
            if len(date.split('-')) == 2:
                thismonth = date
            else:
                thismonth = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m')

            if thismonth == month:
                self.__date_index = self.__time_list.index(date)
                break
            else:
                self.__date_index = None

        return self

    def get_obs(self):

        if self.__date_index is None:
            return None
        else:
            return self.__obs[self.__date_index]

    def get_cov(self):

        if self.__date_index is None:
            return None
        else:
            return self.__cov[self.__date_index]

    def get_obs_aux(self):
        obs_aux = {
            'time_epoch': self.__time_list.copy(),
            'duration': self.__duration_list.copy()
        }
        return obs_aux

def demo1():
    gr = GRACE_obs()
    hh = gr.get_time_list()
    pass


if __name__ == '__main__':
    demo1()
