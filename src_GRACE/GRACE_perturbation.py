import sys

sys.path.append('../')

import numpy as np
import h5py
from src_hydro.GeoMathKit import GeoMathKit
import os
from pathlib import Path


class GRACE_perturbed_obs:

    def __init__(self, ens=30, basin_name='MDB'):
        self.ens = ens
        self.basin_name = basin_name
        self.TWS = None
        pass

    def configure_dir(self, input_dir='/media/user/My Book/Fan/src_GRACE/output',
                      obs_dir='/media/user/My Book/Fan/src_GRACE/obs'):

        self.__input_dir = Path(input_dir)
        self.__obs_dir = Path(obs_dir)
        return self

    def configure_time(self, month_begin='2002-04', month_end='2002-04'):
        self.__monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        return self

    def perturb_TWS(self):

        in_dir = self.__input_dir

        '''signal'''
        signal_fn = in_dir / ('%s_signal.hdf5' % self.basin_name)
        S_h5fn = h5py.File(signal_fn, 'r')

        '''cov'''
        cov_fn = in_dir / ('%s_cov.hdf5' % self.basin_name)
        C_h5fn = h5py.File(cov_fn, 'r')

        basin_num = len(list(S_h5fn.keys())) - 1
        self.TWS = {}

        time_epochs = list(S_h5fn['time_epoch'][:].astype('str'))
        time_epochs_2 = list(C_h5fn['time_epoch'][:].astype('str'))

        TWS = []
        un_TWS = []
        COV = []
        new_time = []
        print('')
        print('Start to perturb src_GRACE to obtain appropriate observations...')
        for month in self.__monthlist:

            '''to confirm this month exists in the list'''
            index = -1
            for tt in time_epochs:
                if month.strftime('%Y-%m') in tt:
                    print(month.strftime('%Y-%m'))
                    index = time_epochs.index(tt)
            if index < 0:
                continue

            '''to confirm if cov is consistent with signal. '''
            assert month.strftime('%Y-%m') in time_epochs_2[index], 'src_GRACE signal is likely incompatible with its cov!'

            '''obtain the cov of this month'''
            cov = C_h5fn['data'][index]

            '''obtain the TWS of each basin'''
            a = []
            for id in range(1, basin_num + 1):
                a.append(S_h5fn['sub_basin_%d' % id][index])

            a = np.array(a)

            '''perturb the signal'''
            perturbed_TWS = np.random.multivariate_normal(a, cov, self.ens)

            new_time.append(time_epochs[index])
            COV.append(cov)
            un_TWS.append(a)
            TWS.append(perturbed_TWS)

            pass

        self.TWS['time'] = new_time
        self.TWS['cov'] = COV
        self.TWS['unperturbation'] = np.array(un_TWS)
        self.TWS['ens'] = np.array(TWS)

        print('Finished')
        return self

    def remove_temporal_mean(self):

        meanTWs = np.mean(self.TWS['unperturbation'], 0)

        self.TWS['ens'] -= meanTWs

        return self

    def add_temporal_mean(self):
        return self

    def save(self):
        """
        Save the data
        """

        fn = self.__obs_dir / ('%s_obs_GRACE.hdf5' % self.basin_name)

        obs = h5py.File(fn, 'w')

        obs.create_dataset(name='cov', data=self.TWS['cov'])

        dt = h5py.special_dtype(vlen=str)
        obs.create_dataset('time_epoch', data=self.TWS['time'], dtype=dt)

        for i in np.arange(self.ens + 1):
            if i == 0:
                obs.create_dataset(name='ens_%s' % i, data=self.TWS['unperturbation'])
            else:
                obs.create_dataset(name='ens_%s' % i, data=self.TWS['ens'][:, i - 1, :])
                pass

        obs.close()
        pass


def demo2():
    # ob = GRACE_obs(ens=30, basin_name='MDB')
    ob = GRACE_perturbed_obs(ens=30, basin_name='Brahmaputra')
    ob.configure_dir(input_dir='/home/user/test/output', obs_dir='/home/user/test/obs'). \
        configure_time(month_begin='2002-04', month_end='2023-09')

    ob.perturb_TWS().save()
    pass


def visualization():
    import pygmt
    from datetime import datetime
    fn = '/home/user/test/obs/MDB_obs_GRACE.hdf5'
    # fn = '/home/user/test/obs/Brahmaputra_obs_GRACE.hdf5'

    ens_all = 30

    obs = h5py.File(fn, 'r')

    time_epoch = obs['time_epoch'][:].astype('str')

    time = []

    for tt in time_epoch:
        hh = datetime.strptime(tt, '%Y-%m-%d')
        time.append(hh.year + (hh.month - 1) / 12 + hh.day / 365.25)
        pass

    time = np.array(time)

    basin_num = len(obs['cov'][0, 0])

    fig = pygmt.Figure()
    for id in range(basin_num):

        dd = obs['ens_%s' % 0][:, id]

        dmax = np.max(dd)*1.3
        dmin = np.min(dd)*1.3
        sp_1 = int(np.round((dmax - dmin) / 10))
        if sp_1 == 0:
            sp_1 = 0.5
        sp_2 = sp_1 * 2

        fig.basemap(region=[time[0] - 0.2, time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                    frame=["WSne+tbasin_%s"%(id+1), "xa2f1", 'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

        for ens in np.arange(ens_all + 1):
            vv = obs['ens_%s' % ens][:, id]

            if ens == 1:
                fig.plot(x=time, y=vv, pen="0.3p,grey", label='Perturbed src_GRACE')
            else:
                fig.plot(x=time, y=vv, pen="0.3p,grey")

        ens = 0
        vv = obs['ens_%s' % ens][:, id]
        fig.plot(x=time, y=vv, pen="0.8p,blue", label='Unperturbed src_GRACE')

        fig.legend(position='jTR', box='+gwhite+p0.5p')
        fig.shift_origin(yshift='-4.5c')

        pass

    fig.savefig('result.png')
    fig.show()
    pass


if __name__ == '__main__':
    # demo2()
    visualization()
