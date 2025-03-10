import sys

sys.path.append('../')

import numpy as np
import h5py
from src_GHM.GeoMathKit import GeoMathKit
import os
from pathlib import Path
from datetime import datetime
from src_OBS.GRACE_perturbation import GRACE_perturbed_obs
from src_OBS.obs_auxiliary import aux_ESAsing_5daily, obs_auxiliary


class ESAsing_5daily_perturbed_obs(GRACE_perturbed_obs):

    def __init__(self, ens=30, basin_name='MDB'):
        super().__init__(ens=ens, basin_name=basin_name)
        self.obs_aux = None
        pass

    def configure_obs_aux(self, obs_aux: aux_ESAsing_5daily):
        '''get a complete time list for all ESA simulation dataset'''
        self.obs_aux = obs_aux
        return self

    def perturb_TWS(self):

        in_dir = self._input_dir

        '''signal'''
        signal_fn = in_dir / ('%s_signal.hdf5' % self.basin_name)
        S_h5fn = h5py.File(signal_fn, 'r')

        '''cov'''
        cov_fn = in_dir / ('%s_cov.hdf5' % self.basin_name)
        C_h5fn = h5py.File(cov_fn, 'r')

        basin_num = len(list(S_h5fn.keys())) - 1
        self.TWS = {}

        time_epochs_1 = list(S_h5fn['time_epoch'][:].astype('str'))
        time_epochs_2 = list(C_h5fn['time_epoch'][:].astype('str'))

        TWS = []
        un_TWS = []
        COV = []
        new_time = []
        print('')
        print('Start to perturb GRACE to obtain appropriate observations...')

        Time_list = self.obs_aux.getTimeReference()['time_epoch']

        '''get covariance matrix: this is static for ESA simulation data'''
        cov = C_h5fn['data']

        for count, time in enumerate(Time_list):
            '''confirm data consistency'''
            assert time == time_epochs_1[count] and time == time_epochs_2[count]
            '''obtain the TWS of each basin'''
            a = []
            for id in range(1, basin_num + 1):
                a.append(S_h5fn['sub_basin_%d' % id][count])
            a = np.array(a)

            '''perturb the signal'''
            if basin_num == 1:
                '''in case there is only one subbasin'''
                perturbed_TWS = np.random.normal(a, np.sqrt(cov), self.ens)[:, None]
            else:
                perturbed_TWS = np.random.multivariate_normal(a, cov, self.ens)

            new_time.append(time)
            COV.append(cov)
            un_TWS.append(a)
            TWS.append(perturbed_TWS)

            pass

        self.TWS['time'] = new_time
        self.TWS['cov'] = COV
        self.TWS['unperturbation'] = np.array(un_TWS)
        self.TWS['ens'] = np.array(TWS)

        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return self


def demo3():
    # ob = GRACE_obs(ens=30, basin_name='MDB')
    ob = ESAsing_5daily_perturbed_obs(ens=30, basin_name='Brahmaputra_ExpZero')
    ob.configure_dir(input_dir='/media/user/My Book/Fan/ESA_SING/TestRes',
                     obs_dir='/media/user/My Book/Fan/ESA_SING/TestRes/obs')
    day_begin = '2003-09-01'
    day_end = '2006-12-31'
    obs_aux = aux_ESAsing_5daily().setTimeReference(day_begin=day_begin, day_end=day_end,
                                                    dir_in='/media/user/My Book/Fan/ESA_SING/ESA_5daily')
    ob.configure_obs_aux(obs_aux=obs_aux)
    # ob.perturb_TWS().save()
    ob.perturb_TWS().save()
    pass


def visualization():
    import pygmt
    from datetime import datetime
    fn = '/media/user/My Book/Fan/ESA_SING/TestRes/obs/Brahmaputra_ExpZero_obs_GRACE.hdf5'
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

        dmax = np.max(dd) * 1.3
        dmin = np.min(dd) * 1.3
        sp_1 = int(np.round((dmax - dmin) / 10))
        if sp_1 == 0:
            sp_1 = 0.5
        sp_2 = sp_1 * 2

        fig.basemap(region=[time[0] - 0.2, time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                    frame=["WSne+tbasin_%s" % (id + 1), "xa2f1", 'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

        for ens in np.arange(ens_all + 1):
            vv = obs['ens_%s' % ens][:, id]

            if ens == 1:
                fig.plot(x=time, y=vv, pen="0.3p,grey", label='Perturbed src_OBS')
            else:
                fig.plot(x=time, y=vv, pen="0.3p,grey")

        ens = 0
        vv = obs['ens_%s' % ens][:, id]
        fig.plot(x=time, y=vv, pen="0.8p,blue", label='Unperturbed src_OBS')

        fig.legend(position='jTR', box='+gwhite+p0.5p')
        fig.shift_origin(yshift='-4.5c')

        pass

    # fig.savefig('result.png')
    fig.show()
    pass


if __name__ == '__main__':
    # demo3()
    visualization()
