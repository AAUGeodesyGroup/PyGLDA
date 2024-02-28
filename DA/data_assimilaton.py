from src.hotrun import model_run_daily
from DA.configure_DA import config_DA
from src.GeoMathKit import GeoMathKit
from DA.observations import GRACE_obs
from DA.ExtracStates import EnsStates
from DA.ObsDesignMatrix import DM_basin_average
import numpy as np
from mpi4py import MPI


class DataAssimilation:

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):

        self._states_predict = None
        self._states_update = None

        self._DM = None
        self._model = model
        self.DA_setting = DA_setting
        self._obs = obs
        self._sv = sv
        self._today = '2000-01-01'

        '''obtain info'''

        pass

    def configure_design_matrix(self, DM: DM_basin_average):
        """
        design matrix for the observation equations. It should be constant over time.
        Here we pass the function (operator) instead of design matrix to gain flexibility in the future
        """
        self._DM = DM.operator
        return self

    def predict(self, states, today='2002-01-01'):
        """
        prediction is for one ensemble
        """
        self._states_predict = self._model.update(is_first_day=False, previous_states=states, day=today)
        self._today = today
        return self

    def update(self, obs, obs_cov, ens_states):
        """
        update is for all ensembles
        reference: WIKI
        """
        R = obs_cov

        '''calculate the deviation of ens_states'''
        A = ens_states - np.mean(ens_states, 1)

        '''propagate it into obs-equivalent variable'''
        HX = self._DM(states=ens_states)

        '''to calculate the deviation'''
        HA = HX - np.mean(HX, 1)

        '''calculate matrix P'''
        P = np.cov(HA) + R

        '''calculate the gain factor K'''
        N = self.DA_setting.basic.ensemble
        '''method-1: straight-forward'''
        # K = 1 / (N - 1) * A @ HA.T @ np.linalg.inv(P)
        '''method-2: via linear solver'''
        Bt = np.linalg.lstsq(P.T, HA, rcond=None)[0]
        K = 1 / (N - 1) * A @ Bt.T

        '''update the states'''
        self._states_update = ens_states + K @ (obs - HX)

        return self

    def run_mpi(self):
        """
        parallelized running with MPI4py
        """

        '''main process'''
        main_thread = 1

        '''OL process: no perturbation'''
        OL_thread = 0

        '''preparation'''
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        '''make sure that each ensemble is given a process'''
        assert size == self.DA_setting.basic.ensemble+1

        '''configure setting'''
        daylist = GeoMathKit.dayListByDay(begin=self.DA_setting.basic.fromdate,
                                          end=self.DA_setting.basic.todate)

        '''assign job to each ensemble'''
        for day in daylist:

            today = day.strftime('%Y-%m-%d')

            '''kalman filter: prediction step'''
            self.predict(today=today, states=self._states_predict)

            '''obtain obs'''
            self._obs.set_date(date=today)
            obs = self._obs.get_obs()
            obs_cov = self._obs.get_cov()
            if obs is None:
                pass
                continue

            '''synchronization to prepare for the data assimilation'''
            this_day = comm.gather(today, root=main_thread)
            if rank == main_thread:
                for i in this_day:
                    '''confirm the synchronization again'''
                    assert this_day[i] == today

            '''collect states from each ensemble'''
            ens_states = comm.gather(root=main_thread, sendobj=self._states_predict)

            if rank != main_thread:
                pass
            else:
                '''kalman filter: update step'''
                '''load ensemble states'''
                ens_states = self._sv.get_states_by_transfer(states_ens=ens_states)
                self.update(obs=obs, obs_cov=obs_cov, ens_states=ens_states)

            '''every thread should wait until the main thread finishes its job'''
            states_predict = comm.scatter(sendobj=self._states_update.T, root=main_thread)
            if rank != OL_thread:
                '''do not update the OL thread'''
                self._states_predict = states_predict.flatten()

        pass


def demo1():
    da = DataAssimilation()

    pass


if __name__ == '__main__':
    demo1()
