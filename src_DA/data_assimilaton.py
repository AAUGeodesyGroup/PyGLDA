from src_hydro.hotrun import model_run_daily
from src_DA.configure_DA import config_DA
from src_hydro.GeoMathKit import GeoMathKit
from src_DA.observations import GRACE_obs
from src_DA.ExtracStates import EnsStates
from src_DA.ObsDesignMatrix import DM_basin_average
import numpy as np
from mpi4py import MPI


class DataAssimilation:

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):

        self._states_predict = None

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

    def predict(self, states, today='2002-01-01', is_first_day=False, issave=True):
        """
        prediction is for one ensemble
        """
        self._states_predict = self._model.update(is_first_day=is_first_day, previous_states=states, day=today,
                                                  issave=issave)
        self._today = today
        return self

    def update(self, obs, obs_cov, ens_states):
        """
        update is for all ensembles
        reference: WIKI
        """
        R = obs_cov

        '''calculate the deviation of ens_states'''
        A = ens_states - np.mean(ens_states, 1)[:, None]

        '''propagate it into obs-equivalent variable'''
        HX = self._DM(states=ens_states)

        '''to calculate the deviation'''
        HA = HX - np.mean(HX, 1)[:, None]

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
        states_update = ens_states + K @ (obs - HX)

        return states_update

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
        assert size == self.DA_setting.basic.ensemble + 1

        '''configure setting'''
        daylist = GeoMathKit.dayListByDay(begin=self.DA_setting.basic.fromdate,
                                          end=self.DA_setting.basic.todate)

        '''assign job to each ensemble'''
        firstday = True
        for day in daylist:

            today = day.strftime('%Y-%m-%d')

            '''kalman filter: prediction step'''
            self.predict(today=today, states=self._states_predict, is_first_day=firstday)
            firstday = False

            '''obtain obs'''
            self._obs.set_date(date=today)
            obs = self._obs.get_obs()
            obs_cov = self._obs.get_cov()

            if obs is None:
                '''src_GRACE is not available at this time epoch'''
                pass
                continue

            '''synchronization to prepare for the data assimilation'''
            this_day = comm.gather(today, root=main_thread)
            if rank == main_thread:
                for iday in this_day:
                    '''confirm the synchronization again'''
                    assert iday == today

            '''collect obs from each ensemble'''
            ens_obs = comm.gather(root=main_thread, sendobj=obs)

            '''collect states from each ensemble'''
            ens_states = comm.gather(root=main_thread, sendobj=self._states_predict)

            if rank != main_thread:
                states_update = None
                pass
            else:
                '''delete the OL process: it does not participate in data assimilation'''
                del ens_obs[OL_thread]
                del ens_states[OL_thread]

                '''load ensemble states'''
                ens_states = self._sv.get_states_by_transfer(states_ens=ens_states)
                ens_obs = np.array(ens_obs).T

                '''kalman filter: update step'''
                states_update = self.update(obs=ens_obs, obs_cov=obs_cov, ens_states=ens_states)
                # print(np.shape(states_update.T), '=============================')
                states_update = list(states_update.T)

                '''add an arbitrary matrix to states to be able to use comm.scatter: 
                by default, the process id of OL_process must be 0'''
                states_update = [np.zeros(np.shape(states_update[0]))] + states_update

            '''every thread should wait until the main thread finishes its job, and redistribute the updated states'''
            states_ens_update = comm.scatter(sendobj=states_update, root=main_thread)

            '''possible negative value exists in the updated states. replace the negative value with zero'''
            states_ens_update[states_ens_update < 0] = 0.0001

            if rank != OL_thread:
                '''do not update the OL thread'''
                self._states_predict = self._sv.restore_states(old_states=self._states_predict,
                                                               new_states=states_ens_update.flatten())
                '''save the new states, and overlap the old one'''
                self._model.save(states=self._states_predict, day=today)
        pass


class DataAssimilation_monthly(DataAssimilation):

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):
        super().__init__(DA_setting, model, obs, sv)
        pass

    def run_mpi(self):
        """
        parallelized running with MPI4py
        """
        from datetime import datetime, timedelta

        '''main process'''
        main_thread = 1

        '''OL process: no perturbation'''
        OL_thread = 0

        '''preparation'''
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        '''make sure that each ensemble is given a process'''
        assert size == self.DA_setting.basic.ensemble + 1

        '''configure setting'''
        daylist = GeoMathKit.dayListByDay(begin=self.DA_setting.basic.fromdate,
                                          end=self.DA_setting.basic.todate)

        '''assign job to each ensemble'''
        firstday = True

        pre_month = '1000-01-01'
        historic_states_daylist = []
        historic_states = []
        historic_mean_states = None
        obs = None
        obs_cov = None

        for day in daylist:

            today = day.strftime('%Y-%m-%d')
            this_month = day.strftime('%Y-%m')

            '''kalman filter: prediction step'''
            self.predict(today=today, states=self._states_predict, is_first_day=firstday, issave=True)
            firstday = False

            '''obtain obs'''
            if this_month != pre_month:
                self._obs.set_month(month=this_month)
                obs = self._obs.get_obs()
                obs_cov = self._obs.get_cov()
                pre_month = this_month
                '''record historic states to calculate the monthly mean and update the daily state'''
                historic_states_daylist = []
                historic_states = []
                historic_mean_states = None
                '''calculate the last day of this month'''
                year, month = day.year, day.month
                if month == 12:
                    last_day = datetime(year, month, 31)
                else:
                    last_day = datetime(year, month + 1, 1) + timedelta(days=-1)

                pass

            if obs is None:
                '''src_GRACE is not available at this month'''
                pass
                continue

            '''record the previous states over areas of interest'''
            historic_states_daylist.append(today)
            sv = self._sv.get_states_by_transfer_single(states=self._states_predict)
            # historic_states.append(sv)
            if historic_mean_states is None:
                historic_mean_states = sv
            else:
                historic_mean_states += sv

            if day != last_day:
                '''no update'''
                continue

            '''Kalman filter by the end of the month'''
            historic_mean_states = historic_mean_states / len(historic_states_daylist)
            # print(historic_states_daylist)
            '''synchronization to prepare for the data assimilation'''
            this_day = comm.gather(today, root=main_thread)
            if rank == main_thread:
                for iday in this_day:
                    '''confirm the synchronization again'''
                    assert iday == today

            '''collect obs from each ensemble'''
            ens_obs = comm.gather(root=main_thread, sendobj=obs)

            '''collect states from each ensemble'''
            ens_states = comm.gather(root=main_thread, sendobj=historic_mean_states)

            if rank != main_thread:
                delta_state = None
                pass
            else:
                '''delete the OL process: it does not participate in data assimilation'''
                del ens_obs[OL_thread]
                del ens_states[OL_thread]

                '''load ensemble states'''
                ens_obs = np.array(ens_obs).T
                ens_states = np.array(ens_states).T

                '''kalman filter: update step'''
                states_update = self.update(obs=ens_obs, obs_cov=obs_cov, ens_states=ens_states)
                # print(np.shape(states_update.T), '=============================')

                '''delta for monthly mean'''
                delta_state = states_update - ens_states
                delta_state = list(delta_state.T)

                '''add an arbitrary matrix to states to be able to use comm.scatter: 
                by default, the process id of OL_process must be 0'''
                delta_state = [np.zeros(np.shape(delta_state[0]))] + delta_state

            '''every thread should wait until the main thread finishes its job, and redistribute the updated states'''
            states_ens_update_delta = comm.scatter(sendobj=delta_state, root=main_thread)
            delta_state = None  # free the memory
            '''update the states for each ensemble: equal increment for each day'''
            for his_day in historic_states_daylist:
                states_old = self._sv.load_state_dict(date=his_day)

                if rank != OL_thread:
                    '''do not update the OL thread'''
                    new_state = self._sv.restore_states(old_states=states_old,
                                                        new_states=states_ens_update_delta.flatten(), isdelta=True)

                    '''possible negative value exists in the updated states. replace the negative value with zero'''
                    for key, vv in new_state.items():
                        vv[vv < 0] = 0.0001

                    '''save the new states, and overwrite the old one'''
                    self._model.save(states=new_state, day=his_day)

            if rank != OL_thread:
                '''to assign the last day to the current prediction vector to enable another round of kalman filter'''
                self._states_predict = new_state

            '''free memory'''
            states_old = None
            vv = None
            states_ens_update_delta = None
        pass


class DataAssimilation_monthly_diag(DataAssimilation_monthly):
    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):
        super().__init__(DA_setting, model, obs, sv)
        pass

    def update(self, obs, obs_cov, ens_states):
        """
        update is for all ensembles
        reference: WIKI
        """
        R = obs_cov

        '''calculate the deviation of ens_states'''
        A = ens_states - np.mean(ens_states, 1)[:, None]

        '''propagate it into obs-equivalent variable'''
        HX = self._DM(states=ens_states)

        '''to calculate the deviation'''
        HA = HX - np.mean(HX, 1)[:, None]

        '''calculate matrix P'''
        # print(R)
        P = np.cov(HA) + np.diag(np.diag(R))

        '''calculate the gain factor K'''
        N = self.DA_setting.basic.ensemble
        '''method-1: straight-forward'''
        # K = 1 / (N - 1) * A @ HA.T @ np.linalg.inv(P)
        '''method-2: via linear solver'''
        Bt = np.linalg.lstsq(P.T, HA, rcond=None)[0]
        K = 1 / (N - 1) * A @ Bt.T

        '''update the states'''
        states_update = ens_states + K @ (obs - HX)
        print(obs)
        print(HX)
        return states_update


class DataAssimilation_monthlymean_dailyupdate(DataAssimilation):

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):
        super().__init__(DA_setting, model, obs, sv)
        pass

    def configure_design_matrix(self, DM: DM_basin_average):
        """
        design matrix for the observation equations. It should be constant over time.
        Here we pass the function (operator) instead of design matrix to gain flexibility in the future
        """
        self._DM = DM.getDM()
        return self

    def update(self, obs, obs_cov, ens_states):
        """
        update is for all ensembles
        reference: WIKI
        """
        R = obs_cov

        '''calculate the deviation of ens_states'''
        sh = np.shape(ens_states)
        ens_states_2d = ens_states.T.reshape((sh[-1], -1)).T
        A = ens_states_2d - np.mean(ens_states_2d, 1)[:, None]

        '''propagate it into obs-equivalent variable'''
        H = self._DM
        # print(np.shape(H))
        # print(np.shape(ens_states))
        HX = np.einsum('ij, jkl -> ikl', H, ens_states)
        # HX = H @ ens_states
        HX = np.mean(HX, axis=1)
        # HX = self._DM(states=ens_states)

        '''to calculate the deviation'''
        HA = HX - np.mean(HX, 1)[:, None]

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
        states_update = ens_states_2d + K @ (obs - HX)

        return states_update

    def run_mpi(self):
        """
        parallelized running with MPI4py
        """
        from datetime import datetime, timedelta

        '''main process'''
        main_thread = 1

        '''OL process: no perturbation'''
        OL_thread = 0

        '''preparation'''
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        '''make sure that each ensemble is given a process'''
        assert size == self.DA_setting.basic.ensemble + 1

        '''configure setting'''
        daylist = GeoMathKit.dayListByDay(begin=self.DA_setting.basic.fromdate,
                                          end=self.DA_setting.basic.todate)

        '''assign job to each ensemble'''
        firstday = True

        pre_month = '1000-01-01'
        historic_states_daylist = []
        historic_states = []
        historic_mean_states = None
        obs = None
        obs_cov = None

        for day in daylist:

            today = day.strftime('%Y-%m-%d')
            this_month = day.strftime('%Y-%m')

            '''kalman filter: prediction step'''
            self.predict(today=today, states=self._states_predict, is_first_day=firstday, issave=True)
            firstday = False

            '''obtain obs'''
            if this_month != pre_month:
                self._obs.set_month(month=this_month)
                obs = self._obs.get_obs()
                obs_cov = self._obs.get_cov()
                pre_month = this_month
                '''record historic states to calculate the monthly mean and update the daily state'''
                historic_states_daylist = []
                historic_states = []

                '''calculate the last day of this month'''
                year, month = day.year, day.month
                if month == 12:
                    last_day = datetime(year, month, 31)
                else:
                    last_day = datetime(year, month + 1, 1) + timedelta(days=-1)

                pass

            if obs is None:
                '''src_GRACE is not available at this month'''
                pass
                continue

            '''record the previous states over areas of interest'''
            historic_states_daylist.append(today)
            sv = self._sv.get_states_by_transfer_single(states=self._states_predict)
            historic_states.append(sv)

            if day != last_day:
                '''no update'''
                continue

            historic_states = np.array(historic_states)
            '''Kalman filter by the end of the month'''

            # print(historic_states_daylist)
            '''synchronization to prepare for the data assimilation'''
            this_day = comm.gather(today, root=main_thread)
            if rank == main_thread:
                for iday in this_day:
                    '''confirm the synchronization again'''
                    assert iday == today

            '''collect obs from each ensemble'''
            ens_obs = comm.gather(root=main_thread, sendobj=obs)

            '''collect states from each ensemble'''
            ens_states = comm.gather(root=main_thread, sendobj=historic_states)

            '''release the memory'''
            historic_states = []

            if rank != main_thread:
                states_update = None
                pass
            else:
                '''delete the OL process: it does not participate in data assimilation'''
                del ens_obs[OL_thread]
                del ens_states[OL_thread]

                '''load ensemble states'''
                ens_obs = np.array(ens_obs).T
                ens_states = np.array(ens_states).T

                '''kalman filter: update step'''
                states_update = self.update(obs=ens_obs, obs_cov=obs_cov, ens_states=ens_states)
                # print(np.shape(states_update.T), '=============================')

                states_update = list(states_update.T)

                '''add an arbitrary matrix to states to be able to use comm.scatter: 
                by default, the process id of OL_process must be 0'''
                states_update = [np.zeros(np.shape(states_update[0]))] + states_update

            '''every thread should wait until the main thread finishes its job, and redistribute the updated states'''
            states_ens_update = comm.scatter(sendobj=states_update, root=main_thread)
            states_update = None  # free the memory
            states_ens_update = np.array(states_ens_update).reshape((len(historic_states_daylist), -1))
            '''update the states for each ensemble: equal increment for each day'''
            for i in range(len(historic_states_daylist)):
                his_day = historic_states_daylist[i]
                states_old = self._sv.load_state_dict(date=his_day)
                states_ens_update_delta = states_ens_update[i]

                if rank != OL_thread:
                    '''do not update the OL thread'''
                    new_state = self._sv.restore_states(old_states=states_old,
                                                        new_states=states_ens_update_delta, isdelta=False)

                    '''possible negative value exists in the updated states. replace the negative value with zero'''
                    for key, vv in new_state.items():
                        vv[vv < 0] = 0.0001

                    '''save the new states, and overwrite the old one'''
                    self._model.save(states=new_state, day=his_day)

            if rank != OL_thread:
                '''to assign the last day to the current prediction vector to enable another round of kalman filter'''
                self._states_predict = new_state

            '''free the memory'''
            vv = None
            states_ens_update = None
            states_ens_update_delta = None
            states_old = None

        pass
