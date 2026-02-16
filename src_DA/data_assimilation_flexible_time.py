from src_GHM.hotrun import model_run_daily
from src_DA.configure_DA import config_DA
from src_GHM.GeoMathKit import GeoMathKit
from src_DA.observations import GRACE_obs
from src_DA.ExtractStates import EnsStates
from src_DA.ObsDesignMatrix import DM_basin_average
import numpy as np
from mpi4py import MPI
from datetime import datetime


class DataAssimilation:
    """
    Monthly mean update: each day has the same update;
    This is a version to allow one to set observation at any time interval.
    """

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):

        self._states_predict = None

        self._DM = None
        self._model = model
        self.DA_setting = DA_setting
        self._obs = obs
        self._sv = sv
        self._today = '2000-01-01'

        '''obtain info'''
        self._obs_helper = self.helper_resolve_time(obs.get_obs_aux())

        '''inflation factor'''
        self._inflation = 1.0
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

        '''Inflation to increase the model perturbation'''
        A = A*self._inflation

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

        historic_states = []
        historic_mean_states = None
        obs = None
        obs_cov = None
        rr = -1
        previous_month = -1

        print('=====================Data assimilation=========================')
        for count, day in enumerate(daylist):
            "print information"
            if day.month != previous_month:
                previous_month = day.month
                print('\nDoing year/month: %04d/%02d' % (day.year, day.month))
            today = day.strftime('%Y-%m-%d')
            print('.', end='')
            '''==================kalman filter: prediction step=================================='''
            self.predict(today=today, states=self._states_predict, is_first_day=firstday, issave=True)
            firstday = False

            newRecord = self._obs_helper['NewRecord'][count]
            keepRecord = self._obs_helper['KeepRecord'][count]
            assimilationRecord = self._obs_helper['AssimilationRecord'][count]
            # print(newRecord, keepRecord, assimilationRecord)
            if newRecord:
                """create a new vector to prepare for assimilation of next time"""
                historic_mean_states = None

            if keepRecord:
                '''record the previous states over areas of interest'''
                sv = self._sv.get_states_by_transfer_single(states=self._states_predict)
                if historic_mean_states is None:
                    historic_mean_states = sv
                else:
                    historic_mean_states += sv
            else:
                historic_mean_states = None

            if not assimilationRecord:
                continue

            '''====================start assimilation================================'''
            # print('====> GRACE data has been assimilated.')
            print('|', end='')
            rr += 1
            '''get obs and cov'''
            info = self._obs_helper['AssimilationInfo'][rr]
            self._obs.set_date(date=info[0])
            obs = self._obs.get_obs()
            obs_cov = self._obs.get_cov()
            '''EnKF'''
            historic_mean_states = historic_mean_states / len(info[1])

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
            for his_day_datetime in info[1]:
                his_day = his_day_datetime.strftime("%Y-%m-%d")
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

    def helper_resolve_time(self, obs_aux: dict):
        """
        It is assumed that the date increases at daily basis.
        This tool helps to decide when and how the update takes place
        """

        '''configure time period'''
        daylist = GeoMathKit.dayListByDay(begin=self.DA_setting.basic.fromdate,
                                          end=self.DA_setting.basic.todate)

        '''configure observation reference'''
        duration = obs_aux['duration']
        data_first = []
        data_end = []
        for data in duration:
            a, b = data.split('_')
            data_first.append(datetime.strptime(a, "%Y-%m-%d"))
            data_end.append(datetime.strptime(b, "%Y-%m-%d"))
            pass

        ''''''
        NewRecord = []
        KeepRecord = []
        AssimilationRecord = []
        AssimilationInfo = []

        newRecord = False
        keepRecord = False
        assimilation = False

        nod = 0
        dates_in_list = []

        '''find the match of the first data set'''
        data_index = None
        for day in daylist:
            if day in data_first:
                data_index = data_first.index(day)
                break

        assert data_index is not None
        current_index = data_index

        Nlen = len(data_first)

        '''start loop'''
        for day in daylist:

            if day > data_end[-1]:
                '''this means that no observation is available'''
                NewRecord.append(False)
                KeepRecord.append(False)
                AssimilationRecord.append(False)
                continue

            if data_index == Nlen:
                '''this is to avoid the mistake of the last observation'''
                data_first.append(-1)

            if day == data_first[data_index]:
                newRecord = True
                keepRecord = True

                current_index = data_index
                data_index += 1
                nod = 1
                dates_in_list = [day]

            else:
                newRecord = False

                if data_end[current_index] >= day > data_first[current_index]:
                    keepRecord = True
                    nod += 1
                    dates_in_list.append(day)
                else:
                    keepRecord = False

            if day == data_end[current_index]:
                assimilation = True
                time_epoch = obs_aux['time_epoch'][current_index]
                number_of_days = nod
                AssimilationInfo.append([time_epoch, dates_in_list, current_index])
            else:
                assimilation = False

            NewRecord.append(newRecord)
            KeepRecord.append(keepRecord)
            AssimilationRecord.append(assimilation)

            pass

        return {'NewRecord': NewRecord,
                'KeepRecord': KeepRecord,
                'AssimilationRecord': AssimilationRecord,
                'AssimilationInfo': AssimilationInfo}
