from src.hotrun import model_run_daily
from DA.configure_DA import config_DA
from src.GeoMathKit import GeoMathKit
from DA.observations import GRACE_obs


class DataAssimilation:

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs):
        self._states = None
        self._states_cov = None
        self._DM = None
        self._model = model
        self.DA_setting = DA_setting
        self._obs = obs

        self._today = '2000-01-01'

        '''obtain info'''




        pass

    def configure(self, DM):
        """
        design matrix for the observation equations. It should be constant over time
        """
        self._DM = DM
        return self

    def initialise(self, states, cov):
        self._states = states
        self._states_cov = cov

        return self

    def predict(self, today='2002-01-01'):
        self._states = self._model.update(is_first_day=False, previous_states=self._states, day=today)
        self._today = today
        return self

    def update(self, obs, obs_cov, DM=None):
        if DM is None:
            DM_A = DM
        else:
            DM_A = self._DM

        return self

    def run(self):
        daylist = GeoMathKit.dayListByDay(begin=self.DA_setting.basic.fromdate,
                                          end=self.DA_setting.basic.todate)

        for day in daylist:

            today = day.strftime('%Y-%m-%d')

            '''kalman filter: prediction step'''
            self.predict(today=today)

            '''obtain obs'''
            self._obs.set_date(today)
            obs = self._obs.get_obs()
            obs_cov = self._obs.get_cov()
            if obs is None:
                '''no obs, no update'''
                continue

            '''kalman filter: update step'''
            self.update(obs=obs, obs_cov=obs_cov)

        pass


def demo1():
    da = DataAssimilation()

    pass


if __name__ == '__main__':
    demo1()
