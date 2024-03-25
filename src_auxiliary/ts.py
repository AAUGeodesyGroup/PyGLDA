from enum import Enum

import h5py
import numpy as np


class decomposition(Enum):
    trend = 0
    annual = 1
    semi_annual = 2
    quarter_annual = 3


class ts:

    def __init__(self):
        self.time = None
        pass

    def set_period(self, time: np.ndarray):
        """
        Time is year fraction, e.g., 2005.22
        """
        self.time = time - time[0]
        return self

    def setDecomposition(self, signal=None):
        """
        put whatever you would like to analyze into the list
        """

        if signal is None:
            signal = [decomposition.trend, decomposition.annual, decomposition.semi_annual,
                      decomposition.quarter_annual]

        self.__signal = signal

        '''remove possible duplicate element'''
        signal = list(dict.fromkeys(signal))

        DM = np.ones(len(self.time)).T[:, None]
        for ss in decomposition:
            if ss not in signal:
                continue

            dm = self.__get_sub_dm(ss)

            DM = np.hstack((DM, dm.T))

        self.__DM = DM
        return self

    def __get_sub_dm(self, signal=decomposition.trend):

        if signal == decomposition.trend:
            dm = self.time.copy()
            return dm[None, :]

        if signal == decomposition.annual:
            dm = [np.cos((2 * np.pi) * self.time), np.sin((2 * np.pi) * self.time)]
            return np.array(dm)

        if signal == decomposition.semi_annual:
            dm = [np.cos((2 * np.pi / 0.5) * self.time), np.sin((2 * np.pi / 0.5) * self.time)]
            return np.array(dm)

        if signal == decomposition.quarter_annual:
            dm = [np.cos((2 * np.pi / 0.25) * self.time), np.sin((2 * np.pi / 0.25) * self.time)]
            return np.array(dm)

        return None

    def getSignal(self, obs):
        """
        obs is the time series of data to be analyzed.
        """

        x = np.linalg.lstsq(a=self.__DM, b=obs, rcond=-1)[0]

        res = {'bias': x[0]}
        i = 1
        for m in range(4):
            ss = decomposition(m)
            if ss not in self.__signal:
                continue
            if ss == decomposition.trend:
                res[ss.name] = x[i]
                i += 1
                continue
            A = np.sqrt(x[i] ** 2 + x[i + 1] ** 2)
            phi_1 = np.arccos(x[i]/A)
            phi_2 = 2*np.pi - phi_1
            phi = phi_1
            phi[ (-x[i+1]/A) < 0] = phi_2[(-x[i+1]/A) < 0]
            res[ss.name] = np.array([A, np.rad2deg(phi)])

            i += 2

        return res


def demo1():
    tsd = ts().set_period(time=np.array([2002.1, 2002.2, 2002.3, 2002.4, 2002.5]))

    tsd.setDecomposition()

    pass


def demo2():
    from datetime import datetime
    from src_hydro.GeoMathKit import GeoMathKit
    hf = h5py.File('/home/user/Desktop/res/monthly_mean_TWS_DRB_DA.h5', 'r')

    time = []
    vv = []

    for x, v in hf.items():
        m = x.split('-')
        n = datetime(year=int(m[0]), month=int(m[1]), day=15)
        time.append(GeoMathKit.year_fraction(n))
        vv.append(v[:])
        pass

    tsd = ts().set_period(time=np.array(time)).setDecomposition()
    res = tsd.getSignal(obs=vv)
    pass


if __name__ == '__main__':
    demo1()
    # demo2()
