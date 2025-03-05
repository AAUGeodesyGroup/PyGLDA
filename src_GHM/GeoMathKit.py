"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/4 12:02
@Description:
"""
import calendar
import datetime
import gzip
import math

import numpy as np


class GeoMathKit:

    def __init__(self):
        pass

    @staticmethod
    def nutarg(tdb_mjd, THETA):
        """
        As TT -TDB is quite small, Consequently, TT can be used in practice in place of TDB in the expressions for the
        fundamental nutation arguments. See IERS2010 page 66

        Doodson's argument calculation
        ! PURPOSE    :  COMPUTE DOODSON'S FUNDAMENTAL ARGUMENTS (BETA)
        !               AND FUNDAMENTAL ARGUMENTS FOR NUTATION (FNUT)
        !               BETA=(B1,B2,B3,B4,B5,B6)
        !               FNUT=(F1,F2,F3,F4,F5)
        !               F1=MEAN ANOMALY (MOON)
        !               F2=MEAN ANOMALY (SUN)
        !               F3=F=MOON'S MEAN LONGITUDE-LONGITUDE OF LUNAR ASC. NODE
        !               F4=D=MEAN ELONGATION OF MOON FROM SUN
        !               F5=MEAN LONGITUDE OF LUNAR ASC. NODE
        !               B2=S=F3+F5
        !               B3=H=S-F4=S-D
        !               B4=P=S-F1
        !               B5=NP=-F5
        !               B6=PS=S-F4-F2
        !               B1=THETA+PI-S
        ! PARAMETERS :.
        !        IN  :  TMJD   : TIME IN MJD                               R*8
        !               THETA  : CORRESPONDING MEAN SID.TIME GREENWICH     R*8
        !        OUT :  BETA   : DOODSON ARGUMENTS                         R*8
        !               FNUT   : FUNDAMENTAL ARGUMENTS FOR NUTATION        R*8
        ! SR CALLED  :
        ! REMARKS    :
        !
        ! AUTHOR     : FROM BERNESE5.0
        :param tdb_mjd: tdb time in MJD convention
        :param theta : CORRESPONDING MEAN SID.TIME GREENWICH
        :return:
        """
        # TIME INTERVAL (IN JUL. CENTURIES) BETWEEN TMJD AND J2000.
        TU = (tdb_mjd - 51544.5) / 36525.
        # !  FUNDAMENTAL ARGUMENTS (IN RAD)
        ROH = np.pi / 648000.
        R = 1296000.

        F1 = (485868.249036 + (1325. * R + 715923.217799902) * TU + 31.8792 * TU * TU +
              .051635 * TU * TU * TU - .0002447 * TU * TU * TU * TU) * ROH

        F2 = (1287104.79305 + (99. * R + 1292581.0480999947) * TU - .5532 * TU * TU +
              .000136 * TU * TU * TU - .00001149 * TU * TU * TU * TU) * ROH

        F3 = (335779.526232 + (1342. * R + 295262.8478000164) * TU - 12.7512 * TU * TU -
              .001037 * TU * TU * TU + .00000417 * TU * TU * TU * TU) * ROH

        F4 = (1072260.7036900001 + (1236. * R + 1105601.2090001106) * TU - 6.3706 * TU * TU +
              .006593 * TU * TU * TU - .00003169 * TU * TU * TU * TU) * ROH

        F5 = (450160.398036 - (5. * R + 482890.5431000004) * TU + 7.4722 * TU * TU +
              .007702 * TU * TU * TU - .00005939 * TU * TU * TU * TU) * ROH

        FNUT = np.array([F1, F2, F3, F4, F5])

        BETA = np.zeros((6, 1))

        BETA[1, 0] = F3 + F5
        S = BETA[1, 0]
        BETA[2, 0] = S - F4
        BETA[3, 0] = S - F1
        BETA[4, 0] = -F5
        BETA[5, 0] = S - F4 - F2
        BETA[0, 0] = THETA + np.pi - S

        return BETA

    @staticmethod
    def doodsonArguments(mjd):
        """
        This is provided by Matlab code from EOT11a
        :param mjd: TDB time in MJD convention
        :return:
        """
        dood = np.zeros((6, 1))

        # compute GMST
        Tu0 = (np.floor(mjd) - 51544.5) / 36525.0
        gmst0 = (6.0 / 24 + 41.0 / (24 * 60) + 50.54841 / (24 * 60 * 60))
        gmst0 = gmst0 + (8640184.812866 / (24 * 60 * 60)) * Tu0
        gmst0 = gmst0 + (0.093104 / (24 * 60 * 60)) * Tu0 * Tu0
        gmst0 = gmst0 + (-6.2e-6 / (24 * 60 * 60)) * Tu0 * Tu0 * Tu0
        r = 1.002737909350795 + 5.9006e-11 * Tu0 - 5.9e-15 * Tu0 * Tu0
        gmst = np.mod(2 * np.pi * (gmst0 + r * np.mod(mjd, 1)), 2 * np.pi)

        t = (mjd - 51544.5) / 365250.0

        dood[1, 0] = (218.31664562999 + (
                4812678.81195750 + (-0.14663889 + (0.00185140 + -0.00015355 * t) * t) * t) * t) * np.pi / 180
        dood[2, 0] = (280.46645016002 + (
                360007.69748806 + (0.03032222 + (0.00002000 + -0.00006532 * t) * t) * t) * t) * np.pi / 180
        dood[3, 0] = (83.35324311998 + (
                40690.13635250 + (-1.03217222 + (-0.01249168 + 0.00052655 * t) * t) * t) * t) * np.pi / 180
        dood[4, 0] = (234.95544499000 + (
                19341.36261972 + (-0.20756111 + (-0.00213942 + 0.00016501 * t) * t) * t) * t) * np.pi / 180
        dood[5, 0] = (282.93734098001 + (
                17.19457667 + (0.04568889 + (-0.00001776 + -0.00003323 * t) * t) * t) * t) * np.pi / 180
        dood[0, 0] = gmst + np.pi - dood[1, 0]

        return dood

    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        :param lon1:  point 1
        :param lat1:
        :param lon2: point 2
        :param lat2:
        :return:
        """

        '''Degree to radian'''
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        ''' haversine'''
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # average radius of the earth, unit [km]

        return c * r * 1000

    @staticmethod
    def getPnm(lat, Nmax: int, option=0):
        """
        get legendre function up to degree/order Nmax in Lat.
        :param lat: Co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param Nmax:
        :param option:
        :return:
        """

        if option != 0:
            lat = (90. - lat) / 180. * np.pi

        NMmax = int((Nmax + 1) * (Nmax + 2) / 2)

        if type(lat) is np.ndarray:
            Nsize = np.size(lat)
        else:
            Nsize = 1

        Pnm = np.zeros((NMmax, Nsize))

        Pnm[GeoMathKit.getIndex(0, 0)] = 1

        Pnm[GeoMathKit.getIndex(1, 1)] = np.sqrt(3) * np.sin(lat)

        '''For the diagonal element'''
        for n in range(2, Nmax + 1):
            Pnm[GeoMathKit.getIndex(n, n)] = np.sqrt((2 * n + 1) / (2 * n)) * np.sin(lat) * Pnm[
                GeoMathKit.getIndex(n - 1, n - 1)]

        for n in range(1, Nmax + 1):
            Pnm[GeoMathKit.getIndex(n, n - 1)] = np.sqrt(2 * n + 1) * np.cos(lat) * Pnm[
                GeoMathKit.getIndex(n - 1, n - 1)]

        for n in range(2, Nmax + 1):
            for m in range(n - 2, -1, -1):
                Pnm[GeoMathKit.getIndex(n, m)] = \
                    np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (2 * n - 1)) \
                    * np.cos(lat) * Pnm[GeoMathKit.getIndex(n - 1, m)] \
                    - np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (n - m - 1) * (n + m - 1) / (2 * n - 3)) \
                    * Pnm[GeoMathKit.getIndex(n - 2, m)]

        return Pnm

    @staticmethod
    def getIndex(n: int, m: int):
        """
        index of Cnm at one-dimension array
        :param n: degree
        :param m: order
        :return:
        """
        assert m <= n

        return int(n * (n + 1) / 2 + m)

    @staticmethod
    def getCoLatLoninRad(lat, lon):
        '''
        :param lat: geophysical coordinate in degree
        :param lon: geophysical coordinate in degree
        :return: Co-latitude and longitude in rad
        '''

        theta = (90. - lat) / 180. * np.pi
        phi = lon / 180. * np.pi

        return theta, phi

    @staticmethod
    def CS_1dTo2d(CS: np.ndarray):
        """

        C00 C10 C20 =>     C00
                           C10 C20

        :param CS: one-dimension array
        :return: two dimension array
        """

        def index(N):
            n = (np.round(np.sqrt(2 * N))).astype(np.int) - 1
            m = N - (n * (n + 1) / 2).astype(np.int) - 1
            return n, m

        CS_index = np.arange(len(CS)) + 1
        n, m = index(CS_index)

        dim = index(len(CS))[0] + 1
        CS2d = np.zeros((dim, dim))
        CS2d[n, m] = CS

        return CS2d

    @staticmethod
    def CS_2dTo1d(CS: np.ndarray):
        """
        Transform the CS in 2-dimensional matrix to 1-dimemsion vectors
        example:
        00
        10 11
        20 21 22
        30 31 32 33           =>           00 10 11 20 21 22 30 31 32 33 ....

        :param CS:
        :return:
        """
        shape = np.shape(CS)
        assert len(shape) == 2
        index = np.nonzero(np.tril(np.ones(shape)))

        return CS[index]

    @staticmethod
    def dayListByMonth(begin, end):
        """
        get the date of every day between the given 'begin' month and 'end' month

        :param begin: year, month
        :param end: year,month
        :return:
        """

        daylist = []
        begin_date = datetime.date(begin[0], begin[1], 1)
        end_date = datetime.date(end[0], end[1], calendar.monthrange(end[0], end[1])[1])

        while begin_date <= end_date:
            date_str = begin_date
            daylist.append(date_str)
            begin_date += datetime.timedelta(days=1)

        return daylist

    @staticmethod
    def monthListByMonth(begin, end):
        """
        get the date of every day between the given 'begin' month and 'end' month

        :param begin: '2008-01'
        :param end: '2009-07'
        :return:
        """

        begin = [int(x) for x in begin.split('-')]
        end = [int(x) for x in end.split('-')]
        daylist = []
        begin_date = datetime.date(begin[0], begin[1], 1)
        end_date = datetime.date(end[0], end[1], calendar.monthrange(end[0], end[1])[1])

        while begin_date <= end_date:
            date_str = begin_date
            daylist.append(date_str)
            begin_date += datetime.timedelta(days=1)

        monthlist = []
        for day in daylist:
            if day.day == 1:
                monthlist.append(day)

        return monthlist

    @staticmethod
    def dayListByDay(begin, end):
        """
        get the date of every day between the given 'begin' day and 'end' day

        :param begin: year, month, day. '2009-01-01'
        :param end: year,month,day. '2010-01-01'
        :return:
        """

        daylist = []
        begin_date = datetime.datetime.strptime(begin, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d")

        while begin_date <= end_date:
            date_str = begin_date
            daylist.append(date_str)
            begin_date += datetime.timedelta(days=1)

        return daylist

    @staticmethod
    def un_gz(file_name):

        # aquire the filename and remove the postfix
        f_name = file_name.replace(".gz", "")
        # start uncompress
        g_file = gzip.GzipFile(file_name)
        # read uncompressed files and write down a copy without postfix
        open(f_name, "wb+").write(g_file.read())
        g_file.close()

    @staticmethod
    def year_fraction(date):
        start = datetime.date(date.year, 1, 1).toordinal()
        year_length = datetime.date(date.year + 1, 1, 1).toordinal() - start
        return date.year + float(date.toordinal() - start) / year_length


if __name__ == '__main__':
    CS = np.array([23, 56, 78, 90, 32, 34])
    GeoMathKit.CS_1dTo2d(CS)
