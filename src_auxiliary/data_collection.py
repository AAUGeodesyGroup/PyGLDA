#!/usr/bin/env python
import cdsapi

# c = cdsapi.Client()
# c.retrieve("reanalysis-era5-pressure-levels",
# {
# "variable": "temperature",
# "pressure_level": "1000",
# "product_type": "reanalysis",
# "year": "2008",
# "month": "01",
# "day": "01",
# "time": "12:00",
# "format": "grib"
# }, "download.grib")

import sys

sys.path.append('../')

import calendar
import json
import cdsapi
from pathlib import Path
from src_hydro.GeoMathKit import GeoMathKit
import requests


class ERA5_for_W3:

    def __init__(self):

        '''download path'''
        self.__path = "../data/ERA5/"
        self.__ERA_parameters = {}
        self.__daylist = None
        self.__server = cdsapi.Client()

        self.__TimeEpoch = ['00:00']
        self.__par = self.__defaultPar()
        # TimeEpoch = [
        #     '00:00', '01:00', '02:00',
        #     '03:00', '04:00', '05:00',
        #     '06:00', '07:00', '08:00',
        #     '09:00', '10:00', '11:00',
        #     '12:00', '13:00', '14:00',
        #     '15:00', '16:00', '17:00',
        #     '18:00', '19:00', '20:00',
        #     '21:00', '22:00', '23:00',
        # ]

    def setDate(self, begin='2000-01', end='2001-01'):
        """
        Manually set the date of data to be downloaded.
        Notice: this func has to be used after 'configure', otherwise the date will be automatically selected
        from the configuration file.
        :param begin:
        :param end:
        :return:
        """
        str1, str2 = begin.split('-'), end.split('-')

        begin = [int(t) for t in str1]
        end = [int(t) for t in str2]
        self.__daylist = GeoMathKit.dayListByMonth(begin, end)
        return self

    def setPath(self, output_path='/media/user/Backup Plus/ERA5/W3'):

        assert Path(output_path).is_dir()
        self.__path = Path(output_path)
        return self

    def getDataByMonth(self):
        """
        for PSFC
        :param kind:
        :return:
        """
        begin = None
        for day in self.__daylist:
            if day.day == 1:
                '''record the first day of the month'''
                begin = day
            if day.day == calendar.monthrange(begin.year, begin.month)[1]:
                '''record the last day of the month and start downloading'''
                par = self.__par
                par['details']['date'] = '%04d%02d%02d' % (begin.year, begin.month, begin.day) + "/" \
                                         + '%04d%02d%02d' % (day.year, day.month, day.day)
                par['target'] = str(self.__path / ("W3_ERA5_daily_%04d%02d.grib" % (day.year, day.month)))

                self.__server.retrieve(par['class'], par['details'], par['target'])
                pass

            continue
        pass

    def __defaultPar(self):

        par = {
            'class': 'reanalysis-era5-land',
            'details': {
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                             '2m_temperature', 'snowfall', 'surface_pressure', 'surface_solar_radiation_downwards',
                             'surface_thermal_radiation_downwards', 'total_precipitation'],
                'date': "2018-01-01",
                'time': '00:00',
                'format': 'grib',  # 'netcdf'
                # 'format': 'netcdf'
                # 'grid': '0.05/0.05'
            },
            'target': 'download.grib'
        }

        return par


class ERA5_for_W3RA:

    def __init__(self):
        self.__path = None
        pass

    def setPath(self, output_path='/media/user/Backup Plus/ERA5/W3'):

        assert Path(output_path).is_dir()
        self.__path = Path(output_path)
        return self

    def run(self):
        # CDS API script to use CDS service to retrieve daily ERA5* variables and iterate over
        # all months in the specified years.

        # Requires:
        # 1) the CDS API to be installed and working on your system
        # 2) You have agreed to the ERA5 Licence (via the CDS web page)
        # 3) Selection of required variable, daily statistic, etc

        # Output:
        # 1) separate netCDF file for chosen daily statistic/variable for each month

        c = cdsapi.Client(timeout=300)

        # Uncomment years as required
        # years = ['2006']
        years = [
            # '1979'
            #           ,'1980', '1981',
            #            '1982', '1983', '1984',
            #            '1985', '1986', '1987',
            #            '1988', '1989', '1990',
            #            '1991', '1992', '1993',
            # '1994', '1995', '1996',
            # '1997', '1998', '1999',
            # '2000', '2001',
            # '2002','2003', '2004', '2005','2006',
            '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021', '2022', '2023'
        ]

        # Retrieve all months for a given year.
        # months = ['11', '12']
        months = ['01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12']

        # For valid keywords, see Table 2 of:
        # https://datastore.copernicus-climate.eu/documents/app-c3s-daily-era5-statistics/C3S_Application-Documentation_ERA5-daily-statistics-v2.pdf

        # select your variable; name must be a valid ERA5 CDS API name.
        var = "2m_temperature"

        # Select the required statistic, valid names given in link above
        stat1 = "daily_mean"
        stat2 = "daily_maximum"
        stat3 = "daily_minimum"
        stat = stat1

        # Loop over years and months

        for yr in years:
            for mn in months:
                result = c.service(
                    "tool.toolbox.orchestrator.workflow",
                    params={
                        "realm": "user-apps",
                        "project": "app-c3s-daily-era5-statistics",
                        "version": "master",
                        "kwargs": {
                            "dataset": "reanalysis-era5-land",
                            "product_type": "reanalysis",
                            "variable": var,
                            "statistic": stat,
                            "year": yr,
                            "month": mn,
                            "time_zone": "UTC+00:0",
                            "frequency": "1-hourly",
                            "grid": "0.1/0.1"
                            #
                            # Users can change the output grid resolution and selected area
                            #
                            #                "grid": "1.0/1.0",
                            #                "area":{"lat": [10, 60], "lon": [65, 140]}

                        },
                        "workflow_name": "application"
                    })

                # set name of output file for each month (statistic, variable, year, month

                file_name = self.__path / (stat + "_" + var + "_" + yr + "_" + mn + ".nc")

                location = result[0]['location']
                res = requests.get(location, stream=True)
                print("Writing data to " + str(file_name))
                with open(file_name, 'wb') as fh:
                    for r in res.iter_content(chunk_size=1024):
                        fh.write(r)
                fh.close()


def demo1():
    ERA5_download = ERA5_for_W3().setPath().setDate(begin='1995-01', end='1999-12')
    ERA5_download.getDataByMonth()
    pass


def demo2():
    ERA5_download = ERA5_for_W3RA().setPath('/media/user/My Book/Fan/ERA5/DailyMeanTemp')
    ERA5_download.run()
    pass


if __name__ == '__main__':
    # demo1()
    demo2()
