import sys
sys.path.append('../')


def demo1():
    from src_auxiliary.data_collection import ERA5_for_W3
    '''This demo is to show the update of necessary meteorological forcing field to drive W3RA model.'''
    '''ERA-5 land data of 0.1 degree will be downloaded'''
    '''More information about downloading ERA5-land refers to https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5'''

    '''set the path where ERA5 will be downloaded'''
    '''set the time period when ERA will be downloaded'''
    ERA5_download = ERA5_for_W3().setPath('/media/user/My Book/Fan/test').setDate(begin='2023-06', end='2024-05')
    '''start downloading data automatically on monthly basis'''
    ERA5_download.getDataByMonth()
    pass


if __name__ == '__main__':
    demo1()
