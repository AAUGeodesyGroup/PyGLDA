import numpy as np


def snow_submodel(Precipitation, Temperature, FreeWater, DrySnow):
    """    % derived from HBV-96 shared by Jaap Schellekens (Deltares) in May 2011
    % original in PCraster, adapted to Matlab by Albert van Dijk"""

    Mat = np.ones(np.shape(Precipitation))
    '''Snow routine parameters
    % parameters'''
    Cfmax = np.array([0.6, 1])[None,:].T * 3.75653 * Mat  # % meltconstant in temperature-index - 0.6 correction is for forests
    TT = -1.41934 * Mat  # % critical temperature for snowmelt and refreezing
    TTI = 1.00000 * Mat  # % defines interval in which precipitation falls as rainfall and snowfall
    CFR = 0.05000 * Mat  # % refreezing efficiency constant in refreezing of freewater in snow
    WHC = 0.10000 * Mat  # % fraction of Snowvolume that can store water

    '''Partitioning into fractions rain and snow'''
    RainFrac = np.fmax(0, np.fmin((Temperature - (TT - TTI / 2)) / TTI,
                                  1))  # %fraction of precipitation which falls as rain
    SnowFrac = 1 - RainFrac;  # %fraction of precipitation which falls as snow
    '''Snowfall/melt calculations'''
    SnowFall = SnowFrac * Precipitation  # %snowfall depth
    RainFall = RainFrac * Precipitation  # %rainfall depth
    PotSnowMelt = Cfmax * np.fmax(0, Temperature - TT)  # %Potential snow melt, based on temperature
    PotRefreezing = Cfmax * CFR * np.fmax(TT - Temperature, 0)  # %Potential refreezing, based on temperature
    Refreezing = np.fmin(PotRefreezing, FreeWater)  # %actual refreezing
    SnowMelt = np.fmin(PotSnowMelt, DrySnow)  # %actual snow melt
    DrySnow = DrySnow + SnowFall + Refreezing - SnowMelt  # %dry snow content
    FreeWater = FreeWater - Refreezing  # %free water content in snow
    MaxFreeWater = DrySnow * WHC
    FreeWater = FreeWater + SnowMelt + RainFall
    InSoil = np.fmax(FreeWater - MaxFreeWater, 0)  # %abundant water in snow pack which goes into soil
    FreeWater = FreeWater - InSoil

    return FreeWater, DrySnow, InSoil
