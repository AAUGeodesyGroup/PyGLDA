"""
function [state,out]=W3RA_timestep_model(in,state,par)

% This cript contains the World-Wide Water Resources Assessment (W3RA) time step model
%
% The model is modified from the Australian Water Resources Assessment
% Landscape (AWRA-L) model version 0.5
%
% W3RA is documented in
% van Dijk et al. (2013), Water Resour. Res., 49, 2729�2746, doi:10.1002/wrcr.20251
% URL: http://onlinelibrary.wiley.com/doi/10.1002/wrcr.20251/abstract
%
% More comprehensive documentation of AWRA-L version 0.5 can be found in:
%
% Van Dijk, A.I.J.M. (2010) The Australian water resources assessment system
% (version 0.5), 3.0.5.Technical description of the landscape hydrology model
% (AWRA-L). WIRADA Technical Report, CSIRO Water for a Healthy Country
% Flagship, Canberra.
% URL: http://www.clw.csiro.au/publications/waterforahealthycountry/2010/wfhc-aus-water-resources-assessment-system.pdf
%
% The section references below refer to the sections in the AWRA-L report.
% Changes compared to that code are indicated, e.g. by commenting out
% redundant code.
%
% Further question please contact albert.vandijk@anu.edu.au
"""

"""
Modified from matlab to python by Fan Yang (fany@plan.aau.dk) in Aalborg University, Denmark. 2024.01.25
"""

import numpy as np

from src_hydro.config_settings import config_settings
from src_hydro.config_parameters import config_parameters
from src_hydro.EnumType import states_var, output_var, forcing
from src_hydro.GeoMathKit import GeoMathKit
from src_hydro.ext_adapter import ext_adapter
from src_hydro.snow_submodel import snow_submodel


class timestep_model:
    """This version is adapted by AAU: removing all src_DA-relevant matters """

    def __init__(self, par: config_parameters, settings: config_settings, ext: ext_adapter):
        self.__output = {}

        self.__par = par

        self.__settings = settings

        self.__ext = ext

        pass

    def updatePar(self, par):
        self.__par = par
        pass

    def updateState(self, state: dict):
        """
        update the state on daily basis
        :param ext: a dict that contains input data
        :param state: a dict that contains state in previous step
        :return: new state
        """
        par = self.__par
        settings = self.__settings

        ext = self.__ext

        '''% ASSIGN STATE VARIABLES'''
        S0 = state[states_var.S0]
        Ss = state[states_var.Ss]
        Sd = state[states_var.Sd]
        Sg = state[states_var.Sg]
        Sr = state[states_var.Sr]
        Mleaf = state[states_var.Mleaf]
        # % LAI        =   state.LAI
        # % EVI         =   state.EVI
        FreeWater = state[states_var.FreeWater]  # % added variables, part of snow model
        DrySnow = state[states_var.DrySnow]  # % added variables, part of snow model

        '''% ASSIGN INPUT VARIABLES'''
        Pg = ext.Pg  # % prcp
        Rg = ext.Rg  # % rad
        Ta = ext.Ta  # % effective average air temp
        T24 = ext.T24
        pe = ext.pe
        pair = ext.pair
        u2 = ext.u2  # % effective wind speed at 2m
        fday = ext.fday  # % is now calculated rather than assumed
        ns_alb = ext.ns_alb

        '''% ASSIGN PARAMETERS'''
        Nhru = par.Nhru
        Fhru = par.Fhru
        SLA = par.SLA  # % (5.3)
        LAIref = par.LAIref  # % (5.3)
        Sgref = par.Sgref
        S0FC = par.S0FC  # % accessible deep soil water storage at field capacity
        SsFC = par.SsFC
        SdFC = par.SdFC
        # % fday        =   par.fday          % (3.1)
        Vc = par.Vc  # %
        alb_dry = par.alb_dry  # % (3.2)
        alb_wet = par.alb_wet  # % (3.2)
        w0ref_alb = par.w0ref_alb  # % (3.2)
        Gfrac_max = par.Gfrac_max  # % (3.5)
        fvegref_G = par.fvegref_G  # % (3.5)
        hveg = par.hveg  # % (3.7)
        Us0 = par.Us0  # % (2.3)
        Ud0 = par.Ud0  # % (2.3)
        wslimU = par.wslimU
        wdlimU = par.wdlimU
        cGsmax = par.cGsmax
        FsoilEmax = par.FsoilEmax  # % (4.5)
        w0limE = par.w0limE  # % (4.5)
        FwaterE = par.FwaterE  # % (4.7)
        S_sls = par.S_sls  # % (4.2)
        ER_frac_ref = par.ER_frac_ref  # % (4.2)
        InitLoss = par.InitLoss  # % (2.2)
        PrefR = par.PrefR  # % (2.2)
        FdrainFC = par.FdrainFC  # % (2.4)
        beta = par.beta  # % (2.4)
        Fgw_conn = par.Fgw_conn  # % (2.6)
        K_gw = par.K_gw  # % (2.5)
        K_rout = par.K_rout  # % (2.7)
        LAImax = par.LAImax  # % (5.5)
        Tgrow = par.Tgrow  # % (5.4)
        Tsenc = par.Tsenc  # % (5.4)

        '''diagnostic equations'''
        LAI = SLA * Mleaf  # % (5.3)
        # % Mleaf        =   LAI./SLA;    #% (5.3)
        fveg = 1 - np.exp(-LAI / LAIref)  # % (5.3)
        # % Vc          =   max(0,EVI-0.07)./fveg;
        fsoil = 1 - fveg
        w0 = S0 / S0FC  # % (2.1)
        ws = Ss / SsFC  # % (2.1)
        wd = Sd / SdFC  # % (2.1)

        TotSnow = FreeWater + DrySnow
        wSnow = FreeWater / (TotSnow + 1e-5)

        '''% Spatialise catchment fractions'''
        fwater = np.fmin(0.005, 0.007 * Sr ** 0.75)
        Sgfree = np.fmax(Sg, 0)
        fsat = np.fmin(1, np.fmax(np.fmin(0.005, 0.007 * Sr ** 0.75), Sgfree / Sgref))
        fwater = np.tile(fwater, (par.Nhru, 1))
        fsat = np.tile(fsat, (par.Nhru, 1))
        Sghru = np.tile(Sg, (par.Nhru, 1))

        '''CALCULATION OF PET
        # Conversions and coefficients (3.1)'''
        pes = 610.8 * np.exp(17.27 * Ta / (237.3 + Ta))  # saturated vapour pressure
        fRH = pe / pes  # relative air humidity
        cRE = 0.03449 + 4.27e-5 * Ta
        Caero = fday * 0.176 * (1 + Ta / 209.1) * (pair - 0.417 * pe) * (1 - fRH)
        del fRH
        keps = 1.4e-3 * ((Ta / 187) ** 2 + Ta / 107 + 1) * (6.36 * pair + pe) / pes
        Rgeff = Rg / fday  ## this is original
        '''% shortwave radiation balance (3.2)'''
        # %alb_veg     =   0.452.*Vc;
        # %alb_soil    =   alb_wet+(alb_dry-alb_wet).*exp(-w0./w0ref_alb)
        '''new equations for snow albedo'''
        alb_snow = 0.65 - 0.2 * wSnow  # % assumed; ideally some lit research needed
        fsnow = np.fmin(1, 0.05 * TotSnow)  # % assumed; ideally some lit research needed
        # %alb         =   fveg.*alb_veg+(fsoil-fsnow).*alb_soil +fsnow.*alb_snow;
        # %alb         =   albedo;
        alb = (1 - fsnow) * ns_alb + fsnow * alb_snow
        RSn = (1 - alb) * Rgeff
        '''# long wave radiation balance (3.3 to 3.5)'''
        StefBolz = 5.67e-8
        Tkelv = Ta + 273.16
        RLin = (0.65 * (pe / Tkelv) ** 0.14) * StefBolz * Tkelv ** 4  # (3.3)
        RLout = 1 * StefBolz * Tkelv ** 4  # v0.5   # (3.4)
        RLn = RLin - RLout
        # RLn         =   RLn*(24*3600/1e6) # v5 (MJ/m2/d) only
        fGR = Gfrac_max * (1 - np.exp(-fsoil / fvegref_G))  # (3.5)
        # fGR       =0   ## v5
        Rneff = (RSn + RLn) * (1 - fGR)
        '''# Aerodynamic conductance (3.7)'''
        fh = np.log(813 / hveg - 5.45)
        ku2 = 0.305 / (fh * (fh + 2.3))
        ga = ku2 * u2
        '''%  Potential evaporation'''
        kalpha = 1 + Caero * ga / Rneff
        E0 = cRE * (1. / (1 + keps)) * kalpha * Rneff * fday
        E0 = np.fmax(E0, 0)

        '''CALCULATION OF ET FLUXES AND ROOT WATER UPTAKE
        # Root water uptake constraint (4.4)'''
        Usmax = np.fmax(0, Us0 * np.fmin(1, ws / wslimU))
        Udmax = np.fmax(0, Ud0 * np.fmin(1, wd / wdlimU))
        U0max = 0
        Utot = np.fmax(Usmax, np.fmax(Udmax, U0max))

        '''max transpiration (4.3)'''
        Gsmax = cGsmax * Vc
        gs = fveg * Gsmax
        ft = 1 / (1 + (keps / (1 + keps)) * ga / gs)
        Etmax = ft * E0
        '''Actual transpiration (4.1)'''
        Et = np.fmin(Utot, Etmax)
        '''Root water uptake distribution (2.3)'''
        Umax = U0max + Usmax + Udmax
        U0 = np.fmax(0, np.fmin((U0max / Umax) * Et, S0 - 1e-2))
        Us = np.fmax(0, np.fmin((Usmax / Umax) * Et, Ss - 1e-2))
        Ud = np.fmax(0, np.fmin((Udmax / Umax) * Et, Sd - 1e-2))
        Et = U0 + Us + Ud  # to ensure mass balance
        '''Soil evaporation (4.5)'''
        S0 = np.fmax(0, S0 - U0)
        w0 = S0 / S0FC
        fsoilE = FsoilEmax * np.fmin(1, w0 / w0limE)
        Es = np.fmax(0, np.fmin((1 - fsat) * fsoilE * (E0 - Et), S0 - 1e-2))
        '''# Groundwater evaporation (4.6)'''
        Eg = np.fmin((fsat - fwater) * FsoilEmax * (E0 - Et), Sghru)
        '''# Open water evaporation (4.7) # uses Priestley-Taylor'''
        Er = np.fmin(fwater * FwaterE * np.fmax(0, E0 - Et), Sr)
        '''# Rainfall interception evaporation (4.2)'''
        Sveg = S_sls * LAI
        fER = ER_frac_ref * fveg
        '''# # ## end v6 ##'''
        Pwet = -np.log(1 - fER / fveg) * Sveg / fER
        Ei = (Pg < Pwet) * fveg * Pg + (Pg >= Pwet) * (fveg * Pwet + fER * (Pg - Pwet))

        ETtot = Et + Es + Er + Ei  # for output only

        '''HBV snow routine'''
        Pn = Pg - Ei
        FreeWater, DrySnow, InSoil = snow_submodel(Pn, T24, FreeWater, DrySnow)

        '''# CALCULATION OF WATER BALANCES
        # surface water fluxes (2.2)'''
        NetInSoil = np.fmax(0, InSoil - InitLoss)
        Rhof = (1 - fsat) * (NetInSoil / (NetInSoil + PrefR)) * NetInSoil
        Rsof = fsat * NetInSoil
        QR = Rhof + Rsof  # % estimated event surface runoff [mm] (2.3)
        I = InSoil - QR
        '''SOIL WATER BALANCES (2.1 & 2.4)
        % Topsoil water balance (S0)'''
        S0 = S0 + I - Es - U0
        SzFC = S0FC
        Sz = S0
        wz = np.fmax(1e-2, Sz) / SzFC
        fD = (wz > 1) * np.fmax(FdrainFC, 1 - 1. / wz) + (wz <= 1) * FdrainFC * np.exp(beta * (wz - 1))
        Dz = np.fmax(0, np.fmin(fD * Sz, Sz - 1e-2))
        D0 = Dz
        S0 = S0 - D0
        '''Shallow root zone water balance (Ss)'''
        Ss = Ss + D0 - Us
        SzFC = SsFC
        Sz = Ss
        wz = np.fmax(1e-2, Sz) / SzFC
        fD = (wz > 1) * np.fmax(FdrainFC, 1 - 1. / wz) + (wz <= 1) * FdrainFC * np.exp(beta * (wz - 1))
        Dz = np.fmax(0, np.fmin(fD * Sz, Sz - 1e-2))
        Ds = Dz
        Ss = Ss - Ds
        '''Deep root zone water balance (Sd) (2.6)'''
        Sd = Sd + Ds - Ud
        SzFC = SdFC
        Sz = Sd
        wz = np.fmax(1e-2, Sz) / SzFC
        fD = (wz > 1) * np.fmax(FdrainFC, 1 - 1. / wz) + (wz <= 1) * FdrainFC * np.exp(beta * (wz - 1))
        Dz = np.fmax(0, np.fmin(fD * Sz, Sz - 1e-2))  # % drainage from layer z [mm d^-1] (2.5)
        Dd = Dz
        Sd = Sd - Dd
        Y = np.fmin(Fgw_conn * np.fmax(0, wdlimU * SdFC - Sd),
                    Sghru - Eg)  # % capillary rise of groundwater into deeper root zone (2.7)
        # %Y           =   Fgw_conn*np.fmax(0,wdlimU*SdFC-Sd)
        Sd = Sd + Y

        '''# CATCHMENT WATER BALANCE
        # Groundwater store water balance (Sg) (2.5)'''
        NetGf = np.sum(Fhru * (Dd - Eg - Y), 0)
        Sg += NetGf
        Sgfree = np.fmax(Sg, 0)
        Qg = np.fmin(Sgfree, (1 - np.exp(-K_gw)) * Sgfree)  # % groundwate discharge into stream (2.6)
        Sg = Sg - Qg  # % groundwater reservoir storage
        '''# Surface water store water balance (Sr) (2.7)'''
        Sr = Sr + np.sum(Fhru * (QR - Er), 0) + Qg
        Qtot = np.fmin(Sr, (1 - np.exp(-K_rout)) * Sr)
        Sr = Sr - Qtot

        '''# # VEGETATION ADJUSTMENT (5)'''
        fveq = (1 / np.fmax((E0 / Utot) - 1, 1e-3)) * (keps / (1 + keps)) * (ga / Gsmax)  # % (5.5)
        fvmax = 1 - np.exp(-LAImax / LAIref)  # % maximum achievable canopy cover
        fveq = np.fmin(fveq, fvmax)
        dMleaf = -np.log(1 - fveq) * LAIref / SLA - Mleaf  # % equilibrium dry leaf biomass
        Mleafnet = (dMleaf > 0) * (dMleaf / Tgrow) + (
                dMleaf < 0) * dMleaf / Tsenc  # % net biomass change of living leaves [kg/m^2] (5.2)
        Mleaf = Mleaf + Mleafnet

        '''Updating diagnostics'''
        LAI = SLA * Mleaf  # % (5.3) SLA=specific leaf area
        fveg = 1 - np.exp(-LAI / LAIref)  # % (5.3) canopy fractional cover
        fsoil = 1 - fveg
        w0 = S0 / S0FC  # % (2.2)
        ws = Ss / SsFC  # % (2.2)
        wd = Sd / SdFC  # % (2.2)

        '''ASSIGN OUTPUT VARIABLES'''
        out = {}
        '''fluxes'''
        out['Pg'] = np.sum(Fhru * Pg, 0)
        out['E0'] = np.sum(Fhru * E0, 0)
        out['Ee'] = np.sum(Fhru * (Es + Eg + Er + Ei), 0)
        # %   out['Eg']      =   sum(Fhru*Eg)
        out['Et'] = np.sum(Fhru * Et, 0)
        out['Ei'] = np.sum(Fhru * Ei, 0)
        out['Etot'] = out['Et'] + out['Ee']
        out['Qtot'] = Qtot
        # %   out['Qg']      =   Qg;
        # %   out['QR']      =   sum(Fhru.*QR)
        out['gwflux'] = NetGf
        out['D'] = np.sum(Fhru * Dd, 0)
        '''% HRU specific drainage'''
        # %   out['D1']    = Dd(1,:)
        # %   out['D2']    = Dd(2,:)
        # %   out['Et1']   = Et(1,:)
        # %   out['Et2']   = Et(2,:)
        ETtot = Es + Eg + Er + Ei + Et
        out['ET1'] = ETtot[0, :]
        out['ET2'] = ETtot[1, :]
        '''states'''
        out['S0'] = np.sum(Fhru * S0, 0)
        out['Ss'] = np.sum(Fhru * Ss, 0)
        out['Sd'] = np.sum(Fhru * Sd, 0)
        out['Sg'] = Sg
        out['Sr'] = Sr
        out['Ssnow'] = np.sum(Fhru * (FreeWater + DrySnow), 0)
        out['Stot'] = out['S0'] + out['Ss'] + out['Sd'] + Sg + Sr + out['Ssnow'] + np.sum(
            Fhru * Mleaf * 4, 0)  # % assume 80% water  in biomass
        out['Mleaf'] = np.sum(Fhru * Mleaf, 0)
        out['LAI'] = np.sum(Fhru * LAI, 0)
        out['fveg'] = np.sum(Fhru * fveg, 0)
        # %   out['fveq']    =   sum(Fhru * fveq)
        '''satellite equivalent'''
        out['albedo'] = np.sum(Fhru * alb, 0)
        out['EVI'] = np.sum(Fhru * (Vc * fveg + 0.07), 0)  # % assume 0.07 is EVI for bare soil
        out['fsat'] = np.sum(Fhru * fsat, 0)  # % fraction of area covered by saturated soil
        out['wunsat'] = np.sum(Fhru * w0, 0)

        # Update states'''
        state[states_var.S0] = S0
        state[states_var.Ss] = Ss
        state[states_var.Sd] = Sd
        state[states_var.Sr] = Sr
        state[states_var.Sg] = Sg
        state[states_var.Mleaf] = Mleaf
        state[states_var.LAI] = LAI
        state[states_var.FreeWater] = FreeWater
        state[states_var.DrySnow] = DrySnow

        '''## ASSIGN OUTPUT VARIABLES
        # comment out what you don't need.'''

        return out
