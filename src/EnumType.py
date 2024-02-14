from enum import Enum


class init_mode(Enum):
    warm = 0
    cold = 1
    resume = 2


class forcing(Enum):
    Pg = 0
    Rg = 1
    Ta = 2
    pe = 3
    T24 = 4
    u1 = 5
    pair = 6
    LWdown = 7
    SnowFrac = 8
    fday = 9
    CO2 = 10


class perturbation_choice_forcing_field(Enum):
    triangle_distribution = 0
    normal_distribution = 1


class forcingSource(Enum):
    ERA5 = 0
    E20WFDEI = 1


class states_var(Enum):
    S0 = 0
    Ss = 1
    Sd = 2
    Sr = 3
    Sg = 4
    LAI = 5
    Mleaf = 6
    DrySnow = 7
    EVI = 8
    FreeWater = 9


class output_var(Enum):
    ETtot = 0
    Qtot = 1
    Qnet = 2
    E0 = 3
    Ei = 4
    Et = 5
    Es = 6
    Er = 7
    RSn = 8
    RLn = 9
    LE = 10
    H = 11
    albedo = 12
    LAI = 13
    Ssnow = 14
    z_g = 15
    S0 = 16
    Sroot = 17
    Ssoil = 18
    Sg = 19
    Ts = 20
    fsnow = 21
    Sr = 22
