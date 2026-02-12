from enum import Enum


class data_var(Enum):
    GroundWater = 0
    TotalWater = 1
    SoilWater = 2
    DeepSoilWater = 3
    SurfaceWater = 4
    Discharge = 5
    TopSoil = 6


class data_dim(Enum):
    one_dimension = 1
    two_dimension = 2

