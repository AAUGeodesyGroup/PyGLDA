from enum import Enum


class HydroModel(Enum):
    w3 = 0
    w3ra_v0 = 1
    w3ra_v1 = 2


class FusionMethod(Enum):
    EnKF_v0 = 0
    EnKF_v1 = 1
    EnKF_v2 = 2




