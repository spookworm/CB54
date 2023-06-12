def epsilon0():
    # Vacuum Permittivity (F/m)
    return 8.854187817e-12


def mu0():
    # Vacuum Permeability
    import math
    return 4.0 * math.pi * 1.0e-7


def realmax():
    # Equivalent of realmax in MATLAB. Used to find the minimum resolution required by the materials present in the scene.
    import sys
    return sys.float_info.max


def input_carrier_frequency():
    return 60e6


def input_disc_per_lambda():
    return 10


def angular_frequency(input_carrier_frequency):
    import math
    return 2.0 * math.pi * input_carrier_frequency

