import numpy as np


def freq_maker(general_freq: float, band_width: float):
    number_freq = 5
    frequencies = np.linspace(general_freq - band_width / 2, general_freq + band_width / 2, number_freq, endpoint=True)
    return frequencies


parametrization_hyperparams = {
    'matrix_size': (3, 3),
    'layers_num': 1,
    'tau': 20 * 1e-3,
    'delta': 10 * 1e-3,
    'asymmetry_factor': None
}

scattering_hyperparams = {
    'theta': 180,
    'eta': 0,
    'phi': 90,
    'band_width': 1000
}

optimization_hyperparams = {
    'iterations': 1,
    'seed': 42,
    'frequencies': tuple(freq_maker(7000, scattering_hyperparams['band_width'])),
    'scattering_angle': tuple([180])
}

object_hyperparams = {
    'type': 'wire',
    'obj_length': 1 * 1e-2,
    'dist_from_obj_to_surf': 3 * 1e-2,
    'wire_radius': 0.5 * 1e-4
}
