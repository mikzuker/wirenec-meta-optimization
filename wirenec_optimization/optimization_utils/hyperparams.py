import numpy as np


def freq_maker(general_freq: float, band_width: float):
    frequencies = []
    number_freq = 4
    half_freq_1 = np.arange(general_freq, general_freq + 3*band_width/4, band_width/number_freq)
    half_freq_2 = np.arange(general_freq - band_width / 2, general_freq, band_width/number_freq)
    half_freq_2 = list(half_freq_2)
    half_freq_1 = list(half_freq_1)
    frequencies = half_freq_2 + half_freq_1
    frequencies = [round(num) for num in frequencies]
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
    'frequencies': tuple(freq_maker(10000, scattering_hyperparams['band_width'])),
    'scattering_angle': tuple([180])
}

object_hyperparams = {
    'type': 'wire',
    'obj_length': 1 * 1e-2,
    'dist_from_obj_to_surf': 3 * 1e-2,
    'wire_radius': 0.5 * 1e-4
}