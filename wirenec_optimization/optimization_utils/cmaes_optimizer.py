import numpy as np
import matplotlib.pyplot as plt

from cmaes import CMA
from tqdm import tqdm
from scipy.stats import linregress
from typing import Tuple

from wirenec.scattering import get_scattering_in_frequency_range
from wirenec_optimization.parametrization.base_parametrization import BaseStructureParametrization

from wirenec.geometry import Wire, Geometry

from wirenec_optimization.optimization_utils.hyperparams import parametrization_hyperparams, optimization_hyperparams, \
    scattering_hyperparams, object_hyperparams

def create_wire_bundle_geometry(lengths, tau):
    m, n = lengths.shape

    wires = []
    x0, y0 = -(m - 1) * tau / 2, -(n - 1) * tau / 2
    for i in range(m):
        for j in range(n):
            x, y = x0 + i * tau, y0 + j * tau
            p1, p2 = np.array([x, y, -lengths[i, j]/2]), np.array([x, y, lengths[i, j]/2])
            wires.append(Wire(p1, p2))
    return Geometry(wires)

def objective_function(
        parametrization: BaseStructureParametrization,
        params: np.ndarray,
        freq: [list, tuple, np.ndarray] = tuple([9000, 1000]),
        geometry: bool = False, scattering_angle: tuple = (90)
):
    length = object_hyperparams['obj_length']
    wire_radius = object_hyperparams['wire_radius']
    height = object_hyperparams['dist_from_obj_to_surf']
    unmov_g = Geometry([Wire((0., -length, height),
                       (0., length, height),
                        wire_radius)])
    g = parametrization.get_geometry(params=params)
    g.wires.extend(unmov_g.wires)
    scat_on_freq = []
    if not geometry:
        for angle in scattering_angle:
            scattering, _ = get_scattering_in_frequency_range(unmov_g, optimization_hyperparams['frequencies'], angle,
                                                              scattering_hyperparams['phi'], scattering_hyperparams['eta'], angle)
            scat_on_freq.append(scattering)
        return np.mean(scat_on_freq)
    else:
        return g


def check_convergence(progress, num_for_progress: int = 100, slope_for_progress: float = 1e-8):
    if len(progress) > num_for_progress:
        slope1 = linregress(range(len(progress[-num_for_progress:])), progress[-num_for_progress:]).slope
        slope2 = linregress(range(len(progress[-3:])), progress[-3:]).slope

        if abs(slope1) <= slope_for_progress and abs(slope2) <= slope_for_progress:
            print('Minimum slope converged')
            return True

    return False


def cma_optimizer(
        structure_parametrization: BaseStructureParametrization,
        iterations: int = 200,
        seed: int = 48,
        frequencies: Tuple = tuple([9_000]),
        plot_progress: bool = False,
        scattering_angle: float = 90,
        population_size_factor: float = 1,
):
    np.random.seed(seed)
    bounds = structure_parametrization.bounds
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
    mean = lower_bounds + (np.random.rand(len(bounds)) * (upper_bounds - lower_bounds))
    sigma = 2 * (upper_bounds[0] - lower_bounds[0]) / 3

    optimizer = CMA(
        mean=mean,
        sigma=sigma,
        bounds=bounds,
        seed=seed,
        population_size=int(len(bounds) * population_size_factor)
    )

    cnt = 0
    max_value, max_params = 100000, []

    pbar = tqdm(range(iterations))
    progress = []

    for generation in pbar:
        solutions = []
        values = []
        for _ in range(optimizer.population_size):
            params = optimizer.ask()

            value = objective_function(
                structure_parametrization, params,
                freq=frequencies, scattering_angle=scattering_angle
            )
            values.append(value)
            if value < max_value:
                max_value = value
                max_params = params
                cnt += 1
            # if abs(value) < abs(max_value):
            #     max_value = abs(value)
            #     max_params = params
            #     cnt += 1

            solutions.append((params, value))

        progress.append(np.around(np.mean(values), 15))
        if check_convergence(progress):
            break

        pbar.set_description(
            "Processed %s generation\t max %s mean %s" % (
                generation, np.around(max_value, 15),
                np.around(np.mean(values), 15))
        )

        optimizer.tell(solutions)

    if plot_progress:
        plt.plot(progress, marker='.', linestyle=':')
        plt.show()

    results = {
        'params': max_params,
        'optimized_value': -max_value,
        'progress': progress,
    }
    return results
