import numpy as np
import matplotlib.pyplot as plt

from cmaes import CMA
from omegaconf import DictConfig
from tqdm import tqdm
from scipy.stats import linregress
from typing import Tuple, Optional, Any

from wirenec.scattering import get_scattering_in_frequency_range
from wirenec_optimization.parametrization.base_parametrization import (
    BaseStructureParametrization,
)

from wirenec.geometry import Wire, Geometry


from wirenec_optimization.parametrization.sample_objects import make_wire, make_srr, make_ssrr, \
    SphereParametrization, SpatialParametrization


def create_wire_bundle_geometry(lengths, tau):
    m, n = lengths.shape

    wires = []
    x0, y0 = -(m - 1) * tau / 2, -(n - 1) * tau / 2
    for i in range(m):
        for j in range(n):
            x, y = x0 + i * tau, y0 + j * tau
            p1, p2 = np.array([x, y, -lengths[i, j] / 2]), np.array(
                [x, y, lengths[i, j] / 2]
            )
            wires.append(Wire(p1, p2))
    return Geometry(wires)


def get_reference_object(object_params: DictConfig) -> Geometry:
    if object_params.type == "wire":
        return make_wire(
            len_obj=object_params.obj_length, height=object_params.dist_from_obj_to_surf,
            wire_radius=object_params.wire_radius
        )
    elif object_params.type == "srr":
        return make_srr(size_ratio=object_params.size_ratio, orientation=object_params.orientation,
                        height=object_params.dist_from_obj_to_surf)
    elif object_params.type == "ssrr":
        return make_ssrr(size_ratio=object_params.size_ratio, orientation=object_params.orientation,
                         height=object_params.dist_from_obj_to_surf, wire_radius=object_params.wire_radius)
    elif object_params.type == 'sphere':
        sphere = SphereParametrization()
        g = sphere.get_geometry(size_ratio=object_params.size_ratio, orientation=object_params.orientation,
                                phi_segments=object_params.config.phi_segments,
                                theta_segments=object_params.config.theta_segments)
        g.translate((0, 0, object_params.dist_from_obj_to_surf))
        return g
    elif object_params.type == "spatial":
        g = SpatialParametrization(**object_params.config).get_random_geometry(seed=object_params.seed)
        g.translate((0, 0, object_params.dist_from_obj_to_surf))
        return g
        # return make_sphere(size_ratio=object_params.size_ratio, orientation=object_params.orientation,
        #                    height=object_params.dist_from_obj_to_surf, phi_segments=object_params.config.phi_segments,
        #                    theta_segments=object_params.config.theta_segments)
    else:
        raise Exception("Unknown object type")


def objective_function(
    parametrization: BaseStructureParametrization,
    params: np.ndarray,
    freq: [list, tuple, np.ndarray] = tuple([9000, 1000]),
    geometry: bool = False,
    scattering_angle: tuple = (90,),
    scattering_theta_angle: int = 180,
    theta: int = 180,
    phi: int = 90,
    eta: int = 0,
    object_params: Optional[DictConfig] = None,
):
    g = parametrization.get_geometry(params=params)

    if object_params is not None:
         unmoving_g = get_reference_object(object_params)
         g.wires.extend(unmoving_g.wires)

    scat_on_freq = []
    if not geometry:
        for angle in scattering_angle:
            scattering, _ = get_scattering_in_frequency_range(
                g,
                freq,
                theta,
                phi,
                eta,
                angle,
                scattering_theta_angle=scattering_theta_angle
            )
            scat_on_freq.append(scattering)
        # return np.mean(scat_on_freq)
        return np.max(scat_on_freq)
    else:
        return g


def check_convergence(
    progress, num_for_progress: int = 100, slope_for_progress: float = 1e-8
):
    if len(progress) > num_for_progress:
        slope1 = linregress(
            range(len(progress[-num_for_progress:])), progress[-num_for_progress:]
        ).slope
        slope2 = linregress(range(len(progress[-3:])), progress[-3:]).slope

        if abs(slope1) <= slope_for_progress and abs(slope2) <= slope_for_progress:
            print("Minimum slope converged")
            return True

    return False


def cma_optimizer(
    structure_parametrization: BaseStructureParametrization,
    iterations: int = 200,
    seed: int = 48,
    frequencies: Tuple = tuple([9_000]),
    plot_progress: bool = False,
    scattering_angle: tuple = (90,),
    scattering_theta_angle: int = 180,
    population_size_factor: float = 1,
    config: Optional[DictConfig] = None,
    **kwargs: Any
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
        population_size=int(len(bounds) * population_size_factor),
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
                structure_parametrization,
                params,
                freq=frequencies,
                scattering_angle=scattering_angle,
                scattering_theta_angle=scattering_theta_angle,
                phi=config.scattering_hyperparams.phi,
                eta=config.scattering_hyperparams.eta,
                object_params=config.object_hyperparams,
            )
            values.append(value)
            if value < max_value:
                max_value = value
                max_params = params
                cnt += 1

            solutions.append((params, value))

        progress.append(np.around(np.mean(values), 15))
        if check_convergence(progress):
            break

        pbar.set_description(
            "Processed %s generation\t max %s mean %s"
            % (generation, np.around(max_value, 15), np.around(np.mean(values), 15))
        )

        optimizer.tell(solutions)

    if plot_progress:
        plt.plot(progress, marker=".", linestyle=":")
        plt.show()

    results = {
        "params": max_params,
        "optimized_value": -max_value,
        "progress": progress,
    }
    return results
