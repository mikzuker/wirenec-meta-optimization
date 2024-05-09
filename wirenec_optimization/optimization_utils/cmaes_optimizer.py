from typing import Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
from cmaes import CMA
from omegaconf import DictConfig
from scipy.stats import linregress
from tqdm import tqdm
from wirenec.geometry import Wire, Geometry
from wirenec.scattering import get_scattering_in_frequency_range

from wirenec_optimization.parametrization.base_parametrization import (
    BaseObjectParametrization,
)
from wirenec_optimization.parametrization.base_parametrization import (
    BaseStructureParametrization,
)
from wirenec_optimization.parametrization.sample_objects import make_wire, SRRParametrization, SSRRParametrization
from wirenec_optimization.parametrization.spatial_parametrization import SpatialParametrization


def create_wire_bundle_geometry(lengths, tau):
    m, n = lengths.shape

    wires = []
    x0, y0 = -(m - 1) * tau / 2, -(n - 1) * tau / 2
    for i in range(m):
        for j in range(n):
            x, y = x0 + i * tau, y0 + j * tau
            p1, p2 = np.array([x, y, -lengths[i, j] / 2]), np.array([x, y, lengths[i, j] / 2])
            wires.append(Wire(p1, p2))
    return Geometry(wires)


class SphereParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 25 * 1e-3, min_size: float = 25 * 1e-3):
        super().__init__("Sphere", max_size, min_size)

    @staticmethod
    def __parametrize_sphere__(r, phi_lines, theta_lines):
        phi = np.linspace(0, np.pi, phi_lines)
        theta = np.linspace(0, 2 * np.pi, theta_lines)
        segments = []

        # Go through each line of constant latitude (phi)
        for i in range(phi_lines):
            for j in range(theta_lines - 1):
                # Convert to cartesian coordinates
                x1, y1, z1 = (
                    r * np.sin(phi[i]) * np.cos(theta[j]),
                    r * np.sin(phi[i]) * np.sin(theta[j]),
                    r * np.cos(phi[i]),
                )
                x2, y2, z2 = (
                    r * np.sin(phi[i]) * np.cos(theta[j + 1]),
                    r * np.sin(phi[i]) * np.sin(theta[j + 1]),
                    r * np.cos(phi[i]),
                )

                # Store the segment
                segment = [(x1, y1, z1), (x2, y2, z2)]
                segments.append(segment)

        # Go through each line of constant longitude (theta)
        for i in range(theta_lines):
            for j in range(phi_lines - 1):
                # Convert to cartesian coordinates
                x1, y1, z1 = (
                    r * np.sin(phi[j]) * np.cos(theta[i]),
                    r * np.sin(phi[j]) * np.sin(theta[i]),
                    r * np.cos(phi[j]),
                )
                x2, y2, z2 = (
                    r * np.sin(phi[j + 1]) * np.cos(theta[i]),
                    r * np.sin(phi[j + 1]) * np.sin(theta[i]),
                    r * np.cos(phi[j + 1]),
                )

                # Store the segment
                segment = [(x1, y1, z1), (x2, y2, z2)]
                segments.append(segment)

        segments = segments[: len(segments) - phi_lines + 1]
        return segments

    def get_geometry(
        self,
        size_ratio,
        orientation,
        wire_radius: float = 0.5 * 1e-4,
        phi_segments: int = 5,
        theta_segments: int = 5,
    ):
        radius = self.min_size + (self.max_size - self.min_size) * size_ratio

        segments = self.__parametrize_sphere__(radius, phi_segments, theta_segments)

        g = Geometry([Wire(*np.around(s, 10), radius=wire_radius) for s in segments])
        g.rotate(*orientation)
        return g


def get_reference_object(object_params: DictConfig) -> Geometry:
    if object_params.type == "wire":
        return make_wire(len_obj=object_params.obj_length, height=object_params.dist_from_obj_to_surf)
    elif object_params.type == "spatial":
        g = SpatialParametrization(**object_params.config).get_random_geometry(seed=object_params.seed)
        g.translate((0, 0, object_params.dist_from_obj_to_surf))
        return g
    elif object_params.type == "sphere":
        g = SphereParametrization(min_size=object_params.radius).get_geometry(0, (0, 0, 0), **object_params.config)
        g.translate((0, 0, object_params.dist_from_obj_to_surf))
        return g
    elif object_params.type == "srr":
        g = SRRParametrization().get_geometry(object_params.size_ratio, (0, 0, 0))
        g.translate((0, 0, object_params.dist_from_obj_to_surf))
        return g
    elif object_params.type == "ssr":
        g = SSRRParametrization().get_geometry(object_params.size_ratio, (0, 0, 0), num=1)
        g.translate((0, 0, object_params.dist_from_obj_to_surf))
        return g
    else:
        raise Exception("Unknown object type")


def objective_function(
    parametrization: BaseStructureParametrization,
    params: np.ndarray,
    freq: [list, tuple, np.ndarray] = tuple([9000, 1000]),
    geometry: bool = False,
    scattering_angle: tuple = (90,),
    phi: int = 90,
    eta: int = 90,
    object_params: Optional[DictConfig] = None,
):
    try:
        g = parametrization.get_geometry(params=params)
        if object_params is not None:
            unmoving_g = get_reference_object(object_params)
            g.wires.extend(unmoving_g.wires)
    except:
        unmoving_g = get_reference_object(object_params)
        g = unmoving_g

    finally:
        scat_on_freq = []
        if not geometry:
            for angle in scattering_angle:
                scattering, _ = get_scattering_in_frequency_range(
                    g,
                    freq,
                    angle,
                    phi,
                    eta,
                    angle,
                )
                scat_on_freq.append(scattering)
            return np.mean(scat_on_freq)
        else:
            return g


def check_convergence(progress, num_for_progress: int = 100, slope_for_progress: float = 1e-8):
    if len(progress) > num_for_progress:
        slope1 = linregress(range(len(progress[-num_for_progress:])), progress[-num_for_progress:]).slope
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
    population_size_factor: float = 1,
    config: Optional[DictConfig] = None,
    **kwargs: Any,
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
