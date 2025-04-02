from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import json
import pickle

from wirenec.geometry import Geometry

from wirenec_optimization.optimization_configs.utils import parse_config

from wirenec_optimization.export_utils.utils import get_macros

import matplotlib.pyplot as plt
import numpy as np

from wirenec.scattering import get_scattering_in_frequency_range

from wirenec_optimization.spatial_spectra_creation.parametrization import LayersParametrization

from wirenec_optimization.optimization_utils.visualization import plot_geometry


def global_data_make(geometry,
                     config: DictConfig | None = None,
                     path: str = "data"):
    angles = np.linspace(0, 360)

    parametrization_hyperparams = OmegaConf.to_container(config.parametrization_hyperparams, resolve=True)
    scattering_hyperparams = OmegaConf.to_container(config.scattering_hyperparams, resolve=True)
    seed_hyperparams = OmegaConf.to_container(config.seed_hyperparams, resolve=True)

    seeds = list(range(seed_hyperparams["starting_seed"],
                  seed_hyperparams["ending_seed"]+1,
                  seed_hyperparams["step"]))

    for i in seeds:
        iter_path = Path(f"{path}/{'seed'}_{str(i)}")

        iter_path.mkdir(parents=True, exist_ok=True)

        path_hyperparams = Path(f"{iter_path}/hyperparams")
        path_macros = Path(f"{iter_path}/macros")
        path_vectors = Path(f"{iter_path}/vectors")

        path_hyperparams.mkdir(parents=True, exist_ok=True)
        path_macros.mkdir(parents=True, exist_ok=True)
        path_vectors.mkdir(parents=True, exist_ok=True)

        rcs = []

        g = geometry.get_random_geometry(seed=i)

        for angle in angles:
            g_rotating = g
            g_rotating.rotate(alpha=angle)

            scatter = get_scattering_in_frequency_range(geometry=g_rotating,
                                                        frequency_range=scattering_hyperparams["frequency"],
                                                        theta=scattering_hyperparams["theta"],
                                                        phi=scattering_hyperparams["phi"],
                                                        eta=scattering_hyperparams["eta"],
                                                        scattering_theta_angle=scattering_hyperparams["theta_angle"],
                                                        scattering_phi_angle=scattering_hyperparams["phi_angle"])

            rcs.append(scatter[0][0])

        with open(f"{iter_path}/scat_data.json", "w+") as fp:
            fp.write(
                "angle"
                + "\t"
                + "RCS"
                + "\n"
            )
            for j in range(len(angles)):
                fp.write(str(angles[j]) + "\t" + str(rcs[j]) + "\n")

        plt.figure(figsize=(8, 8), dpi=64)
        plt.plot(angles, rcs, label='RCS from random dipoles')
        plt.xlabel("angles, degree")
        plt.ylabel("RCS, $m^2$")
        plt.savefig(f"{iter_path}/scattering_progress.png", bbox_inches='tight')
        plt.savefig(f"{iter_path}/scattering_progress.pdf", bbox_inches='tight')
        plt.close('all')

        plot_geometry(g, from_top=True, save_to=iter_path / "spatial_geometry.png", is_shown=False)
        plot_geometry(g, from_top=True, save_to=iter_path / "spatial_geometry.pdf", is_shown=False)

        with open(f"{path_hyperparams}/parametrization_hyperparams.json", "w+") as fp:
            json.dump(parametrization_hyperparams, fp)
        with open(f"{path_hyperparams}/scattering_hyperparams.json", "w+") as fp:
            json.dump(scattering_hyperparams, fp)
        with open(f"{path_hyperparams}/seed_hyperparams.json", "w+") as fp:
            json.dump(seed_hyperparams, fp)
        with open(f"{path_macros}/structure_macros.txt", "w+") as fp:
            fp.write(get_macros(g))
        with open(f"{path_vectors}/spatial_structure.pkl", "wb") as handle:
            pickle.dump(g.wires, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    config_path: Path = Path("spatial_producer_config.yaml")
    config = parse_config(config_path)

    param = LayersParametrization(**config.get("parametrization_hyperparams"))
    global_data_make(param, config=config)
