import json
import yaml
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from wirenec.geometry import Geometry, Wire
from wirenec.scattering import get_scattering_in_frequency_range
from wirenec.visualization import scattering_plot

from export_utils.utils import get_macros
from optimization_utils.cmaes_optimizer import (
    cma_optimizer,
    objective_function,
    get_reference_object,
)
from parametrization.layers_parametrization import LayersParametrization
from wirenec_optimization.optimization_configs.utils import parse_config
from wirenec_optimization.optimization_utils.bandwidth_creating import (
    freq_maker,
)
from wirenec_optimization.optimization_utils.visualization import plot_geometry
from wirenec_optimization.parametrization.base_parametrization import (
    BaseStructureParametrization,
)
from wirenec_optimization.parametrization.sample_objects import make_wire


def dipolar_limit(freq):
    c = 299_792_458
    lbd = c / (freq * 1e6)
    lengths = lbd / 2

    res = []
    for (
        i,
        l,
    ) in enumerate(lengths):
        g = Geometry([Wire((0, 0, -l / 2), (0, 0, l / 2), 0.5 * 1e-3)])
        f = freq[i]
        scattering = get_scattering_in_frequency_range(g, [f], 90, 90, 0, 270)
        res.append(scattering[0][0])

    return freq, np.array(res)


def save_results(
    parametrization: BaseStructureParametrization,
    config: DictConfig,
    path: str = "data/bandwidth_optimization/",
):
    parametrization_hyperparams = OmegaConf.to_container(
        config.parametrization_hyperparams, resolve=True
    )
    optimization_hyperparams = OmegaConf.to_container(
        config.optimization_hyperparams, resolve=True
    )
    scattering_hyperparams = OmegaConf.to_container(
        config.scattering_hyperparams, resolve=True
    )
    object_hyperparams = OmegaConf.to_container(config.object_hyperparams, resolve=True)

    path += f"{parametrization.structure_name}__"
    for param, value in parametrization_hyperparams.items():
        path += f"{param}_{str(value)}__"
    for param, value in optimization_hyperparams.items():
        path += f"{param}_{str(value)}__"
    for param, value in scattering_hyperparams.items():
        path += f"{param}_{str(value)}__"
    path = Path(path.rstrip("_"))

    path.mkdir(parents=True, exist_ok=True)

    path_hyperparams = Path(f"{path}/hyperparams")
    path_macros = Path(f"{path}/macros")
    path_vectors = Path(f"{path}/vectors")

    path_hyperparams.mkdir(parents=True, exist_ok=True)
    path_macros.mkdir(parents=True, exist_ok=True)
    path_vectors.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, figsize=(6, 8))

    g_optimized = objective_function(
        parametrization, params=optimized_dict["params"], object_params=config.object_hyperparams, geometry=True
    )

    test_obj = get_reference_object(config.object_hyperparams)
    opt_structure = Geometry(g_optimized.wires[: -len(test_obj.wires)])

    scatter = scattering_plot(
        ax[0],
        g_optimized,
        theta=scattering_hyperparams["theta"],
        eta=scattering_hyperparams["eta"],
        num_points=100,
        scattering_phi_angle=optimization_hyperparams["scattering_angle"][0],
        color="firebrick",
        lw=2,
        label="Scattering angle:"
        + " "
        + str(optimization_hyperparams["scattering_angle"][0])
        + "$\degree$",
    )
    scatter_initial = scattering_plot(
        ax[0],
        make_wire(
            object_hyperparams["obj_length"],
            object_hyperparams["dist_from_obj_to_surf"],
        ),
        theta=scattering_hyperparams["theta"],
        eta=scattering_hyperparams["eta"],
        num_points=100,
        scattering_phi_angle=optimization_hyperparams["scattering_angle"][0],
        color="olive",
        lw=2,
        ls=(5, (5, 5)),
        label="Initial object:" + " " + str(object_hyperparams["type"]),
    )

    parameters_count = (
        int(len(optimized_dict["params"]) / 5)  # two more parameters for deltas
        if parametrization_hyperparams["asymmetry_factor"] is not None
        else int(len(optimized_dict["params"]) / 3)
    )

    ax[0].set_xlim(4_000, 10_000)
    ax[0].set_ylim(-max(scatter_initial[1]) * 0.05, max(scatter_initial[1]) * 1.1)
    ax[0].axhline(0, color="k", lw=1)
    ax[0].scatter(
        optimization_hyperparams["frequencies"],
        [0] * len(optimization_hyperparams["frequencies"]),
        color="darkgreen",
        marker="s",
        alpha=0.5,
        label="Optimized frequencies",
    )
    ax[0].fill_between(
        optimization_hyperparams["frequencies"],
        0,
        max(scatter_initial[1]) * 1.1,
        color="darkgreen",
        alpha=0.1,
        label="Optimized area",
    )
    ax[0].legend()

    ax[1].plot(optimized_dict["progress"], marker=".", linestyle=":")

    fig.savefig(path / "scattering_progress.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    plot_geometry(g_optimized, from_top=False, save_to=path / "optimized_geometry.pdf")

    with open(f"{path}/scat_data.json", "w+") as fp:
        fp.write(
            "frequency"
            + "\t"
            + "scaterring_"
            + str(optimization_hyperparams["scattering_angle"][0])
            + "\t"
            + "initial"
            + "\n"
        )
        for i in range(len(scatter[0])):
            fp.write(
                str(scatter[0][i])
                + "\t"
                + str(scatter[1][i])
                + "\t"
                + str(scatter_initial[1][i])
                + "\n"
            )

    with open(f"{path_hyperparams}/parametrization_hyperparams.json", "w+") as fp:
        json.dump(parametrization_hyperparams, fp)
    with open(f"{path_hyperparams}/optimization_hyperparams.json", "w+") as fp:
        json.dump(optimization_hyperparams, fp)
    with open(f"{path_hyperparams}/scattering_hyperparams.json", "w+") as fp:
        json.dump(scattering_hyperparams, fp)
    with open(f"{path_hyperparams}/object_hyperparams.json", "w+") as fp:
        json.dump(object_hyperparams, fp)
    with open(f"{path_hyperparams}/optimized_params.json", "w+") as fp:
        optimized_dict["params"] = optimized_dict["params"].tolist()
        json.dump(optimized_dict, fp)
    with open(f"{path}/progress.npy", "wb") as fp:
        np.save(fp, np.array(optimized_dict["progress"]))
    with open(f"{path_macros}/obj_macros.txt", "w+") as fp:
        fp.write(get_macros(test_obj))
    with open(f"{path_macros}/opt_without_obj_macros.txt", "w+") as fp:
        fp.write(get_macros(opt_structure))
    with open(f"{path_macros}/opt_with_obj_macros.txt", "w+") as fp:
        fp.write(get_macros(g_optimized))
    with open(f"{path_vectors}/test_object.pkl", 'wb') as handle:
        pickle.dump(test_obj.wires, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{path_vectors}/optimized_structure.pkl", 'wb') as handle:
        pickle.dump(opt_structure.wires, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{path_vectors}/opt_structure_with_object.pkl", 'wb') as handle:
        pickle.dump(g_optimized.wires, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    config_path: Path = Path("optimization_configs/single_layer_config.yaml")
    config = parse_config(config_path)

    if config.get("parametrization_class") == "diffuser":
        assert config.optimization_hyperparams.bandwidth is not None
        config.optimization_hyperparams.frequencies = tuple(
            freq_maker(config.optimization_hyperparams.general_frequency, config.optimization_hyperparams.bandwidth)
        )

    parametrization = LayersParametrization(**config.get("parametrization_hyperparams"))
    optimized_dict = cma_optimizer(
        parametrization, **config.optimization_hyperparams, config=config
    )
    save_results(parametrization, config)
