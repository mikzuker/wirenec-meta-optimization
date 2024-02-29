import json
import pickle
from pathlib import Path
import pandas as pd
import sklearn.metrics

import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from wirenec.geometry import Geometry
from wirenec.visualization import scattering_plot

from wirenec_optimization.optimization_configs.utils import parse_config

from wirenec_optimization.optimization_utils.visualization import plot_geometry
from wirenec_optimization.parametrization.base_parametrization import (
    BaseStructureParametrization,
)

from typing import Optional

# def dir_checker():
#     folder_path = Path('data/reproduced_experiments')
#     # if folder_path.exists() and folder_path.is_dir():
#     #     None
#     #     print('Exist')
#     # else:
#     folder_path.mkdir(parents=True, exist_ok=True)
#     # print('Was made')


def reproduce_experiment(optimization_hyperparams: dict,
                         scattering_hyperparams: dict,
                         object_hyperparams: dict,
                         optimized_dict: dict,
                         test_obj: Geometry,
                         opt_structure: Optional[Geometry] = None,
                         ):

    parametrization: BaseStructureParametrization


    folder_path: str = 'data/reproduced_experiments'
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    g_optimized = Geometry(opt_structure + test_obj)
    test_obj = Geometry(test_obj)

    fig, ax = plt.subplots(2, figsize=(6, 8))

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
        test_obj,
        theta=scattering_hyperparams["theta"],
        eta=scattering_hyperparams["eta"],
        num_points=100,
        scattering_phi_angle=optimization_hyperparams["scattering_angle"][0],
        color="olive",
        lw=2,
        ls=(5, (5, 5)),
        label="Initial object:" + " " + str(object_hyperparams["type"]),
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

    plt.show()

    plot_geometry(g_optimized, from_top=False)

    with open(f"{folder_path}/scat_data_{object_hyperparams['type']}_{optimization_hyperparams['bandwidth']}_{optimization_hyperparams['seed']}.csv", "w+") as file:
        file.write(
            "frequency"
            + "\t"
            + "scaterring_"
            + str(optimization_hyperparams["scattering_angle"][0])
            + "\t"
            + "initial"
            + "\n"
        )
        for i in range(len(scatter[0])):
            file.write(
                str(scatter[0][i])
                + "\t"
                + str(scatter[1][i])
                + "\t"
                + str(scatter_initial[1][i])
                + "\n"
            )
def metric_maker(scattering_data):
    seed = optimization_hyperparams['seed']
    bandwidth = optimization_hyperparams['bandwidth']

    max_value = scattering_data['scaterring_180'].max()
    min_value = scattering_data['scaterring_180'].min()
    mean_value = scattering_data['scaterring_180'].mean()
    std_value = scattering_data['scaterring_180'].std()

    min_freq = min(optimization_hyperparams['frequencies'])
    max_freq = max(optimization_hyperparams['frequencies'])
    mae_data = []

    for i in range(scattering_data['scaterring_180'].count()):
        if min_freq <= scattering_data['frequency'][i] <= max_freq:
            mae_data.append((scattering_data['frequency'][i], i))
    min_line_number = mae_data[0][1]
    max_line_number = mae_data[-1][1]
    mae_value = sklearn.metrics.mean_absolute_error(scattering_data['scaterring_180'][min_line_number:max_line_number], scattering_data['initial'][min_line_number:max_line_number])

    with open(r'data\metrics_aggregation.txt', 'a') as fp:
        # head = fp.readlines()
        # if head[0] == str('max_value'):
        #     None
        # else:
        #     fp.write(
        #         'max_value'
        #         + "\t"
        #           'min_value'
        #         + "\t"
        #           'mean_value'
        #         + "\t"
        #           'std_value'
        #         + "\t"
        #           'mae_value'
        #         + "\t"
        #           'seed'
        #         + "\t"
        #           'bandwidth'
        #     )
        fp.write(
            str(max_value)
            + "\t"
            + str(min_value)
            + "\t"
            + str(mean_value)
            + "\t"
            + str(std_value)
            + "\t"
            + str(mae_value)
            + "\t"
            + str(seed)
            + "\t"
            + str(bandwidth)
            + "\n"
        )

if __name__ == '__main__':
    with open(r'C:\Users\mikzu\Downloads\exp1_10_seeds\6\vectors\test_object.pkl', 'rb') as handle:
        test_object = pickle.load(handle)
    with open(r'C:\Users\mikzu\Downloads\exp1_10_seeds\6\vectors\optimized_structure.pkl', 'rb') as handle:
        opt_structure = pickle.load(handle)
    with open(r'C:\Users\mikzu\Downloads\exp1_10_seeds\6\hyperparams\optimized_params.json', 'rb') as fp:
        optimized_dict = json.load(fp)
    with open(r'C:\Users\mikzu\Downloads\exp1_10_seeds\6\hyperparams\optimization_hyperparams.json', 'rb') as fp:
        optimization_hyperparams = json.load(fp)
    with open(r'C:\Users\mikzu\Downloads\exp1_10_seeds\6\hyperparams\scattering_hyperparams.json', 'rb') as fp:
        scattering_hyperparams = json.load(fp)
    with open(r'C:\Users\mikzu\Downloads\exp1_10_seeds\6\hyperparams\object_hyperparams.json', 'rb') as fp:
        object_hyperparams = json.load(fp)
    with open(r'C:\Users\mikzu\Downloads\exp1_10_seeds\6\scat_data.txt') as fp:
        scattering_data = pd.read_csv(fp, sep='\t')

    #reproduce_experiment(optimization_hyperparams, scattering_hyperparams, object_hyperparams, optimized_dict, test_object, opt_structure)
    metric_maker(scattering_data)