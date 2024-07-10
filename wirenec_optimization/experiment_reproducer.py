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

from matplotlib import ticker

import csv
from itertools import islice

# def dir_checker():
#     folder_path = Path('data/reproduced_experiments')
#     # if folder_path.exists() and folder_path.is_dir():
#     #     None
#     #     print('Exist')
#     # else:
#     folder_path.mkdir(parents=True, exist_ok=True)
#     # print('Was made')


def reproduce_experiment(
                         optimization_hyperparams: dict,
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
    #opt_structure = Geometry(opt_structure)
    test_obj = Geometry(test_obj)

    fig, ax = plt.subplots(2, figsize=(6, 8))
    #fig.subplots_adjust(hspace=0.5)

    scatter = scattering_plot(
        ax[0],
        g_optimized,
        theta=scattering_hyperparams["theta"],
        phi=scattering_hyperparams["phi"],
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
        phi=scattering_hyperparams["phi"],
        eta=scattering_hyperparams["eta"],
        num_points=100,
        scattering_phi_angle=optimization_hyperparams["scattering_angle"][0],
        color="olive",
        lw=2,
        ls=(5, (5, 5)),
        label="Initial object:" + " " + str(object_hyperparams["type"]),
    )

    X_CST_str = []
    Y_CST_str = []
    X_CST = []
    Y_CST = []

    with open(CST_Path) as datafile:
        plotting = csv.reader(datafile, delimiter="\t")

        for line in islice(datafile, 24, None):

            for ROWS in plotting:
                X_CST_str.append(ROWS[0])
                Y_CST_str.append(ROWS[1])
        X_CST_str = [i.replace(',', '.') for i in X_CST_str]
        Y_CST_str = [i.replace(',', '.') for i in Y_CST_str]

        for i in range(len(X_CST_str)):
            X_CST.append(float(X_CST_str[i]) * 10 ** 3)
            Y_CST.append(float(Y_CST_str[i]) * 10 ** 4)

    ax[0].plot(X_CST, Y_CST, lw=2, color='navy', label='CST modeling')

    ax[0].set_xlim(4_000, 10_000)
    ax[0].set_ylim(-max(scatter_initial[1]) * 0.05, max(scatter_initial[1]) * 1.7)
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
        max(scatter_initial[1]) * 1.7,
        color="darkgreen",
        alpha=0.1,
        label="Optimized area",
    )
    ax[0].legend()
    ax[0].yaxis.set_major_locator(ticker.MaxNLocator(4))

    progress = [element * 10**4 for element in optimized_dict['progress']]
    ax[1].plot(progress, marker=".", linestyle=":", color='k')
    ax[1].set_xlabel('Iterations').set_fontsize(14)
    ax[1].set_ylabel('Mean population scattering, $cm^2$').set_fontsize(14)
    ax[1].set_yticks(list(ax[1].get_yticks()) + [0])

    plt.savefig(f"{folder_path}/reproduce_{object_hyperparams['type']}_{optimization_hyperparams['bandwidth']}_{optimization_hyperparams['number_of_frequencies']}_{optimization_hyperparams['seed']}.pdf", bbox_inches='tight')
    plt.show()

    plot_geometry(g_optimized, from_top=False, save_to=f"{folder_path}/reproduce_{object_hyperparams['type']}_optimized.svg")

    # with open(f"{folder_path}/scat_data_{object_hyperparams['type']}_{optimization_hyperparams['bandwidth']}_{optimization_hyperparams['seed']}.csv", "w+") as file:
    #     file.write(
    #         "frequency"
    #         + "\t"
    #         + "scaterring_"
    #         + str(optimization_hyperparams["scattering_angle"][0])
    #         + "\t"
    #         + "initial"
    #         + "\n"
    #     )
    #     for i in range(len(scatter[0])):
    #         file.write(
    #             str(scatter[0][i])
    #             + "\t"
    #             + str(scatter[1][i])
    #             + "\t"
    #             + str(scatter_initial[1][i])
    #             + "\n"
    #         )


if __name__ == '__main__':
    sphere_path = Path()
    wire_path = Path()
    random_path = Path()
    srr_path = Path()
    ssrr_path = Path()
    with open(ssrr_path/r'vectors\test_object.pkl', 'rb') as handle:
        test_object = pickle.load(handle)
    with open(ssrr_path/r'vectors\optimized_structure.pkl', 'rb') as handle:
        opt_structure = pickle.load(handle)
    with open(ssrr_path/r'hyperparams\optimized_params.json', 'rb') as fp:
        optimized_dict = json.load(fp)
    with open(ssrr_path/r'hyperparams\optimization_hyperparams.json', 'rb') as fp:
        optimization_hyperparams = json.load(fp)
    with open(ssrr_path/r'hyperparams\scattering_hyperparams.json', 'rb') as fp:
        scattering_hyperparams = json.load(fp)
    with open(ssrr_path/r'hyperparams\object_hyperparams.json', 'rb') as fp:
        object_hyperparams = json.load(fp)
    with open(ssrr_path/r'scat_data.json') as fp:
        scattering_data = pd.read_csv(fp, sep='\t')
    CST_Path = Path()
    reproduce_experiment(optimization_hyperparams, scattering_hyperparams, object_hyperparams, optimized_dict, test_object, opt_structure)