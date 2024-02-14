import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wirenec.geometry import Geometry, Wire
from wirenec.scattering import get_scattering_in_frequency_range
from wirenec.visualization import scattering_plot  # , plot_geometry

from export_utils.utils import get_macros
from optimization_utils.cmaes_optimizer import cma_optimizer, objective_function
from parametrization.layers_parametrization import LayersParametrization
from wirenec_optimization.optimization_utils.visualization import plot_geometry

from wirenec_optimization.optimization_utils.hyperparams import parametrization_hyperparams, optimization_hyperparams, \
    scattering_hyperparams, object_hyperparams

from wirenec_optimization.parametrization.sample_objects import make_wire


def dipolar_limit(freq):
    c = 299_792_458
    lbd = c / (freq * 1e6)
    lengths = lbd / 2

    res = []
    for i, l, in enumerate(lengths):
        g = Geometry([Wire((0, 0, -l / 2), (0, 0, l / 2), 0.5 * 1e-3)])
        f = freq[i]
        scattering = get_scattering_in_frequency_range(g, [f], 90, 90, 0, 270)
        res.append(scattering[0][0])

    return freq, np.array(res)


def save_results(
        parametrization,
        param_hyperparams: dict,
        opt_hyperparams: dict,
        scat_hyperparams: dict,
        optimized_dict: dict,
        path: str = "data/bandwidth_optimization/",
):
    path += f'{parametrization.structure_name}__'
    for param, value in param_hyperparams.items():
        path += f"{param}_{str(value)}__"
    for param, value in opt_hyperparams.items():
        path += f"{param}_{str(value)}__"
    for param, value in scat_hyperparams.items():
        path += f"{param}_{str(value)}__"
    path = Path(path.rstrip("_"))

    path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, figsize=(6, 8))

    g_optimized = objective_function(parametrization, params=optimized_dict['params'], geometry=True)

    scatter = scattering_plot(
        ax[0], g_optimized, theta=scat_hyperparams['theta'], eta=scat_hyperparams['eta'], num_points=100,
        scattering_phi_angle=optimization_hyperparams['scattering_angle'][0],
        label='Scattering angle:' + ' ' + str(optimization_hyperparams['scattering_angle'][0]) + '$\degree$'
    )
    scatter_initial = scattering_plot(ax[0], make_wire(object_hyperparams['obj_length'], object_hyperparams['dist_from_obj_to_surf']),
        theta=scat_hyperparams['theta'], eta=scat_hyperparams['eta'], num_points=100,
        scattering_phi_angle=optimization_hyperparams['scattering_angle'][0],
        label='initial object:' + ' ' + str(object_hyperparams['type']))

    parameters_count = (
        int(len(optimized_dict['params']) / 5)  # two more parameters for deltas
        if param_hyperparams["asymmetry_factor"] is not None
        else int(len(optimized_dict['params']) / 3)
    )
    # ax[0].plot(x, np.array(y) * parameters_count, color='b', linestyle='--', label=f'{parameters_count} Bound')
    # ax[0].plot(x, np.array(y), color='k', linestyle='--', label=f'Single dipole bound')

    ax[0].set_xlim(2_000, 10_000)

    ax[1].plot(optimized_dict['progress'], marker='.', linestyle=':')
    ax[0].legend()

    fig.savefig(path / 'scattering_progress.pdf', dpi=200, bbox_inches='tight')
    plt.show()

    plot_geometry(g_optimized, from_top=False, save_to=path / 'optimized_geometry.pdf')

    with open(f'{path}/scat_data.txt', "w+") as file:
            file.write('frequency' + '\t' + 'scaterring_' + str(optimization_hyperparams['scattering_angle'][0]) + '\t' + 'initial' + '\n')
            for i in range(len(scatter[0])):
                file.write(str(scatter[0][i]) + '\t' + str(scatter[1][i]) + '\t' + str(scatter_initial[1][i]) + '\n')

    with open(f'{path}/parametrization_hyperparams.json', 'w+') as fp:
        json.dump(param_hyperparams, fp)
    with open(f'{path}/optimization_hyperparams.json', 'w+') as fp:
        json.dump(opt_hyperparams, fp)
    with open(f'{path}/scattering_hyperparams.json', 'w+') as fp:
        json.dump(scat_hyperparams, fp)
    with open(f'{path}/object_hyperparams.json', 'w+') as fp:
        json.dump(object_hyperparams, fp)
    with open(f'{path}/optimized_params.json', 'w+') as fp:
        optimized_dict['params'] = optimized_dict['params'].tolist()
        json.dump(optimized_dict, fp)
    with open(f'{path}/progress.npy', 'wb') as fp:
        np.save(fp, np.array(optimized_dict['progress']))
    with open(f'{path}/macros.txt', 'w+') as fp:
        fp.write(get_macros(g_optimized))


if __name__ == "__main__":
    parametrization_hyperparams = parametrization_hyperparams
    optimization_hyperparams = optimization_hyperparams
    scattering_hyperparams = scattering_hyperparams
    object_hyperparams = object_hyperparams

    parametrization = LayersParametrization(**parametrization_hyperparams)
    # parametrization = SpatialParametrization(**parametrization_hyperparams)
    optimized_dict = cma_optimizer(parametrization, **optimization_hyperparams) 
    save_results(parametrization, parametrization_hyperparams, optimization_hyperparams, scattering_hyperparams, optimized_dict)
