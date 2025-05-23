import numpy as np
from wirenec.geometry import Wire, Geometry

from wirenec_optimization.export_utils.utils import get_macros
# from wirenec.visualization import plot_geometry
from wirenec_optimization.optimization_utils.visualization import plot_geometry
from wirenec_optimization.parametrization.base_parametrization import BaseStructureParametrization
from wirenec_optimization.parametrization.sample_objects import (
    get_geometry_dimensions, WireParametrization,
    SSRRParametrization, SRRParametrization
)
from wirenec_optimization.parametrization import MAIN_FOLDER
from pathlib import Path

folder_path = Path('data/reproduced_experiments')
folder_path.mkdir(parents=True, exist_ok=True)
class LayersParametrization(BaseStructureParametrization):
    def __init__(self, matrix_size, layers_num, tau, delta, asymmetry_factor: float | None = 0.9):
        super().__init__("layers")

        self.type_mapping = {
            0: WireParametrization,
            1: SSRRParametrization,
            # 1: SRRParametrization,
        }

        self.matrix_size = matrix_size
        self.layers_num = layers_num

        self.tau = tau
        self.delta = delta
        self.asymmetry_factor = asymmetry_factor

    @property
    def bounds(self) -> np.ndarray:
        m, n = self.matrix_size

        size_bounds = [(0, 1) for _ in range(n * m)] * self.layers_num
        orientation_bounds = [(0, 2 * np.pi) for _ in range(n * m)] * self.layers_num
        types_bounds = [(0, len(self.type_mapping.keys()) - 1) for _ in range(n * m)] * self.layers_num

        if self.asymmetry_factor:
            delta_bounds = [(0, 1) for _ in range(n * m)] * self.layers_num * 2
        else:
            delta_bounds = []

        return np.array(types_bounds + size_bounds + orientation_bounds + delta_bounds)

    def get_random_geometry(self, seed: int = 42) -> Geometry:
        np.random.seed(seed)
        bounds = self.bounds
        random_parameters = [np.random.uniform(low=mn, high=mx) for (mn, mx) in bounds]
        return self.get_geometry(random_parameters)

    def get_geometry(self, params: [np.ndarray, list]) -> Geometry:
        m, n = self.matrix_size
        split_size = m * n * self.layers_num
        types_params, size_params, orientation_params = (
            np.array_split(params[:split_size], self.layers_num),
            np.array_split(params[split_size:2 * split_size], self.layers_num),
            np.array_split(params[2 * split_size:3 * split_size], self.layers_num),
        )
        if self.asymmetry_factor:
            delta_params = np.array_split(params[3 * split_size:], self.layers_num)

        wires = []
        a_x, a_y = self.tau * n, self.tau * m
        x0, y0 = -a_x / 2 + self.tau / 2, -a_y / 2 + self.tau / 2

        for l in range(self.layers_num):
            for i in range(m):
                for j in range(n):
                    tp, size_ratio, orientation = (
                        int(np.around(types_params[l].reshape((m, n))[i, j])),
                        size_params[l].reshape((m, n))[i, j],
                        orientation_params[l].reshape((m, n))[i, j]
                    )
                    orientation = (orientation, 0, 0)
                    g_tmp = self.type_mapping[tp]().get_geometry(size_ratio, orientation)
                    if self.asymmetry_factor:
                        phi_rel, dr_rel = delta_params[l].reshape((m, n, 2))[i, j]
                        obj_size_max = get_geometry_dimensions(g_tmp)
                        phi, dr = phi_rel * 2 * np.pi, (self.tau - obj_size_max) / 2 * self.asymmetry_factor * dr_rel
                        dx, dy = dr * np.cos(phi), dr * np.sin(phi)
                    else:
                        dx = dy = 0

                    x, y, z = x0 + self.tau * i + dx, y0 + self.tau * j + dy, self.delta * l
                    g_tmp.translate((x, y, z))
                    wires += g_tmp.wires

        # plot_geometry(wires)
        # plot_geometry(g_tmp)
        return Geometry(wires)
        # return g_tmp


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


if __name__ == '__main__':
    hyper_params = {
        'matrix_size': (3, 3),
        'layers_num': 1,
        'tau': 40 * 1e-3,
        'delta': 10 * 1e-3,
        'asymmetry_factor': None
    }
    param = LayersParametrization(**hyper_params)
    g = param.get_random_geometry(seed=45)
    #g_w = g.wires()
    #plot_geometry(*g_w, from_top=True)

    # Save as CST macros
    with (MAIN_FOLDER / "../data/sample_layers_macros.txt").open('w+') as f:
        f.write(get_macros(g))
