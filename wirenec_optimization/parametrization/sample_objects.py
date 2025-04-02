import numpy as np
from wirenec.geometry import Wire, Geometry
from wirenec.geometry.samples import double_SRR
from wirenec.visualization import plot_geometry

from wirenec_optimization.parametrization.base_parametrization import (
    BaseObjectParametrization, BaseStructureParametrization
)


def get_geometry_dimensions(geom: Geometry):
    wires = geom.wires
    points = [w.p1 for w in wires] + [w.p2 for w in wires]
    points = np.array(points).T
    mx = 0
    for dim in points:
        mx = max(mx, dim.max() - dim.min())
    return mx


class WireParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 25 * 1e-3, min_size: float = 5 * 1e-3):
    # def __init__(self, max_size: float = 10 * 1e-3, min_size: float = 5 * 1e-3):
        super().__init__("Wire", max_size, min_size)

    def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.75 * 1e-3):
        length = self.min_size + (self.max_size - self.min_size) * size_ratio
        g = Geometry([Wire((0, -length / 2, 0), (0, length / 2, 0), wire_radius)])
        g.rotate(*orientation)
        return g


def make_wire(len_obj, height, wire_radius: float = 0.75 * 1e-3):
    coord_len = len_obj / 2
    g = Geometry([Wire((0., -coord_len, height),
                       (0., coord_len, height),
                       wire_radius, segments=12)])
    return g


def double_srr_updated(r=3.25 * 1e-3, p0=(0, 0, 0), wr=0.25 * 1e-3, num=20):
    g = double_SRR(
        inner_radius=r, outer_radius=r + 5 * wr, wire_radius=wr, num_of_wires=num
    )
    g.translate(p0)

    return g


class SRRParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 12 * 1e-3, min_size: float = 3.5 * 1e-3): #max_size: 9 * 1e-3
        super().__init__("SRR", max_size, min_size)

    def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.25 * 1e-3):
        r = self.min_size + (self.max_size - self.min_size) * size_ratio
        g = double_srr_updated(r=r, wr=wire_radius)
        g.rotate(*orientation)
        return g


def make_srr(size_ratio, orientation, height):
    srr = SRRParametrization()
    g = srr.get_geometry(size_ratio, orientation)
    g.translate((0, 0, height))
    return g


class SSRRParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 30 * 1e-3, min_size: float = 23 * 1e-3):
    # def __init__(self, max_size: float = 14 * 1e-3, min_size: float = 4 * 1e-3):
        super().__init__("SSRR", max_size, min_size)

    def get_geometry(
            self,
            size_ratio,
            orientation,
            wire_radius: float = 0.5 * 1e-3,
            # wire_radius: float = 0.05 * 1e-3,
            num: int = 2,
            segments_count: int = 1,
            G_ratio: float = 0.1,
    ):
        L = self.min_size + (self.max_size - self.min_size) * size_ratio
        G = G_ratio * L
        d = max(2 * wire_radius, G)
        wires = []
        for layer in range(num):
            parity = 1 if layer % 2 else -1
            steps = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

            # Create list of corners coordinates
            side_length = L - 2 * d * layer
            cords = (
                    np.array(
                        [
                            (G / 2, side_length / 2, 0.0),
                            *[
                                (
                                    x_factor * side_length / 2,
                                    y_factor * side_length / 2,
                                    0.0,
                                )
                                for (x_factor, y_factor) in steps
                            ],
                            (-G / 2, side_length / 2, 0.0),
                        ]
                    ).astype(float)
                    * parity
            )

            # Create extended list with `segments_count` wires along each side
            for c1, c2 in zip(cords, cords[1:]):
                cords_extended = np.linspace(c1, c2, segments_count + 1, endpoint=True)
                wires += [
                    Wire(p1, p2, radius=wire_radius, kind=self.object_type)
                    for p1, p2 in zip(cords_extended, cords_extended[1:])
                ]

        g = Geometry(wires)
        g.rotate(*orientation)
        return g


def make_ssrr(size_ratio, orientation, height, wire_radius):
    ssrr = SSRRParametrization()
    g = ssrr.get_geometry(size_ratio, orientation, wire_radius)
    g.translate((0, 0, height))
    return g


class SphereParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 40 * 1e-3, min_size: float = 40 * 1e-3):
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
            wire_radius: float = 0.75 * 1e-3,
            phi_segments: int = 8,
            theta_segments: int = 8,
    ):
        radius = self.min_size + (self.max_size - self.min_size) * size_ratio

        segments = self.__parametrize_sphere__(radius, phi_segments, theta_segments)

        g = Geometry([Wire(*np.around(s, 10), radius=wire_radius) for s in segments])
        g.rotate(*orientation)
        return g


class SpatialParametrization(BaseStructureParametrization):
    def __init__(self, matrix_size, tau_x, tau_y, tau_z):
        super().__init__("spatial")

        self.type_mapping = {
            0: WireParametrization,
            1: WireParametrization,
            # 2: SSRRParametrization
        }

        self.matrix_size = matrix_size

        self.tau_x = tau_x
        self.tau_y = tau_y
        self.tau_z = tau_z

    @property
    def bounds(self) -> np.ndarray:
        m, n, k = self.matrix_size

        size_bounds = [(0, 1) for _ in range(n * m)] * k
        orientation_bounds = [(0, 2 * np.pi) for _ in range(n * m)] * 3 * k
        types_bounds = [(0, len(self.type_mapping.keys()) - 1) for _ in range(n * m)] * k
        return np.array(types_bounds + size_bounds + orientation_bounds)

    def get_random_geometry(self, seed: int = 42) -> Geometry:
        np.random.seed(seed)
        bounds = self.bounds
        random_parameters = [np.random.uniform(low=mn, high=mx) for (mn, mx) in bounds]
        return self.get_geometry(random_parameters)

    def get_geometry(self, params: [np.ndarray, list]) -> Geometry:
        m, n, k = self.matrix_size
        split_size = m * n * k
        types_params, size_params, orientation_params = (
            np.array_split(params[:split_size], k),
            np.array_split(params[split_size:2 * split_size], k),
            np.array_split(params[2 * split_size:], k)
        )

        wires = []
        a_x, a_y, a_z = self.tau_x * n, self.tau_y * m, self.tau_z * k
        x0, y0, z0 = -a_x / 2 + self.tau_x / 2, -a_y / 2 + self.tau_y / 2, -a_y / 2 + self.tau_z / 2

        for l in range(k):
            for i in range(m):
                for j in range(n):
                    tp, size_ratio, orientation = (
                        int(np.around(types_params[l].reshape((m, n))[i, j])),
                        size_params[l].reshape((m, n))[i, j],
                        orientation_params[l].reshape((m, n, 3))[i, j]
                    )
                    orientation = tuple(orientation)
                    g_tmp = self.type_mapping[tp]().get_geometry(size_ratio, orientation)
                    x, y, z = x0 + self.tau_x * i, y0 + self.tau_y * j, self.tau_z * l
                    g_tmp.translate((x, y, z))
                    wires += g_tmp.wires

        return Geometry(wires)

sample_object_mapping = {
    "wire": WireParametrization,
    "srr": SRRParametrization,
    "ssrr": SSRRParametrization,
    "sphere": SphereParametrization,
    "spatial": SpatialParametrization
}

if __name__ == "__main__":
    wire_param = WireParametrization(20 * 1e-3)
    ssrr_param = SSRRParametrization()
    sphere_param = SphereParametrization()

    g = sphere_param.get_geometry(0, (0, 0, 0))
    print(len(g.wires))
    plot_geometry(g)
