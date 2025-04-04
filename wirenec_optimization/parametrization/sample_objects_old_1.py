import numpy as np
from wirenec.geometry import Wire, Geometry
from wirenec.geometry.samples import double_srr_6GHz

# from wirenec.visualization import plot_geometry
from wirenec_optimization.optimization_utils.visualization import plot_geometry

from wirenec_optimization.parametrization.base_parametrization import BaseObjectParametrization
# from wirenec_optimization.parametrization.plot_incorrect_geometry import plot_geometry

def get_geometry_dimensions(geom: Geometry):
    wires = geom.wires
    points = [w.p1 for w in wires] + [w.p2 for w in wires]
    points = np.array(points).T
    mx = 0
    for dim in points:
        mx = max(mx, dim.max() - dim.min())
    return mx


class WireParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 20 * 1e-3, min_size: float = 2 * 1e-3):
        super().__init__("Wire", max_size, min_size)

    def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.5*1e-3):
        length = self.min_size + (self.max_size - self.min_size) * size_ratio
        g = Geometry([Wire(
            (0, -length / 2, 0),
            (0, length / 2, 0),
            radius=wire_radius, kind=self.object_type)])
        g.rotate(*orientation)
        return g
def make_wire(len_obj, height, wire_radius: float = 0.5*1e-3):
    coord_len = len_obj/2
    g = Geometry([Wire((0., -coord_len, height),
                             (0., coord_len, height),
                             wire_radius, segments=12)])
    return g


class SSRRParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 1 * 1e-3, min_size: float = 14 * 1e-3):
        super().__init__("SSRR", max_size, min_size)

    def get_geometry(
            self, size_ratio, orientation,
            wire_radius: float = 0.15 * 1e-3, num: int = 2,
            segments_count: int = 2, G_ratio: float = 0.1
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
            coords = np.array([
                (G / 2, side_length / 2, 0.),
                *[(x_factor * side_length / 2, y_factor * side_length / 2, 0.) for (x_factor, y_factor) in steps],
                (-G / 2, side_length / 2, 0.)
            ]).astype(float) * parity

            # Create extended list with `segments_count` wires along each side
            for c1, c2 in zip(coords, coords[1:]):
                coords_extended = np.linspace(c1, c2, segments_count + 1, endpoint=True)
                wires += [Wire(p1, p2, radius=wire_radius, kind=self.object_type) for p1, p2 in zip(coords_extended, coords_extended[1:])]

        # plot_geometry(wires)

        # g = Geometry(wires)
        # g.rotate(*orientation)
        return wires

def make_ssrr(size_ratio, orientation, height):
    ssrr = SSRRParametrization()
    g = ssrr.get_geometry(size_ratio, orientation)
    g.translate((0, 0, height))
    return g

class SRRParametrization(BaseObjectParametrization):
    def __init__(self, max_size: float = 5.75 * 1e-3, min_size: float = 7 * 1e-3):
        super().__init__("SRR", max_size, min_size)

    def get_geometry(self, size_ratio, orientation, wire_radius: float = 0.5*1e-3):
        r = self.min_size + (self.max_size - self.min_size) * size_ratio
        g = double_srr_6GHz(r=r)
        wires = []
        for wire in g.wires:
            wire.kind = self.object_type
            wires.append(wire)
        g = Geometry(wires)
        g.rotate(*orientation)
        return g
def make_srr(size_ratio, orientation, height):
    srr = SRRParametrization()
    g = srr.get_geometry(size_ratio, orientation)
    g.translate((0, 0, height))
    return g

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
    wire_param = WireParametrization(20 * 1e-4)
    ssrr_param = SSRRParametrization()
    srr_param = SRRParametrization()

    g = ssrr_param.get_geometry(0.5, (0, 0, 0))
    plot_geometry(g)
