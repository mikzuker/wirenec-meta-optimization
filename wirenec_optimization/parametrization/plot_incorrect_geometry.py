import matplotlib.pyplot as plt
from wirenec.geometry import Wire

from wirenec_optimization.parametrization.sample_objects import SSRRParametrization
from wirenec_optimization.parametrization.layers_parametrization import LayersParametrization

kind_colors = {None: "b", "SRR": "r", "SSRR": "green"}


def plot_geometry(wires: list[Wire], from_top=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_params = {
        "linewidth": 1,
        "alpha": 0.8,
        # "path_effects": [path_effects.SimpleLineShadow(), path_effects.Normal()]
    }

    if from_top:
        ax.view_init(elev=90, azim=270)

    for wire in wires:
        p1, p2 = wire.p1, wire.p2
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        ax.plot(
            [x1, x2], [y1, y2], [z1, z2], color=kind_colors[wire.kind], **plot_params
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.ticklabel_format(style="sci", scilimits=(0, 0))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    plt.show()


if __name__ == "__main__":
    # hyper_params = {
    #     'matrix_size': (3, 3),
    #     'layers_num': 1,
    #     'tau': 20 * 1e-3,
    #     'delta': 10 * 1e-3,
    #     'asymmetry_factor': None
    # }
    # param = LayersParametrization(**hyper_params)
    # g = param.get_random_geometry(seed=42)

    wire1 = Wire((0.0, 0.0, 0.0), (1.0 / 100, 1.0 / 100, 0.0))
    wire2 = Wire((1.0 / 100, 0.0, 0.0), (0.0, 1.0 / 100, 0.0))
    srr = SSRRParametrization().get_geometry(0.5, (0, 0, 0))
    # g = Geometry([*srr.wires, wire1, wire2])
    # # RuntimeError: GEOMETRY DATA ERROR -- WIRE #1 (TAG ID #1) INTERSECTS WIRE #2 (TAG ID #2)

    plot_geometry([*srr, wire1, wire2])
    # plot_geometry(g)
