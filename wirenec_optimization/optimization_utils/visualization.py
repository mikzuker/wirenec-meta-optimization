import matplotlib.pyplot as plt
from wirenec.geometry import Wire
import matplotlib.patheffects as path_effects

kind_colors = {
    None: 'b',
    'Wire': 'b',
    'SRR': 'r',
    'SSRR': 'g'
}


def plot_geometry(g, from_top=False, save_to=None, is_shown=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_params = {
        "linewidth": 1,
        "alpha": 0.8,
        # "path_effects": [path_effects.SimpleLineShadow(), path_effects.Normal()]
    }

    if from_top:
        ax.view_init(elev=90, azim=270)

    wires = g.wires
    for wire in wires:
        p1, p2 = wire.p1, wire.p2
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        ax.plot([x1 * 100, x2 * 100], [y1 * 100, y2 * 100], [z1 * 100, z2 * 100],
                color=kind_colors[wire.kind],
                **plot_params)


        ax.set_xlabel('X, cm').set_fontsize(12)
        ax.set_ylabel('Y, cm').set_fontsize(12)
        ax.set_zlabel('Z, cm').set_fontsize(12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.ticklabel_format(style='sci', scilimits=(0, 0))

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    if save_to is not None:
        plt.savefig(save_to, dpi=200)

    if is_shown is True:
        plt.show()
    else:
        plt.close('all')
