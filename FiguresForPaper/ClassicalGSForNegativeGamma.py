"""
Classical ground state spin configurations for pure negative Gamma model.
"""


import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice

from FontSize import *

SQRT3 = np.sqrt(3)
POINTS = np.array(
    [
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
        [0.5, 0.5 * SQRT3], [1.5, 0.5 * SQRT3],
        [2.5, 0.5 * SQRT3], [3.5, 0.5 * SQRT3],
        [1.0, SQRT3], [2.0, SQRT3], [3.0, SQRT3], [4.0, SQRT3],
        [1.5, 1.5 * SQRT3], [2.5, 1.5 * SQRT3],
        [3.5, 1.5 * SQRT3], [4.5, 1.5 * SQRT3],
    ], dtype=np.float64
)
CLUSTER = Lattice(POINTS, np.array([[4.0, 0.0], [2.0, 2 * SQRT3]]))
INTRA_BONDS, INTER_BONDS = CLUSTER.bonds(nth=1)

_SPIN_UP = [-1, 1, 1]
_SPIN_DOWN = [1, -1, -1]
VECTORS0 = np.array(
    [
        _SPIN_UP, _SPIN_UP, _SPIN_UP, _SPIN_UP,
        _SPIN_DOWN, _SPIN_DOWN, _SPIN_DOWN, _SPIN_DOWN,
        _SPIN_UP, _SPIN_UP, _SPIN_UP, _SPIN_UP,
        _SPIN_DOWN, _SPIN_DOWN, _SPIN_DOWN, _SPIN_DOWN,
    ]
) / SQRT3

_SPIN_UP = [-1, 1, -1]
_SPIN_DOWN = [1, -1, 1]
VECTORS1 = np.array(
    [
        _SPIN_DOWN, _SPIN_UP, _SPIN_DOWN, _SPIN_UP,
        _SPIN_UP, _SPIN_DOWN, _SPIN_UP, _SPIN_DOWN,
        _SPIN_DOWN, _SPIN_UP, _SPIN_DOWN, _SPIN_UP,
        _SPIN_UP, _SPIN_DOWN, _SPIN_UP, _SPIN_DOWN,
    ]
) / SQRT3

_SPIN_UP = [1, 1, -1]
_SPIN_DOWN = [-1, -1, 1]
VECTORS2 = np.array(
    [
        _SPIN_UP, _SPIN_DOWN, _SPIN_UP, _SPIN_DOWN,
        _SPIN_UP, _SPIN_DOWN, _SPIN_UP, _SPIN_DOWN,
        _SPIN_UP, _SPIN_DOWN, _SPIN_UP, _SPIN_DOWN,
        _SPIN_UP, _SPIN_DOWN, _SPIN_UP, _SPIN_DOWN,
    ]
) / SQRT3

# Yellow, Gray, Pink, Cyan
_SPIN0 = [1, 1, -1]
_SPIN1 = [1, 1, 1]
_SPIN2 = [1, -1, 1]
_SPIN3 = [-1, 1, 1]
VECTORS3 = np.array(
    [
        _SPIN0, _SPIN2, _SPIN0, _SPIN2,
        _SPIN1, _SPIN3, _SPIN1, _SPIN3,
        _SPIN0, _SPIN2, _SPIN0, _SPIN2,
        _SPIN1, _SPIN3, _SPIN1, _SPIN3,
    ]
) / SQRT3

sub_fig_tags = ("(a)", "(b)", "(c)", "(d)")
tmp = [VECTORS0, VECTORS1, VECTORS2, VECTORS3]

fig, axes = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all")
for index in range(4):
    spin_vectors = tmp[index]
    ax = axes[divmod(index, 2)]
    sub_fig_tag = sub_fig_tags[index]

    for bond in INTRA_BONDS:
        (x0, y0), (x1, y1) = bond.endpoints
        ax.plot([x0, x1], [y0, y1], ls="solid", lw=2, color="black", zorder=0)

    colors = 0.5 * spin_vectors + 0.5
    ax.quiver(
        POINTS[:, 0], POINTS[:, 1], spin_vectors[:, 0], spin_vectors[:, 1],
        color=colors, units="xy", scale_units="xy", scale=1.45,
        width=0.06, pivot="mid", clip_on=False, zorder=1,
    )

    ax.text(
        1.0, 1.5 * SQRT3, sub_fig_tag,
        ha="center", va="center", fontsize=LARGE,
    )

    ax.set_axis_off()
    ax.set_aspect("equal")

fig.subplots_adjust(
    top=0.98, bottom=0.02, left=0.125, right=0.9, hspace=0.08, wspace=0.0
)
plt.get_current_fig_manager().window.showMaximized()

plt.show()
fig.savefig("figures/SpinConfigForNegativeGamma.pdf", transparent=True)
plt.close("all")
