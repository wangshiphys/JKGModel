import matplotlib.pyplot as plt
import numpy as np

from FontSize import *

line_width = 4
color_gses = "tab:blue"
color_d2gses = "tab:red"


def derivation(xs, ys, nth=1):
    """
    Calculate the nth derivatives of `ys` versus `xs` discretely.

    The derivatives are calculated using the following formula:
        dy / dx = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])

    Parameters
    ----------
    xs : 1-D array
        The independent variables.
        `xs` is assumed to be sorted in ascending order and there are no
        identical values in `xs`.
    ys : 1-D array
        The dependent variables.
        `ys` should be of the same length as `xs`.
    nth : int, optional
        The nth derivatives.
        Default: 1.

    Returns
    -------
    xs : 1-D array
        The independent variables.
    ys : 1-D array
        The nth derivatives corresponding to the returned `xs`.
    """

    assert isinstance(nth, int) and nth >= 0
    assert isinstance(xs, np.ndarray) and xs.ndim == 1
    assert isinstance(ys, np.ndarray) and ys.shape == xs.shape

    for i in range(nth):
        ys = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        xs = (xs[1:] + xs[:-1]) / 2
    return xs, ys


names = [
    "data/GEs_numx=4_numy=6_alpha=0.0500.npz",
    "data/GEs_numx=4_numy=6_alpha=0.3000.npz",
    # "data/GEs_numx=4_numy=6_alpha=0.5000.npz",
    "data/GEs_numx=4_numy=6_alpha=0.7500.npz",
    "data/GEs_numx=4_numy=6_beta=0.0000.npz",
    "data/GEs_numx=4_numy=6_beta=0.5000.npz",
    # "data/GEs_numx=4_numy=6_beta=0.7500.npz",
    "data/GEs_numx=4_numy=6_beta=1.5000.npz",
]
container = []
for name in names:
    with np.load(name) as ld:
        gses = ld["gses"]
        params = ld["params"]
    d2params, d2gses = derivation(params, gses, nth=2)
    container.append((params, gses, d2params, d2gses))

fig, axes_gses = plt.subplots(2, 3)

axes_d2gses = []
axes_gses = axes_gses.reshape((-1,))
sub_fig_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
for index in range(len(names)):
    ax_gses = axes_gses[index]
    sub_fig_tag = sub_fig_tags[index]
    params, gses, d2params, d2gses = container[index]

    ax_d2gses = ax_gses.twinx()
    axes_d2gses.append(ax_d2gses)

    ax_gses.plot(params, gses, lw=line_width, color=color_gses)
    ax_d2gses.plot(d2params, -d2gses, lw=line_width, color=color_d2gses)
    ax_gses.text(
        0.06, 0.98, sub_fig_tag,
        ha="left", va="top", fontsize=XXLARGE+10, transform=ax_gses.transAxes
    )
    ax_d2gses.set_yticks([])

axes_gses[0].set_xlim(-0.5, 1.5)
axes_gses[0].set_ylim(-12.0, -8.5)
axes_d2gses[0].set_ylim(-30, 140)
axes_gses[0].set_xlabel(r"$\beta/\pi$", fontsize=XXLARGE+4)
axes_gses[0].tick_params("x", labelsize=XXLARGE+2)
axes_gses[0].tick_params("y", colors=color_gses, labelsize=XXLARGE+2)
axes_gses[0].set_ylabel("$E$", rotation=0, fontsize=XXLARGE+4, color=color_gses)

axes_gses[1].set_xlim(0, 2)
axes_gses[1].set_ylim(-20.0, -6.0)
axes_d2gses[1].set_ylim(-180, 1000)
axes_gses[1].set_xlabel(r"$\beta/\pi$", fontsize=XXLARGE+4)
axes_gses[1].tick_params("x", labelsize=XXLARGE+2)
axes_gses[1].tick_params("y", colors=color_gses, labelsize=XXLARGE+2)

axes_gses[2].set_xlim(0, 2)
axes_gses[2].set_ylim(-23.0, -9.0)
axes_d2gses[2].set_ylim(-200, 3100)
axes_gses[2].set_xlabel(r"$\beta/\pi$", fontsize=XXLARGE+4)
axes_gses[2].tick_params("x", labelsize=XXLARGE+2)
axes_gses[2].tick_params("y", colors=color_gses, labelsize=XXLARGE+2)

axes_gses[3].set_xlim(0, 1)
axes_gses[3].set_ylim(-15, -7.0)
axes_d2gses[3].set_ylim(-120, 850)
axes_gses[3].set_xlabel(r"$\alpha/\pi$", fontsize=XXLARGE+4)
axes_gses[3].tick_params("x", labelsize=XXLARGE+2)
axes_gses[3].tick_params("y", colors=color_gses, labelsize=XXLARGE+2)
axes_gses[3].set_ylabel("$E$", rotation=0, fontsize=XXLARGE+4, color=color_gses)

axes_gses[4].set_xlim(0.25, 0.75)
axes_gses[4].set_ylim(-14, -11.0)
axes_d2gses[4].set_ylim(-120, 1000)
axes_gses[4].set_xticks([0.25, 0.50, 0.75])
axes_gses[4].set_xlabel(r"$\alpha/\pi$", fontsize=XXLARGE+4)
axes_gses[4].tick_params("x", labelsize=XXLARGE+2)
axes_gses[4].tick_params("y", colors=color_gses, labelsize=XXLARGE+2)

axes_gses[5].set_xlim(0.25, 0.75)
axes_gses[5].set_ylim(-22.5, -17.5)
axes_d2gses[5].set_ylim(-1000, 40000)
axes_gses[5].set_xticks([0.25, 0.50, 0.75])
axes_gses[5].set_xlabel(r"$\alpha/\pi$", fontsize=XXLARGE+4)
axes_gses[5].tick_params("x", labelsize=XXLARGE+2)
axes_gses[5].tick_params("y", colors=color_gses, labelsize=XXLARGE+2)

plt.show()
print(fig.get_size_inches())
fig.savefig("figures/SecondDerivatives.pdf", transparent=True)
plt.close("all")
