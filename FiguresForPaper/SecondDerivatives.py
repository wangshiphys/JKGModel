import matplotlib.pyplot as plt
import numpy as np

from FontSize import *


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


name0 = "data/GEs_numx=4_numy=6_beta=1.5000.npz"
name1 = "data/GEs_numx=4_numy=6_beta=0.0000.npz"
name2 = "data/GEs_numx=4_numy=6_alpha=0.3000.npz"
with np.load(name0) as ld:
    gses0 = ld["gses"]
    params0 = ld["params"]
with np.load(name1) as ld:
    gses1 = ld["gses"]
    params1 = ld["params"]
with np.load(name2) as ld:
    gses2 = ld["gses"]
    params2 = ld["params"]
d2params0, d2gses0 = derivation(params0, gses0, nth=2)
d2params1, d2gses1 = derivation(params1, gses1, nth=2)
d2params2, d2gses2 = derivation(params2, gses2, nth=2)

line_width = 4
color_gses = "tab:blue"
color_d2gses = "tab:red"

fig, (ax0_gses, ax1_gses, ax2_gses) = plt.subplots(1, 3)
ax0_d2gses = ax0_gses.twinx()
ax1_d2gses = ax1_gses.twinx()
ax2_d2gses = ax2_gses.twinx()

ax0_gses.plot(params0, gses0, lw=line_width, color=color_gses)
ax0_d2gses.plot(d2params0, -d2gses0, lw=line_width, color=color_d2gses)
ax0_gses.set_xlim(0, 0.65)
ax0_gses.set_ylim(-22, -9.5)
ax0_d2gses.set_ylim(-700, 40000)
ax0_gses.set_xlabel(r"$\alpha/\pi$", fontsize=LARGE)
ax0_gses.set_ylabel("$E$", rotation=0, fontsize=LARGE, color=color_gses)
ax0_gses.set_yticks([-22, -18, -14, -10])
ax0_gses.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax0_d2gses.set_yticks([0, 10000, 20000, 30000, 40000])
ax0_d2gses.set_yticklabels(["0", "1E4", "2E4", "3E4", "4E4"])
ax0_gses.tick_params("x", colors="black", labelsize=SMALL)
ax0_gses.tick_params("y", colors=color_gses, labelsize=SMALL)
ax0_d2gses.tick_params("y", colors=color_d2gses, labelsize=SMALL)

ax1_gses.plot(params1, gses1, lw=line_width, color=color_gses)
ax1_d2gses.plot(d2params1, -d2gses1, lw=line_width, color=color_d2gses)
ax1_gses.set_xlim(0, 0.65)
ax1_gses.set_ylim(-10, -7.6)
ax1_d2gses.set_ylim(-60, 850)
ax1_gses.set_xlabel(r"$\alpha/\pi$", fontsize=LARGE)
ax1_gses.set_yticks([-10, -9, -8])
ax1_gses.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax1_d2gses.set_yticks([0, 200, 400, 600, 800])
ax1_d2gses.set_yticklabels(["0", "2E2", "4E2", "6E2", "8E2"])
ax1_gses.tick_params("x", labelsize=SMALL)
ax1_gses.tick_params("y", colors=color_gses, labelsize=SMALL)
ax1_d2gses.tick_params("y", colors=color_d2gses, labelsize=SMALL)

ax2_gses.plot(params2, gses2, lw=line_width, color=color_gses)
ax2_d2gses.plot(d2params2, -d2gses2, lw=line_width, color=color_d2gses)
ax2_gses.set_ylim(-20, -8)
ax2_gses.set_xlim(0.25, 1.25)
ax2_d2gses.set_ylim(-130, 850)
ax2_gses.set_xlabel(r"$\beta/\pi$", fontsize=LARGE)
ax2_d2gses.set_ylabel(r"$E''$", rotation=0, fontsize=LARGE, color=color_d2gses)
ax2_gses.set_yticks([-20, -16, -12, -8])
ax2_gses.set_xticks([0.25, 0.50, 0.75, 1.00, 1.25])
ax2_d2gses.set_yticks([0, 200, 400, 600, 800])
ax2_d2gses.set_yticklabels(["0", "2E2", "4E2", "6E2", "8E2"])
ax2_gses.tick_params("x", labelsize=SMALL)
ax2_gses.tick_params("y", colors=color_gses, labelsize=SMALL)
ax2_d2gses.tick_params("y", colors=color_d2gses, labelsize=SMALL)

ax0_gses.text(
    0.98, 0.98, "(b)", ha="right", va="top",
    fontsize=LARGE, transform=ax0_gses.transAxes
)
ax1_gses.text(
    0.98, 0.98, "(c)", ha="right", va="top",
    fontsize=LARGE, transform=ax1_gses.transAxes
)
ax2_gses.text(
    0.98, 0.98, "(d)", ha="right", va="top",
    fontsize=LARGE, transform=ax2_gses.transAxes
)

fig.set_size_inches(18, 4)
plt.tight_layout()
plt.show()
fig.savefig("figures/SecondDerivatives.pdf", transparent=True)
plt.close("all")
