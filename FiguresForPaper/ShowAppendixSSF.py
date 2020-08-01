"""
Show the static magnetic structure factors (SMSF) for some representative model
parameters.
"""

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import TRIANGLE_CELL_MS as MS

from FontSize import *

BZLW = 5
COLORMAP = "hot_r"
BZCOLOR = "tab:green"
SF_DATA_NAME_TEMP = "data/SF_num1=4_num2=6_which=avg_" \
                    "alpha={0:.4f}_beta={1:.4f}.npz"

sub_fig_tags = ["(a)", "(b)", "(c)", "(d)"]
params = [(0.50, 0.50), (0.50, 1.86), (0.30, 1.30), (0.7, 1.65)]

fig, axes = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all")
for index in range(len(params)):
    alpha, beta = params[index]
    ax = axes[divmod(index, 2)]
    sub_fig_tag = sub_fig_tags[index]
    title = r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(alpha, beta)

    sf_data_name = SF_DATA_NAME_TEMP.format(alpha, beta)
    with np.load(sf_data_name) as ld:
        BZBoundary = ld["BZBoundary"]
        factors = ld["factors"][0::5, 0::5]
        kpoints = ld["kpoints"][0::5, 0::5, :]
    vmin = np.min(factors)
    vmax = np.max(factors)

    im = ax.pcolormesh(
        kpoints[:, :, 0], kpoints[:, :, 1], factors,
        cmap=COLORMAP, shading="gouraud", zorder=0,
    )
    im.set_edgecolor("face")

    colorbar = fig.colorbar(im, ax=ax, pad=0.01, format="%.1f")
    colorbar.ax.tick_params(axis="y", labelsize="large")
    colorbar.set_ticks(np.linspace(vmin, vmax, 5, endpoint=True))
    colorbar.ax.tick_params(axis="y", labelsize=SMALL)

    ax.plot(
        BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
        ls="dashed", lw=BZLW, color=BZCOLOR, alpha=1.0,
    )

    ax.text(
        0.0, 0.98, sub_fig_tag,
        fontsize=LARGE, ha="left", va="top", transform=ax.transAxes,
    )
    ax.set_title(title, fontsize="xx-large", loc="center")
    ax.set_title(title, fontsize=LARGE, loc="center")
    ticks = np.array([-1, 0, 1], dtype=np.int64)
    tick_labels = ["{0}".format(tick) for tick in ticks]
    ax.set_xticks(ticks * np.pi)
    ax.set_yticks(ticks * np.pi)
    ax.set_xticklabels(tick_labels, fontsize="large")
    ax.set_yticklabels(tick_labels, fontsize="large")
    ax.set_xticklabels(tick_labels, fontsize=SMALL)
    ax.set_yticklabels(tick_labels, fontsize=SMALL)
    ax.grid(True, ls="dashed", color="gray")
    ax.set_aspect("equal")

axes[1, 0].set_xlabel(r"$k_x/\pi$", fontsize=LARGE)
axes[1, 1].set_xlabel(r"$k_x/\pi$", fontsize=LARGE)
axes[0, 0].set_ylabel(r"$k_y/\pi$", fontsize=LARGE)
axes[1, 0].set_ylabel(r"$k_y/\pi$", fontsize=LARGE)

plt.get_current_fig_manager().window.showMaximized()
plt.show()
fig.savefig("figures/AppendixSSF.pdf", transparent=True)
plt.close("all")

