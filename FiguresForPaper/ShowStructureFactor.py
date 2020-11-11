"""
Show the static magnetic structure factors (SMSF) for some representative model
parameters.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import TRIANGLE_CELL_MS as MS
from HamiltonianPy import TRIANGLE_CELL_KS as KS

from FontSize import *

BZLW = 5
MARKERSIZE = 16
COLORMAP = "hot_r"
BZCOLOR = "tab:green"
FIG_Path = "figures/StructureFactors/"
Path(FIG_Path).mkdir(exist_ok=True, parents=True)
SF_DATA_PATH = "data/"
SF_DATA_NAME_TEMP = "SF_num1=4_num2=6_direction=avg_" \
                    "alpha={0:.4f}_beta={1:.4f}.npz"

params = [
    (0.50, 0.00), (0.30, 0.75), (0.75, 0.25), (0.30, 0.00),
    (0.30, 0.50), (0.30, 1.30), (0.70, 1.65), (0.05, 0.10),
]
sub_fig_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
for index in range(len(params)):
    alpha, beta = params[index]
    sub_fig_tag = sub_fig_tags[index]
    fig_name = "SF_alpha={0:.4f},beta={1:.4f}.pdf".format(alpha, beta)
    title = r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(alpha, beta)

    sf_data_name = SF_DATA_PATH + SF_DATA_NAME_TEMP.format(alpha, beta)
    with np.load(sf_data_name) as ld:
        BZBoundary = ld["BZBoundary"]
        factors = ld["factors"][0::5, 0::5]
        kpoints = ld["kpoints"][0::5, 0::5, :]
    vmin = np.min(factors)
    vmax = np.max(factors)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(
        kpoints[:, :, 0], kpoints[:, :, 1], factors,
        cmap=COLORMAP, shading="gouraud", zorder=0,
    )
    im.set_edgecolor("face")

    colorbar = fig.colorbar(im, ax=ax, pad=0.05, format="%.1f")
    colorbar.set_ticks(np.linspace(vmin, vmax, 5, endpoint=True))
    colorbar.ax.tick_params(axis="y", labelsize=MEDIUM)

    ax.plot(
        BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
        ls="dashed", lw=BZLW, color=BZCOLOR, alpha=1.0,
    )
    ax.text(
        0.02, 0.98, sub_fig_tag,
        fontsize=XLARGE, ha="left", va="top", transform=ax.transAxes,
    )

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=XLARGE, loc="center")
    ticks = np.array([-1, 0, 1], dtype=np.int64)
    tick_labels = ["{0}".format(tick) for tick in ticks]
    ax.set_xticks(ticks * np.pi)
    ax.set_yticks(ticks * np.pi)

    if index in (0, 3, 6):
        ax.set_yticklabels(tick_labels, fontsize=MEDIUM)
        ax.set_ylabel(r"$k_y/\pi$", fontsize=XLARGE, labelpad=0)
    else:
        ax.set_yticklabels([])
    if index in (6, 7):
        ax.set_xticklabels(tick_labels, fontsize=MEDIUM)
        ax.set_xlabel(r"$k_x/\pi$", fontsize=XLARGE, labelpad=0)
    else:
        ax.set_xticklabels([])

    fig.set_size_inches(5.4, 4.75)
    fig.subplots_adjust(top=0.925, bottom=0.166, left=0.150, right=0.980)
    fig.savefig(FIG_Path + fig_name, transparent=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

fig, ax = plt.subplots()
ax.plot([-4.6, 4.6], [0.0, 0.0], color="k", lw=BZLW / 2, zorder=0)
ax.plot([0.0, 0.0], [-4.1, 4.1], color="k", lw=BZLW / 2, zorder=0)
ax.plot(BZBoundary[:, 0], BZBoundary[:, 1], lw=BZLW, color=BZCOLOR, zorder=1)
marker0, = ax.plot(
    0, 0, ls="", marker="o", ms=MARKERSIZE, color="tab:blue", zorder=2,
)
marker1, = ax.plot(
    MS[:, 0], MS[:, 1],
    ls="", marker="s", ms=MARKERSIZE, color="tab:orange", zorder=2,
)
marker2, = ax.plot(
    KS[:, 0], KS[:, 1],
    ls="", marker="^", ms=MARKERSIZE, color="tab:red", zorder=2,
)
ax.text(
    0.5, 0.5, r"$\~\Gamma$",
    fontsize=XLARGE, color="tab:blue", ha="center", va="center",
)
ax.text(
    MS[2, 0] + 0.8, MS[2, 1], r"$\~M^{\prime}$",
    fontsize=XLARGE, color="tab:orange", ha="center", va="center",
)
ax.text(
    MS[3, 0] + 0.5, MS[3, 1] + 0.5, r"$\~M$",
    fontsize=XLARGE, color="tab:orange", ha="center", va="center",
)
ax.text(
    MS[4, 0] - 0.8, MS[4, 1], r"$\~M^{\prime\prime}$",
    fontsize=XLARGE, color="tab:orange", ha="center", va="center",
)
ax.text(
    KS[2, 0] + 0.8, KS[2, 1], r"$\~K^{\prime}$",
    fontsize=XLARGE, color="tab:red", ha="center", va="center",
)
ax.text(
    KS[3, 0] + 0.5, KS[3, 1] + 0.5, r"$\~K$",
    fontsize=XLARGE, color="tab:red", ha="center", va="center",
)
ax.legend(
    [marker0, marker1], ["FM", "Stripe"],
    bbox_to_anchor=(1.08, 0.94), bbox_transform=ax.transAxes,
    fontsize=MEDIUM, loc="upper right", frameon=True, shadow=True,
    borderpad=0.1, handletextpad=0.1, borderaxespad=0.0
)
ax.text(
    0.1, 0.93, "(i)",
    fontsize=XLARGE, ha="left", va="top", transform=ax.transAxes,
)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_axis_off()
ax.set_aspect("equal")
fig.set_size_inches(5.4, 4.75)
fig.subplots_adjust(top=1.000, bottom=0.000, left=0.000, right=0.930)
fig.savefig(FIG_Path + "BrillouinZone.pdf", transparent=True)

plt.show()
plt.close("all")
