"""
Show the static magnetic structure factors (SMSF) for some representative model
parameters.
"""


import matplotlib.pyplot as plt
import numpy as np


BZLW = 5
COLORMAP = "hot_r"
BZCOLOR = "tab:green"
SF_DATA_NAME_TEMP = "data/SF_num1=4_num2=6_direction=avg_" \
                    "alpha={0:.4f}_beta={1:.4f}.npz"

params = [
    (0.50, 0.00), (0.00, 0.00), (1.00, 0.00),
    (0.70, 0.66), (0.30, 0.25), (0.05, 0.10),
]
sub_fig_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
titles = [
    r"$J=\Gamma=0,K=1$",
    r"$J=K=0,\Gamma=1$",
    r"$J=K=0,\Gamma=-1$",
    r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(*params[3]),
    r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(*params[4]),
    r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(*params[5]),
]

fig, axes = plt.subplots(
    nrows=2, ncols=3, sharex="all", sharey="all", num="StructureFactors",
)
for index in range(len(params)):
    title = titles[index]
    alpha, beta = params[index]
    ax = axes[divmod(index, 3)]
    sub_fig_tag = sub_fig_tags[index]

    sf_data_name = SF_DATA_NAME_TEMP.format(alpha, beta)
    with np.load(sf_data_name) as ld:
        BZBoundary = ld["BZBoundary"]
        factors = ld["factors"][0::5, 0::5]
        kpoints = ld["kpoints"][0::5, 0::5, :]

    im = ax.pcolormesh(
        kpoints[:, :, 0], kpoints[:, :, 1], factors,
        cmap=COLORMAP, shading="gouraud", zorder=0,
    )
    im.set_edgecolor("face")

    colorbar = fig.colorbar(im, ax=ax, pad=0.01, format="%.1f")
    colorbar.ax.tick_params(axis="y", labelsize="large")

    ax.plot(
        BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
        ls="dashed", lw=BZLW, color=BZCOLOR, alpha=1.0,
    )

    ax.text(
        0.0, 1.0, sub_fig_tag,
        fontsize="xx-large", ha="left", va="bottom", transform=ax.transAxes,
    )
    ax.set_title(title, fontsize="xx-large", loc="center")
    ticks = np.array([-1, 0, 1], dtype=np.int64)
    tick_labels = ["{0}".format(tick) for tick in ticks]
    ax.set_xticks(ticks * np.pi)
    ax.set_yticks(ticks * np.pi)
    ax.set_xticklabels(tick_labels, fontsize="large")
    ax.set_yticklabels(tick_labels, fontsize="large")
    ax.grid(True, ls="dashed", color="gray")
    ax.set_aspect("equal")

axes[1, 0].set_xlabel(r"$k_x/\pi$", fontsize="large")
axes[1, 1].set_xlabel(r"$k_x/\pi$", fontsize="large")
axes[1, 2].set_xlabel(r"$k_x/\pi$", fontsize="large")
axes[0, 0].set_ylabel(r"$k_y/\pi$", fontsize="large")
axes[1, 0].set_ylabel(r"$k_y/\pi$", fontsize="large")

plt.get_current_fig_manager().window.showMaximized()
plt.show()
# fig.savefig("figures/StructureFactors.pdf", dpi=100, transparent=True)
# fig.savefig("figures/StructureFactors.png", dpi=100, transparent=True)
plt.close("all")
