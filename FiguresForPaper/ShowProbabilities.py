"""
Show the probabilities of the cluster spin-coherent state in exact cluster
ground states for some representative model parameters.
"""


import matplotlib.pyplot as plt
import numpy as np


INDICATOR_LW = 5
INDICATOR_MS = 10
COLORMAP = "hot_r"
INDICATOR_ALPHA = 0.9
INDICATOR_COLOR = "tab:green"
PS_DATA_NAME_TEMP = "data/PS_num1=4_num2=6_direction=avg_" \
                    "alpha={0:.4f}_beta={1:.4f}.npz"

params = [
    (0.50, 0.75), (0.50, 0.00), (0.30, 1.30),
    (0.70, 1.65), (0.70, 0.66), (0.30, 0.25),
]
sub_fig_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

theta_ticks = np.array([0.0, 0.25, 0.50])
phi_ticks = np.array([0.0, 0.5, 1.0, 1.5])

fig, axes = plt.subplots(
    nrows=2, ncols=3, subplot_kw={"polar": True}, num="Probabilities",
)

for index in range(len(params)):
    alpha, beta = params[index]
    ax = axes[divmod(index, 3)]
    sub_fig_tag = sub_fig_tags[index]

    ps_data_name = PS_DATA_NAME_TEMP.format(alpha, beta)
    with np.load(ps_data_name) as ld:
        phis = ld["phis"][0::2]
        thetas = ld["thetas"][0:51:2]
        probabilities = ld["probabilities"][0:51:2, 0::2]
    max_probs = np.max(probabilities)

    im = ax.pcolormesh(
        phis, thetas, probabilities, zorder=0,
        cmap=COLORMAP, shading="gouraud", vmin=0.0, vmax=max_probs,
    )
    im.set_edgecolor("face")

    colorbar = fig.colorbar(im, ax=ax, pad=0.03)
    colorbar_ticks = np.linspace(0.0, max_probs, 5, endpoint=True)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(
        ["{0:.1f}".format(tick * 100) for tick in colorbar_ticks],
    )
    colorbar.ax.tick_params(axis="y", labelsize="large")
    colorbar.ax.set_title("P[%]", pad=8, fontsize="large")

    ax.set_aspect("equal")
    ax.set_ylim(0, 0.5 * np.pi)
    ax.set_xticks(phi_ticks * np.pi)
    ax.set_yticks(theta_ticks * np.pi)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, ls="dashed", color="gray")

    ax.set_title(
        r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(alpha, beta),
        fontsize="xx-large", pad=15,
    )
    ax.text(
        0.0, 1.125, sub_fig_tag,
        fontsize="xx-large", ha="right", va="center", transform=ax.transAxes,
    )

    ax.annotate(
        "", xy=(0.30 * np.pi, 0.53 * np.pi),
        xytext=(0.20 * np.pi, 0.53 * np.pi),
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.1"},
        annotation_clip=False,
    )
    ax.text(
        0.25 * np.pi, 0.58 * np.pi, r"$\phi$",
        ha="center", va="center", fontsize="large", rotation=-45,
    )


tmp = [
    (0.0, "left", "center"), (0.5, "center", "bottom"),
    (1.0, "right", "center"), (1.5, "center", "top"),
]
for phi, ha, va in tmp:
    axes[0, 0].text(
        phi * np.pi, 0.51 * np.pi,
        r"${0:.0f}^\circ$".format(phi * 180),
        fontsize="large", ha=ha, va=va,
    )
for theta in [0.00, 0.25, 0.50]:
    axes[0, 0].text(
        1.35 * np.pi, theta * np.pi,
        r"$\theta={0:.0f}^\circ$".format(theta * 180),
        fontsize="large", ha="center", va="center", rotation=0,
    )

axes[0, 0].plot(
    [0.0, np.pi], [0.5 * np.pi, 0.5 * np.pi], zorder=3,
    ls="", marker="o", ms=INDICATOR_MS,
    color=INDICATOR_COLOR, alpha=INDICATOR_ALPHA, clip_on=False,
)

axes[0, 1].plot(
    [0.5 * np.pi, 1.5 * np.pi], [0.5 * np.pi, 0.5 * np.pi], zorder=3,
    ls="dashed", lw=INDICATOR_LW, color=INDICATOR_COLOR, alpha=INDICATOR_ALPHA,
)

tmp = np.arctan2(1, -np.sin(phis) - np.cos(phis))
indices = np.where(tmp <= 0.5 * np.pi)
axes[0, 2].plot(
    phis[indices], tmp[indices], zorder=3,
    ls="dashed", lw=INDICATOR_LW, color=INDICATOR_COLOR, alpha=INDICATOR_ALPHA,
)

alpha, beta = params[4]
K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
G = np.cos(alpha * np.pi)
tmp = -(G + 2 * K - np.sqrt(9 * G * G - 4 * K * G + 4 * K * K)) / 4
sy = sz = np.sqrt(G * G / (4 * tmp * (tmp + G) + 3 * G * G))
sx = sy * (2 * tmp + G) / G

spin_vectors = [(1, 1, 1), (sx, sy, sz), (0, -1, 1)]
for index, (sx, sy, sz) in zip([3, 4, 5], spin_vectors):
    ax = axes[divmod(index, 3)]
    ax.plot(
        np.arctan2(sy, sx), np.arctan2(np.sqrt(sx * sx + sy * sy), sz),
        zorder=3, ls="", marker="o", ms=INDICATOR_MS,
        color=INDICATOR_COLOR, alpha=INDICATOR_ALPHA,
    )

plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
# fig.savefig("figures/Probabilities.pdf", dpi=200)
# fig.savefig("figures/Probabilities.png", dpi=200)
plt.close("all")
