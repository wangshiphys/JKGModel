import matplotlib.pyplot as plt
import numpy as np


NUMX = 4
NUMY = 6
LW = 4
MS = 10
INDICATOR_COLOR = "green"
INDICATOR_ALPHA = 1.0
COLORMAP = "hot_r"
DATA_PATH = "data/"
DATA_NAME_TEMP = "PS_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"


fig, axes = plt.subplots(
    nrows=2, ncols=3,
    subplot_kw={"polar": True}, num="Probabilities",
)
sub_fig_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
params = [
    (0.30, 1.30), (0.70, 1.65), (0.50, 0.75),
    (0.30, 0.25), (0.70, 0.66), (0.50, 0.00),
]
phi_ticks = np.array([0.0, 0.5, 1.0, 1.5])
theta_ticks = np.array([0.0, 0.25, 0.50])

for index, (tag, (alpha, beta)) in enumerate(zip(sub_fig_tags, params)):
    data_file_name = DATA_PATH + DATA_NAME_TEMP.format(NUMX, NUMY, alpha, beta)
    with np.load(data_file_name) as ld:
        thetas = ld["thetas"][0:51]
        phis = ld["phis"]
        probabilities = ld["probabilities"][0:51, :]

    i, j = divmod(index, 3)
    cs_ij = axes[i, j].pcolormesh(
        phis, thetas, probabilities, zorder=0,
        cmap=COLORMAP, shading="gouraud", vmin=0.0,
    )
    colorbar = fig.colorbar(cs_ij, ax=axes[i, j], pad=0.06)
    colorbar_ticks = np.linspace(0.0, np.max(probabilities), 5, endpoint=True)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(
        ["{0:.1f}".format(tick * 100) for tick in colorbar_ticks],
    )
    colorbar.ax.tick_params(labelsize="medium")
    colorbar.ax.set_title("P[%]", pad=10, fontsize="medium")

    axes[i, j].set_aspect("equal")
    axes[i, j].set_ylim(0, 0.5 * np.pi)
    axes[i, j].set_xticks(phi_ticks * np.pi)
    axes[i, j].set_yticks(theta_ticks * np.pi)
    axes[i, j].set_xticklabels([])
    axes[i, j].set_yticklabels([])
    axes[i, j].grid(True, ls="dashed", color="gray")

    axes[i, j].set_title(
        r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(alpha, beta),
        fontsize="large", pad=15,
    )
    axes[i, j].text(
        0.0, 1.13, tag,
        fontsize="large", ha="center", va="center",
        transform=axes[i, j].transAxes
    )

    axes[i, j].annotate(
        "", xy=(0.30 * np.pi, 0.53 * np.pi),
        xytext=(0.20 * np.pi, 0.53 * np.pi),
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.1"},
        annotation_clip=False,
    )
    axes[i, j].text(
        0.25 * np.pi, 0.58 * np.pi, r"$\phi$",
        ha="center", va="center", fontsize="medium", rotation=-45,
    )


tmp = [
    (0.0, "left", "center"), (0.5, "center", "bottom"),
    (1.0, "right", "center"), (1.5, "center", "top"),
]
for phi, ha, va in tmp:
    axes[0, 1].text(
        phi * np.pi, 0.51 * np.pi,
        r"${0:.0f}^\circ$".format(phi * 180),
        fontsize="medium", ha=ha, va=va,
    )
for theta in [0.00, 0.25, 0.50]:
    axes[0, 1].text(
        1.25 * np.pi, theta * np.pi,
        r"$\theta={0:.0f}^\circ$".format(theta * 180),
        fontsize="medium", ha="center", va="center",
    )

tmp = np.arctan2(1, -np.sin(phis) - np.cos(phis))
indices = np.where(tmp <= 0.5 * np.pi)
axes[0, 0].plot(
    phis[indices], tmp[indices], zorder=3,
    ls="dashed", lw=LW, color=INDICATOR_COLOR, alpha=INDICATOR_ALPHA,
)
axes[0, 2].plot(
    [0.0, np.pi], [0.5 * np.pi, 0.5 * np.pi], zorder=3,
    ls="", marker="o", ms=MS,
    color=INDICATOR_COLOR, alpha=INDICATOR_ALPHA, clip_on=False,
)
axes[1, 2].plot(
    [0.5 * np.pi, 1.5 * np.pi], [0.5 * np.pi, 0.5 * np.pi], zorder=3,
    ls="dashed", lw=LW, color=INDICATOR_COLOR, alpha=INDICATOR_ALPHA,
)

alpha, beta = params[4]
K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
G = np.cos(alpha * np.pi)
tmp = -(G + 2 * K - np.sqrt(9 * G * G - 4 * K * G + 4 * K * K)) / 4
sy = sz = np.sqrt(G * G / (4 * tmp * (tmp + G) + 3 * G * G))
sx = sy * (2 * tmp + G) / G
spin_vectors = [(1, 1, 1), (0, -1, 1), (sx, sy, sz)]
for index, (sx, sy, sz) in zip([1, 3, 4], spin_vectors):
    i, j = divmod(index, 3)
    axes[i, j].plot(
        np.arctan2(sy, sx), np.arctan2(np.sqrt(sx * sx + sy * sy), sz),
        zorder=3, ls="", marker="o", ms=MS,
        color=INDICATOR_COLOR, alpha=INDICATOR_ALPHA,
    )

fig.set_size_inches(18, 9)
plt.show()
print(fig.get_size_inches())
fig.savefig("figures/Probabilities.pdf", dpi=100)
fig.savefig("figures/Probabilities.png", dpi=100)
fig.savefig("figures/Probabilities.jpg", dpi=100)
plt.close("all")
