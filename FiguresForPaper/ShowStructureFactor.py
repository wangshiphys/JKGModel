import matplotlib.pyplot as plt
import numpy as np


NUMX = 4
NUMY = 6
DATA_PATH = "data/"
DATA_NAME_TEMP = "SF_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"

fig, axes = plt.subplots(
    nrows=2, ncols=3, sharex="all", sharey="all", num="StructureFactors",
)
sub_fig_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
params = [
    (0.00, 0.00), (1.00, 0.00), (0.50, 0.00),
    (0.30, 0.25), (0.75, 0.25), (0.05, 0.02),
]
for index, (tag, (alpha, beta)) in enumerate(zip(sub_fig_tags, params)):
    data_file_name = DATA_PATH + DATA_NAME_TEMP.format(NUMX, NUMY, alpha, beta)
    with np.load(data_file_name) as ld:
        kpoints = ld["kpoints"]
        factors = ld["factors"]
        BZBoundary = ld["BZBoundary"]

    i, j = divmod(index, 3)
    im_ij = axes[i, j].pcolormesh(
        kpoints[:, :, 0], kpoints[:, :, 1], factors, zorder=0,
        cmap="Reds", shading="gouraud",
    )
    fig.colorbar(im_ij, ax=axes[i, j], format="%.1f")
    axes[i, j].plot(
        BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
        lw=3, ls="dashed", color="tab:blue", alpha=1.0,
    )
    axes[i, j].text(
        0.0, 0.96, tag,
        fontsize="large", ha="left", va="top",
        transform=axes[i, j].transAxes
    )

    ticks = np.array([-1, 0, 1], dtype=np.int64)
    axes[i, j].set_xticks(ticks * np.pi)
    axes[i, j].set_yticks(ticks * np.pi)
    axes[i, j].set_xticklabels(["{0}".format(tick) for tick in ticks])
    axes[i, j].set_yticklabels(["{0}".format(tick) for tick in ticks])
    axes[i, j].grid(True, ls="dashed", color="gray")
    axes[i, j].set_aspect("equal")

axes[1, 0].set_xlabel(r"$k_x/\pi$", fontsize="large")
axes[1, 1].set_xlabel(r"$k_x/\pi$", fontsize="large")
axes[1, 2].set_xlabel(r"$k_x/\pi$", fontsize="large")
axes[0, 0].set_ylabel(r"$k_y/\pi$", fontsize="large")
axes[1, 0].set_ylabel(r"$k_y/\pi$", fontsize="large")

tmp = [(0, 0, 1), (0, 0, -1), (0, 1, 0)]
for j, (J, K, G) in enumerate(tmp):
    axes[0, j].text(
        0.5, 0.96, r"$J={0:d},K={1:d},\Gamma={2:d}$".format(J, K, G),
        fontsize="large", ha="center", va="top",
        transform=axes[0, j].transAxes
    )
tmp = [(0.30, 0.25), (0.75, 0.25), (0.05, 0.02)]
for j, (alpha, beta) in enumerate(tmp):
    axes[1, j].text(
        0.5, 0.96, r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(alpha, beta),
        fontsize="large", ha="center", va="top",
        transform=axes[1, j].transAxes
    )

fig.set_size_inches(18, 9)
plt.show()
print(fig.get_size_inches())
# fig.savefig("figures/StructureFactors.pdf", dpi=100)
# fig.savefig("figures/StructureFactors.png", dpi=100)
# fig.savefig("figures/StructureFactors.jpg", dpi=100)
plt.close("all")
