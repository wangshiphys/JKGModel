import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy import  TRIANGLE_CELL_KS


numx = 4
numy = 6
alpha = 0.05
beta = 0.02
J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
G = np.cos(alpha * np.pi)
sub_fig_tag = "(f)"
params = r"$\alpha={0:.2f}\pi, \beta={1:.2f}\pi$".format(alpha, beta)
# params = r"$J={0:.0f},K={1:.0f},\Gamma={2:.0f}$".format(J, K, G)

step = 0.005
kx_ratios = np.arange(-0.7, 0.7 + step, step)
ky_ratios = np.arange(-0.7 + step, 0.7, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(kx_ratios, ky_ratios, indexing="ij"), axis=-1),
    4 * np.pi * np.identity(2) / np.sqrt(3)
)
BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]

sf_file_name = "SF_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}".format(
    numx, numy, alpha, beta
)
factors = np.load("data/" + sf_file_name + ".npy")

fig, ax = plt.subplots(num=sf_file_name)
im = ax.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], factors, zorder=0,
    cmap="Reds", shading="gouraud",
)
fig.colorbar(im, ax=ax, format="%.1f")

ax.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=3, ls="dashed", color="tab:blue", alpha=1.0,
)
ax.text(
    0.0, 1.0, sub_fig_tag,
    fontsize="xx-large", ha="left", va="top", transform=ax.transAxes
)
ax.text(
    0.5, 0.96, params,
    fontsize="xx-large", ha="center", va="top", transform=ax.transAxes
)

ticks = np.array([-1, 0, 1], dtype=np.int64)
ax.set_xticks(ticks * np.pi)
ax.set_yticks(ticks * np.pi)
ax.set_xticklabels(["{0}".format(tick) for tick in ticks])
ax.set_yticklabels(["{0}".format(tick) for tick in ticks])
ax.set_xlabel(r"$k_x/\pi$", fontsize="large")
ax.set_ylabel(r"$k_y/\pi$", fontsize="large")
ax.grid(True, ls="dashed", color="gray")
ax.set_aspect("equal")

fig.set_size_inches(9, 7.2)
plt.show()
print(fig.get_size_inches())
fig.savefig("figures/" + sf_file_name + ".pdf", dpi=200)
fig.savefig("figures/" + sf_file_name + ".eps", dpi=200)
fig.savefig("figures/" + sf_file_name + ".jpg", dpi=200)
fig.savefig("figures/" + sf_file_name + ".png", dpi=200)
plt.close("all")
