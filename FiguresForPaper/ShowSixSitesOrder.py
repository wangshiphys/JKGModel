from pathlib import Path

import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping

from ClassicalModel import *
from FontSize import *
from StructureFactor import ClassicalSpinStructureFactor

POINTS = np.array(
    [
        [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2],
        [1.5, np.sqrt(3) / 2], [1.0, np.sqrt(3)], [2.0, np.sqrt(3)],
    ]
)
VECTORS = np.array([[1.5, -np.sqrt(3) / 2], [1.5, 1.5 * np.sqrt(3)]])
CELL = HP.Lattice(POINTS, VECTORS)

num0 = 7
num1 = 4
alpha = 0.30
beta = 0.50

step = 0.01
ratios = np.arange(-0.7, 0.7 + step, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(ratios, ratios, indexing="ij"), axis=-1),
    4 * np.pi * np.identity(2) / np.sqrt(3)
)
BZBoundary = HP.TRIANGLE_CELL_KS[[*range(6), 0]]

Hx, Hy, Hz = HMatrixGenerator(alpha, beta)
cluster, category, x_bonds, y_bonds, z_bonds = ClusterGenerator(
    num0=num0, num1=num1, cell=CELL
)
args = (cluster.point_num, category, x_bonds, y_bonds, z_bonds, Hx, Hy, Hz)

res = basinhopping(
    EnergyCore0, np.pi * np.random.random(2 * len(category)),
    niter=200, minimizer_kwargs={"args": args}
)
phis = res.x[0::2]
thetas = res.x[1::2]
sin_phis = np.sin(phis)
cos_phis = np.cos(phis)
sin_thetas = np.sin(thetas)
cos_thetas = np.cos(thetas)

cell_vectors = np.array(
    [sin_phis * cos_thetas, sin_phis * sin_thetas, cos_phis]
)
cluster_vectors = np.empty((cluster.point_num, 3), dtype=np.float64)
for cell_index, cluster_indices in category.items():
    cluster_vectors[cluster_indices] = cell_vectors[:, cell_index]

factors = ClassicalSpinStructureFactor(kpoints, cluster.points, cluster_vectors)
factors = factors.real
vmin = np.min(factors)
vmax = np.max(factors)

fig_config, ax_config = plt.subplots(num="SpinConfig")
ax_config.plot(
    cluster.points[:, 0], cluster.points[:, 1],
    color="k", ls="", marker="o", ms=5,
)
ax_config.quiver(
    cluster.points[:, 0], cluster.points[:, 1],
    cluster_vectors[:, 0], cluster_vectors[:, 1],
    units="xy", scale_units="xy", scale=1.45, width=0.08,
    pivot="middle", color=0.5 * cluster_vectors + 0.5, clip_on=False,
)
ax_config.text(
    0.02, 0.98, "(a)",
    fontsize=LARGE, ha="left", va="top", transform=ax_config.transAxes,
)
ax_config.set_axis_off()
ax_config.set_aspect("equal")
fig_config.set_size_inches(9.9, 9.26)
fig_config.subplots_adjust(top=1, bottom=0, left=0, right=1)

fig_ssf, ax_ssf = plt.subplots(num="SSF")
im = ax_ssf.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], factors,
    cmap="hot_r", shading="gouraud", zorder=0,
)
im.set_edgecolor("face")

colorbar = fig_ssf.colorbar(im, ax=ax_ssf, pad=0.01, format="%.1f")
colorbar.set_ticks(np.linspace(vmin, vmax, 5, endpoint=True))
colorbar.ax.tick_params(axis="y", labelsize=SMALL)

ax_ssf.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=5, ls="dashed", color="tab:green", alpha=1.0,
)
ax_ssf.text(
    0.02, 0.98, "(b)",
    fontsize=LARGE, ha="left", va="top", transform=ax_ssf.transAxes,
)
ax_ssf.set_aspect("equal")
# ax_ssf.grid(True, ls="dashed", color="gray")
ticks = np.array([-1, 0, 1])
ax_ssf.set_xticks(ticks * np.pi)
ax_ssf.set_yticks(ticks * np.pi)
tick_labels = ["{0}".format(tick) for tick in ticks]
ax_ssf.set_xticklabels(tick_labels, fontsize=SMALL)
ax_ssf.set_yticklabels(tick_labels, fontsize=SMALL)
ax_ssf.set_xlabel(r"$k_x/\pi$", fontsize=LARGE)
ax_ssf.set_ylabel(r"$k_y/\pi$", fontsize=LARGE)
fig_ssf.set_size_inches(11.0, 9.26)
fig_ssf.subplots_adjust(top=0.973, bottom=0.094, left=0.014, right=0.986)

fig_path = "figures/SixSitesOrder/"
Path(fig_path).mkdir(exist_ok=True, parents=True)
fig_ssf.savefig(fig_path + "SSF.pdf", transparent=True)
fig_config.savefig(fig_path + "SpinConfig.pdf", transparent=True)

plt.show()
plt.close("all")
