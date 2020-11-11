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
        [0.0, 0.0], [0.5, 0.5 * np.sqrt(3)],
        [1.0, 0.0], [1.5, 0.5 * np.sqrt(3)],
        [2.0, 0.0], [2.5, 0.5 * np.sqrt(3)],
    ]
)
VECTORS = np.array([[0.0, np.sqrt(3)], [3.0, 0.0]])
CELL = HP.Lattice(POINTS, VECTORS)

num0 = 5
num1 = 3
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

angle = 4.705588762457632
axis = (1.268856573288606, 5.486167665013975)
Rotation = HP.RotationGeneral(axis, angle)

fig_config, ax_config = plt.subplots(num="SpinConfig")
intra, inter = cluster.bonds(nth=1)
for bond in intra:
    (x0, y0), (x1, y1) = bond.endpoints
    ax_config.plot(
        [x0, x1], [y0, y1], ls="dashed", color="gray", lw=1.0, zorder=0
    )

cluster_vectors_temp = np.dot(cluster_vectors, Rotation.T)
ax_config.quiver(
    cluster.points[:, 0], cluster.points[:, 1],
    cluster_vectors_temp[:, 0], cluster_vectors_temp[:, 1],
    units="xy", scale_units="xy", scale=1.2, width=0.1, zorder=1,
    pivot="middle", color=0.5 * cluster_vectors_temp + 0.5, clip_on=False,
)
ax_config.plot(
    cluster.points[:, 0], cluster.points[:, 1],
    color="k", ls="", marker="o", ms=5, zorder=2
)
ax_config.text(
    0.00, 0.98, "(a)",
    fontsize=SMALL, ha="center", va="top", transform=ax_config.transAxes,
)
ax_config.set_axis_off()
ax_config.set_aspect("equal")
fig_config.set_size_inches(3.8, 3.0)
fig_config.subplots_adjust(top=1, bottom=0, left=0, right=1)

fig_ssf, ax_ssf = plt.subplots(num="SSF")
im = ax_ssf.pcolormesh(
    kpoints[0::3, 0::3, 0], kpoints[0::3, 0::3, 1], factors[0::3, 0::3],
    cmap="hot_r", shading="gouraud", zorder=0,
)
im.set_edgecolor("face")

colorbar = fig_ssf.colorbar(im, ax=ax_ssf, pad=0.05, format="%.0f")
colorbar.set_ticks(np.linspace(0, vmax, 5, endpoint=True))
colorbar.ax.tick_params(axis="y", labelsize=SMALL)

ax_ssf.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=2.5, ls="dashed", color="tab:green", alpha=1.0,
)
ax_ssf.text(
    0.00, 0.98, "(b)",
    fontsize=SMALL, ha="right", va="top", transform=ax_ssf.transAxes,
)
ax_ssf.set_aspect("equal")
ticks = np.array([-1, 0, 1])
ax_ssf.set_xticks(ticks * np.pi)
ax_ssf.set_yticks(ticks * np.pi)
tick_labels = ["{0}".format(tick) for tick in ticks]
ax_ssf.set_xticklabels(tick_labels, fontsize=SMALL)
ax_ssf.set_yticklabels(tick_labels, fontsize=SMALL)
ax_ssf.set_xlabel(r"$k_x/\pi$", fontsize=SMALL)
ax_ssf.set_ylabel(r"$k_y/\pi$", fontsize=SMALL)
fig_ssf.set_size_inches(3.8, 3.0)
fig_ssf.subplots_adjust(top=0.989, bottom=0.227, left=0, right=1)

fig_path = "figures/SixSitesOrder/"
Path(fig_path).mkdir(exist_ok=True, parents=True)
fig_ssf.savefig(fig_path + "SSF.pdf", transparent=True)
fig_config.savefig(fig_path + "SpinConfig.pdf", transparent=True)

plt.show()
print(fig_config.get_size_inches())
print(fig_ssf.get_size_inches())
plt.close("all")
