import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping

from ClassicalModel import *
from StructureFactor import ClassicalSpinStructureFactor

CELL_POINTS3 = np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2], [1.0, 0.0]])
CELL_POINTS4 = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [1.5, np.sqrt(3) / 2]]
)
CELL_POINTS6 = np.array(
    [
        [0.0, 0.0], [1.0, 0.0],
        [0.5, np.sqrt(3) / 2], [1.5, np.sqrt(3) / 2],
        [1.0, np.sqrt(3)], [2.0, np.sqrt(3)],
    ]
)
CELL_VECTORS3 = np.array([[1.5, -np.sqrt(3) / 2], [1.5, np.sqrt(3) / 2]])
CELL_VECTORS4 = np.array([[2.0, 0.0], [1.0, np.sqrt(3)]])
CELL_VECTORS6 = np.array([[1.5, -np.sqrt(3) / 2], [1.5, 1.5 * np.sqrt(3)]])
CELL3 = HP.Lattice(CELL_POINTS3, CELL_VECTORS3)
CELL4 = HP.Lattice(CELL_POINTS4, CELL_VECTORS4)
CELL6 = HP.Lattice(CELL_POINTS6, CELL_VECTORS6)


num0 = 10
num1 = 6
alpha = 0.25
beta = 0.50
cell = CELL6

step = 0.01
ratios = np.arange(-0.7, 0.7 + step, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(ratios, ratios, indexing="ij"), axis=-1),
    4 * np.pi * np.identity(2) / np.sqrt(3)
)
BZBoundary = HP.TRIANGLE_CELL_KS[[*range(6), 0]]

Hx, Hy, Hz = HMatrixGenerator(alpha, beta)
cluster, category, x_bonds, y_bonds, z_bonds = ClusterGenerator(
    num0=num0, num1=num1, cell=cell
)
args = (cluster.point_num, category, x_bonds, y_bonds, z_bonds, Hx, Hy, Hz)

initial_spin_angles = np.pi * np.random.random(2 * len(category))
initial_spin_angles[1::2] *= 2
res = basinhopping(
    EnergyCore0, initial_spin_angles,
    niter=200, minimizer_kwargs={"args": args}
)
phis = res.x[0::2]
thetas = res.x[1::2]
sin_phis = np.sin(phis)
cos_phis = np.cos(phis)
sin_thetas = np.sin(thetas)
cos_thetas = np.cos(thetas)

energy_per_site = res.fun / cluster.point_num
cell_vectors = np.array(
    [sin_phis * cos_thetas, sin_phis * sin_thetas, cos_phis]
)
print(res.message[0])
print("Cell Vectors:")
print(repr(cell_vectors.T))
print("Energy Per-Site: {0}".format(energy_per_site))

cluster_vectors = np.empty((cluster.point_num, 3), dtype=np.float64)
for cell_index, cluster_indices in category.items():
    cluster_vectors[cluster_indices] = cell_vectors[:, cell_index]

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
ax_config.set_axis_off()
ax_config.set_aspect("equal")
fig_config.subplots_adjust(top=1.0, bottom=0.0)
plt.get_current_fig_manager().window.showMaximized()

factors = ClassicalSpinStructureFactor(kpoints, cluster.points, cluster_vectors)
msg = "Maximum imaginary part of factors: {0}"
print(msg.format(np.max(np.abs(factors.imag))))
factors = factors.real

fig_ssf, ax_ssf = plt.subplots(num="SSF")
im = ax_ssf.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], factors,
    cmap="magma", shading="gouraud", zorder=0,
)
fig_ssf.colorbar(im, ax=ax_ssf)
ax_ssf.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=3, ls="dashed", color="tab:blue", alpha=1.0,
)
ticks = np.array([-1, 0, 1])
ax_ssf.set_xticks(ticks * np.pi)
ax_ssf.set_yticks(ticks * np.pi)
ax_ssf.set_xticklabels(["{0}".format(tick) for tick in ticks])
ax_ssf.set_yticklabels(["{0}".format(tick) for tick in ticks])
ax_ssf.set_xlabel(r"$k_x/\pi$", fontsize="large")
ax_ssf.set_ylabel(r"$k_y/\pi$", fontsize="large")
ax_ssf.grid(True, ls="dashed", color="gray")
ax_ssf.set_aspect("equal")

plt.get_current_fig_manager().window.showMaximized()
plt.show()
plt.close("all")
