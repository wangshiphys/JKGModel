from time import time

import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

from StructureFactor import ClassicalSpinStructureFactor

num_cell = 2
num_cluster = 30
cell = HP.lattice_generator("triangle", num0=num_cell, num1=num_cell)
cluster = HP.lattice_generator("triangle", num0=num_cluster, num1=num_cluster)

cell_spin_vectors = 2 * np.random.random((num_cell * num_cell, 3)) - 1
cell_spin_vectors /= np.linalg.norm(cell_spin_vectors, axis=1, keepdims=True)

cluster_spin_vectors = []
for point in cluster.points:
    index = cell.getIndex(site=point, fold=True)
    cluster_spin_vectors.append(cell_spin_vectors[index])
cluster_spin_vectors = np.array(cluster_spin_vectors, dtype=np.float64)

step = 0.01
ratios = np.arange(-0.7, 0.7 + step, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(ratios, ratios, indexing="ij"), axis=-1),
    4 * np.pi * np.identity(2) / np.sqrt(3)
)
BZBoundary = HP.TRIANGLE_CELL_KS[[*range(6), 0]]

t0 = time()
factors = ClassicalSpinStructureFactor(
    kpoints, cluster.points, cluster_spin_vectors
)
assert np.all(np.abs(factors.imag) < 1E-10)
factors = factors.real
t1 = time()
print("Time spend on SMSF: {0:.3f}s".format(t1 - t0))

fig, ax = plt.subplots()
im = ax.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], factors, zorder=0,
    cmap="magma", shading="gouraud",
)
fig.colorbar(im, ax=ax)
ax.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=3, ls="dashed", color="tab:blue", alpha=1.0,
)
ticks = np.array([-1, 0, 1])
ax.set_xticks(ticks * np.pi)
ax.set_yticks(ticks * np.pi)
ax.set_xticklabels(["{0}".format(tick) for tick in ticks])
ax.set_yticklabels(["{0}".format(tick) for tick in ticks])
ax.set_xlabel(r"$k_x/\pi$", fontsize="large")
ax.set_ylabel(r"$k_y/\pi$", fontsize="large")
ax.grid(True, ls="dashed", color="gray")
ax.set_aspect("equal")
plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
plt.close("all")
