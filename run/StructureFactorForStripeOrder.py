from time import time

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import lattice_generator, Lattice
from HamiltonianPy import TRIANGLE_CELL_KS

from StructureFactor import ClassicalSpinStructureFactor
from utilities import ShowVectorField3D


bs = 4 * np.pi * np.identity(2) / np.sqrt(3)
step = 0.01
kx_ratios = np.arange(-0.7, 0.7 + step, step)
ky_ratios = np.arange(-0.7 + step, 0.7, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(kx_ratios, ky_ratios, indexing="ij"), axis=-1), bs
)
BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]

numx = numy = 8
cluster = lattice_generator("triangle", num0=numx, num1=numy)

which = "z"
if which == "x":
    # FM along x-bond
    cell_points = np.array([[0, 0], [0.5, np.sqrt(3)/2]], dtype=np.float64)
    cell_vectors = np.array([[1, 0], [1, np.sqrt(3)]], dtype=np.float64)
elif which == "y":
    # FM along y-bond
    cell_points = np.array([[0, 0], [1, 0]], dtype=np.float64)
    cell_vectors = np.array([[2, 0], [-0.5, np.sqrt(3)/2]], dtype=np.float64)
elif which == "z":
    # FM along z-bond
    cell_points = np.array([[0, 0], [1, 0]], dtype=np.float64)
    cell_vectors = np.array([[2, 0], [0.5, np.sqrt(3)/2]], dtype=np.float64)
else:
    raise ValueError("Invalid `which` variable!")
cell = Lattice(points=cell_points, vectors=cell_vectors)

theta = np.pi * np.random.random()
phi = 2 * np.pi * np.random.random()
vx = np.sin(theta) * np.cos(phi)
vy = np.sin(theta) * np.sin(phi)
vz = np.cos(theta)
spin_vectors = []
for point in cluster.points:
    index = cell.getIndex(site=point, fold=True)
    if index == 0:
        spin_vectors.append([vx, vy, vz])
    elif index == 1:
        spin_vectors.append([-vx, -vy, -vz])
    else:
        raise RuntimeError("Something is wrong in `cell.getIndex`!")
spin_vectors = np.array(spin_vectors, dtype=np.float64)

t0 = time()
factors = ClassicalSpinStructureFactor(kpoints, cluster.points, spin_vectors)
assert np.all(np.abs(factors.imag) < 1E-12)
t1 = time()
print("The time spend: {0:.4f}s".format(t1 - t0))

fig0 = plt.figure("original")
ax0 = fig0.add_subplot(111, projection="3d")
ShowVectorField3D(ax0, cluster.points, spin_vectors)
ax0.set_xlabel("X", fontsize="xx-large")
ax0.set_ylabel("Y", fontsize="xx-large")
ax0.set_zlabel("Z", fontsize="xx-large")
ax0.set_zlim(-0.5, 0.5)

fig1, ax1 = plt.subplots()
im1 = ax1.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], factors.real,
    cmap="magma", shading="gouraud",
)
fig1.colorbar(im1, ax=ax1)
ax1.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=3, ls="dashed", color="tab:blue", alpha=1.0,
)
ticks = np.array([-1, 0, 1], dtype=np.int64)
ax1.set_xticks(ticks * np.pi)
ax1.set_yticks(ticks * np.pi)
ax1.set_xticklabels(["{0}".format(tick) for tick in ticks])
ax1.set_yticklabels(["{0}".format(tick) for tick in ticks])
ax1.set_xlabel(r"$k_x/\pi$", fontsize="large")
ax1.set_ylabel(r"$k_y/\pi$", fontsize="large")
ax1.grid(True, ls="dashed", color="gray")
ax1.set_aspect("equal")
plt.get_current_fig_manager().window.showMaximized()
plt.show()
plt.close("all")
