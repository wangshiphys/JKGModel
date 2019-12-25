import logging
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import lattice_generator, TRIANGLE_CELL_KS
from HamiltonianPy.rotation3d import RotationGeneral

from StructureFactor import ClassicalSpinStructureFactor
from DualTransformation import *

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logging.info("Program start running")


bs = 4 * np.pi * np.identity(2) / np.sqrt(3)
step = 0.01
kx_ratios = np.arange(-0.7, 0.7 + step, step)
ky_ratios = np.arange(-0.7 + step, 0.7, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(kx_ratios, ky_ratios, indexing="ij"), axis=-1), bs
)
BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]

num = 16
cluster = lattice_generator("triangle", num0=num, num1=num)
points = cluster.points
vx, vy, vz = 2 * np.random.random(3) - 1
FMCellVectors = np.array([vx, vy, vz], dtype=np.float64)
NeelCellVectors = np.array(
    [
        np.array([vx, vy, vz]),
        np.dot(RotationGeneral((0, -vz, vy), 120, deg=True), [vx, vy, vz]),
        np.dot(RotationGeneral((0, -vz, vy), 240, deg=True), [vx, vy, vz]),
    ], dtype=np.float64
)

FMSpinConfig = GenerateFMOrder(points, FMCellVectors)
T1FMSpinConfig = T1(points, FMSpinConfig)
T4FMSpinConfig = T4(points, FMSpinConfig)
T1T4FMSpinConfig = T1T4(points, FMSpinConfig)
# ShowFMT1T4(points, FMCellVectors)

NeelSpinConfig = GenerateNeelOrder(points, NeelCellVectors)
T1NeelSpinConfig = T1(points, NeelSpinConfig)
T4NeelSpinConfig = T4(points, NeelSpinConfig)
T1T4NeelSpinConfig = T1T4(points, NeelSpinConfig)
# ShowNeelT1T4(points, NeelCellVectors)

vectors = T1T4NeelSpinConfig
t0 = time()
factors = ClassicalSpinStructureFactor(kpoints, cluster.points, vectors)
assert np.all(np.abs(factors.imag) < 1E-12)
t1 = time()
logging.info("The time spend: %.4fs", t1 - t0)

# Plot the static structure factors
fig, ax = plt.subplots()
im = ax.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], factors.real, zorder=0,
    cmap="Reds", shading="gouraud",
)
fig.colorbar(im, ax=ax)
ax.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=3, ls="dashed", color="tab:blue", alpha=1.0,
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

plt.get_current_fig_manager().window.showMaximized()
plt.show()
plt.close("all")
logging.info("Program start running")
