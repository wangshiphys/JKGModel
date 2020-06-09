from time import time

import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

from DualTransformation import *
from StructureFactor import ClassicalSpinStructureFactor
from utilities import TriangularLattice

num = 30
order = "Neel"
transformation = ""
points = TriangularLattice(num1=num, num2=num).cluster.points

if order == "FM":
    spin_vectors = GenerateFMOrder(points)
elif order == "Neel":
    spin_vectors = GenerateNeelOrder(points)
else:
    raise ValueError("Invalid `order`: {0}".format(order))

if transformation == "T1":
    spin_vectors = T1(points, spin_vectors)
elif transformation == "T4":
    spin_vectors = T4(points, spin_vectors)
elif transformation == "T1T4":
    spin_vectors = T1T4(points, spin_vectors)

step = 0.01
ratios = np.arange(-0.7, 0.7 + step, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(ratios, ratios, indexing="ij"), axis=-1),
    4 * np.pi * np.identity(2) / np.sqrt(3)
)
BZBoundary = HP.TRIANGLE_CELL_KS[[*range(6), 0]]

t0 = time()
factors = ClassicalSpinStructureFactor(kpoints, points, spin_vectors)
assert np.all(np.abs(factors.imag) < 1E-12)
factors = factors.real
t1 = time()
print("Time spend on SMSF: {0:.3f}s".format(t1 - t0))

fig, ax = plt.subplots(num=transformation+order)
im = ax.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], factors, zorder=0,
    cmap="hot_r", shading="gouraud",
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
plt.tight_layout()
plt.show()
plt.close("all")
