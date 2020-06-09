from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import TRIANGLE_CELL_KS

from StructureFactor import QuantumSpinStructureFactor
from utilities import TriangularLattice

print("Program start running")
log_msg = "SMSF for alpha={0:.4f}, beta={1:.4f}, direction={2}: dt={3:.3f}s"

ES_DATA_PATH = "data/QuantumSpinModel/ES/"
SF_DATA_PATH = "data/QuantumSpinModel/SMSF/"
Path(SF_DATA_PATH).mkdir(exist_ok=True, parents=True)

lattice_id = "num1={0}_num2={1}_direction={2}"
ES_DATA_NAME_TEMP = "ES_" + lattice_id + "_alpha={3:.4f}_beta={4:.4f}.npz"
SF_DATA_NAME_TEMP = "SF_" + lattice_id + "_alpha={3:.4f}_beta={4:.4f}.npz"

step = 0.01
ratios = np.arange(-0.7, 0.7 + step, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(ratios, ratios, indexing="ij"), axis=-1),
    4 * np.pi * np.identity(2) / np.sqrt(3)
)
BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]

num1 = 4
num2 = 6
alpha = 0.50
beta = 0.00

total_factors = 0.0
for direction in ("xy", "xz", "yx", "yz", "zx", "zy"):
    points = TriangularLattice(num1, num2, direction).cluster.points
    es_data_name = ES_DATA_NAME_TEMP.format(num1, num2, direction, alpha, beta)
    sf_data_name = SF_DATA_NAME_TEMP.format(num1, num2, direction, alpha, beta)

    with np.load(ES_DATA_PATH + es_data_name) as ld:
        gs_ket = ld["vectors"][:, 0]

    t0 = time()
    factors = QuantumSpinStructureFactor(kpoints, points, gs_ket)
    assert np.all(np.abs(factors.imag) < 1E-12)
    factors = factors.real
    np.savez(
        SF_DATA_PATH + sf_data_name,
        size=[num1, num2], direction=[direction], parameters=[alpha, beta],
        kpoints=kpoints, BZBoundary=BZBoundary, factors=factors,
    )
    t1 = time()
    total_factors += factors
    print(log_msg.format(alpha, beta, direction, t1 - t0))

avg_factors = total_factors / 6
np.savez(
    SF_DATA_PATH + SF_DATA_NAME_TEMP.format(num1, num2, "avg", alpha, beta),
    size=[num1, num2], direction=["avg"], parameters=[alpha, beta],
    kpoints=kpoints, BZBoundary=BZBoundary, factors=factors,
)

name = SF_DATA_NAME_TEMP.format(num1, num2, "avg", alpha, beta)[:-4]
fig, ax = plt.subplots(num=name)
im = ax.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], avg_factors, zorder=0,
    cmap="magma", shading="gouraud",
)
fig.colorbar(im, ax=ax)
ax.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=3, ls="dashed", color="tab:blue", alpha=1.0,
)
title = r"$\alpha={0:.4f}\pi,\beta={1:.4f}\pi$".format(alpha, beta)
ax.set_title(title, fontsize="xx-large")

ticks = np.array([-1, 0, 1])
ax.set_xticks(ticks * np.pi)
ax.set_yticks(ticks * np.pi)
ax.set_xticklabels(["{0}".format(tick) for tick in ticks])
ax.set_yticklabels(["{0}".format(tick) for tick in ticks])
ax.set_xlabel(r"$k_x/\pi$", fontsize="x-large")
ax.set_ylabel(r"$k_y/\pi$", fontsize="x-large")
ax.grid(True, ls="dashed", color="gray")
ax.set_aspect("equal")

plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
plt.close("all")

print("Program stop running")
