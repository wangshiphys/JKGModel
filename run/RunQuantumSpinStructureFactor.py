import logging
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import lattice_generator, TRIANGLE_CELL_KS

from StructureFactor import QuantumSpinStructureFactor


logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logging.info("Program start running")
LOG_MSG = "SFs for alpha=%.4f, beta=%.4f: dt=%.4fs"

# GS_DATA_PATH = "data/QuantumSpinModel/GS/"
GS_DATA_PATH = "E:/JKGModel/data/QuantumSpinModel/GS/"
# GS_DATA_PATH = "C:/Users/swang/Desktop/Eigenstates/"
GS_DATA_NAME_TEMP = "GS_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"
SF_DATA_NAME_TEMP = "SF_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"
################################################################################

# Prepare k-points and boundary of first Brillouin Zone
numx = 4
numy = 6
cluster = lattice_generator("triangle", num0=numx, num1=numy)
bs = 4 * np.pi * np.identity(2) / np.sqrt(3)
step = 0.01
kx_ratios = np.arange(-0.7, 0.7 + step, step)
ky_ratios = np.arange(-0.7 + step, 0.7, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(kx_ratios, ky_ratios, indexing="ij"), axis=-1), bs
)
BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]

# Load ground state data
alpha = 0.70
beta = 0.66
gs_data_name = GS_DATA_NAME_TEMP.format(numx, numy, alpha, beta)
with np.load(GS_DATA_PATH + gs_data_name) as ld:
    ket = ld["ket"][:, 0]

# Calculate and save static structure factors
t0 = time()
factors = QuantumSpinStructureFactor(kpoints, cluster.points, ket)
assert np.all(np.abs(factors.imag) < 1E-12)
sf_data_name = SF_DATA_NAME_TEMP.format(numx, numy, alpha, beta)
np.savez(
    sf_data_name, size=[numx, numy], parameters=[alpha, beta],
    kpoints=kpoints, BZBoundary=BZBoundary, factors=factors.real
)
t1 = time()
logging.info(LOG_MSG, alpha, beta, t1 - t0)

# Plot the static structure factors
fig, ax = plt.subplots(num=sf_data_name[:-4])
im = ax.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], factors.real, zorder=0,
    cmap="Reds", shading="gouraud",
)
fig.colorbar(im, ax=ax)
ax.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=3, ls="dashed", color="tab:blue", alpha=1.0,
)
ax.set_title(
    r"$\alpha={0:.4f}\pi,\beta={1:.4f}\pi$".format(alpha, beta),
    fontsize="xx-large",
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

logging.info("Program stop running")
