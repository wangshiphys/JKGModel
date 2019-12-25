import logging
import sys

from time import time

import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from HamiltonianPy import lattice_generator, TRIANGLE_CELL_KS

from StructureFactor import ClassicalSpinStructureFactor


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

data_path = "data/ClassicalSpinModel/OptimizedSpinConfig/"
data_name_temp = "OSC_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.h5"

alpha = 0.05
beta = 0.05
numx = 12
numy = 12
cluster = lattice_generator("triangle", num0=numx, num1=numy)
data_full_name = data_path + data_name_temp.format(numx, numy, alpha, beta)
h5_file = tb.open_file(data_full_name, mode="r")
for carray in h5_file.iter_nodes("/"):
    vectors = carray.read()
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
h5_file.close()
logging.info("Program start running")
