import logging
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np

from utilities import mykron0


def ProbabilitiesOnFM24(thetas, phis, ket):
    num_thetas = thetas.shape[0]
    num_phis = phis.shape[0]
    sin_half_thetas = np.sin(thetas / 2)
    cos_half_thetas = np.cos(thetas / 2)
    exp_1j_half_phis = np.exp(1j * phis / 2)
    exp_1j_half_phis_conj = exp_1j_half_phis.conjugate()
    probabilities = np.zeros((num_thetas, num_phis), dtype=np.float64)

    spinor2 = np.zeros(2 ** 2, dtype=np.complex128)
    spinor4 = np.zeros(2 ** 4, dtype=np.complex128)
    spinor8 = np.zeros(2 ** 8, dtype=np.complex128)
    spinor16 = np.zeros(2 ** 16, dtype=np.complex128)
    spinor24 = np.zeros(2 ** 24, dtype=np.complex128)
    for i in range(num_thetas):
        t0 = time()
        for j in range(num_phis):
            spinor1 = np.array(
                [
                    cos_half_thetas[i] * exp_1j_half_phis_conj[j],
                    sin_half_thetas[i] * exp_1j_half_phis[j],
                ], dtype=np.complex128,
            )

            mykron0(spinor1, spinor1, spinor2)
            mykron0(spinor2, spinor2, spinor4)
            mykron0(spinor4, spinor4, spinor8)
            mykron0(spinor8, spinor8, spinor16)
            mykron0(spinor8, spinor16, spinor24)

            inner = np.vdot(spinor24, ket)
            probabilities[i, j] = (inner * inner.conjugate()).real
        t1 = time()
        logging.info("%03dth theta, dt=%.4fs", i, t1 - t0)
    return probabilities


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    logging.info("Program start running")

    # GS_DATA_PATH = "data/QuantumSpinModel/GS/"
    GS_DATA_PATH = "E:/JKGModel/data/QuantumSpinModel/GS/"
    # GS_DATA_PATH = "C:/Users/swang/Desktop/Eigenstates/"
    PS_DATA_PATH = ""
    GS_DATA_NAME_TEMP = "GS_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"
    PS_DATA_NAME_TEMP = "PS_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"
    ############################################################################

    # Prepare thetas and phis
    step = 0.01
    thetas = np.pi * np.arange(0.0, 1.0 + step, step)
    phis = np.pi * np.arange(0.0, 2.0 + step, step)

    numx = 4
    numy = 6
    alpha = 0.35
    beta  = 1.63
    gs_data_name = GS_DATA_NAME_TEMP.format(numx, numy, alpha, beta)
    ps_data_name = PS_DATA_NAME_TEMP.format(numx, numy, alpha, beta)

    # Load ground state data
    with np.load(GS_DATA_PATH + gs_data_name) as ld:
        ket = ld["ket"][:, 0]

    # Calculate and save probabilities data
    t0 = time()
    probabilities = ProbabilitiesOnFM24(thetas, phis, ket)
    np.savez(
        PS_DATA_PATH + ps_data_name,
        size=[numx, numy], parameters=[alpha, beta],
        thetas=thetas, phis=phis, probabilities=probabilities,
        )
    t1 = time()
    logging.info(
        "The total time spend on alpha=%.3f, beta=%.3f: %.4fs",
        alpha, beta, t1 - t0
    )

    # Load overlaps data
    with np.load(PS_DATA_PATH + ps_data_name) as ld:
        probabilities = ld["probabilities"]

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    cs = ax.pcolormesh(
        phis, thetas, probabilities, zorder=0, cmap="magma", shading="gouraud",
    )
    colorbar = fig.colorbar(cs, ax=ax)

    ax.plot(phis, np.arctan2(1, -np.sin(phis) - np.cos(phis)), ls="dashed")
    ax.plot(np.arctan2(1, 1), np.arctan2(np.sqrt(2), 1), ls="", marker="o")

    ax.set_aspect("equal")
    ax.grid(True, ls="dashed", color="gray")
    ax.set_title(r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(alpha, beta))
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")

    logging.info("Program stop running")
