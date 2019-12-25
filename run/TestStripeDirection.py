import logging
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from utilities import mykron0


def ProbabilitiesOnStripe24(thetas, phis, ket, which="x"):
    num_thetas = thetas.shape[0]
    num_phis = phis.shape[0]
    sin_half_thetas_up = np.sin(thetas / 2)
    cos_half_thetas_up = np.cos(thetas / 2)
    exp_1j_half_phis_up = np.exp(1j * phis / 2)
    exp_1j_half_phis_conj_up = exp_1j_half_phis_up.conjugate()

    sin_half_thetas_down = np.sin((np.pi - thetas) / 2)
    cos_half_thetas_down = np.cos((np.pi - thetas) / 2)
    exp_1j_half_phis_down = np.exp(1j * (phis + np.pi) / 2)
    exp_1j_half_phis_conj_down = exp_1j_half_phis_down.conjugate()

    probabilities = np.zeros((num_thetas, num_phis), dtype=np.float64)
    if which == "x":
        spinor2 = np.zeros(2 ** 2, dtype=np.complex128)
        spinor4 = np.zeros(2 ** 4, dtype=np.complex128)
        spinor8 = np.zeros(2 ** 8, dtype=np.complex128)
        spinor16 = np.zeros(2 ** 16, dtype=np.complex128)
        spinor24 = np.zeros(2 ** 24, dtype=np.complex128)
        for i in range(num_thetas):
            t0 = time()
            for j in range(num_phis):
                spinor1_up = np.array(
                    [
                        cos_half_thetas_up[i] * exp_1j_half_phis_conj_up[j],
                        sin_half_thetas_up[i] * exp_1j_half_phis_up[j],
                    ], dtype=np.complex128,
                )
                spinor1_down = np.array(
                    [
                        cos_half_thetas_down[i] * exp_1j_half_phis_conj_down[j],
                        sin_half_thetas_down[i] * exp_1j_half_phis_down[j],
                    ], dtype=np.complex128,
                )
                mykron0(spinor1_down, spinor1_up, spinor2)
                mykron0(spinor2, spinor2, spinor4)
                mykron0(spinor4, spinor4, spinor8)
                mykron0(spinor8, spinor8, spinor16)
                mykron0(spinor16, spinor8, spinor24)

                inner = np.vdot(spinor24, ket)
                probabilities[i, j] = (inner * inner.conjugate()).real
            t1 = time()
            logging.info("%03dth theta, dt=%.4fs", i, t1 - t0)
    elif which == "y":
        spinor2_down_up = np.zeros(2 ** 2, dtype=np.complex128)
        spinor4_down_up = np.zeros(2 ** 4, dtype=np.complex128)
        spinor6_down_up = np.zeros(2 ** 6, dtype=np.complex128)
        spinor2_up_down = np.zeros(2 ** 2, dtype=np.complex128)
        spinor4_up_down = np.zeros(2 ** 4, dtype=np.complex128)
        spinor6_up_down = np.zeros(2 ** 6, dtype=np.complex128)
        spinor12 = np.zeros(2 ** 12, dtype=np.complex128)
        spinor24 = np.zeros(2 ** 24, dtype=np.complex128)
        for i in range(num_thetas):
            t0 = time()
            for j in range(num_phis):
                spinor1_up = np.array(
                    [
                        cos_half_thetas_up[i] * exp_1j_half_phis_conj_up[j],
                        sin_half_thetas_up[i] * exp_1j_half_phis_up[j],
                    ], dtype=np.complex128,
                )
                spinor1_down = np.array(
                    [
                        cos_half_thetas_down[i] * exp_1j_half_phis_conj_down[j],
                        sin_half_thetas_down[i] * exp_1j_half_phis_down[j],
                    ], dtype=np.complex128,
                )

                mykron0(spinor1_down, spinor1_up, spinor2_down_up)
                mykron0(spinor2_down_up, spinor2_down_up, spinor4_down_up)
                mykron0(spinor4_down_up, spinor2_down_up, spinor6_down_up)
                mykron0(spinor1_up, spinor1_down, spinor2_up_down)
                mykron0(spinor2_up_down, spinor2_up_down, spinor4_up_down)
                mykron0(spinor4_up_down, spinor2_up_down, spinor6_up_down)
                mykron0(spinor6_down_up, spinor6_up_down, spinor12)
                mykron0(spinor12, spinor12, spinor24)

                inner = np.vdot(spinor24, ket)
                probabilities[i, j] = (inner * inner.conjugate()).real
            t1 = time()
            logging.info("%03dth theta, dt=%.4fs", i, t1 - t0)
    elif which == "z":
        spinor2_up = np.zeros(2 ** 2, dtype=np.complex128)
        spinor4_up = np.zeros(2 ** 4, dtype=np.complex128)
        spinor6_up = np.zeros(2 ** 6, dtype=np.complex128)
        spinor2_down = np.zeros(2 ** 2, dtype=np.complex128)
        spinor4_down = np.zeros(2 ** 4, dtype=np.complex128)
        spinor6_down = np.zeros(2 ** 6, dtype=np.complex128)
        spinor12 = np.zeros(2 ** 12, dtype=np.complex128)
        spinor24 = np.zeros(2 ** 24, dtype=np.complex128)
        for i in range(num_thetas):
            t0 = time()
            for j in range(num_phis):
                spinor1_up = np.array(
                    [
                        cos_half_thetas_up[i] * exp_1j_half_phis_conj_up[j],
                        sin_half_thetas_up[i] * exp_1j_half_phis_up[j],
                    ], dtype=np.complex128,
                )
                spinor1_down = np.array(
                    [
                        cos_half_thetas_down[i] * exp_1j_half_phis_conj_down[j],
                        sin_half_thetas_down[i] * exp_1j_half_phis_down[j],
                    ], dtype=np.complex128,
                )

                mykron0(spinor1_up, spinor1_up, spinor2_up)
                mykron0(spinor2_up, spinor2_up, spinor4_up)
                mykron0(spinor4_up, spinor2_up, spinor6_up)
                mykron0(spinor1_down, spinor1_down, spinor2_down)
                mykron0(spinor2_down, spinor2_down, spinor4_down)
                mykron0(spinor4_down, spinor2_down, spinor6_down)
                mykron0(spinor6_down, spinor6_up, spinor12)
                mykron0(spinor12, spinor12, spinor24)

                inner = np.vdot(spinor24, ket)
                probabilities[i, j] = (inner * inner.conjugate()).real
            t1 = time()
            logging.info("%03dth theta, dt=%.4fs", i, t1 - t0)
    else:
        raise ValueError("Invalid `which` parameter: {0}".format(which))
    return probabilities


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s - %(message)s",
    )
    logging.info("Program start running")


    PS_DATA_PATH = ""
    # GS_DATA_PATH = "C:/Users/swang/Desktop/Eigenstates/"
    GS_DATA_PATH = "E:/JKGModel/data/QuantumSpinModel/GS/"
    GS_DATA_NAME_TEMP = "GS_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"
    PS_DATA_NAME_TEMP = "PS_numx={0}_numy={1}_" \
                        "alpha={2:.4f}_beta={3:.4f}_which={4}.npz"
    ############################################################################

    # Prepare thetas and phis
    step = 0.01
    thetas = np.pi * np.arange(0.0, 1.0 + step, step)
    phis = np.pi * np.arange(0.0, 2.0 + step, step)

    numx = 4
    numy = 6
    alpha = 0.30
    beta  = 0.27
    which = "x"
    gs_data_name = GS_DATA_NAME_TEMP.format(numx, numy, alpha, beta)
    ps_data_name = PS_DATA_NAME_TEMP.format(numx, numy, alpha, beta, which)

    # Load ground state data
    with np.load(GS_DATA_PATH + gs_data_name) as ld:
        ket = ld["ket"][:, 0]

    # Calculate and save probabilities data
    t0 = time()
    probabilities = ProbabilitiesOnStripe24(thetas, phis, ket, which=which)
    np.savez(
        PS_DATA_PATH + ps_data_name,
        size=[numx, numy], parameters=[alpha, beta], which=[which],
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

    ax.set_aspect("equal")
    ax.grid(True, ls="dashed", color="gray")
    ax.set_title(r"$\alpha={0:.2f}\pi,\beta={1:.2f}\pi$".format(alpha, beta))
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")

    logging.info("Program stop running")
