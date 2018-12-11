"""
Calculate the ground state averages of spin operators of the system
"""


from datetime import datetime
from pathlib import Path
from time import time

from scipy.sparse import identity, kron

import numpy as np


__all__ = [
    "TotalSpin",
    "GSAverages",
]


SX_MATRIX = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
# The real SY matrix is np.array([[0.0, -0.5j], [0.5j, 0.0]])
# The following  matrix is the imaginary part
SY_MATRIX_IMAG = np.array([[0.0, -0.5], [0.5, 0.0]], dtype=np.float64)
SZ_MATRIX = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64)


LOG_TEMPLATE = "{now:%Y-%m-%d %H:%M:%S} - {message} - Time Used: {dt:.3f}s"

gs_path_template = "data/SpinModel/GS/alpha={alpha:.3f}/"
avg_path_template = gs_path_template.replace("GS", "Averages")
gs_name_template = "GS_numx={numx}_numy={numy}_" \
                   "alpha={alpha:.3f}_beta={beta:.3f}.npz"
avg_name_template = gs_name_template.replace("GS", "Averages")


def TotalSpin(spin_num):
    """
    Calculate the total spin matrices of a system with N spins

    Parameters
    spin_num : int
        The number of spin in the system

    Returns
    -------
    total_sx : csr_matrix
        The total S^x matrix
    total_sy_imag : csr_matrix
        The imaginary part of the total S^y matrix
        Since the S^y matrix is pure imaginary, so total_sy = 1j * total_sy_imag
    total_sz : csr_matrix
        The total S^z matrix
    """

    assert isinstance(spin_num, int) and spin_num > 0

    total_sx = total_sy_imag = total_sz = 0.0
    for index in range(spin_num):
        I0 = identity(1 << index, np.float64, "csr")
        I1 = identity(1 << (spin_num - index - 1), np.float64, "csr")
        total_sx += kron(I1, kron(SX_MATRIX, I0, "csr"), "csr")
        total_sz += kron(I1, kron(SZ_MATRIX, I0, "csr"), "csr")
        total_sy_imag += kron(I1, kron(SY_MATRIX_IMAG, I0, "csr"), "csr")
        del I0, I1
    return total_sx, total_sy_imag, total_sz


def GSAverages(params, numx=4, numy=6):
    """
    Calculate the ground state averages of spin operators of the system

    Parameters
    ----------
    params : sequence
        A collection of (alpha, beta), which specifies the model parameter
    numx : int, optional
        The number of lattice site along the first translation vector
        default: 4
    numy : int, optional
        The number of lattice site along the second translation vector
        default: 6
    """

    t0 = time()
    total_sx, total_sy_imag, total_sz = TotalSpin(numx * numy)
    t1 = time()
    log = LOG_TEMPLATE.format(
        now=datetime.now(), message="Calculating Spin Matrices", dt=t1 - t0
    )
    print(log, flush=True)

    for alpha, beta in params:
        t0 = time()
        gs_full_name = (gs_path_template + gs_name_template).format(
            numx=numx, numy=numy, alpha=alpha, beta=beta
        )
        with np.load(gs_full_name) as ld:
            ket = ld["ket"]

        total_sx_dot_ket = total_sx.dot(ket)
        total_sy_imag_dot_ket = total_sy_imag.dot(ket)
        total_sz_dot_ket = total_sz.dot(ket)

        total_sx_avg = np.vdot(ket, total_sx_dot_ket)
        total_sz_avg = np.vdot(ket, total_sz_dot_ket)
        total_sy_avg = 1j * np.vdot(ket, total_sy_imag_dot_ket)
        del ket

        total_s2_avg = np.vdot(total_sy_imag_dot_ket, total_sy_imag_dot_ket)
        del total_sy_imag_dot_ket
        total_s2_avg += np.vdot(total_sx_dot_ket, total_sx_dot_ket)
        del total_sx_dot_ket
        total_s2_avg += np.vdot(total_sz_dot_ket, total_sz_dot_ket)
        del total_sz_dot_ket

        avg_path = Path(avg_path_template.format(alpha=alpha))
        avg_path.mkdir(parents=True, exist_ok=True)
        avg_full_name = avg_path / avg_name_template.format(
            numx=numx, numy=numy, alpha=alpha, beta=beta
        )
        np.savez(
            avg_full_name,
            total_sx_avg=[total_sx_avg], total_sy_avg=[total_sy_avg],
            total_sz_avg=[total_sz_avg], total_s2_avg=[total_s2_avg],
        )
        t1 = time()

        log = LOG_TEMPLATE.format(
            now=datetime.now(), dt=t1 - t0,
            message="alpha={0:.3f}, beta={1:.3f} done".format(alpha, beta)
        )
        print(log, flush=True)
