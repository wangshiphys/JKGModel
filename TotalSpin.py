"""
Calculate the total spin matrices of a system with N spins
"""


from pathlib import Path

from scipy.sparse import identity, kron, save_npz

import sys
import time

import numpy as np


SPIN_MATRICES = {
    "x": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
    # The real S^y matrix is np.array([[0.0, -0.5j], [0.5j, 0.0]])
    # The following matrix is (S^y / 1j), called it SY
    "y": np.array([[0.0, -0.5], [0.5, 0.0]], dtype=np.float64),
    "z": np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64),
}


def TotalSpin(spin_num, data_path="tmp/"):
    """
    Calculate the total spin matrices of a system with N spins

    Parameters
    spin_num : int
        The number of spin in the system
    data_path : str, optional
        Where to save the result
        The default value "tmp/" means tmp directory in the current working
        directory.
    """

    assert isinstance(spin_num, int) and spin_num > 0

    data_dir = Path(data_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    name_template = "Total_{0} with spin_number={1:02}.npz"
    sx_name = data_dir / name_template.format("SX", spin_num)
    sy_name = data_dir / name_template.format("SY", spin_num)
    sz_name = data_dir / name_template.format("SZ", spin_num)
    s2_name = data_dir / name_template.format("S2", spin_num)

    total_sx = total_sy = total_sz = 0.0
    for index in range(spin_num):
        I0 = identity(1 << index, np.float64, "csr")
        I1 = identity(1 << (spin_num-index-1), np.float64, "csr")
        total_sx += kron(I1, kron(SPIN_MATRICES["x"], I0, "csr"), "csr")
        total_sy += kron(I1, kron(SPIN_MATRICES["y"], I0, "csr"), "csr")
        total_sz += kron(I1, kron(SPIN_MATRICES["z"], I0, "csr"), "csr")
        del I0, I1

    save_npz(sx_name, total_sx, compressed=False)
    save_npz(sy_name, 1j * total_sy, compressed=False)
    save_npz(sz_name, total_sz, compressed=False)

    total_s2 = total_sx.dot(total_sx)
    del total_sx
    # Since the matrix used for calculation SY = (S^y / 1j) so
    # (S^y) * (S^y) = -SY * SY
    total_s2 -= total_sy.dot(total_sy)
    del total_sy
    total_s2 += total_sz.dot(total_sz)
    del total_sz
    save_npz(s2_name, total_s2, compressed=False)


if __name__ == "__main__":
    spin_num = int(sys.argv[1])
    t0 = time.time()
    TotalSpin(spin_num, data_path="tmp/")
    t1 = time.time()
    print("Time used: {0:.5f}s".format(t1 - t0))
