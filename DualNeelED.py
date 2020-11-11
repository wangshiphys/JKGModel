"""
Exact diagonalization of the J-K-Gamma model on a 24-sites cluster that is
compatible with 120 Neel order.
"""


import logging
import sys
from pathlib import Path
from time import time

import HamiltonianPy as HP
import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import eigsh


def BondsGenerator(cluster):
    x_bonds = []
    y_bonds = []
    z_bonds = []
    intra, inter = cluster.bonds(nth=1)
    for bond in intra + inter:
        p0, p1 = bond.endpoints
        index0 = cluster.getIndex(site=p0, fold=True)
        index1 = cluster.getIndex(site=p1, fold=True)
        bond_index = (index0, index1)

        azimuth = bond.getAzimuth(ndigits=0)
        # The definition of x, y, z bond in a trio is counterclockwise.
        if azimuth in (-180, 0, 180):
            x_bonds.append(bond_index)
        elif azimuth in (-120, 60):
            z_bonds.append(bond_index)
        elif azimuth in (-60, 120):
            y_bonds.append(bond_index)
        else:
            raise RuntimeError("Invalid bond azimuth: {0}".format(azimuth))
    return x_bonds, y_bonds, z_bonds


def TermMatrix(cluster):
    directory = Path("tmp/")
    name_template = "TRIANGLE_DualNeel_H{0}.npz"
    file_HJ = directory / name_template.format("J")
    file_HK = directory / name_template.format("K")
    file_HG = directory / name_template.format("G")

    if file_HJ.exists() and file_HK.exists() and file_HG.exists():
        t0 = time()
        HJ = load_npz(file_HJ)
        HK = load_npz(file_HK)
        HG = load_npz(file_HG)
        t1 = time()
        logging.info("Load HJ, HK, HG, dt=%.3fs", t1 - t0)
    else:
        site_num = cluster.point_num
        all_bonds = BondsGenerator(cluster)

        HJ = HK = HG = 0.0
        msg = "%s-bond: %2d/%2d, dt=%.3fs"
        m_func = HP.SpinInteraction.matrix_function
        configs = (("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"))
        for (gamma, alpha, beta), bonds in zip(configs, all_bonds):
            bond_num = len(bonds)
            for count, (index0, index1) in enumerate(bonds, start=1):
                t0 = time()
                SKM = m_func([(index0, gamma), (index1, gamma)], site_num)
                # Kitaev term
                HK += SKM
                # Heisenberg term
                HJ += SKM
                HJ += m_func([(index0, alpha), (index1, alpha)], site_num)
                HJ += m_func([(index0, beta), (index1, beta)], site_num)
                # Gamma term
                HG += m_func([(index0, alpha), (index1, beta)], site_num)
                HG += m_func([(index0, beta), (index1, alpha)], site_num)
                t1 = time()
                logging.info(msg, gamma, count, bond_num, t1 - t0)
        directory.mkdir(parents=True, exist_ok=True)
        save_npz(file_HJ, HJ, compressed=True)
        save_npz(file_HK, HK, compressed=True)
        save_npz(file_HG, HG, compressed=True)
    return HJ, HK, HG


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        stream=sys.stdout, level=logging.INFO,
    )

    points = np.array(
        [
            [0.0, 0.0], [0.5, 0.5 * np.sqrt(3)],
            [1.0, 0.0], [1.5, 0.5 * np.sqrt(3)],
            [2.0, 0.0], [2.5, 0.5 * np.sqrt(3)],
            [3.0, 0.0], [3.5, 0.5 * np.sqrt(3)],
            [4.0, 0.0], [4.5, 0.5 * np.sqrt(3)],
            [5.0, 0.0], [5.5, 0.5 * np.sqrt(3)],
            [3.0, np.sqrt(3)], [3.5, 1.5 * np.sqrt(3)],
            [4.0, np.sqrt(3)], [4.5, 1.5 * np.sqrt(3)],
            [5.0, np.sqrt(3)], [5.5, 1.5 * np.sqrt(3)],
            [6.0, np.sqrt(3)], [6.5, 1.5 * np.sqrt(3)],
            [7.0, np.sqrt(3)], [7.5, 1.5 * np.sqrt(3)],
            [8.0, np.sqrt(3)], [8.5, 1.5 * np.sqrt(3)],
        ]
    )
    vectors = np.array([[6.0, 0.0], [6.0, 2 * np.sqrt(3)]])
    cluster = HP.Lattice(points, vectors)
    cluster.show(scope=1)

    alpha = 0.5
    beta = 1.85
    J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
    K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
    G = np.cos(alpha * np.pi)
    J = 0.0 if np.abs(J) <= 1E-10 else J
    K = 0.0 if np.abs(K) <= 1E-10 else K
    G = 0.0 if np.abs(G) <= 1E-10 else G

    HJ, HK, HG = TermMatrix(cluster)
    HJ *= J
    HK *= K
    HG *= G
    HM = HJ + HK
    del HJ, HK
    HM += HG
    del HG

    t0 = time()
    values, vectors = eigsh(HM, k=1, which="SA")
    t1 = time()
    msg = "ES for alpha=%.4f, beta=%.4f, dt=%.3fs"
    logging.info(msg, alpha, beta, t1 - t0)

    np.savez_compressed(
        "ES_DualNeel_alpha={0:.4f}_beta={1:.4f}.npz".format(alpha, beta),
        parameters=[alpha, beta], values=values, vectors=vectors,
    )
