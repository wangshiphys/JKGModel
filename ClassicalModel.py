"""
Analysing the classical J-K-Gamma-Gamma' (J-K-G-GP) model.

The spins are viewed as unit-vectors in 3-dimension.
"""


__all__ = [
    "HMatrixGenerator", "ClusterGenerator", "EnergyCore0", "EnergyCore1"
]


import HamiltonianPy as HP
import numpy as np


def HMatrixGenerator(alpha, beta, GP=0.0):
    """
    Generate the spin interaction matrices for x-, y-, z-type bonds.

    For J-K-Gamma-Gamma' (J-K-G-GP) model on triangular lattice, the spin
    interaction term on a nearest-neighbor bond is:
        (S_i^x, S_i^y, S_i^z) HM (S_j^x, S_j^y, S_j^z)^T
    This function returns HMs for x-, y-, z-type bonds.

    Parameters
    ----------
    alpha, beta : float
        Model parameters.
        J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
        K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
        G = np.cos(alpha * np.pi)
        J is the coefficient of the Heisenberg term;
        K is the coefficient of the Kitaev term;
        G is the coefficient of the Gamma term.
    GP : float, optional
        The coefficient of the Gamma' term.
        Default: 0.0.

    Returns
    -------
    Hx, Hy, Hz : array with shape (3, 3)
        HMs for x-, y-, z-type bonds, respectively.
    """

    G = np.cos(alpha * np.pi)
    J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
    K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
    Hx = np.array([[J + K, GP, GP], [GP, J, G], [GP, G, J]], dtype=np.float64)
    Hy = np.array([[J, GP, G], [GP, J + K, GP], [G, GP, J]], dtype=np.float64)
    Hz = np.array([[J, G, GP], [G, J, GP], [GP, GP, J + K]], dtype=np.float64)
    return Hx, Hy, Hz


def ClusterGenerator(num0, num1=None, cell=None):
    if num1 is None:
        num1 = num0

    if cell is None:
        cluster = cell = HP.lattice_generator("triangle", num0=num0, num1=num1)
    else:
        cluster_points = [
            cell.points + np.dot([i, j], cell.vectors)
            for i in range(num0) for j in range(num1)
        ]
        cluster_points = np.concatenate(cluster_points)
        cluster_vectors = cell.vectors * np.array([[num0], [num1]])
        cluster = HP.Lattice(cluster_points, cluster_vectors)

    category = {key: list() for key in range(cell.point_num)}
    for point in cluster.points:
        cell_index = cell.getIndex(point, fold=True)
        cluster_index = cluster.getIndex(point, fold=False)
        assert cell_index in category
        category[cell_index].append(cluster_index)

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
        if azimuth in (-180, 0, 180):
            x_bonds.append(bond_index)
        elif azimuth in (-120, 60):
            z_bonds.append(bond_index)
        elif azimuth in (-60, 120):
            y_bonds.append(bond_index)
        else:
            raise RuntimeError("Invalid bond azimuth: {0}.".format(azimuth))
    return cluster, category, x_bonds, y_bonds, z_bonds


def EnergyCore0(
        spin_angles, site_num, category,
        x_bonds, y_bonds, z_bonds, Hx, Hy, Hz
):
    phis = spin_angles[0::2]
    thetas = spin_angles[1::2]
    sin_phis = np.sin(phis)
    cos_phis = np.cos(phis)
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)
    cell_vectors = np.array(
        [sin_phis * cos_thetas, sin_phis * sin_thetas, cos_phis]
    )
    vectors = np.empty((site_num, 3), dtype=np.float64)
    for cell_index, cluster_indices in category.items():
        vectors[cluster_indices] = cell_vectors[:, cell_index]

    energy = 0.0
    for bonds, H in [(x_bonds, Hx), (y_bonds, Hy), (z_bonds, Hz)]:
        tmp = vectors[bonds, :]
        energy += np.sum(tmp[:, 0, :] * np.dot(tmp[:, 1, :], H))
    return energy


def EnergyCore1(spin_angles, x_bonds, y_bonds, z_bonds, Hx, Hy, Hz):
    phis = spin_angles[0::2]
    thetas = spin_angles[1::2]
    sin_phis = np.sin(phis)
    cos_phis = np.cos(phis)
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)
    vectors = np.array([sin_phis * cos_thetas, sin_phis * sin_thetas, cos_phis])

    energy = 0.0
    for bonds, H in [(x_bonds, Hx), (y_bonds, Hy), (z_bonds, Hz)]:
        tmp = vectors[:, bonds]
        energy += np.sum(tmp[:, :, 0] * np.dot(H, tmp[:, :, 1]))
    return energy
