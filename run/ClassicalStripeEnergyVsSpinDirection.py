"""
Energy of the triangular lattice J-K-Gamma model for classical stripe states.
"""

import matplotlib.pyplot as plt
import numpy as np


# Spins along the x-bond direction are parallel and along y-bond and z-bond
# direction are anti-parallel.
def _StripeXEnergy(sx, sy, sz, J, K, G):
    return -J - K + 2 * K * sx * sx + 2 * G * (sy * sz - sx * sy - sx * sz)


# Spins along the y-bond direction are parallel and along x-bond and z-bond
# direction are anti-parallel.
def _StripeYEnergy(sx, sy, sz, J, K, G):
    return -J - K + 2 * K * sy * sy + 2 * G * (sx * sz - sx * sy - sy * sz)


# Spins along the z-bond direction are parallel and along x-bond and y-bond
# direction are anti-parallel.
def _StripeZEnergy(sx, sy, sz, J, K, G):
    return -J - K + 2 * K * sz * sz + 2 * G * (sx * sy - sx * sz - sy * sz)


def ExtremePoint(K, G, config="StripeX"):
    """
    Find the moment direction of the classical stripe state when the energy
    of the triangular lattice J-K-Gamma model takes extreme values.

    Parameters
    ----------
    K, G : float
        The coefficients of the Kitaev and Gamma interactions.
    config : str, optional
        The configuration of the stripe states in real space.
        If set to "StripeX", the spins along the x-bond direction are parallel
        and along the y-bond and z-bond direction are anti-parallel;
        If set to "StripeY", the spins along the y-bond direction are parallel
        and along the x-bond and z-bond direction are anti-parallel;
        If set to "StripeZ", the spins along the z-bond direction are parallel
        and along the x-bond and y-bond direction are anti-parallel.
        Default: "StripeX".

    Returns
    -------
    p0 : 5-tuple, (sx0, sy0, sz0, theta0, phi0)
    p1 : 5-tuple, (sx1, sy1, sz1, theta1, phi1)
    p2 : 5-tuple, (sx2, sy2, sz2, theta2, phi2)
        `sx`, `sy`, `sz` are the x, y, z components of the ordered moment;
        `theta` is the angle between the ordered moment direction and the
        z-axis; `phi` is the angle between the projection of the ordered
        moment in the xy-plane and the x-axis.
    """

    tmp = np.sqrt(9 * G * G - 4 * K * G + 4 * K * K)
    lambda_b = -(G + 2 * K - tmp) / 4
    lambda_c = -(G + 2 * K + tmp) / 4

    x_a = 0.0
    y_a = -1.0 / np.sqrt(2)
    z_a = 1.0 / np.sqrt(2)

    y_b = z_b = np.sqrt(G * G / (4 * lambda_b * (lambda_b + G) + 3 * G * G))
    x_b = y_b * (2 * lambda_b + G) / G

    y_c = z_c = np.sqrt(G * G / (4 * lambda_c * (lambda_c + G) + 3 * G * G))
    x_c = y_c * (2 * lambda_c + G) / G

    if config == "StripeX":
        x0, y0, z0 = x_a, y_a, z_a
        x1, y1, z1 = x_b, y_b, z_b
        x2, y2, z2 = x_c, y_c, z_c
    elif config == "StripeY":
        x0, y0, z0 = z_a, x_a, y_a
        x1, y1, z1 = z_b, x_b, y_b
        x2, y2, z2 = z_c, x_c, y_c
    elif config == "StripeZ":
        x0, y0, z0 = y_a, z_a, x_a
        x1, y1, z1 = y_b, z_b, x_b
        x2, y2, z2 = y_c, z_c, x_c
    else:
        raise ValueError("Invalid `config`: {0}".format(config))

    phi0 = np.arctan2(y0, x0)
    phi1 = np.arctan2(y1, x1)
    phi2 = np.arctan2(y2, x2)
    theta0 = np.arctan2(np.sqrt(x0 * x0 + y0 * y0), z0)
    theta1 = np.arctan2(np.sqrt(x1 * x1 + y1 * y1), z1)
    theta2 = np.arctan2(np.sqrt(x2 * x2 + y2 * y2), z2)
    extreme_point0 = (x0, y0, z0, theta0, phi0)
    extreme_point1 = (x1, y1, z1, theta1, phi1)
    extreme_point2 = (x2, y2, z2, theta2, phi2)
    return extreme_point0, extreme_point1, extreme_point2


def StripeEnergies(thetas, phis, J=2.0, K=-1.0, G=0.0, config="StripeX"):
    """
    Energy of the triangular lattice J-K-Gamma model for classical stripe
    states.

    Parameters
    ----------
    thetas, phis : 1D array
        `thetas` and `phis` specify a mesh of ordered moment directions.
        `theta` is the angle between the ordered moment direction and the
        z-axis; `phi` is the angle between the projection of the ordered
        moment in the xy-plane and the x-axis.
    J, K, G : float, optional
        The coefficients of the Heisenberg, Kitaev and Gamma interactions.
    config : str, optional
        The configuration of the stripe states in real space.
        If set to "StripeX", the spins along the x-bond direction are parallel
        and along the y-bond and z-bond direction are anti-parallel;
        If set to "StripeY", the spins along the y-bond direction are parallel
        and along the x-bond and z-bond direction are anti-parallel;
        If set to "StripeZ", the spins along the z-bond direction are parallel
        and along the x-bond and y-bond direction are anti-parallel.
        Default: "StripeX".

    Returns
    -------
    energies : 2D array with shape (N0, N1)
        The energies for stripe states with different ordered moment directions.
        The 1st and 2nd dimension correspond to different thetas and phis
        respectively.
    """

    if config == "StripeX":
        core = _StripeXEnergy
    elif config == "StripeY":
        core = _StripeYEnergy
    elif config == "StripeZ":
        core = _StripeZEnergy
    else:
        raise ValueError("Invalid `config`: {0}".format(config))

    sin_phis = np.sin(phis)
    cos_phis = np.cos(phis)
    num_phis = phis.shape[0]
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)
    num_thetas = thetas.shape[0]
    energies = np.empty((num_thetas, num_phis), dtype=np.float64)
    for i in range(num_thetas):
        for j in range(num_phis):
            sx = sin_thetas[i] * cos_phis[j]
            sy = sin_thetas[i] * sin_phis[j]
            sz = cos_thetas[i]
            energies[i, j] = core(sx, sy, sz, J, K, G)
    return energies


if __name__ == "__main__":
    alpha = 0.30
    beta  = 0.80
    J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
    K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
    G = np.cos(alpha * np.pi)

    step = 0.005
    thetas = np.arange(0, 1 + step, step)
    phis = np.arange(-1, 1 + step, step)

    config = "StripeX"
    point0, point1, point2 = ExtremePoint(K, G, config=config)
    energies = StripeEnergies(
        thetas * np.pi, phis * np.pi, J, K, G, config=config
    )

    fig, ax = plt.subplots()
    cs = ax.pcolormesh(phis, thetas, energies, cmap="hot", shading="gouraud")
    fig.colorbar(cs, ax=ax)
    circle, = ax.plot(
        point0[-1] / np.pi, point0[-2] / np.pi,
        ls="", marker="o", ms=10, color="tab:green",
    )
    square, = ax.plot(
        point1[-1] / np.pi, point1[-2] / np.pi,
        ls="", marker="s", ms=10, color="tab:green",
    )
    triangle, = ax.plot(
        point2[-1] / np.pi, point2[-2] / np.pi,
        ls="", marker="<", ms=10, color="tab:green",
    )
    ax.legend([circle, square, triangle], ["1st", "2nd", "3rd"], loc="best")
    title = r"$\alpha={0:.3f}\pi, \beta={1:.3f}\pi$".format(alpha, beta)
    ax.set_title(title, fontsize="xx-large")
    ax.set_xlabel(r"$\phi/\pi$", fontsize="xx-large")
    ax.set_ylabel(r"$\theta/\pi$", fontsize="xx-large")
    ax.grid(True, ls="dashed", color="gray")
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()
    plt.close("all")
