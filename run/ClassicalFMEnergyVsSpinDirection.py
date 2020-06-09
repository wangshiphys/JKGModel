"""
Energy of the triangular lattice J-K-Gamma model for classical FM states.
"""


import matplotlib.pyplot as plt
import numpy as np


def FMEnergies(thetas, phis, J=-1.0, K=0.0, G=0.0):
    """
    Energy of the triangular lattice J-K-Gamma model for classical FM states.

    Parameters
    ----------
    thetas, phis : 1D array
        `thetas` and `phis` specify a mesh of ordered moment directions.
        `theta` is the angle between the ordered moment direction and the
        z-axis; `phi` is the angle between the projection of the ordered
        moment in the xy-plane and the x-axis.
    J, K, G : float, optional
        The coefficients of the Heisenberg, Kitaev and Gamma interactions.

    Returns
    -------
    energies : 2D array with shape (N0, N1)
        The energies for FM states with different ordered moment directions.
        The 1st and 2nd dimension correspond to different thetas and phis
        respectively.
    """

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
            energies[i, j] = 3 * J + K + 2 * G * (sx * sy + sx * sz + sy * sz)
    return energies


if __name__ == "__main__":
    alpha = 0.0
    beta = 1.3
    J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
    K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
    G = np.cos(alpha * np.pi)

    step = 0.005
    thetas = np.arange(0, 1 + step, step)
    phis = np.arange(-1, 1 + step, step)

    energies = FMEnergies(thetas * np.pi, phis * np.pi, J, K, G)

    fig, ax = plt.subplots()
    cs = ax.pcolormesh(phis, thetas, energies, cmap="hot", shading="gouraud")
    fig.colorbar(cs, ax=ax)
    ax.plot(
        np.arctan2(-1, -1) / np.pi, np.arctan2(np.sqrt(2), -1) / np.pi,
        marker="o", ms=10, color="tab:blue", alpha=0.75,
    )
    ax.plot(
        np.arctan2(1, 1) / np.pi, np.arctan2(np.sqrt(2), 1) / np.pi,
        marker="o", ms=10, color="tab:blue", alpha=0.75,
    )
    tmp = np.arctan2(1, -np.sin(phis * np.pi) - np.cos(phis * np.pi)) / np.pi
    ax.plot(phis, tmp, lw=3, color="tab:green", alpha=0.75)
    title = r"$\alpha={0:.3f}\pi, \beta={1:.3f}\pi$".format(alpha, beta)
    ax.set_title(title, fontsize="xx-large")
    ax.set_xlabel(r"$\phi/\pi$", fontsize="xx-large")
    ax.set_ylabel(r"$\theta/\pi$", fontsize="xx-large")
    ax.grid(True, ls="dashed", color="gray")
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()
    plt.close("all")
