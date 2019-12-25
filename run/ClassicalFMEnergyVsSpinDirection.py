import matplotlib.pyplot as plt
import numpy as np


def FMEnergies(thetas, phis, J=-1.0, K=0.0, G=0.0):
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)
    sin_phis = np.sin(phis)
    cos_phis = np.cos(phis)
    num_thetas = thetas.shape[0]
    num_phis = phis.shape[0]
    energies = np.zeros((num_thetas, num_phis), dtype=np.float64)
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
    fig.colorbar(cs, ax=ax, shrink=0.85)
    ax.plot(
        np.arctan2(-1, -1) / np.pi, np.arctan2(np.sqrt(2), -1) / np.pi,
        ls="", marker="o", color="tab:blue", alpha=0.75,
    )
    ax.plot(
        np.arctan2(1, 1) / np.pi, np.arctan2(np.sqrt(2), 1) / np.pi,
        ls="", marker="o", color="tab:blue", alpha=0.75,
    )
    tmp = np.arctan2(1, -np.sin(phis * np.pi) - np.cos(phis * np.pi)) / np.pi
    ax.plot(phis, tmp, color="tab:green", alpha=0.75)
    ax.set_title(r"$\alpha={0:.3f}\pi, \beta={1:.3f}\pi$".format(alpha, beta))
    ax.set_xlabel(r"$\phi/\pi$")
    ax.set_ylabel(r"$\theta/\pi$")
    ax.set_aspect("equal")
    ax.grid(True, ls="dashed", color="gray")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
