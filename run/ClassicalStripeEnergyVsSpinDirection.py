import matplotlib.pyplot as plt
import numpy as np


def _StripeXEnergy(sx, sy, sz, J, K, G):
    return -J - K + 2 * K * sx * sx + 2 * G * (sy * sz - sx * sy - sx * sz)

def _StripeYEnergy(sx, sy, sz, J, K, G):
    return -J - K + 2 * K * sy * sy + 2 * G * (sx * sz - sx * sy - sy * sz)

def _StripeZEnergy(sx, sy, sz, J, K, G):
    return -J - K + 2 * K * sz * sz + 2 * G * (sx * sy - sx * sz - sy * sz)


def ExtremePoint(K, G, which="x"):
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

    if which == "x":
        x0, y0, z0 = x_a, y_a, z_a
        x1, y1, z1 = x_b, y_b, z_b
        x2, y2, z2 = x_c, y_c, z_c
    elif which == "y":
        x0, y0, z0 = z_a, x_a, y_a
        x1, y1, z1 = z_b, x_b, y_b
        x2, y2, z2 = z_c, x_c, y_c
    elif which == "z":
        x0, y0, z0 = y_a, z_a, x_a
        x1, y1, z1 = y_b, z_b, x_b
        x2, y2, z2 = y_c, z_c, x_c
    else:
        raise ValueError("Invalid `which` parameter: {0}".format(which))

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


def StripeEnergies(thetas, phis, J=2.0, K=-1.0, G=0.0, which="x"):
    if which == "x":
        core = _StripeXEnergy
    elif which == "y":
        core = _StripeYEnergy
    elif which == "z":
        core = _StripeZEnergy
    else:
        raise ValueError("Invalid `which` parameter: {0}".format(which))

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
            energies[i, j] = core(sx, sy, sz, J, K, G)
    return energies


if __name__ == "__main__":
    alpha = 0.30
    beta  = 0.20
    J = np.sin(alpha * np.pi) * np.sin(beta * np.pi)
    K = np.sin(alpha * np.pi) * np.cos(beta * np.pi)
    G = np.cos(alpha * np.pi)

    step = 0.005
    thetas = np.arange(0, 1 + step, step)
    phis = np.arange(-1, 1 + step, step)

    which = "x"
    point0, point1, point2 = ExtremePoint(K, G, which=which)
    energies = StripeEnergies(
        thetas * np.pi, phis * np.pi, J, K, G, which=which
    )

    fig, ax = plt.subplots()
    cs = ax.pcolormesh(phis, thetas, energies, cmap="hot", shading="gouraud")
    fig.colorbar(cs, ax=ax, shrink=0.83)
    circle, = ax.plot(
        point0[-1] / np.pi, point0[-2] / np.pi,
        ls="", marker="o", color="tab:green", alpha=0.75,
    )
    square, = ax.plot(
        point1[-1] / np.pi, point1[-2] / np.pi,
        ls="", marker="s", color="tab:green", alpha=0.75,
    )
    triangle, = ax.plot(
        point2[-1] / np.pi, point2[-2] / np.pi,
        ls="", marker="<", color="tab:green", alpha=0.75,
    )
    ax.legend(
        [circle, square, triangle], ["1st", "2nd", "3rd"], loc=0,
    )
    ax.set_title(r"$\alpha={0:.3f}\pi, \beta={1:.3f}\pi$".format(alpha, beta))
    ax.set_xlabel(r"$\phi/\pi$")
    ax.set_ylabel(r"$\theta/\pi$")
    ax.set_aspect("equal")
    ax.grid(True, ls="dashed", color="gray")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
