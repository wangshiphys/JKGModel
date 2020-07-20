import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

from StructureFactor import ClassicalSpinStructureFactor


def NematicOrderGenerator(num, config="NematicX"):
    """
    Generate nematic order.

    Parameters
    ----------
    num : int
        The number of site along the 1st and 2nd translation vector.
    config : str, optional
        The configuration of the nematic order.
        If "NematicX", the spins along the x-bond direction are anti-parallel;
        If "NematicY", the spins along the y-bond direction are anti-parallel;
        If "NematicZ", the spins along the z-bond direction are anti-parallel.
        Default: "NematicX".

    Returns
    -------
    points : 2D array
        The coordinates of the points.
    vectors : 2D array
        The spin vectors defined on the returned `points`.
    """

    if config == "NematicX":
        vectors = np.array([[0.5, np.sqrt(3) / 2], [1.0, 0.0]])
    elif config == "NematicY":
        vectors = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2]])
    elif config == "NematicZ":
        vectors = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    else:
        raise ValueError("Invalid `config`: {0}".format(config))

    spin_up = np.random.random(3)
    spin_up /= np.linalg.norm(spin_up)
    spin_down = -spin_up

    spin_vectors = np.empty((num * num, 3), dtype=np.float64)
    chains = np.arange(num * num, dtype=np.int64).reshape(num, num)
    points = np.dot([[i, j] for i in range(num) for j in range(num)], vectors)
    for chain in chains:
        indices0 = chain[0::2]
        indices1 = chain[1::2]
        if np.random.random() < 0.5:
            spin_vectors[indices0] = spin_up
            spin_vectors[indices1] = spin_down
        else:
            spin_vectors[indices1] = spin_up
            spin_vectors[indices0] = spin_down
    return points, spin_vectors


if __name__ == "__main__":
    step = 0.01
    ratios = np.arange(-0.7, 0.7 + step, step)
    kpoints = np.matmul(
        np.stack(np.meshgrid(ratios, ratios, indexing="ij"), axis=-1),
        4 * np.pi * np.identity(2) / np.sqrt(3)
    )
    BZBoundary = HP.TRIANGLE_CELL_KS[[*range(6), 0]]

    num = 30
    config = "NematicX"
    points, vectors = NematicOrderGenerator(num=num, config=config)

    factors = ClassicalSpinStructureFactor(kpoints, points, vectors)
    assert np.all(np.abs(factors.imag) < 1E-10)
    factors = factors.real

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()

    ax0.plot(
        points[:, 0], points[:, 1],
        ls="", marker="o", ms=6, color="black", zorder=0,
    )
    ax0.quiver(
        points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1],
        color=0.5 * vectors + 0.5, units="xy", scale_units="xy",
        scale=1.45, width=0.06, pivot="mid", clip_on=False, zorder=1,
    )
    ax0.set_axis_off()
    ax0.set_aspect("equal")

    im1 = ax1.pcolormesh(
        kpoints[:, :, 0], kpoints[:, :, 1], factors,
        zorder=0, cmap="magma", shading="gouraud",
    )
    fig1.colorbar(im1, ax=ax1)
    ax1.plot(
        BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
        lw=3, ls="dashed", color="tab:blue", alpha=1.0,
    )
    ticks = np.array([-1, 0, 1])
    ax1.set_xticks(ticks * np.pi)
    ax1.set_yticks(ticks * np.pi)
    ax1.set_xticklabels(["{0}".format(tick) for tick in ticks])
    ax1.set_yticklabels(["{0}".format(tick) for tick in ticks])
    ax1.set_xlabel(r"$k_x/\pi$", fontsize="large")
    ax1.set_ylabel(r"$k_y/\pi$", fontsize="large")
    ax1.grid(True, ls="dashed", color="gray")
    ax1.set_aspect("equal")

    plt.show()
    plt.close("all")
