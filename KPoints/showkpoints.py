import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import TRIANGLE_CELL_KS

from database import AllBs

BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]


def ShowSingle(num1=4, num2=6, which="xy", ms=8):
    title = "num1={0}_num2={1}_which={2}".format(num1, num2, which)
    fig, ax = plt.subplots(num=title)
    Bs = AllBs[which]
    b0, b1 = Bs
    for i in range(-num1, num1 + 1):
        for j in range(-num2, num2 + 1):
            x, y = np.dot([i / num1, j / num2], Bs)
            ax.plot(x, y, ls="", marker="o", ms=ms, color="tab:blue", zorder=3)
            ax.text(
                x, y, "({0},{1})".format(i, j),
                va="top", ha="center", clip_on=True,
            )
    ax.plot([0, b0[0] / 2], [0, b0[1] / 2], color="tab:red")
    ax.plot([0, b1[0] / 2], [0, b1[1] / 2], color="tab:green")
    ax.text(b0[0] / 2, b0[1] / 2, "b0", fontsize="xx-large")
    ax.text(b1[0] / 2, b1[1] / 2, "b1", fontsize="xx-large")
    ax.plot(
        BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
        lw=3, ls="dashed", alpha=1.0,
    )
    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-4.0, 4.0)
    ax.set_aspect("equal")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")


def ShowMultiple(whichs, num1=4, num2=6, ms=8):
    fig, ax = plt.subplots()
    for which in whichs:
        kpoints = np.dot(
            [
                [i / num1, j / num2]
                for i in range(-num1, num1 + 1) for j in range(-num2, num2 + 1)
            ], AllBs[which]
        )
        ax.plot(kpoints[:, 0], kpoints[:, 1], ls="", marker="o", ms=ms)
    ax.plot(
        BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
        lw=3, ls="dashed", alpha=1.0,
    )
    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-4.0, 4.0)
    ax.set_aspect("equal")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    ShowSingle()
    ShowMultiple(["xy", "yz"])
