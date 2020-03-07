import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import TRIANGLE_CELL_KS

from database import AllBs, KPOINT_IDS

BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]


ms = 8
num1 = 4
num2 = 6
which = "zy"

fig, ax = plt.subplots()
Bs = AllBs[which]
b0, b1 = Bs
for i in range(-num1, num1 + 1):
    for j in range(-num2, num2 + 1):
        x, y = np.dot([i / num1, j / num2], Bs)
        ax.plot(x, y, ls="", marker="o", ms=2 * ms, color="tab:blue")
        ax.text(
            x, y, "({0},{1})".format(i, j), fontsize="large",
            va="top", ha="center", clip_on=True,
        )
kpoints = np.dot([[i / num1, j / num2] for i, j in KPOINT_IDS[which]], Bs)
ax.plot(kpoints[:, 0], kpoints[:, 1], marker="o", ms=ms, color="tab:red")
ax.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=3, ls="dashed", alpha=1.0,
)
ax.set_xlim(-4.3, 4.3)
ax.set_ylim(-4.0, 4.0)
ax.set_title("num1={0}_num2={1}_which={2}".format(num1, num2, which))
ax.set_aspect("equal")
fig.set_size_inches(4.6, 4, 2)
plt.show()
fig.savefig(
    "KPath_num1={0}_num2={1}_which={2}.png".format(num1, num2, which), dpi=300
)
plt.close("all")
