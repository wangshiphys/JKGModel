"""
Demonstrate the J-K-Gamma model on triangular lattice.
"""


import matplotlib.pyplot as plt
import numpy as np

from FontSize import *

LW = 6
MS = 25

point0 = np.array([0.0, 0.0])
point1 = np.array([1.0, 0.0])
point2 = np.array([0.5, np.sqrt(3) / 2])
point3 = np.array([-0.5, np.sqrt(3) / 2])
point4 = np.array([-1.0, 0.0])
point5 = np.array([-0.5, -np.sqrt(3) / 2])
point6 = np.array([0.5, -np.sqrt(3) / 2])


fig, ax = plt.subplots(num="ModelDefinition")

# Draw x-type bond
bond_x = np.array([point1, point4])
ax.plot(bond_x[:, 0], bond_x[:, 1], lw=LW, color="tab:red")
bond_x = np.array([point2, point3])
ax.plot(bond_x[:, 0], bond_x[:, 1], lw=LW, color="tab:red")
bond_x = np.array([point5, point6])
ax.plot(bond_x[:, 0], bond_x[:, 1], lw=LW, color="tab:red")

# Draw y-type bond
bond_y = np.array([point1, point2])
ax.plot(bond_y[:, 0], bond_y[:, 1], lw=LW, color="tab:green")
bond_y = np.array([point3, point6])
ax.plot(bond_y[:, 0], bond_y[:, 1], lw=LW, color="tab:green")
bond_y = np.array([point4, point5])
ax.plot(bond_y[:, 0], bond_y[:, 1], lw=LW, color="tab:green")

# Draw z-type bond
bond_z = np.array([point1, point6])
ax.plot(bond_z[:, 0], bond_z[:, 1], lw=LW, color="tab:blue")
bond_z = np.array([point2, point5])
ax.plot(bond_z[:, 0], bond_z[:, 1], lw=LW, color="tab:blue")
bond_z = np.array([point3, point4])
ax.plot(bond_z[:, 0], bond_z[:, 1], lw=LW, color="tab:blue")

# Draw the points
ax.plot(point0[0], point0[1], ls="", marker="o", ms=MS, color="tab:orange")
ax.plot(point1[0], point1[1], ls="", marker="o", ms=MS, color="tab:purple")
ax.plot(point2[0], point2[1], ls="", marker="o", ms=MS, color="tab:pink")
ax.plot(point3[0], point3[1], ls="", marker="o", ms=MS, color="tab:cyan")
ax.plot(point4[0], point4[1], ls="", marker="o", ms=MS, color="tab:purple")
ax.plot(point5[0], point5[1], ls="", marker="o", ms=MS, color="tab:pink")
ax.plot(point6[0], point6[1], ls="", marker="o", ms=MS, color="tab:cyan")

for index, point in zip(
    [0, 1, 3, 2, 1, 3, 2],
    [point0, point1, point2, point3, point4, point5, point6]
):
    ax.text(
        point[0], point[1], str(index),
        fontsize=XXLARGE, ha="center", va="center", color="black",
    )

ax.text(
    -0.50, 0.0, "x",
    fontsize=XXLARGE+8, ha="center", va="bottom", color="tab:red",
)
ax.text(
    0.25, -np.sqrt(3)/4, "y",
    fontsize=XXLARGE+8, ha="left", va="bottom", color="tab:green",
)
ax.text(
    0.25, np.sqrt(3)/4, "z",
    fontsize=XXLARGE+8, ha="left", va="top", color="tab:blue",
)

ax.text(
    0, 1.0, "(c)",
    fontsize=XXLARGE+8, ha="left", va="top", transform=ax.transAxes
)

ax.set_axis_off()
ax.set_aspect("equal")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.0, 1.0)
fig.set_size_inches(4, 4)
plt.tight_layout()
plt.show()
fig.savefig("figures/ModelDefinition.pdf", transparent=True)
plt.close("all")
