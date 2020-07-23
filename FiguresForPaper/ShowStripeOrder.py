"""
Demonstrate stripe orders.
"""


import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import lattice_generator, Lattice

from FontSize import *


def Show2DVectors(ax, xs, ys, vectors, colors):
    ax.quiver(
        xs, ys, vectors[:, 0], vectors[:, 1], color=colors,
        units="xy", scale_units="xy",
        scale=1.45, width=0.08,
        pivot="middle", zorder=2,
    )


fig, ax = plt.subplots(num="StripeOrder")

# The clusters on which the spin vectors are plotted
numx = numy = 4
StripeXCluster = lattice_generator("triangle", num0=numx, num1=numy)
StripeYCluster = lattice_generator("triangle", num0=numx, num1=numy)
StripeZCluster = lattice_generator("triangle", num0=numx, num1=numy)

# Draw the clusters
x_deltas = (0.0, 4.0, 8.0)
clusters = (StripeXCluster, StripeYCluster, StripeZCluster)
for x_delta, clusters in zip(x_deltas, clusters):
    intra_bonds, inter_bonds = clusters.bonds(nth=1)
    for bond in intra_bonds:
        (x0, y0), (x1, y1) = bond.endpoints
        ax.plot([x0 + x_delta, x1 + x_delta], [y0, y1], color="black", zorder=0)

# Draw the StripeX order
StripeXCell = Lattice(
    np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2]]),
    np.array([[1.0, 0.0], [1.0, np.sqrt(3)]]),
)
StripeXCellSpinColors = ("tab:orange", "tab:purple")
StripeXCellSpinVectors = np.array([[0.0, 1.0], [0.0, -1.0]])

x_delta = x_deltas[0]
spin_colors = []
spin_vectors = []
points = StripeXCluster.points
for point in points:
    index = StripeXCell.getIndex(site=point, fold=True)
    spin_vectors.append(StripeXCellSpinVectors[index])
    spin_colors.append(StripeXCellSpinColors[index])
spin_vectors = np.array(spin_vectors)
Show2DVectors(
    ax, points[:, 0] + x_delta, points[:, 1], spin_vectors, spin_colors
)

# Draw the StripeY order
StripeYCell = Lattice(
    np.array([[0.0, 0.0], [1.0, 0.0]]),
    np.array([[2.0, 0.0], [-0.5, np.sqrt(3) / 2]]),
)
StripeYCellSpinColors = ("tab:orange", "tab:purple")
StripeYCellSpinVectors = np.array([[0.0, 1.0], [0.0, -1.0]])

x_delta = x_deltas[1]
spin_colors = []
spin_vectors = []
points = StripeYCluster.points
for point in points:
    index = StripeYCell.getIndex(site=point, fold=True)
    spin_vectors.append(StripeYCellSpinVectors[index])
    spin_colors.append(StripeYCellSpinColors[index])
spin_vectors = np.array(spin_vectors)
Show2DVectors(
    ax, points[:, 0] + x_delta, points[:, 1], spin_vectors, spin_colors
)

# Draw the StripeZ order
StripeZCell = Lattice(
    np.array([[0.0, 0.0], [1.0, 0.0]]),
    np.array([[2.0, 0.0], [0.5, np.sqrt(3) / 2]]),
)
StripeZCellSpinColors = ("tab:orange", "tab:purple")
StripeZCellSpinVectors = np.array([[0.0, 1.0], [0.0, -1.0]])

x_delta = x_deltas[2]
spin_colors = []
spin_vectors = []
points = StripeZCluster.points
for point in points:
    index = StripeZCell.getIndex(site=point, fold=True)
    spin_vectors.append(StripeZCellSpinVectors[index])
    spin_colors.append(StripeZCellSpinColors[index])
spin_vectors = np.array(spin_vectors)
Show2DVectors(
    ax, points[:, 0] + x_delta, points[:, 1], spin_vectors, spin_colors
)

# Add sub-figure tag to these clusters
anchor_x = 0.75
anchor_y = 3.00
for x_delta, tag in zip(x_deltas, ["(a)", "(b)", "(c)"]):
    ax.text(
        anchor_x + x_delta, anchor_y, tag,
        fontsize=LARGE, ha="left", va="top"
    )

ax.set_axis_off()
ax.set_aspect("equal")
ax.set_xlim(-0.2, 12.8)
ax.set_ylim(-0.4, 3.0)
fig.set_size_inches(18, 5.2)
plt.tight_layout()
plt.show()
fig.savefig("figures/StripeOrder.pdf", transparent=True)
plt.close("all")
