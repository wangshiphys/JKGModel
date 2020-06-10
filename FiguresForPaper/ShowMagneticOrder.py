"""
Demonstrate Stripe, 120-degree Neel and Dual-Neel magnetic orders.
"""

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import lattice_generator, Lattice


def Show2DVectors(ax, xs, ys, vectors, colors):
    ax.quiver(
        xs, ys, vectors[:, 0], vectors[:, 1], color=colors,
        units="xy", scale_units="xy",
        scale=1.45, width=0.08,
        pivot="middle", zorder=2,
    )


fig, ax = plt.subplots(num="MagneticOrder")

NUM = 4
# The clusters on which the spin vectors are plotted
StripeXCluster = lattice_generator("triangle", num0=NUM, num1=NUM)
StripeYCluster = lattice_generator("triangle", num0=NUM, num1=NUM)
StripeZCluster = lattice_generator("triangle", num0=NUM, num1=NUM)
NeelCluster = lattice_generator("triangle", num0=NUM, num1=NUM)
DualNeelCluster = lattice_generator("triangle", num0=2 * NUM, num1=NUM)

# Draw the clusters
x_deltas = (0.0, 4.0, 8.0, 12.0, 16.0)
clusters = (
    StripeXCluster, StripeYCluster, StripeZCluster,
    NeelCluster, DualNeelCluster
)
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

# Draw the 120 degree Neel order
NeelCell = Lattice(
    np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2], [1.0, 0.0]]),
    np.array([[-1.5, np.sqrt(3) / 2], [1.5, np.sqrt(3) / 2]]),
)
NeelCellSpinColors = ("tab:orange", "tab:purple", "tab:red")
NeelCellSpinVectors = np.array(
    [[-np.sqrt(3) / 2, -0.5], [0.0, 1.0], [np.sqrt(3) / 2, -0.5]]
)

x_delta = x_deltas[3]
spin_colors = []
spin_vectors = []
points = NeelCluster.points
for point in points:
    index = NeelCell.getIndex(site=point, fold=True)
    spin_colors.append(NeelCellSpinColors[index])
    spin_vectors.append(NeelCellSpinVectors[index])
spin_vectors = np.array(spin_vectors)
Show2DVectors(
    ax, points[:, 0] + x_delta, points[:, 1], spin_vectors, spin_colors
)

# Draw the dual 120 degree Neel order
DualNeelCell = Lattice(
    np.array(
        [
            np.dot([x, y], np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]]))
            for x in range(6) for y in range(2)
        ]
    ),
    np.array([[6.0, 0.0], [3.0, np.sqrt(3)]]),
)
color_map_set1 = plt.get_cmap("Set1")
color_map_set2 = plt.get_cmap("Set2")
tmp1 = color_map_set1(range(color_map_set1.N))
tmp2 = color_map_set2(range(color_map_set2.N))
DualCellSpinColors = np.concatenate([tmp1, tmp2])

FourSublatticeTransformationCell = lattice_generator("triangle", num0=2, num1=2)
FourSublatticeTransformation = [
    # 0th site, identity
    [[ 1, 0], [0,  1]],
    # 1st site, Rotation about the z-axis by 180 degree
    [[-1, 0], [0, -1]],
    # 2nd site, Rotation about the x-axis by 180 degree
    [[ 1, 0], [0, -1]],
    # 3rd site, Rotation about the y-axis by 180 degree
    [[-1, 0], [0,  1]],
]

x_delta = x_deltas[4]
spin_colors = []
spin_vectors = []
points = DualNeelCluster.points
for point in points:
    index0 = NeelCell.getIndex(site=point, fold=True)
    index1 = FourSublatticeTransformationCell.getIndex(site=point, fold=True)
    index2 = DualNeelCell.getIndex(site=point, fold=True)
    transform = FourSublatticeTransformation[index1]
    spin_vectors.append(np.dot(transform, NeelCellSpinVectors[index0]))
    spin_colors.append(DualCellSpinColors[index2])
spin_vectors = np.array(spin_vectors)
Show2DVectors(
    ax, points[:, 0] + x_delta, points[:, 1], spin_vectors, spin_colors
)

# Draw a dashed frame which mark the magnetic unit cell of
# the dual 120 degree order
width = 5.8
height = np.sqrt(3)
anchor_x = 15.4
anchor_y = -np.sqrt(3) / 4 + 0.08
DashedFrame = np.array(
    [
        [anchor_x, anchor_y],
        [anchor_x + 1, anchor_y + height],
        [anchor_x + width + 1, anchor_y + height],
        [anchor_x + width, anchor_y],
        [anchor_x, anchor_y],
    ]
)
ax.plot(
    DashedFrame[:, 0], DashedFrame[:, 1],
    ls="dashed", lw=2, color="tab:green", alpha=0.85, zorder=1,
)

# Add sub-figure tag to these clusters
anchor_x = 0.75
anchor_y = 3.00
for x_delta, tag in zip(x_deltas, ["(b)", "(c)", "(d)", "(e)", "(f)"]):
    ax.text(
        anchor_x + x_delta, anchor_y, tag,
        fontsize="xx-large", ha="left", va="top"
    )

ax.set_axis_off()
ax.set_aspect("equal")
ax.set_xlim(-0.5, 25.0)
ax.set_ylim(-0.5, 3.20)
fig.set_size_inches(18, 3.1)
plt.tight_layout()
plt.show()
print(fig.get_size_inches())
# fig.savefig("figures/MagneticOrder.pdf", dpi=200)
# fig.savefig("figures/MagneticOrder.png", dpi=200)
plt.close("all")
