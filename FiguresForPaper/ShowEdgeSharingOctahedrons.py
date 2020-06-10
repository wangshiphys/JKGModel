"""
Draw the triangular lattice composed of edge-sharing octahedrons.
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def Rotation2D(theta=0.0, deg=True):
    """
    Rotation about the axis perpendicular to the plane by theta angle.

    Parameters
    ----------
    theta : float, optional
        The angle to rotate.
        Default: 0.0.
    deg : bool, optional
        Whether the given theta is in degree or radian.
        Default: True.

    Returns
    -------
    RZ : array with shape (2, 2)
        The corresponding rotation matrix.
    """

    if deg:
        theta = np.pi * theta / 180
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


SQRT3 = np.sqrt(3)
# Points on the bottom layer
POINTS_BOTTOM = np.array(
    [
        [-1.0, 2 * SQRT3 / 3], [0.0, 2 * SQRT3 / 3], [1.0, 2 * SQRT3 / 3],
        [-1.5, SQRT3 / 6], [-0.5, SQRT3 / 6],
        [0.5, SQRT3 / 6], [1.5, SQRT3 / 6],
        [-1.0, -SQRT3 / 3], [0.0, -SQRT3 / 3], [1.0, -SQRT3 / 3],
        [-0.5, -5 * SQRT3 / 6], [0.5, -5 * SQRT3 / 6],
    ], dtype=np.float64,
)

# Points on the middle layer
POINTS_MIDDLE = np.array(
    [
        [-0.5, SQRT3 / 2], [0.5, SQRT3 / 2],
        [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
        [-0.5, -SQRT3 / 2], [0.5, -SQRT3 / 2],
    ], dtype=np.float64
)

# Points on the top layer
POINTS_TOP = np.array(
    [
        [-0.5, 5 * SQRT3 / 6], [0.5, 5 * SQRT3 / 6],
        [-1.0, SQRT3 / 3], [0.0, SQRT3 / 3], [1.0, SQRT3 / 3],
        [-1.5, -SQRT3 / 6], [-0.5, -SQRT3 / 6],
        [ 0.5, -SQRT3 / 6], [ 1.5, -SQRT3 / 6],
        [-1.0, -2 * SQRT3 / 3], [0.0, -2 * SQRT3 / 3], [1.0, -2 * SQRT3 / 3],
    ], dtype=np.float64
)

# Points' coordinates actually used.
# Rotate the system about z-axis by theta degree
theta = 0.0
points_bottom = np.dot(Rotation2D(theta), POINTS_BOTTOM.T).T
points_middle = np.dot(Rotation2D(theta), POINTS_MIDDLE.T).T
points_top = np.dot(Rotation2D(theta), POINTS_TOP.T).T

LW = 2
MS = 10
FACECOLOR = "orangered"
fig, ax = plt.subplots(num="EdgeSharingOctahedrons")

# Draw points on the bottom layer
ax.plot(
    points_bottom[:, 0], points_bottom[:, 1], zorder=6,
    ls="", marker="o", ms=MS, color="gray", alpha=1.0,
)
# Draw points on the middle layer
ax.plot(
    points_middle[:, 0], points_middle[:, 1], zorder=2,
    ls="", marker="o", ms=2 * MS, color="black", alpha=1.0,
)
# Draw points on the top layer
ax.plot(
    points_top[:, 0], points_top[:, 1], zorder=8,
    ls="", marker="o", ms=MS, color="gray", alpha=1.0,
)
################################################################################

# Draw lines on the bottom layer (dashed lines)
lines_bottom = [
    [0, 2], [3, 6], [7, 9],
    [3, 10], [0, 11], [1, 9],
    [1, 7], [2, 10], [6, 11],
]
for line in lines_bottom:
    (x0, y0), (x1, y1) = points_bottom[line]
    ax.plot(
        [x0, x1], [y0, y1], zorder=0,
        ls="dashed", lw=LW, color="gray", alpha=1.0,
    )

# Draw lines on the middle layer (bold solid lines)
lines_middle = [
    [0, 1], [2, 4], [5, 6],
    [2, 5], [0, 6], [1, 4],
    [0, 2], [1, 5], [4, 6]
]
for line in lines_middle:
    (x0, y0), (x1, y1) = points_middle[line]
    ax.plot(
        [x0, x1], [y0, y1], zorder=1,
        ls="solid", lw=2 * LW, color="black", alpha=1.0,
    )

# Draw lines from bottom layer to top layer (thick solid lines)
lines_bottom_to_top = [
    [0, 0, 2], [1, 0, 3], [4, 2, 3],
    [1, 1, 3], [2, 1, 4], [5, 3, 4],
    [3, 2, 5], [4, 2, 6], [7, 5, 6],
    [4, 3, 6], [5, 3, 7], [8, 6, 7],
    [5, 4, 7], [6, 4, 8], [9, 7, 8],
    [7, 6, 9], [8, 6, 10], [10, 9, 10],
    [8, 7, 10], [9, 7, 11], [11, 10, 11],
]
for i, j1, j2 in lines_bottom_to_top:
    (x0, y0) = points_bottom[i]
    (x1, y1) = points_top[j1]
    (x2, y2) = points_top[j2]
    ax.plot(
        [x0, x1], [y0, y1], zorder=5,
        ls="solid", lw=LW, color="black", alpha=1.0,
    )
    ax.plot(
        [x0, x2], [y0, y2], zorder=5,
        ls="solid", lw=LW, color="black", alpha=1.0,
    )

# Draw lines on the top layer (thick solid lines)
lines_top = [
    [2, 4], [5, 8], [9, 11],
    [1, 8], [0, 11], [2, 10],
    [0, 5], [1, 9], [4, 10],
]
for line in lines_top:
    (x0, y0), (x1, y1) = points_top[line]
    ax.plot(
        [x0, x1], [y0, y1], zorder=7,
        ls="solid", lw=LW, color="black", alpha=1.0,
    )
################################################################################

# Draw triangles between bottom and top layer
# dark colored isosceles triangles
triangles = []
for i, j1, j2 in lines_bottom_to_top:
    p0 = points_bottom[i]
    p1 = points_top[j1]
    p2 = points_top[j2]
    triangles.append(Polygon([p0, p1, p2], closed=True))
collection = PatchCollection(
    triangles, zorder=3,
    facecolors=FACECOLOR, linewidths=0.0, alpha=0.65,
)
ax.add_collection(collection)

# Draw triangles on the top layer
# light colored equilateral triangles
trios = [
    [0, 2, 3], [1, 3, 4],
    [2, 5, 6], [3, 6, 7], [4, 7, 8],
    [6, 9, 10], [7, 10, 11],
]
triangles = [Polygon(points_top[trio], closed=True) for trio in trios]
collection = PatchCollection(
    triangles, zorder=4,
    facecolors=FACECOLOR,  linewidths=0.0, alpha=0.50,
)
ax.add_collection(collection)

ax.text(
    0, 1.0, "(a)",
    fontsize="xx-large", ha="left", va="top", transform=ax.transAxes
)

ax.set_aspect("equal")
ax.set_axis_off()
fig.set_size_inches(8, 8)
plt.tight_layout()
plt.show()
print(fig.get_size_inches())
# fig.savefig("figures/EdgeSharingOctahedrons.pdf", dpi=200)
# fig.savefig("figures/EdgeSharingOctahedrons.png", dpi=200)
plt.close("all")
