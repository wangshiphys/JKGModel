"""
Draw single octahedron.
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from FontSize import *


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
    [[-0.5, SQRT3 / 6], [0.5, SQRT3 / 6], [0.0, -SQRT3 / 3]], dtype=np.float64
)

# Points on the middle layer
POINTS_MIDDLE = np.array([[0.0, 0.0]], dtype=np.float64)

# Points on the top layer
POINTS_TOP = np.array(
    [[0.0, SQRT3 / 3], [-0.5, -SQRT3 / 6], [0.5, -SQRT3 / 6]], dtype=np.float64
)

# Points' coordinates actually used.
# Rotate the system about z-axis by theta degree
theta = 90
points_bottom = np.dot(Rotation2D(theta), POINTS_BOTTOM.T).T
points_middle = np.dot(Rotation2D(theta), POINTS_MIDDLE.T).T
points_top = np.dot(Rotation2D(theta), POINTS_TOP.T).T
arrows = points_top - points_middle[0]

LW = 3
MS = 16
FACECOLOR = "orangered"
fig, ax = plt.subplots(num="SingleOctahedron")

# Draw points on the bottom layer
ax.plot(
    points_bottom[:, 0], points_bottom[:, 1], zorder=6,
    ls="", marker="o", ms=MS, color="gray", alpha=1.0,
)
# Draw points on the middle layer
ax.plot(
    points_middle[:, 0], points_middle[:, 1], zorder=2,
    ls="", marker="o", ms=1.5*MS, color="black", alpha=1.0,
)
# Draw points on the top layer
ax.plot(
    points_top[:, 0], points_top[:, 1], zorder=8,
    ls="", marker="o", ms=MS, color="gray", alpha=1.0,
)
################################################################################

# Draw lines on the bottom layer (dashed lines)
lines_bottom = [[0, 1], [1, 2], [2, 0]]
for line in lines_bottom:
    (x0, y0), (x1, y1) = points_bottom[line]
    ax.plot(
        [x0, x1], [y0, y1], zorder=0,
        ls="dashed", lw=LW, color="gray", alpha=1.0,
    )

# Draw lines from bottom layer to top layer (thick solid lines)
lines_bottom_to_top = [[0, 0, 1], [1, 0, 2], [2, 1, 2]]
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
lines_top = [[0, 1], [1, 2], [2, 0]]
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
trios = [[0, 1, 2]]
triangles = [Polygon(points_top[trio], closed=True) for trio in trios]
collection = PatchCollection(
    triangles, zorder=4,
    facecolors=FACECOLOR, linewidths=0.0, alpha=0.50,
)
ax.add_collection(collection)

ax.quiver(
    [0, 0, 0], [0, 0, 0], arrows[:, 0], arrows[:, 1], zorder=9,
    units="xy", scale_units="xy", pivot="tail", color="black",
    scale=1.02, width=0.02, headwidth=4, headlength=8, headaxislength=6,
)

scale = 1.15
arrow_names = ["x", "y", "z"]
for arrow, name in zip(arrows, arrow_names):
    x, y = points_middle[0] + arrow * scale
    ax.text(x, y, name, ha="center", va="center", fontsize=LARGE)

ax.text(
    0, 1.0, "(b)",
    fontsize=LARGE, ha="left", va="top", transform=ax.transAxes
)

ax.set_axis_off()
ax.set_aspect("equal")
ax.set_xlim(-0.75, 0.65)
ax.set_ylim(-0.65, 0.65)
fig.set_size_inches(4, 4)
plt.tight_layout()
plt.show()
fig.savefig("figures/SingleOctahedron.pdf", transparent=True)
plt.close("all")
