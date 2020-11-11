import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import cKDTree
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


LEFT = 10.0
RIGHT = 30.0
BOTTOM = -4.25
TOP = 4.25

with np.load("data/CMC_num0=24_num1=24_alpha=0.3000_beta=0.5000.npz") as ld:
    all_points = ld["points"]
    all_vectors = ld["vectors"]
all_points = np.dot(all_points, Rotation2D(30))

selected_points = []
selected_vectors = []
for index in range(all_points.shape[0]):
    x, y = all_points[index]
    if (LEFT < x < RIGHT) and (BOTTOM < y < TOP):
        selected_points.append([x, y])
        selected_vectors.append(all_vectors[index])
selected_points = np.array(selected_points)
selected_vectors = np.array(selected_vectors)
# selected_points = all_points
# selected_vectors = all_vectors

tree = cKDTree(selected_points)
pairs = tree.query_pairs(r=1.01)

fig, ax = plt.subplots(num="CMCSpinConfig")
for i, j in pairs:
    x0, y0 = selected_points[i]
    x1, y1 = selected_points[j]
    ax.plot(
        [x0, x1], [y0, y1], ls="dashed", color="gray", lw=1.0, zorder=0
    )
ax.quiver(
    selected_points[:, 0], selected_points[:, 1],
    selected_vectors[:, 0], selected_vectors[:, 1],
    units="xy", scale_units="xy", scale=1.2, width=0.1, zorder=1,
    pivot="middle", color=0.5 * selected_vectors + 0.5, clip_on=False,
)
ax.plot(
    selected_points[:, 0], selected_points[:, 1],
    color="k", ls="", marker="o", ms=5, zorder=2,
)
ax.text(
    0.00, 0.98, "(c)",
    fontsize=SMALL, ha="center", va="top", transform=ax.transAxes,
)
ax.set_axis_off()
ax.set_aspect("equal")
fig.set_size_inches(8.0, 3.2)
fig.subplots_adjust(top=1, bottom=0, left=0, right=1)

plt.show()
fig.savefig("figures/SixSitesOrder/CMCSpinConfig.pdf", transparent=True)
print(fig.get_size_inches())
plt.close("all")
