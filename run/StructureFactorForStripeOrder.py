import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

from StructureFactor import ClassicalSpinStructureFactor

num =  30
config = "StripeZ"
if config == "StripeX":
    # FM along x-bond direction
    cell_points = np.array([[0, 0], [0.5, np.sqrt(3)/2]], dtype=np.float64)
    cell_vectors = np.array([[1, 0], [1, np.sqrt(3)]], dtype=np.float64)
elif config == "StripeY":
    # FM along y-bond direction
    cell_points = np.array([[0, 0], [1, 0]], dtype=np.float64)
    cell_vectors = np.array([[2, 0], [-0.5, np.sqrt(3)/2]], dtype=np.float64)
elif config == "StripeZ":
    # FM along z-bond direction
    cell_points = np.array([[0, 0], [1, 0]], dtype=np.float64)
    cell_vectors = np.array([[2, 0], [0.5, np.sqrt(3)/2]], dtype=np.float64)
else:
    raise ValueError("Invalid `config`: {0}".format(config))
cell = HP.Lattice(points=cell_points, vectors=cell_vectors)
cluster = HP.lattice_generator("triangle", num0=num, num1=num)

tmp = 2 * np.random.random(3) - 1
vx, vy, vz = tmp / np.linalg.norm(tmp)
spin_vectors = []
for point in cluster.points:
    index = cell.getIndex(site=point, fold=True)
    if index == 0:
        spin_vectors.append([vx, vy, vz])
    else:
        spin_vectors.append([-vx, -vy, -vz])
spin_vectors = np.array(spin_vectors, dtype=np.float64)

step = 0.01
ratios = np.arange(-0.7, 0.7 + step, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(ratios, ratios, indexing="ij"), axis=-1),
    4 * np.pi * np.identity(2) / np.sqrt(3)
)
BZBoundary = HP.TRIANGLE_CELL_KS[[*range(6), 0]]

factors = ClassicalSpinStructureFactor(kpoints, cluster.points, spin_vectors)
assert np.all(np.abs(factors.imag) < 1E-12)
factors = factors.real

fig, ax = plt.subplots(num=config)
im = ax.pcolormesh(
    kpoints[:, :, 0], kpoints[:, :, 1], factors, zorder=0,
    cmap="magma", shading="gouraud",
)
fig.colorbar(im, ax=ax)
ax.plot(
    BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
    lw=3, ls="dashed", color="tab:blue", alpha=1.0,
)
ticks = np.array([-1, 0, 1])
ax.set_xticks(ticks * np.pi)
ax.set_yticks(ticks * np.pi)
ax.set_xticklabels(["{0}".format(tick) for tick in ticks])
ax.set_yticklabels(["{0}".format(tick) for tick in ticks])
ax.set_xlabel(r"$k_x/\pi$", fontsize="large")
ax.set_ylabel(r"$k_y/\pi$", fontsize="large")
ax.grid(True, ls="dashed", color="gray")
ax.set_aspect("equal")
plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
plt.close("all")
