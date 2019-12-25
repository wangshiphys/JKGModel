import matplotlib.pyplot as plt
import numpy as np
import tables as tb

from time import time

from HamiltonianPy import lattice_generator, TRIANGLE_CELL_KS
from DualTransformation import *
from utilities import ShowVectorField3D


def DFT(kpoints, points, vectors):
    numkx, numky, space_dim = kpoints.shape
    dft = np.zeros((numkx, numky, 3), dtype=np.complex128)
    for i in range(numkx):
        for j in range(numky):
            tmp = np.exp(1j * np.dot(points, kpoints[i, j])).reshape((-1, 1))
            dft[i, j] = np.sum(vectors * tmp, axis=0)
    return dft


numx = numy = 12
cell = lattice_generator("triangle", num0=1, num1=1)
cluster = lattice_generator("triangle", num0=numx, num1=numy)
ratio_x = np.arange(-2 * numx, 2 * numx) / numx
ratio_y = np.arange(-2 * numy, 2 * numy) / numy
kpoints = np.matmul(
    np.stack(np.meshgrid(ratio_x, ratio_y, indexing="ij"), axis=-1), cell.bs,
)
BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]

# vectors = GenerateFMOrder(cluster.points)
# vectors = T4(cluster.points, GenerateFMOrder(cluster.points))
# vectors = T1T4(cluster.points, GenerateFMOrder(cluster.points))
# vectors = GenerateNeelOrder(cluster.points)
# vectors = T4(cluster.points, GenerateNeelOrder(cluster.points))
# vectors = T1T4(cluster.points, GenerateNeelOrder(cluster.points))

alpha = 0.5
beta = 1.5
index = 10
data_path = "data/ClassicalSpinModel/OptimizedSpinConfig/"
h5f_name_temp = "OSC_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.h5"
h5f_full_name = data_path + h5f_name_temp.format(numx, numy, alpha, beta)
h5f = tb.open_file(h5f_full_name, mode="r")
vectors = h5f.get_node("/", "Run{0:0>4d}".format(index)).read()
h5f.close()

fig_spin_config = plt.figure("SpinConfiguration")
ax_spin_config = fig_spin_config.add_subplot(111, projection="3d")
ShowVectorField3D(ax_spin_config, cluster.points, vectors)
ax_spin_config.set_zlim(-0.5, 0.5)

t0 = time()
dft = DFT(kpoints, cluster.points, vectors)
t1 = time()
print("The time spend on DFT: {0:4f}s".format(t1 - t0))

for index, component in enumerate(["x", "y", "z"]):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    im0 = ax0.pcolormesh(
        kpoints[:, :, 0], kpoints[:, :, 1], dft[:, :, index].real,
        cmap="seismic", shading="gouraud",
    )
    im1 = ax1.pcolormesh(
        kpoints[:, :, 0], kpoints[:, :, 1], dft[:, :, index].imag,
        cmap="seismic", shading="gouraud",
    )
    ax0.plot(BZBoundary[:, 0], BZBoundary[:, 1], lw=1, alpha=0.8, color="red")
    ax1.plot(BZBoundary[:, 0], BZBoundary[:, 1], lw=1, alpha=0.8, color="red")
    ax0.set_title(
        "Real Part for {0} component".format(component),
    )
    ax1.set_title(
        "Imaginary Part for {0} component".format(component),
    )
    fig.colorbar(im0, ax=ax0, shrink=0.80)
    fig.colorbar(im1, ax=ax1, shrink=0.80)
    ax0.grid(True, ls="dashed", color="gray")
    ax1.grid(True, ls="dashed", color="gray")
    ax0.set_aspect("equal")
    ax1.set_aspect("equal")
plt.show()
plt.close("all")
