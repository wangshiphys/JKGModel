import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np
import tables as tb

from StructureFactor import ClassicalSpinStructureFactor

data_path = "C:/Users/swang/Desktop/Working/"
data_path += "JKGModel/data/ClassicalSpinModel/OptimizedSpinConfig/"
data_name_temp = "OSC_num0=12_num1=12_alpha={0:.4f}_beta={1:.4f}.h5"

step = 0.01
ratios = np.arange(-0.7, 0.7 + step, step)
kpoints = np.matmul(
    np.stack(np.meshgrid(ratios, ratios, indexing="ij"), axis=-1),
    4 * np.pi * np.identity(2) / np.sqrt(3)
)
BZBoundary = HP.TRIANGLE_CELL_KS[[*range(6), 0]]
points = HP.lattice_generator("triangle", num0=12, num1=12).points

axis = (np.pi * np.random.random(), 2 * np.pi * np.random.random())
Rotation = HP.RotationGeneral(axis, 2 * np.pi * np.random.random())

alpha = 0.06
for beta in np.arange(0, 1, 0.02):
    data_name = data_name_temp.format(alpha, beta)
    try:
        h5f = tb.open_file(data_path + data_name, mode="r")
    except OSError:
        continue

    container = []
    for carray in h5f.iter_nodes("/"):
        energy = h5f.get_node_attr(carray, "energy")
        container.append((energy, carray))
    container.sort(key=lambda item: item[0])

    print("alpha={0:.2f}, beta={1:.2f}".format(alpha, beta))
    for energy, carray in container[0:1]:
        name = carray.name
        vectors = carray.read()
        print("{0}, E = {1}".format(name, energy))
        factors = ClassicalSpinStructureFactor(kpoints, points, vectors).real

        vectors = np.dot(vectors, Rotation.T)
        fig, (ax_vectors, ax_ssf) = plt.subplots(1, 2, num=name)
        ax_vectors.plot(
            points[:, 0], points[:, 1], color="k", ls="", marker="o", ms=5
        )
        ax_vectors.quiver(
            points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1],
            pivot="middle", color=0.5 * vectors + 0.5,
        )
        ax_vectors.set_title("E = {0:.10f}".format(energy), fontsize="xx-large")
        ax_vectors.set_axis_off()
        ax_vectors.set_aspect("equal")

        im = ax_ssf.pcolormesh(
            kpoints[:, :, 0], kpoints[:, :, 1], factors,
            cmap="magma", shading="gouraud", zorder=0,
        )
        fig.colorbar(im, ax=ax_ssf)
        ax_ssf.plot(
            BZBoundary[:, 0], BZBoundary[:, 1], zorder=1,
            lw=3, ls="dashed", color="tab:blue", alpha=1.0,
        )
        ax_ssf.set_aspect("equal")
        ax_ssf.set_title("alpha={0:.2f}, beta={1:.2f}".format(alpha, beta))
        plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
    print("=" * 60)

    h5f.close()
