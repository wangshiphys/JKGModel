from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tables as tb

from HamiltonianPy import lattice_generator
from utilities import ShowVectorField3D


data_path = "data/ClassicalSpinModel/OptimizedSpinConfig/"
h5f_name_temp = "OSC_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.h5"

alpha = 0.5
beta = 1.85
numx = numy = 12
cluster = lattice_generator("triangle", num0=numx, num1=numy)

h5f_name = h5f_name_temp.format(numx, numy, alpha, beta)
h5f = tb.open_file(data_path + h5f_name, mode="r")
for carray in h5f.iter_nodes("/"):
    energy = h5f.get_node_attr(carray, "energy")
    spin_vectors = carray.read()
    print(spin_vectors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    title = "{0}\n".format(carray.name)
    title += r"$\alpha={0:.4f}\pi,\beta={1:.4f}\pi,E_{{min}}={2:.8f}$".format(
        alpha, beta, energy
    )
    ShowVectorField3D(
        ax, cluster.points, spin_vectors, title=title
    )
    ax.set_zlim(-0.5, 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
h5f.close()
