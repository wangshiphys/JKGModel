import matplotlib.pyplot as plt
import tables as tb

from utilities import TriangularLattice

data_path = "data/ClassicalSpinModel/OptimizedSpinConfig/"
h5f_name_temp = "OSC_num1={0}_num2={1}_direction={2}_" \
                "alpha={3:.4f}_beta={4:.4f}.h5"

alpha = 0.50
beta = 0.50
num1 = num2 = 12
direction = "xz"
lattice = TriangularLattice(num1=num1, num2=num2, direction="xz")
points = lattice.cluster.points

h5f_name = h5f_name_temp.format(num1, num2, direction, alpha, beta)
h5f = tb.open_file(data_path + h5f_name, mode="r")
for carray in h5f.iter_nodes("/"):
    spin_vectors = carray.read()
    energy = h5f.get_node_attr(carray, "energy")

    fig, ax = plt.subplots(num=carray.name)
    ax.plot(
        points[:, 0], points[:, 1],
        color="k", ls="", marker="o", ms=5
    )
    ax.quiver(
        points[:, 0], points[:, 1],
        spin_vectors[:, 0], spin_vectors[:, 1],
        pivot="middle", color=0.5*spin_vectors+0.5,
    )
    ax.set_title("Energy = {0:.10f}".format(energy), fontsize="xx-large")
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
h5f.close()
