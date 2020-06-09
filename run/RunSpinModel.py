import matplotlib.pyplot as plt
import numpy as np

from SpinModel import JKGModelEDSolver


def derivation(xs, ys, nth=1):
    """
    Calculate the nth derivatives of `ys` versus `xs` discretely.

    The derivatives are calculated using the following formula:
        dy / dx = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])

    Parameters
    ----------
    xs : 1-D array
        The independent variables.
        `xs` is assumed to be sorted in ascending order and there are no
        identical values in `xs`.
    ys : 1-D array
        The dependent variables.
        `ys` should be of the same length as `xs`.
    nth : int, optional
        The nth derivatives.
        Default: 1.

    Returns
    -------
    xs : 1-D array
        The independent variables.
    ys : 1-D array
        The nth derivatives corresponding to the returned `xs`.
    """

    assert isinstance(nth, int) and nth >= 0
    assert isinstance(xs, np.ndarray) and xs.ndim == 1
    assert isinstance(ys, np.ndarray) and ys.shape == xs.shape

    for i in range(nth):
        ys = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        xs = (xs[1:] + xs[:-1]) / 2
    return xs, ys


num1 = 3
num2 = 4
direction = "xy"
alpha = 0.5
betas = np.arange(0, 2, 0.01)

solver = JKGModelEDSolver(num1, num2, direction=direction)
for beta in betas:
    solver.EigenStates(alpha=alpha, beta=beta)

gses = []
es_path = "data/QuantumSpinModel/ES/"
es_name_temp = "ES_" + solver.identity + "_alpha={0:.4f}_beta={1:.4f}.npz"
for beta in betas:
    es_full_name = es_path + es_name_temp.format(alpha, beta)
    with np.load(es_full_name) as ld:
        gses.append(ld["values"][0])
gses = np.array(gses, dtype=np.float64)
d2betas, d2gses = derivation(betas, gses, nth=2)

fig, ax_gses = plt.subplots(num=solver.identity)
ax_d2gses = ax_gses.twinx()
color_gses = "tab:blue"
color_d2gses = "tab:orange"
ax_gses.plot(betas, gses, color=color_gses)
ax_d2gses.plot(d2betas, -d2gses / (np.pi ** 2), color=color_d2gses)

ax_gses.set_xlim(betas[0], betas[-1])
ax_gses.set_title(r"$\alpha={0:.4f}\pi$".format(alpha), fontsize="xx-large")
ax_gses.set_xlabel(r"$\beta/\pi$", fontsize="xx-large")
ax_gses.set_ylabel("E", rotation=0, fontsize="xx-large", color=color_gses)
ax_gses.tick_params("y", colors=color_gses)
ax_gses.grid(ls="dashed", axis="x", color="gray")
ax_gses.grid(ls="dashed", axis="y", color=color_gses)

ax_d2gses.set_ylabel(
    r"$E^{''}$", rotation=0, fontsize="xx-large", color=color_d2gses
)
ax_d2gses.tick_params("y", colors=color_d2gses)
plt.get_current_fig_manager().window.showMaximized()
plt.show()
plt.close("all")
