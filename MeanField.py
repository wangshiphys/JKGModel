"""
Mean-field analysis of the hole-doped J-K-Gamma model on triangular lattice
"""


from scipy.optimize import root

import numpy as np

from LatticeData import BS, RS
from SigmaMatrix import PMS


__all__ = [
    "MeanFieldSolver",
]


class MeanFieldSolver:
    """
    Mean-Field analysis of the hole-doped J-K-Gamma model on triangular lattice
    """

    def __init__(self, numkx=100, numky=None):
        """
        Customize the newly created instance

        Parameters
        ----------
        numkx : int, optional
            The number of k-point along the 1st translation vector in k-space
            default: 100
        numky : int, optional
            The number of k-point along the 2nd translation vector in k-space
            default: numky = numkx
        """

        assert isinstance(numkx, int) and numkx > 0
        assert (numky is None) or (isinstance(numky, int) and numky > 0)

        if numky is None:
            numky = numkx

        # Every k-point in the k-space can be expressed as `x * b1 + y * b2`
        # where `b1` and `b2` are the translation vectors of the reciprocal
        # lattice. For k-points in the first Brillouin Zone, x and y are
        # float numbers in the range [0, 1).
        # The `xys` is a collection of all the `(x, y)`s in the first
        # Brillouin zone. `xys.shape = (numkx * numky, 2)`
        xs = np.linspace(0, 1, numkx, endpoint=False)
        ys = np.linspace(0, 1, numky, endpoint=False)
        xys = [[x, y] for x in xs for y in ys]
        ksRS = np.matmul(xys, np.matmul(BS, RS.T))

        # fks = cos(kr1) + cos(kr2) + cos(kr3)
        # pfs = exp(1j * kr1), exp(1j * kr2), exp(1j * kr3)
        fks = np.sum(np.cos(ksRS), axis=-1)
        pfs = np.exp(1j * ksRS)

        self.numkx = numkx
        self.numky = numky
        self.numk = numkx * numky

        # Cache these data for reuse
        self._fks = fks
        self._pfs = pfs

    def _HMSolver(self, t, mu, order_params):
        # Construct the Mean-Field Hamiltonian Matrix

        hopping = t * self._fks + mu / 2
        pairing = np.tensordot(
            self._pfs,
            np.tensordot(order_params, PMS, axes=(0, 0)),
            axes=(-1, 0)
        )
        pairing_dagger = np.transpose(pairing.conj(), axes=(0, 2, 1))

        HMs = np.zeros((self.numk, 4, 4), dtype=np.complex128)
        HMs[:, 0, 0] = -hopping
        HMs[:, 1, 1] = -hopping
        HMs[:, 2, 2] = hopping
        HMs[:, 3, 3] = hopping
        HMs[:, 0:2, 2:] = pairing
        HMs[:, 2:, 0:2] = pairing_dagger

        # Diagonalize the Mean-Field Hamiltonian Matrix and return these
        # matrices that corresponding to the Bogoliubov transformation
        values, vectors = np.linalg.eigh(HMs)
        U0 = vectors[:, 0:2, 0:2]
        U1 = vectors[:, 2:, 0:2]
        U0d = np.transpose(U0.conj(), axes=(0, 2, 1))
        return U0, U0d, U1

    def AllAverages(self, t, mu, order_params):
        """
        Calculate the ground-state averages of these hole-pairing and
        particle-number operators

        Parameters
        ----------
        t : float
            The hopping amplitude
        mu : float
            Chemical potential
        order_params : 2-D array with shape (4, 3)
            Mean-Field order parameters

        Returns
        -------
        filling : float
            The average number of particle per-site
        averages : 2-D array with shape (4, 3)
            The ground-state averages of these hole-pairing operators
        """

        U0, U0d, U1 = self._HMSolver(t, mu, order_params)

        tmp = np.matmul(U0d, U0)
        filling = (np.sum(tmp[:, 0, 0]) + np.sum(tmp[:, 1, 1])).real

        tmp = np.matmul(U0d, np.matmul(PMS[:, np.newaxis], U1))
        particle_pairings = np.matmul(
            tmp[:, :, 0, 0] + tmp[:, :, 1, 1], self._pfs
        )
        hole_pairings = particle_pairings.conj()

        return filling / self.numk, hole_pairings / self.numk

    def OrderParameters(self, J, K, G, averages):
        """
        Construct order parameters from these pairing operators' average

        Parameters
        ----------
        J : float
            The coefficient of Heisenberg term
        K : float
            The coefficient of the Kitaev term
        G : float
            The coefficient of the Gamma term
        averages : 2-D array with shape (4, 3)
            The original ground-state averages of these hole-pairing operators

        Returns
        -------
        res : 2-D array with shape(4, 3)
            The mean field order parameters
        """

        K = K / 4
        G = G / 2

        delta1, delta2, delta3 = -(J + K) * averages[0]

        d1x = -K * averages[1, 0]
        d2x = K * averages[1, 1] + G * averages[3, 1]
        d3x = K * averages[1, 2] + G * averages[2, 2]

        d1y = K * averages[2, 0] + G * averages[3, 0]
        d2y = -K * averages[2, 1]
        d3y = K * averages[2, 2] + G * averages[1, 2]

        d1z = K * averages[3, 0] + G * averages[2, 0]
        d2z = K * averages[3, 1] + G * averages[1, 1]
        d3z = -K * averages[3, 2]

        return np.array([
            [delta1, delta2, delta3],
            [d1x, d2x, d3x],
            [d1y, d2y, d3y],
            [d1z, d2z, d3z]
        ])


    def _Ansatze0_core(self, params, t, J, K, G, filling):
        mu = params[0]
        order_params = np.reshape(
            params[1::2] + params[2::2] * 1j, newshape=(4, 3)
        )

        filling_new, averages = self.AllAverages(t, mu, order_params)
        order_params_new = self.OrderParameters(J, K, G, averages)
        tmp = (order_params_new - order_params).flatten()
        diff = np.zeros(params.shape, dtype=np.float64)
        diff[0] = filling_new - filling
        diff[1::2] = tmp.real
        diff[2::2] = tmp.imag
        return diff

    def Ansatze0(self, delta, J=0.0, K=0.0, G=0.0, maxiter=10):
        extra_args = (delta, J, K, G, 1 - delta)

        params = "J = {0:.2f}, K = {1:.2f}, G = {2:.2f}, delta ={3:.2f}"
        entry = " " * 4 + "(A={0:.4f}, theta={1: .4f})"
        separator = "=" * 80
        print(params.format(J, K, G, delta))
        print(separator)

        for i in range(maxiter):
            x0 = 2 * np.random.random(25) - 1
            res = root(self._Ansatze0_core, x0=x0, args=extra_args)

            if res.success:
                order_params = res.x[1::2] + res.x[2::2] * 1j
                As = np.absolute(order_params)
                thetas = np.angle(order_params) / np.pi
                global_phase = thetas[0]
                thetas = thetas - global_phase

                print("Chemical potential: {0:.4f}".format(res.x[0]))
                print("Order parameters:")
                for i in range(len(As)):
                    end = "" if (i + 1) % 3 else "\n"
                    A, theta = (0, 0) if As[i] < 1e-4 else (As[i], thetas[i])
                    print(entry.format(A, theta), end=end)
                print(" " * 4 + "Global phase: {0:.4f}".format(global_phase))
            else:
                print("Failed to find a self-consistent solution!")
            print(separator, flush=True)


if __name__ == "__main__":
    J, K = 0.0, -1.0
    G, delta = -1.0, 0.04
    solver = MeanFieldSolver(100)
    solver.Ansatze0(delta=delta, J=J, K=K, G=G)