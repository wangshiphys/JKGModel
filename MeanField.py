"""
This module provide mean-field analysis of the t-J-K-Gamma model on
triangular lattice
"""


from scipy.optimize import root
from time import time

import numpy as np

from LatticeInfo import BS, RS
from SigmaMatrix import PMS


class BaseMeanFieldSolver:
    """
    Mean-Field solution of the t-J-K-Gamma model on triangular lattice
    """

    def __init__(self, numkx=100, numky=None):
        """
        Customize the newly created instance

        Parameters
        ----------
        numkx : int, optional
            The number of k-points along the 1st translation vector in k-space
            default: 100
        numky : int, optional
            The number of k-points along the 2nd translation vector in k-space
            default: numky = numkx
        """

        if not isinstance(numkx, int) or numkx < 1:
            raise ValueError("The `numkx` parameter must be positive integer!")
        if numky is None:
            numky = numkx
        else:
            if not isinstance(numky, int) or numky < 1:
                raise ValueError(
                    "The `numky` parameter must be positive integer!"
                )

        # Every k-point in the first Brillouin Zone can be represented as
        # `x * b1 + y * b2` where b1 and b2 are the translation vectors of
        # the reciprocal lattice and x, y are float number in the range [0, 1)
        # The `ratio_mesh` is a collection of all the `(x, y)`s in the first
        # Brillouin zone. `ratio_mesh.shape = (numkx, numky, 2)`
        ratio_x = np.linspace(0, 1, numkx, endpoint=False)
        ratio_y = np.linspace(0, 1, numky, endpoint=False)
        ratio_mesh = np.stack(
            np.meshgrid(ratio_x, ratio_y, indexing="ij"), axis=-1
        )
        ksRS = np.matmul(np.matmul(ratio_mesh, BS), RS.T)

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

    def _HMSolver(self, t, mu, rescaled_averages):
        # Construct the Mean-Field Hamiltonian Matrix according to the input
        # parameters.

        hopping = t * self._fks + mu / 2
        pairing = np.tensordot(
            self._pfs, np.tensordot(rescaled_averages, PMS, axes=(0, 0)),
            axes=(2, 0)
        )
        pairing_dagger = np.transpose(pairing.conj(), axes=(0, 1, 3, 2))

        HMs = np.zeros((self.numkx, self.numky, 4, 4), dtype=np.complex128)
        HMs[:, :, 0, 0] = -hopping
        HMs[:, :, 1, 1] = -hopping
        HMs[:, :, 2, 2] = hopping
        HMs[:, :, 3, 3] = hopping
        HMs[:, :, 0:2, 2:] = pairing
        HMs[:, :, 2:, 0:2] = pairing_dagger

        # Diagonalize the Mean-Field Hamiltonian Matrix and return these
        # matrices that corresponding to the Bogoliubov transformation
        ws, vs = np.linalg.eigh(HMs)
        U0 = vs[:, :, 0:2, 0:2]
        U1 = vs[:, :, 2:, 0:2]
        U0d = np.transpose(U0.conj(), axes=(0, 1, 3, 2))
        return U0, U0d, U1

    def AllAverages(self, t, mu, rescaled_averages):
        """
        Calculate the ground-state averages of these hole-pairing and
        particle-number operators

        Parameters
        ----------
        t : float
            The hopping amplitude
        mu : float
            Chemical potential
        rescaled_averages : 2-D array with shape (4, 3)
            Mean-Field ansatz of the rescaled ground-state averages of these
            hole-pairing operators

        Returns
        -------
        filling : float
            The ground-state average of the particle-number operators
        averages : 2-D array with shape (4, 3)
            The ground-state averages of these hole-pairing operators
        """

        U0, U0d, U1 = self._HMSolver(t, mu, rescaled_averages)

        ppairings = np.stack(
            [self._PairingTermAverage(U0d, U1, which=i) for i in range(4)]
        ) / self.numk
        hpairings = ppairings.conj()

        filling = self._ParticleNumberAverage(U0d, U0) / self.numk
        return filling, hpairings

    def RescaleAverages(self, J, K, G, averages):
        """
        Rescale the ground-state averages of these hole-pairing operators

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
            The rescaled ground-state averages of these hole-pairing operators
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

    def _ParticleNumberAverage(self, Ud, U):
        # Calculate the ground-state average of the particle-number operator

        tmp = np.matmul(Ud, U)
        return (np.sum(tmp[:, :, 0, 0]) + np.sum(tmp[:, :, 1, 1])).real

    def _PairingTermAverage(self, U0d, U1, *, which):
        # Calculate the ground-state averages of these particle-pairing
        # operators
        # The `which` parameter should be only in (0, 1, 2, 3)
        # 0 correspond to singlet-pairing and 1, 2, 3 correspond to
        # three triplet-pairing

        tmp = np.matmul(U0d, np.matmul(PMS[which], U1))
        return np.tensordot(
            self._pfs, tmp[:, :, 0, 0] + tmp[:, :, 1, 1], axes=([0, 1], [0, 1])
        )

    def _RootFunc(self, params_in, dtype, t, J, K, G, filling):
        mu = params_in[0]
        if dtype == "real":
            averages_in = params_in[1:]
        elif dtype == "imag":
            averages_in = params_in[1:] * 1j
        else:
            averages_in = params_in[1::2] * np.exp(2j * np.pi * params_in[2::2])
        tmp = averages_in.reshape((4, 3))

        rescaled_averages = self.RescaleAverages(J, K, G, tmp)
        filling_out, averages_out = self.AllAverages(t, mu, rescaled_averages)

        averages_diff = averages_out.flatten() - averages_in
        diff = np.zeros(params_in.shape, dtype=np.float64)
        diff[0] = filling_out - filling
        if dtype == "real" or dtype == "imag":
            diff[1:] = np.absolute(averages_diff)
        else:
            diff[1::2] = averages_diff.real
            diff[2::2] = averages_diff.imag
        return diff

    def __call__(self, delta, order_params_num, dtype="", J=0.0, K=0.0, G=0.0):
        t, filling = delta, 1 - delta

        mu = np.random.randn()
        amplitude = np.absolute(np.random.randn(order_params_num))
        if dtype == "real" or dtype == "imag":
            x0 = np.zeros(1 + order_params_num, dtype=np.float64)
            x0[0] = mu
            x0[1:] = amplitude
        else:
            x0 = np.zeros(1 + 2 * order_params_num, dtype=np.float64)
            x0[0] = mu
            x0[1::2] = amplitude
            x0[2::2] = np.random.random(order_params_num)
        extra_args = (dtype, t, J, K, G, filling)

        t0 = time()
        res = root(self._RootFunc, x0=x0, args=extra_args)
        t1 = time()

        params = "J={0:.2f}, K={1:.2f}, G={2:.2f}, delta={3:.2f}"
        print("The current model parameters:", params.format(J, K, G, delta))

        indent = " " * 4
        if res.success:
            sol = res.x
            print("The chemical potential: {0: .5f}".format(sol[0]))
            print("The order parameters:")
            if dtype == "real":
                for i in range(1, len(sol)):
                    print(indent + "{0: .5f}".format(sol[i]))
            elif dtype == "imag":
                for i in range(1, len(sol)):
                    print(indent + "{0: .5f}j".format(sol[i]))
            else:
                for i in range(1, len(sol), 2):
                    print(
                        indent + "({0: .5f}, {1: .5f})".format(sol[i], sol[i+1])
                    )
        else:
            print("Failed to find a self-consistent solution!")
        print("The time spend on finding the solution: {0}s".format(t1 - t0))
        print("=" * 80, flush=True)


class SWaveMeanFieldSolver(BaseMeanFieldSolver):
    def _RootFunc(self, params_in, dtype, t, J, K, G, filling):
        mu = params_in[0]
        if dtype == "real":
            averages_in = params_in[1]
        elif dtype == "imag":
            averages_in = params_in[1] * 1j
        else:
            averages_in = params_in[1] * np.exp(2j * np.pi * params_in[2])
        tmp = -(J + K/4) * averages_in
        rescaled_averages =  np.array(
            [[tmp, tmp, tmp], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )

        U0, U0d, U1 = self._HMSolver(t, mu, rescaled_averages)
        ppairing0 = self._PairingTermAverage(U0d, U1, which=0) / self.numk
        hpairing0 = ppairing0.conj()
        filling_out = self._ParticleNumberAverage(U0d, U0) / self.numk

        averages_diff = np.mean(hpairing0) - averages_in
        diff = np.zeros(params_in.shape, dtype=np.float64)
        diff[0] = filling_out - filling
        if dtype == "real" or dtype == "imag":
            diff[1] = np.absolute(averages_diff)
        else:
            diff[1] = averages_diff.real
            diff[2] = averages_diff.imag
        return diff


class PWaveMeanFieldSolver(BaseMeanFieldSolver):
    def _RootFunc(self, params_in, dtype, t, J, K, G, filling):
        mu = params_in[0]
        if dtype == "real":
            averages_in = params_in[1:]
            tmp = np.zeros((4, 3), dtype=np.float64)
        elif dtype == "imag":
            averages_in = params_in[1:] * 1j
            tmp = np.zeros((4, 3), dtype=np.complex128)
        else:
            averages_in = params_in[1::2] * np.exp(2j * np.pi * params_in[2::2])
            tmp = np.zeros((4, 3), dtype=np.complex128)
        tmp[1, :] = averages_in[0]
        tmp[2, :] = averages_in[1]
        tmp[3, :] = averages_in[2]

        rescaled_averages = self.RescaleAverages(J, K, G, tmp)
        U0, U0d, U1 = self._HMSolver(t, mu, rescaled_averages)

        ppairings = np.stack(
            [self._PairingTermAverage(U0d, U1, which=i) for i in range(1, 4)]
        ) / self.numk
        hpairings = ppairings.conj()
        filling_out = self._ParticleNumberAverage(U0d, U0) / self.numk

        averages_diff = np.mean(hpairings, axis=1) - averages_in

        diff = np.zeros(params_in.shape, dtype=np.float64)
        diff[0] = filling_out - filling
        if dtype == "real" or dtype == "imag":
            diff[1:] = np.absolute(averages_diff)
        else:
            diff[1::2] = averages_diff.real
            diff[2::2] = averages_diff.imag
        return diff


if __name__ == "__main__":
    solver = PWaveMeanFieldSolver(100)
    for i in range(10):
        solver(
            delta=0.2, K=-0.5, G=0.0, order_params_num=3, dtype="real"
        )
