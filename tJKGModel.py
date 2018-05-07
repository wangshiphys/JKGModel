"""
This module provide mean-field analysis of the t-J-K-Gamma model on
triangular lattice
"""


import numpy as np
import scipy.optimize as spopt
import sys

from datetime import datetime
from time import time

from SigmaMatrix import PMS


np.set_printoptions(suppress=True)


class MeanFieldSolver:
    """
    Mean-Field solution of the t-J-K-Gamma model on triangular lattice

    Class Attributes
    ----------------
    AS : 2-D array
        The real space translation vectors
    BS : 2-D array
        The reciprocal space (k space) translation vectors
    RS : 2-D array
        The displaces to three nonequivalent nearest neighbors on triangular
        lattice
    """

    AS = np.array([[1, 0], [0.5, np.sqrt(3)/2]], dtype=np.float64)
    RS = np.array(
        [[1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2]], dtype=np.float64
    )
    BS = 2 * np.pi * np.array(
        [[1, -1/np.sqrt(3)], [0, 2/np.sqrt(3)]], dtype=np.float64
    )

    AS.setflags(write=False)
    BS.setflags(write=False)
    RS.setflags(write=False)

    def __init__(self, t=1.0, J=0.0, K=0.0, G=0.0, filling=1.0, kx_num=200,
                 ky_num=None):
        """
        Customize the newly created instance

        Parameters
        ----------
        t : float, optional
            The hopping amplitude
            default: 1.0
        J : float, optional
            The coefficient of the Heisenberg term
            default: 0.0
        K : float, optional
            The coefficient of the Kitaev term
            default: 0.0
        G : float, optional
            The coefficient of the Gamma term
            default: 0.0
        filling : float, optional
            The number of particle per-site
            default: 1.0 (half filling)
        kx_num : int, optional
            The number of k point along the first translation vector in k space
            default: 200
        ky_num : int, optional
            The number of k point along the second translation vector in k space
            default: ky_num = kx_num
        """

        if ky_num is None:
            ky_num = kx_num
        ratio_x = np.linspace(0, 1, kx_num, endpoint=False)
        ratio_y = np.linspace(0, 1, ky_num, endpoint=False)
        ratio_mesh = np.stack(
            np.meshgrid(ratio_x, ratio_y, indexing="ij"), axis=-1
        )

        ks = np.matmul(ratio_mesh, self.BS)
        ksRS = np.matmul(ks, self.RS.T)
        fks = np.sum(np.cos(ksRS), axis=-1)
        pfs = np.exp(1j * ksRS)

        self.t = t
        self.J = J
        self.K = K
        self.G = G
        self.filling = filling
        self.kx_num = kx_num
        self.ky_num = ky_num
        self.ks_num = kx_num * ky_num
        self._fks = fks
        self._pfs = pfs

    def UpdateModelParams(self, t=None, J=None, K=None, G=None, filling=None):
        if t is not None:
            self.t = t
        if J is not None:
            self.J = J
        if K is not None:
            self.K = K
        if G is not None:
            self.G = G
        if filling is not None:
            self.filling = filling

    def _HSolver(self, mu, coeff_matrix):
        hopping_plus = self.t * self._fks + mu / 2
        hopping_minus = -hopping_plus
        pairing = np.tensordot(
            self._pfs,
            np.tensordot(coeff_matrix, PMS, axes=(0, 0)),
            axes=(2, 0)
        )

        HKs = np.zeros((self.kx_num, self.ky_num, 4, 4), dtype=np.complex128)
        HKs[:, :, 0, 0] = hopping_minus
        HKs[:, :, 1, 1] = hopping_minus
        HKs[:, :, 2, 2] = hopping_plus
        HKs[:, :, 3, 3] = hopping_plus
        HKs[:, :, 0:2, 2:] = pairing
        HKs[:, :, 2:, 0:2] = np.transpose(pairing.conj(), axes=(0, 1, 3, 2))

        ws, vs = np.linalg.eigh(HKs)
        U0 = vs[:, :, 0:2, 0:2]
        U1 = vs[:, :, 2:, 0:2]
        U0_dag = np.transpose(U0.conj(), axes=(0, 1, 3, 2))
        return U0, U0_dag, U1

    def _CoeffMatrixGenerator(self, averages):
        # Combine the model parameters J, K, G with these operators' averages
        # to get the total coefficient

        J = self.J
        K = self.K / 4
        G = self.G / 2

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

    @staticmethod
    def _ParticleNumber(Ud, U):
        tmp = np.matmul(Ud, U)
        return (np.sum(tmp[:, :, 0, 0]) + np.sum(tmp[:, :, 1, 1])).real

    def _PairingTermAverage(self, U0_dag, U1, which=0):
        tmp = np.matmul(U0_dag, np.matmul(PMS[which], U1))
        tmp = tmp[:, :, 0, 0] + tmp[:, :, 1, 1]
        return np.tensordot(self._pfs, tmp, axes=([0, 1], [0, 1]))

    def _AllAverages(self, U0_dag, U1):
        As = []
        for PM in PMS:
            tmp = np.matmul(U0_dag, np.matmul(PM, U1))
            tmp = tmp[:, :, 0, 0] + tmp[:, :, 1, 1]
            As.append(np.tensordot(self._pfs, tmp, axes=([0, 1], [0, 1])))
        return np.stack(As)

    def _sPairingCore(self, params_in, find_root=True):
        mu = params_in[0]
        tmp = np.zeros((4, 3), dtype=np.complex128)
        tmp[0, :] = params_in[1] + params_in[2] * 1j

        coeff_matrix = self._CoeffMatrixGenerator(tmp)
        U0, U0_dag, U1 = self._HSolver(mu, coeff_matrix)

        pairing0 = self._PairingTermAverage(U0_dag, U1, which=0)
        pairing0 = pairing0.conj() / self.ks_num
        outputs = np.mean(pairing0)

        filling_out = self._ParticleNumber(U0_dag, U0) / self.ks_num

        if find_root:
            difference = np.zeros(params_in.shape, dtype=np.float64)
            difference[0] = filling_out - self.filling
            difference[1::2] = outputs.real - params_in[1::2]
            difference[2::2] = outputs.imag - params_in[2::2]
            return difference
        else:
            return filling_out, outputs

    def _pPairingCore(self, params_in, find_root=True):
        mu = params_in[0]
        tmp = np.zeros((4, 3), dtype=np.complex128)
        tmp[1, :] = params_in[1] + params_in[2] * 1j
        tmp[2, :] = params_in[3] + params_in[4] * 1j
        tmp[3, :] = params_in[5] + params_in[6] * 1j

        coeff_matrix = self._CoeffMatrixGenerator(tmp)
        U0, U0_dag, U1 = self._HSolver(mu, coeff_matrix)

        As = []
        for i in range(1, 4):
            tmp = np.matmul(U0_dag, np.matmul(PMS[i], U1))
            tmp = tmp[:, :, 0, 0] + tmp[:, :, 1, 1]
            As.append(np.tensordot(self._pfs, tmp, axes=([0, 1], [0, 1])))
        As = np.stack(As).conj() / self.ks_num
        outputs = np.mean(As, axis=-1)

        filling_out = self._ParticleNumber(U0_dag, U0) / self.ks_num

        if find_root:
            difference = np.zeros(params_in.shape, dtype=np.float64)
            difference[0] = filling_out - self.filling
            difference[1::2] = outputs.real - params_in[1::2]
            difference[2::2] = outputs.imag - params_in[2::2]
            return difference
        else:
            return filling_out, outputs

    def _didPairingCore(self, params_in, find_root=True):
        mu = params_in[0]
        tmp = np.zeros((4, 3), dtype=np.complex128)
        tmp[0, 0] = params_in[1] + params_in[2] * 1j
        tmp[0, 1] = np.exp(2j*np.pi/3) * tmp[0, 0]
        tmp[0, 2] = np.exp(-2j*np.pi/3) * tmp[0, 0]

        coeff_matrix = self._CoeffMatrixGenerator(tmp)
        U0, U0_dag, U1 = self._HSolver(mu, coeff_matrix)

        pairing0 = self._PairingTermAverage(U0_dag, U1, which=0)
        pairing0 = pairing0.conj() / self.ks_num

        outputs = pairing0[0]
        filling_out = self._ParticleNumber(U0_dag, U0) / self.ks_num

        if find_root:
            difference = np.zeros(params_in.shape, dtype=np.float64)
            difference[0] = filling_out - self.filling
            difference[1::2] = outputs.real - params_in[1::2]
            difference[2::2] = outputs.imag - params_in[2::2]
            return difference
        else:
            return filling_out, outputs

    def _gPairingCore(self, params_in, find_root=True):
        mu = params_in[0]
        tmp = np.reshape(params_in[1::2] + params_in[2::2]*1j, newshape=(4, 3))

        coeff_matrix = self._CoeffMatrixGenerator(tmp)
        U0, U0_dag, U1 = self._HSolver(mu, coeff_matrix)

        pairings = self._AllAverages(U0_dag, U1).conj()/self.ks_num

        outputs = pairings.flatten()
        filling_out = self._ParticleNumber(U0_dag, U0) / self.ks_num

        if find_root:
            difference = np.zeros(params_in.shape, dtype=np.float64)
            difference[0] = filling_out - self.filling
            difference[1::2] = outputs.real - params_in[1::2]
            difference[2::2] = outputs.imag - params_in[2::2]
            return difference
        else:
            return filling_out, outputs

    def __call__(self, delta=None, J=None, K=None, G=None, ptype=None, file=None):
        self.UpdateModelParams(t=delta, J=J, K=K, G=G, filling=1-delta)

        if ptype == "s":
            CoreFunc = self._sPairingCore
            x0 = np.random.randn(3)
            pairing_type = "s weave"
        elif ptype == "d+id":
            CoreFunc = self._didPairingCore
            x0 = np.random.randn(3)
            pairing_type = "d+id weave"
        elif ptype == "p":
            CoreFunc = self._pPairingCore
            x0 = np.random.randn(7)
            pairing_type = "p weave"
        else:
            CoreFunc = self._gPairingCore
            x0 = np.random.randn(25)
            pairing_type = None

        t0 = time()
        res = spopt.root(CoreFunc, x0)
        t1 = time()

        print(f"J={self.J}, K={self.K}, G={self.G}, delta={delta}", file=file)
        print(f"The assumed pairing type: {pairing_type}", file=file)
        if res.success:
            print("res.fun:", res.fun, file=file)
            print("res.success:", res.success, file=file)
            print("res.x:", res.x, file=file)
        else:
            print("Failed to find a solution!", file=file)
        print(f"The time spend on finding the solution: {t1-t0}s", file=file)
        print('=' * 80, file=file, flush=True)

        return res


if __name__ == "__main__":
    start_time = "Program start running at: {0:%Y-%m-%d %H:%M:%S}"
    msg = "The time spend on initializing the solver: {0}s"
    file_name = "log/log_J_{0:.1f}_K{1:.1f}.txt"

    J = 1.5
    K = -0.5
    t0 = time()
    solver = MeanFieldSolver(J=J, K=K)
    t1 = time()

    with open(file_name.format(J, K), mode='a', buffering=1) as fp:
        print(start_time.format(datetime.now()), file=fp)
        print(msg.format(t1-t0), file=fp, flush=True)
        for i in range(50):
            res = solver(delta=0.1, ptype='s', file=fp)