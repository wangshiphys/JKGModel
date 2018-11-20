"""
Calculate the ground state averages of spin operators of the system
"""


from datetime import datetime
from multiprocessing import current_process, Process, Queue
from pathlib import Path
from time import time

from scipy.sparse import load_npz

import mkl
import numpy as np


mkl.set_num_threads(1)

LOG_FMT = "{now:%Y-%m-%d %H:%M:%S} - {process} - {message}\n"


def LoadGS(params, process_num, GSQueue, log_queue, numx=4, numy=6):
    """
    Load the ground state vector

    Parameters
    ----------
    params : sequence
        A collection of (alpha, beta), which specifies the model parameter
    process_num : int
        The number of process used to calculate the ground state averages
    GSQueue : multiprocessing.Queue
        A queue of ground state vectors waiting to be processed
    log_queue : multiprocessing.Queue
        A queue of log message
    numx : int, optional
        The number of lattice site along the first translation vector
        default: 4
    numy : int, optional
        The number of lattice site along the second translation vector
        default: 6
    """

    gs_path_template = "data/SpinModel/GS/alpha={alpha:.3f}/"
    gs_name_template = "GS_numx={0}_numy={1}".format(numx, numy)
    gs_name_template += "_alpha={alpha:.3f}_beta={beta:.3f}.npz"
    gs_full_name_template = gs_path_template + gs_name_template

    log_msg_template = "alpha={0:.3f}, beta={1:.3f} ready - Time used: {2:.3f}s"

    process = current_process()
    for alpha, beta in params:
        t0 = time()
        gs_full_name = gs_full_name_template.format(alpha=alpha, beta=beta)
        with np.load(gs_full_name) as ld:
            ket = ld["ket"]
        t1 = time()

        GSQueue.put((alpha, beta, ket))
        log = LOG_FMT.format(
            now=datetime.now(), process=process.name,
            message=log_msg_template.format(alpha, beta, t1 - t0)
        )
        log_queue.put(log)

    for i in range(process_num):
        GSQueue.put(None)

    log_queue.put(process.pid)
    GSQueue.close()
    log_queue.close()


def GSAverages(GSQueue, log_queue, numx=4, numy=6):
    """
    Calculate the ground state averages of spin operators of the system

    Parameters
    ----------
    GSQueue : multiprocessing.Queue
        A queue of ground state vectors waiting to be processed
    log_queue : multiprocessing.Queue
        A queue of log message
    numx : int, optional
        The number of lattice site along the first translation vector
        default: 4
    numy : int, optional
        The number of lattice site along the second translation vector
        default: 6
    """

    avg_path_template = "data/SpinModel/Averages/alpha={alpha:.3f}/"
    avg_name_template = "Averages_numx={0}_numy={1}".format(numx, numy)
    avg_name_template += "_alpha={alpha:.3f}_beta={beta:.3f}.npz"

    log_msg_template = "alpha={0:.3f}, beta={1:.3f} done  - Time used: {2:.3f}s"

    SMatrices_name_template = "tmp/Total_{{0}} with spin_number={0}.npz".format(
        numx * numy
    )
    SX = load_npz(SMatrices_name_template.format("SX"))
    SY = load_npz(SMatrices_name_template.format("SY"))
    SZ = load_npz(SMatrices_name_template.format("SZ"))
    S2 = load_npz(SMatrices_name_template.format("S2"))

    process = current_process()
    while True:
        t0 = time()
        task = GSQueue.get()
        if task is None:
            break
        alpha, beta, ket = task

        SX_avg = np.vdot(ket, SX.dot(ket))
        SY_avg = np.vdot(ket, SY.dot(ket))
        SZ_avg = np.vdot(ket, SZ.dot(ket))
        S2_avg = np.vdot(ket, S2.dot(ket))

        avg_path = Path(avg_path_template.format(alpha=alpha))
        avg_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            avg_path / avg_name_template.format(alpha=alpha, beta=beta),
            SX=[SX_avg], SY=[SY_avg], SZ=[SZ_avg], S2=[S2_avg]
        )
        t1 = time()

        log = LOG_FMT.format(
            now=datetime.now(), process=process.name,
            message=log_msg_template.format(alpha, beta, t1 - t0)
        )
        log_queue.put(log)

    log_queue.put(process.pid)
    log_queue.close()


def main(params, process_num=2, numx=4, numy=6):
    """
    The entrance for calculating the ground state averages

    Parameters
    ----------
    params : sequence
        A collection of (alpha, beta), which specifies the model parameter
    process_num : int, optional
        The number of process used to calculate the ground state averages
        default: 2
    numx : int, optional
        The number of lattice site along the first translation vector
        default: 4
    numy : int, optional
        The number of lattice site along the second translation vector
        default: 6
    """

    assert isinstance(process_num, int) and process_num > 0

    log_path = Path("log/SpinModel/")
    log_path.mkdir(parents=True, exist_ok=True)

    GSQueue = Queue(8 * process_num)
    log_queue = Queue()

    process_table = {}
    p = Process(
        target=LoadGS,
        args=(params, process_num, GSQueue, log_queue, numx, numy)
    )
    p.start()
    process_table[p.pid] = p

    for i in range(process_num):
        p = Process(
            target=GSAverages,
            args=(GSQueue, log_queue, numx, numy)
        )
        p.start()
        process_table[p.pid] = p

    with open(log_path / "Averages.log", mode="a", buffering=1) as fp:
        log = LOG_FMT.format(
            now=datetime.now(), message="Start running", process="Main"
        )
        fp.write(log)
        while True:
            log = log_queue.get()
            if log in process_table:
                new_log = LOG_FMT.format(
                    now=datetime.now(), message="Finished",
                    process=process_table[log].name,
                )
                fp.write(new_log)
                process_table[log].join()
                del process_table[log]
            else:
                fp.write(log)
            if len(process_table) == 0:
                break

        log = LOG_FMT.format(
            now=datetime.now(), message="Stop running", process="Main"
        )
        fp.write(log)


if __name__ == "__main__":
    alpha = 0.01
    process_num = 3
    params = [[alpha, beta] for beta in np.arange(0, 0.1, 0.01)]
    main(params=params, process_num=process_num)
