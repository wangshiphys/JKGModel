"""
Calculate the ground state averages of spin operators of the system
"""


from datetime import datetime
from multiprocessing import current_process, Process, Queue
from pathlib import Path
from time import time
from urllib.request import urlopen

from scipy.sparse import load_npz

import argparse
import io

import mkl
import numpy as np


mkl.set_num_threads(1)

LOG_FMT = "{now:%Y-%m-%d %H:%M:%S} - {process} - {message}\n"


def _DownloadGS(url):
    # Download the Ground state vector from the given url
    with urlopen(url) as fp:
        raw_data = fp.read()

    stream = io.BytesIO(raw_data)
    with np.load(stream) as ld:
        ket = ld["ket"]
    stream.close()
    return ket


def _ReadGS(path):
    # Read the ground state vector from the given path
    with np.load(path) as ld:
        ket = ld["ket"]
    return ket


def LoadGS(root, params, process_num, GSQueue, log_queue):
    """
    Load the ground state vector from a remote server or a local file

    Parameters
    ----------
    root : str
        The root directory where the ground state vectors are stored
        It can be an url started with `http://` which specify the address of
        the remote server, or it can be a local directory
    params : sequence
        A collection of (alpha, beta), which specifies the model parameter
    process_num : int
        The number of process used to calculate the ground state averages
    GSQueue : multiprocessing.Queue
        A queue of ground state vectors waiting to be processed
    log_queue : multiprocessing.Queue
        A queue of log message
    """

    gs_path_template = "GS/alpha={alpha:.3f}/"
    gs_name_template = "GS_numx=4_numy=6_alpha={alpha:.3f}_beta={beta:.3f}.npz"
    url_template = root + gs_path_template + gs_name_template

    if url_template.startswith("http://"):
        load_func = _DownloadGS
    else:
        load_func = _ReadGS

    log_msg_template = "Put GS of (alpha={0:.3f}, beta={1:.3f}) into GSQueue"
    log_msg_template += " - Time used: {2:.3f}s"

    process = current_process()
    for alpha, beta in params:
        t0 = time()
        ket = load_func(url_template.format(alpha=alpha, beta=beta))
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


def GSAverages(GSQueue, log_queue, SMatrices_path):
    """
    Calculate the ground state averages of spin operators of the system

    Parameters
    ----------
    GSQueue : multiprocessing.Queue
        A queue of ground state vectors waiting to be processed
    log_queue : multiprocessing.Queue
        A queue of log message
    SMatrices_path : str
        Where to load the spin matrices
    """

    avg_path_template = "Averages/alpha={0:.3f}/"
    avg_name_template = "Averages_numx=4_numy=6_alpha={0:.3f}_beta={1:.3f}.npz"

    log_msg_template = "GS of (alpha={0:.3f}, beta={1:.3f}) processed"
    log_msg_template += " - Time used: {2:.3f}s"

    SMatrices_name_template = "Total_{0} with spin_number=24.npz"
    SX = load_npz(SMatrices_path + SMatrices_name_template.format("SX"))
    SY = load_npz(SMatrices_path + SMatrices_name_template.format("SY"))
    SZ = load_npz(SMatrices_path + SMatrices_name_template.format("SZ"))
    S2 = load_npz(SMatrices_path + SMatrices_name_template.format("S2"))

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

        avg_path = Path(avg_path_template.format(alpha))
        avg_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            avg_path / avg_name_template.format(alpha, beta),
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


def main(root, params, process_num=3, SMatrices_path=None, log_path=None):
    """
    The entrance for calculating the ground state averages

    Parameters
    ----------
    root : str
        The root directory where the ground state vectors are stored
        It can be an url started with `http://` which specify the address of
        the remote server, or it can be a local directory
    params : sequence
        A collection of (alpha, beta), which specifies the model parameter
    process_num : int, optional
        The number of process used to calculate the ground state averages
        default: 3
    SMatrices_path : str, optional
        Where to load the spin matrices
        The default value `None` implies `/home0/wangshiphys/JKGModel/tmp`
    log_path : str, optional
        Where to save the log
        The default value `None` implies `log/Averages.log` relative to the
        current working directory.
    """

    assert isinstance(process_num, int) and process_num > 0

    if SMatrices_path is None:
        SMatrices_path = "/home0/wangshiphys/JKGModel/tmp/"
    if log_path is None:
        log_path = "log/"

    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    GSQueue = Queue(8 * process_num)
    log_queue = Queue()

    process_table = {}
    p = Process(
        target=LoadGS, args=(root, params, process_num, GSQueue, log_queue)
    )
    p.start()
    process_table[p.pid] = p

    for i in range(process_num):
        p = Process(
            target=GSAverages, args=(GSQueue, log_queue, SMatrices_path)
        )
        p.start()
        process_table[p.pid] = p

    with open(log_dir / "Averages.log", mode="a", buffering=1) as fp:
        tmp = LOG_FMT.format(
            now=datetime.now(), message="Start running", process="Main"
        )
        fp.write(tmp)
        while True:
            log = log_queue.get()
            if log in process_table:
                tmp = LOG_FMT.format(
                    now=datetime.now(), message="Finished",
                    process=process_table[log].name,
                )
                fp.write(tmp)
                process_table[log].join()
                del process_table[log]
            else:
                fp.write(log)
            if len(process_table) == 0:
                break

        tmp = LOG_FMT.format(
            now=datetime.now(), message="Stop running", process="Main"
        )
        fp.write(tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command line arguments."
    )
    parser.add_argument(
        "--root", type=str, default="http://210.28.140.119:8000/",
        help="The root directory where the ground state vectors are stored"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="The value of alpha (Default: %(default)s)."
    )
    parser.add_argument(
        "--proc_num", type=int, default=2,
        help="The number of process to use (Default : %(default)s)."
    )

    args = parser.parse_args()
    root = args.root
    alpha = args.alpha
    process_num = args.proc_num

    params = [[alpha, beta] for beta in np.arange(0, 2, 0.01)]

    main(root=root, params=params, process_num=process_num)
