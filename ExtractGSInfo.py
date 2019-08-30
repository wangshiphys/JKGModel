import logging
import warnings
from pathlib import Path
from time import time

import numpy as np
import tables as tb

logging.getLogger(__name__).addHandler(logging.NullHandler())
warnings.filterwarnings("ignore", category=tb.NaturalNameWarning)


def CreateEntrance(
        where="data/SpinModel/", target_path="", numx=4, numy=6,
        step=0.001, alpha_start=0, alpha_end=1, beta_start=0, beta_end=2
):
    """
    Create an entrance hdf5-file to access the stored ground state data.

    Parameters
    ----------
    where : str, optional
        Where to save the generated entrance hdf5-file. It can be an absolute
        path or a path relatively to the current working directory(CWD).
        Default: "data/SpinModel/"(Created if not exist, relative to CWD).
    target_path : str, optional
        Where linked the ground state data files are located. It can be an
        absolute path or a path relative to `where`.
        Default: ""(Locate at the same place as the entrance file).
    numx : int, optional
        The number of lattice site along the 1st translation vector.
        Default: 4.
    numy : int
        The number of lattice site along the 2nd translation vector.
        Default: 6.
    step, alpha_start, alpha_end, beta_start, beta_end : float, optional
        Determine the available model parameters: alphas and betas.
        alphas = np.arange(alpha_start, alpha_end + step, step)
        betas = np.arange(beta_start, beta_end + step, step)
    """

    alphas = np.arange(alpha_start, alpha_end + step, step)
    betas = np.arange(beta_start, beta_end + step, step)

    prefix = "GS_numx={numx}_numy={numy}".format(numx=numx, numy=numy)
    target_file_name_temp = prefix + "_alpha={alpha:.3f}.lzo.h5"
    target_ket_name_temp  = prefix + "_alpha={alpha:.3f}_beta={beta:.3f}"
    target_full_name_temp = "".join(
        [target_path, target_file_name_temp, ":/", target_ket_name_temp]
    )

    # If `where` does not exist, create it.
    Path(where).mkdir(parents=True, exist_ok=True)
    main_file_name = "".join([where, prefix, ".h5"])
    group_name_temp = "alpha={alpha:.3f}"
    link_name_temp= target_ket_name_temp

    h5f = tb.open_file(main_file_name, mode="w")
    h5f.set_node_attr("/", "numx", numx)
    h5f.set_node_attr("/", "numy", numy)

    logger = logging.getLogger(__name__).getChild("CreateEntrance")
    for alpha in alphas:
        start = time()
        group = h5f.create_group("/", group_name_temp.format(alpha=alpha))
        h5f.set_node_attr(group, "alpha", alpha)
        for beta in betas:
            target_name = target_full_name_temp.format(alpha=alpha, beta=beta)
            link_name = link_name_temp.format(alpha=alpha, beta=beta)
            h5f.create_external_link(group, link_name, target_name)
        end = time()
        logger.info("alpha=%.3f: %.6fs", alpha, end - start)
    h5f.close()


# Define the table for storing the ground state information
# The table has four columns
# The 1st and 2nd columns record the model parameters 'alpha' and 'beta'
# The 3rd column store the ground state energies 'gse'
# The 4th column store the ground state averages of spin operators
# The 'averages' column has four sub-columns, which store averages for
# total_s2, total_sx, total_sy and total_sz respectively.
# The structure of the table looks like the following:
# -----------------------------------------------------------------
#  alpha | beta | gse |              averages                     |
# -----------------------------------------------------------------
#                     | total_s2 | total_sx | total_sy | total_sz |
# -----------------------------------------------------------------
class GSInfoRecord(tb.IsDescription):
    alpha = tb.Float64Col(pos=0)
    beta = tb.Float64Col(pos=1)
    gse = tb.Float64Col(pos=2)

    class averages(tb.IsDescription):
        _v_pos = 3
        s2 = tb.ComplexCol(16, pos=0)
        sx = tb.ComplexCol(16, pos=1)
        sy = tb.ComplexCol(16, pos=2)
        sz = tb.ComplexCol(16, pos=3)


def ExtractGSInfo(
        where="data/SpinModel/", entrance_file_name=None, numx=4, numy=6,
):
    """
    Extract ground state information from the stored ground state data.

    Save the extracted information to a table specified by the `GSInfoRecord`.

    Parameters
    ----------
    where : str, optional
        Where to save the extracted information. It can be an absolute path
        or a path relative to the currently working directory(CWD).
        Default: "data/SpinModel/"(Created if not existed, relative to CWD).
    entrance_file_name : str, optional
        The full name of the entrance hd5f-file.
        If not given or None, the value
            where + "GS_numx={0}_numy={1}.h5".format(numx, numy)
        is used.
        Default: None.
    numx : int, optional
        The number of lattice site along the 1st translation vector.
        Default: 4.
    numy : int, optional
        The number of lattice site along the 2nd translation vector.
        Default: 6.
    """

    if entrance_file_name is None:
        entrance_file_name = where + "GS_numx={numx}_numy={numy}.h5".format(
            numx=numx, numy=numy
        )

    # If `where` does not exist, create it
    Path(where).mkdir(parents=True, exist_ok=True)
    gsinfo_file_name = where + "GSInfoDatabase.h5"
    table_name = "GSInfo_numx={numx}_numy={numy}".format(numx=numx, numy=numy)

    h5f_entrance = tb.open_file(entrance_file_name, mode="r")
    h5f_gsinfo = tb.open_file(gsinfo_file_name, mode="a")

    try:
        table = h5f_gsinfo.get_node("/", table_name, classname="Table")
    except tb.NoSuchNodeError:
        table = h5f_gsinfo.create_table(
            "/", table_name, description=GSInfoRecord, expectedrows=25000
        )
        h5f_gsinfo.set_node_attr(table, "numx", numx)
        h5f_gsinfo.set_node_attr(table, "numy", numy)

    count = 0
    gs_record = table.row
    msg = "Append row 'alpha=%.3f, beta=%.3f' into the table"
    logger = logging.getLogger(__name__).getChild("ExtractGSInfo")

    for link in h5f_entrance.walk_nodes("/", classname="ExternalLink"):
        try:
            ket = link(mode="r")
        except (OSError, tb.NoSuchNodeError):
            continue

        alpha = ket.attrs["alpha"]
        beta = ket.attrs["beta"]
        gs_record["alpha"] = alpha
        gs_record["beta"] = beta
        gs_record["gse"] = ket.attrs["gse"]
        gs_record["averages/s2"] = ket.attrs["total_s2_avg"]
        gs_record["averages/sx"] = ket.attrs["total_sx_avg"]
        gs_record["averages/sy"] = ket.attrs["total_sy_avg"]
        gs_record["averages/sz"] = ket.attrs["total_sz_avg"]
        gs_record.append()
        count += 1
        logger.info(msg, alpha, beta)
    table.flush()
    h5f_entrance.close()
    h5f_gsinfo.close()
    logger.info("Append %d rows into the table", count)


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    CreateEntrance(step=0.01)
    ExtractGSInfo()
