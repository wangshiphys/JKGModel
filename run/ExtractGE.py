"""
Extract ground state energy.
"""


from pathlib import Path
from time import time

import numpy as np

gs_data_path = "E:/JKGModel/data/QuantumSpinModel/GS/"
ge_data_path = "E:/JKGModel/data/QuantumSpinModel/GE/"
Path(ge_data_path).mkdir(parents=True, exist_ok=True)
ge_name_temp = "GE_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"

count = 0
t0 = time()
for gs_name in Path(gs_data_path).iterdir():
    # noinspection PyTypeChecker
    with np.load(gs_name) as ld:
        gse = ld["gse"]
        numx, numy = ld["size"]
        alpha, beta = ld["parameters"]
    ge_name = ge_data_path + ge_name_temp.format(numx, numy, alpha, beta)
    np.savez(ge_name, size=[numx, numy], parameters=[alpha, beta], gse=gse)
    count += 1
    print(ge_name)
t1 = time()
print("Total number of extracted GE: {0}".format(count))
print("Time spend on extracting GE: {0:.3f}s".format(t1 - t0))
