from pathlib import Path
from time import time
import numpy as np


# gs_data_path = "data/QuantumSpinModel/GS/"
gs_data_path = "D:/JKGModel/data/QuantumSpinModel/GS/"
# gs_data_path = "E:/JKGModel/data/QuantumSpinModel/GS/"

ge_data_path = "data/QuantumSpinModel/GE/"
ge_name_temp = "GE_numx={0}_numy={1}_alpha={2:.4f}_beta={3:.4f}.npz"
Path(ge_data_path).mkdir(parents=True, exist_ok=True)

count = 0
t0 = time()
for gs_name in Path(gs_data_path).iterdir():
    with np.load(gs_name) as ld:
        numx, numy = ld["size"]
        alpha, beta = ld["parameters"]
        gse = ld["gse"]
    ge_full_name = ge_data_path + ge_name_temp.format(numx, numy, alpha, beta)
    np.savez(
        ge_full_name, size=[numx, numy], parameters=[alpha, beta], gse=gse
    )
    count += 1
    print(ge_full_name)
t1 = time()
print("The total number: {0}".format(count))
print("The time spend on extract ge info: {0:.4f}s".format(t1 - t0))
