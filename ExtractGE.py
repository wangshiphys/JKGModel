"""
Extract the ground state energies from the stored ground state data
"""


from pathlib import Path
from time import time

import numpy as np


gs_dir = Path("data/SpinModel/GS/")

info = "The time spend on processing {0} is: {1:.6f}s"

for sub_dir in gs_dir.iterdir():
    Path(str(sub_dir).replace("GS", "GE")).mkdir(parents=True, exist_ok=True)
    for gs_file in sub_dir.iterdir():
        ge_file = Path(str(gs_file).replace("GS", "GE").replace("npz", "npy"))
        if not ge_file.exists():
            t0 = time()
            with np.load(gs_file) as ld:
                alpha, beta = ld["parameters"]
                gse = ld["gse"][0]
            np.save(ge_file, [alpha, beta, gse])
            t1 = time()
            print(info.format(gs_file.name, t1 - t0))
