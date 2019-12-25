import logging
import sys
from time import time

from ClassicalModel import JKGModelClassicalSolver

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
logging.info("Program start running")

max_run = 10
numx = numy = 6
solver = JKGModelClassicalSolver(numx, numy)
for i in range(1, max_run+1):
    t0 = time()
    solver.MinimizeGeneralSpinConfig(alpha=0.5, beta=0.5)
    t1 = time()
    logging.info("%d/%d, dt=%.4fs", i, max_run, t1 - t0)
logging.info("Program stop running")
