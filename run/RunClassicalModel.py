import logging
import sys
from time import time

from ClassicalModel import JKGModelClassicalSolver

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s",
)
logging.info("Program start running")

max_run = 10
num1 = num2 = 12
direction = "xz"
alpha = 0.50
beta = 0.50

solver = JKGModelClassicalSolver(num1=num1, num2=num2, direction=direction)
for i in range(max_run):
    t0 = time()
    solver.OptimizeSpinConfig(alpha=alpha, beta=beta)
    t1 = time()
    logging.info("%2d/%2d, dt=%.3fs", i + 1, max_run, t1 - t0)
logging.info("Program stop running")
