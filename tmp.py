import numpy as np
from time import time
import cupy as cp
all = time()
for _ in range(250):
    a = np.zeros((60, 180, 32))
    start = time()
    b = np.squeeze(np.random.multivariate_normal([0], np.identity(1) * 1, (60, 180, 32)))
    print(time() - start)
    start = time()
    a += b
    print(time() - start)
print(time() - all)