from numba import jit, guvectorize, float32
import numpy as np

@jit(nopython=True)
def g_func(a, b):
    return np.float32(a if abs(a) > abs(b) else -2 * b**(-a))

@jit(nopython=True)
def harder3_func(z):
    x = np.float32((z[0] + z[1]) / np.max(0.0001,z[2]))
    j = g_func(z[1], z[3])
    k = np.float32(z[0] - z[2])
    if k > j:
        return np.float32(j / k)
    else:
        x *= j
        if k > z[2]:
            return np.float32(x - k)
        else:
            return np.float32(x + k)

# Using the correct signature for float32 input and output
@guvectorize([(float32[:], float32[:])], '(n)->()', nopython=True, target='parallel',fastmath=True, target_backend='host')
def guvectorized_harder3_func(row, result):
    result[0] = harder3_func(row)

import time
t1 = time.time()

# Test the function with a 2D array
arr_2d = np.random.randn(100000, 4).astype(np.float32)  # Small array for testing
print(arr_2d.shape)
results = guvectorized_harder3_func(arr_2d)
print("Results:", results, time.time() - t1)

t1 = time.time()
results = harder3_func(arr_2d)

print("Results:", results, time.time() - t1)