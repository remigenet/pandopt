import cython
%load_ext Cython
%%cython
import numpy as np
cimport numpy as np
def cluster(np.ndarray array, np.float64_t maxdiff):
    cdef np.ndarray[np.float64_t, ndim=1] flat = np.sort(array.flatten())
    cdef list breakpoints = []
    cdef np.float64_t seed = flat[0]
    cdef np.int64_t int = 0
    for i in range(0, len(flat)):
        if (flat[i] - seed) > maxdiff:
            breakpoints.append(i)
            seed = flat[i]
    return np.split(array, breakpoints)