import numpy as np
from enum import Enum
from typing import Callable


def standard_sum(z):
    return np.sum(z)

def standard_mean(z):
    return np.mean(z)

def standard_max(z):
    return np.max(z)

def standard_min(z):
    return np.min(z)

def standard_std(z):
    return np.std(z)

def standard_median(z):
    return np.median(z)

def standard_count(z):
    return np.count(z)


class StandardFunctions(Enum):
    SUM = standard_sum
    MEAN = standard_mean
    MAX = standard_max
    MIN = standard_min
    STD = standard_std
    MEDIAN = standard_median
    COUNT = standard_count
    
def load_function(fname: str):
    if isinstance(fname, Callable):
        return fname
    elif isinstance(fname, str):
        fname = fname.stip().upper()
        if fname == 'SUM':
            return StandardFunctions.SUM.value
        elif fname == 'MEAN':
            return StandardFunctions.MEAN.value
        elif fname == 'MAX':
            return StandardFunctions.MAX.value
        elif fname == 'STD':
            return StandardFunctions.STD.value
        elif fname == 'median':
            return StandardFunctions.MEDIAN.value
        elif fname == 'COUNT':
            return StandardFunctions.COUNT.value
        raise ValueError(f'Unable to find function {fname}')
    else:
        raise TypeError(f'fname cannot be {type(fname)}')