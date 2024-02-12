from pandas.core.groupby.generic import DataFrameGroupBy
import logging
from typing import Callable, Type, Dict, Tuple, Any
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
logger = logging.getLogger()
logger.setLevel(0)
import pandopt
from .transfcoders import _prepare_funcs

class GroupOpt(DataFrameGroupBy):
    _outside_call = True
    def __init__(self, df, *args, **kwargs):
        self.index = range(df.shape[0])
        self._idf = pandopt.DataFrame(df) if not isinstance(df, pandopt.DataFrame) else df
        super().__init__(self._idf, *args, **kwargs)

    def apply(self, func, axis = 0, *args, pandas_fallback=False, raw = True, **kwargs):
        try:
            data = sliding_window_view(self._idf.to_numpy(), self.window, axis=0)
            r = self._idf.apply(func, axis = axis, *args, data = data, pandas_fallback=pandas_fallback, new_index =self._idf.index[self.window-1:], from_rolling = True,**kwargs)
            return r#.reindex(self._idf.index, axis=0)
        except Exception as e:
            logger.warning(f'Error in pandoptRoll apply: {e}')
            if pandas_fallback:
                return super().apply(func, *args, **kwargs)
            raise e
    
    # def aggregate(self, func, *args, **kwargs):
    #     if isinstance(func, dict):
    #         result = {}
    #         for column, func_name in func.items():
    #             f = pandopt.load_function(func_name)  # CAN BE IMPROVED MUCH
    #             result[column] = self.apply(lambda x: x[column].pipe(f), *args, **kwargs)
    #         return pandopt.DataFrame(result)
    #     func = pandopt.math.load_function(func)
    #     return self.apply(func, *args, **kwargs)

    def agg(self, func, *args, **kwargs):
        return self.aggregate(func, *args, **kwargs)

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            try:
                func = pandopt.load_function(name)
            except AttributeError:
                raise AttributeError(f"{name} not found or could not be loaded.")
            return self.apply(func=func, *args, **kwargs)
        return wrapper

