from pandas.core.window.rolling import Rolling
import pandas
import logging
from typing import Callable, Type, Dict, Tuple, Any
from numpy.lib.stride_tricks import sliding_window_view
import pandopt
logger = logging.getLogger()
logger.setLevel(0)


class RollOpt(pandas.core.window.rolling.Rolling):
    _outside_call = True

    def __init__(self, df, window=10, *args, **kwargs):
        self._idf = pandopt.DataFrame(df) if not isinstance(df, (pandopt.DataFrame, pandopt.RollOpt)) else df
        super().__init__(self._idf, window=window, *args, **kwargs)

    def apply(self, func, axis = 0, *args, pandas_fallback=False, raw = True, **kwargs):
        try:
            data = sliding_window_view(self._idf.to_numpy(), self.window, axis=0)
            r = self._idf.apply(func, axis = axis, *args, data = data, pandas_fallback=pandas_fallback, new_index =self._idf.index[self.window-1:], from_rolling = True,**kwargs)
            return r 
        except Exception as e:
            logger.warning(f'Error in pandoptRoll apply: {e}')
            if pandas_fallback:
                return super().apply(func, *args, **kwargs)
            raise e
    
    # def aggregate(self, func, *args, **kwargs):
    #     if isinstance(func, dict):
    #         result = {}
    #         for column, func in func.items():
    #             func = pandopt.load_function(func)  # CAN BE IMPROVED MUCH
    #             result[column] = self.apply(lambda x: x[column].pipe(func_name), *args, **kwargs)
    #         return pandopt.DataFrame(result)
    #     func = pandopt.load_function(func)
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
