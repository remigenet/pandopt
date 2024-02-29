from pandas.core.window.rolling import Rolling
import pandas
import logging
from typing import Callable, Type, Dict, Tuple, Any

import pandopt
logger = logging.getLogger()
logger.setLevel(0)
from .transfcoders import _prepare_func

class RollOpt(pandas.core.window.rolling.Rolling):
    """
    An enhanced version of the pandas Rolling class for optimized rolling window calculations.

    This class extends the pandas Rolling class, providing additional functionalities
    and optimizations for rolling window calculations on DataFrame objects.

    Attributes
    ----------
    _outside_call : bool
        A flag indicating whether a method call is coming from outside the class scope.

    Methods
    -------
    apply(func, axis=0, *args, pandas_fallback=False, raw=True, **kwargs):
        Apply a function to the rolling window with additional parameters and optimizations.
    agg(func, *args, **kwargs):
        Aggregate using one or more operations over the specified axis.

    See Also
    --------
    pandas.core.window.rolling.Rolling : The base class from which this class is derived.

    Examples
    --------
    >>> import pandopt
    >>> custom_df = pandopt.DataFrame(data)
    >>> roll_opt = custom_df.rolling(window=3)
    >>> result = roll_opt.apply(sum)
    """

    _outside_call = True

    def __init__(self, df, window: int, *args, **kwargs):
        """
        Initialize a new instance of the RollOpt class.

        Parameters
        ----------
        df : pandopt.DataFrame
            The custom DataFrame on which to perform the rolling calculations.
        window : int
            The number of periods to use for the rolling window.
        *args, **kwargs
            Additional arguments and keyword arguments passed to the pandas Rolling constructor.

        Examples
        --------
        >>> import pandopt
        >>> custom_df = pandopt.DataFrame(data)
        >>> roll_opt = custom_df.rolling(window=3)
        """

        self._idf = pandopt.DataFrame(df) if not isinstance(df, (pandopt.DataFrame, pandopt.RollOpt)) else df
        self._window = window
        super().__init__(self._idf, window=window, *args, **kwargs)

    def apply(self, funcs, axis = None, *args, **kwargs):
        """
        Apply a function along an axis of the DataFrame with enhanced capabilities.

        Parameters
        ----------
        func : callable
            The function to apply to each column or row.
        axis : {0 or ‘index’, 1 or ‘columns’}, default None
            Axis along which the function is applied:
            - 0 or ‘index’: apply function to each column.
            - 1 or ‘columns’: apply function to each row.
        pandas_fallback : bool, default False
            If True, use the standard pandas apply method for cases where the optimized apply cannot be used.
        data : ndarray, optional
            An optional data parameter for internal use in optimized computations.
        new_index : array-like, optional
            The index for the resulting DataFrame after applying the function.
        from_rolling : bool, default False
            Indicator whether the apply is being called from a rolling window context.
        *args, **kwargs
            Additional arguments and keyword arguments passed to the function.

        Returns
        -------
        DataFrame
            A new DataFrame with the applied function results.

        Examples
        --------
        >>> import pandopt
        >>> custom_df = pandopt.DataFrame(data)
        >>> result_df = custom_df.apply(some_function, axis=1)
        """
        if len((dtypes:=set(self._idf.dtypes)))>1:
            return super().apply(funcs, axis, **kwargs)
        if isinstance(funcs, Callable):
            funcs = [funcs]
        dtype = str(dtypes.pop())
        prepared = _prepare_func(funcs, self._idf.index, axis = axis or 0, ndims = 3, dtype = dtype, globals=self._idf.__class__._compiled_env)
        try:
            result = prepared['function'](self._idf.to_numpy(), self._window)
            result = pandopt.DataFrame(result, index = self._idf.index[self._window - 1:], columns = [func.__name__ for func in funcs] if axis else [(func.__name__, column) for func in funcs for column in self._idf.columns])
            return result
        except Exception as e:
            print(e)
        
        return super().apply(funcs, self.window, axis, **kwargs)


    # def aggregate(self, func, *args, **kwargs):
    #     if isinstance(func, dict):
    #         result = {}
    #         for column, func in func.items():
    #             func = pandopt.load_function(func)  # CAN BE IMPROVED MUCH
    #             result[column] = self.apply(lambda x: x[column].pipe(func_name), *args, **kwargs)
    #         return pandopt.DataFrame(result)
    #     func = pandopt.load_function(func)
    #     return self.apply(func, *args, **kwargs)

    def agg(self, func: Callable, *args, **kwargs) -> 'pandopt.DataFrame':
        """
        Aggregate using one or more operations over the rolling window.

        This method is similar to the standard pandas aggregate (agg) but optimized
        for use with the custom DataFrame class.

        Parameters
        ----------
        func : Callable
            The function or functions to use for aggregating the data.
        *args, **kwargs
            Additional arguments and keyword arguments passed to the aggregation function.

        Returns
        -------
        pandopt.DataFrame
            A DataFrame with the results of the aggregation.

        Examples
        --------
        >>> import pandopt
        >>> custom_df = pandopt.DataFrame(data)
        >>> roll_opt = custom_df.rolling(window=3)
        >>> result = roll_opt.agg(sum)
        """
        return self.aggregate(func, *args, **kwargs)
  
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            try:
                func = pandopt.load_function(name)
            except AttributeError:
                raise AttributeError(f"{name} not found or could not be loaded.")
            return self.apply(func=func, *args, **kwargs)
        return wrapper
