from pandas.core.window.rolling import Rolling
import pandas
import logging
from typing import Callable, Type, Dict, Tuple, Any
from numpy.lib.stride_tricks import sliding_window_view
import pandopt
logger = logging.getLogger()
logger.setLevel(0)


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

    def __init__(self, df: 'pandopt.DataFrame', window: int, *args, **kwargs):
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
        super().__init__(self._idf, window=window, *args, **kwargs)

    def apply(self, func: Callable, axis: int = 0, *args, pandas_fallback: bool = False, raw: bool = True, **kwargs) -> 'pandopt.DataFrame':
        """
        Apply a function to each rolling window.

        Parameters
        ----------
        func : Callable
            The function to apply to each window.
        axis : int, default 0
            The axis along which to apply the function.
        pandas_fallback : bool, default False
            If True, use the standard pandas apply method in case of errors or limitations.
        raw : bool, default True
            Determines if the function receives a window of raw data or a Series/DataFrame.
        *args, **kwargs
            Additional arguments and keyword arguments passed to the function.

        Returns
        -------
        pandopt.DataFrame
            A DataFrame with the results of applying the function over the rolling window.

        Examples
        --------
        >>> import pandopt
        >>> custom_df = pandopt.DataFrame(data)
        >>> roll_opt = custom_df.rolling(window=3)
        >>> result = roll_opt.apply(sum)
        """

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
