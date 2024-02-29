import pandas
import logging
import functools
import numpy as np
import numba as nb
from typing import Callable, Type, Dict, Tuple, Any, Union 
logger = logging.getLogger()
logger.setLevel(0)
import pandopt
from .transfcoders import _prepare_func

import time

class DataFrame(pandas.core.frame.DataFrame):
    """
    A custom DataFrame class that extends pandas.DataFrame with enhanced 
    functionalities for optimized data processing and transformation.

    This class is designed to work seamlessly with Pandas dataframes while 
    providing additional methods and properties for efficient data manipulation, 
    especially in scenarios requiring optimized computational performance.

    Attributes
    ----------
    _compiled_func : dict
        A dictionary to store compiled versions of functions for optimized execution.
    _outside_call : bool
        A flag indicating whether a method call is coming from outside the class scope.

    Methods
    -------
    to_pandas():
        Convert the custom DataFrame back to a pandas DataFrame.
    colname_to_colnum:
        Property to get a mapping of column names to their respective index.
    rowname_to_rownum:
        Property to get a mapping of row names to their respective index.
    apply(func, axis=None, *args, pandas_fallback=False, data=None, new_index=None, from_rolling=False, **kwargs):
        Enhanced apply method with additional parameters for optimized function application.
    rolling(window, *args, **kwargs):
        Provides rolling window calculations.
    __getattr__(name):
        Custom attribute access method to dynamically load and apply functions.
    
    See Also
    --------
    pandas.DataFrame : The base class from which this class is derived.

    Examples
    --------
    >>> import pandopt
    >>> custom_df = pandopt.DataFrame(data)
    >>> custom_df.apply(some_function)
    >>> pandas_df = custom_df.to_pandas()
    """
    _compiled_env = globals().copy()
    _compiled_env.update({'np': np, 'nb': nb})
    _outside_call = True

    def __init__(self, *args, **kwargs):
        """
        Initialize a new instance of the custom DataFrame class.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the pandas DataFrame constructor.
        **kwargs : dict
            Keyword arguments passed to the pandas DataFrame constructor. 
            If 'data' is provided and it is a numpy ndarray that is not F-contiguous, 
            a copy of the data will be made with 'F' order.

        Examples
        --------
        >>> import numpy as np
        >>> import pandopt
        >>> data = np.array([[1, 2], [3, 4]])
        >>> custom_df = pandopt.DataFrame(data=data)
        """
        super().__init__(*args, **kwargs)


    def to_pandas(self):
        return pandas.DataFrame(self)

    @property
    def colname_to_colnum(self):
        """
        Get a mapping of column names to their respective index positions.

        Returns
        -------
        dict
            A dictionary where keys are column names and values are their corresponding index positions.

        Examples
        --------
        >>> import pandopt
        >>> custom_df = pandopt.DataFrame(data)
        >>> col_index_map = custom_df.colname_to_colnum
        """
        return {k: i for i, k in enumerate(self.columns)}

    @property
    def rowname_to_rownum(self):
        """
        Get a mapping of row names or indices to their respective index positions.

        Returns
        -------
        dict
            A dictionary where keys are row names/indices and values are their corresponding index positions.

        Examples
        --------
        >>> import pandopt
        >>> custom_df = pandopt.DataFrame(data)
        >>> row_index_map = custom_df.rowname_to_rownum
        """
        return {k: i for i, k in enumerate(self.index)}

    # def agg(self, *args, **kwargs):
    #     return self.aggregate(*args, **kwargs)

    # def apply(self, *args, **kwargs):

    

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
        if len((dtypes:=set(self.dtypes)))>1:
            return super().apply(funcs, axis, **kwargs)
        if isinstance(funcs, Callable):
            funcs = [funcs]
        dtype = str(dtypes.pop())
        prepared = _prepare_func(funcs, self.columns if axis==1 else self.index, axis = axis or 0, ndims = 2, dtype = dtype, globals=self.__class__._compiled_env)
        try:
            result = prepared['function'](self.to_numpy())
            result = __class__(result, index = self.columns if axis==0 else self.index, columns=[func.__name__ for func in funcs])
            return result
        except Exception as e:
            print(e)
        
        return super().apply(funcs, axis, **kwargs)


    def rolling(self, window, *args, **kwargs):
        """
        Provide rolling window calculations over the DataFrame.

        Parameters
        ----------
        window : int
            Size of the moving window.
        *args, **kwargs
            Additional arguments and keyword arguments passed to the rolling function.

        Returns
        -------
        RollOpt
            A RollOpt object which can apply functions over a rolling window.

        Examples
        --------
        >>> import pandopt
        >>> custom_df = pandopt.DataFrame(data)
        >>> rolling_df = custom_df.rolling(window=3)
        """
        return pandopt.RollOpt(self, window=window, *args, **kwargs)



    # def __getattr__(self, name):
    #     """
    #     Custom method to handle dynamic attribute access.

    #     This method is called if the requested attribute is not found in the usual places 
    #     (i.e., it is not an instance attribute nor is it found in the class tree for self).

    #     Parameters
    #     ----------
    #     name : str
    #         The name of the attribute being accessed.

    #     Returns
    #     -------
    #     Callable
    #         A function wrapper that when called will apply the dynamically loaded function.

    #     Raises
    #     ------
    #     AttributeError
    #         If the function name is not found or could not be loaded.

    #     Examples
    #     --------
    #     >>> import pandopt
    #     >>> custom_df = pandopt.DataFrame(data)
    #     >>> result = custom_df.some_custom_function()
    #     """
    #     def wrapper(*args, **kwargs):
    #         try:
    #             func = pandopt.load_function(name)
    #         except AttributeError:
    #             raise AttributeError(f"{name} not found or could not be loaded.")
    #         return self.apply(func=func, *args, **kwargs)
    #     return wrapper

