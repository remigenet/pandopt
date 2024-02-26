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

    def apply(self, func, axis = None, *args, data = None, new_index = None, from_rolling  = False, **kwargs):
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
            return super().apply(func, axis, **kwargs)
        dtype = str(dtypes.pop())
        prepared = _prepare_func(func, self.columns if axis==1 else self.index, axis = axis or 0, ndims = 2, dtype = dtype, globals = __class__._compiled_env)
        if "vectorized" in prepared:
            try:
                result = prepared['vectorized']['function'](self.to_numpy())
                result = __class__(result, index = self.columns if axis==0 else self.index)
                return result
            except Exception as e:
                print(e)
        try:
            result = prepared['modified']['function'](self.to_numpy())
            result = __class__(result, index = self.columns if axis==0 else self.index)
            return result
        except Exception as e:
            print(e)
        
        return super().apply(func, axis, **kwargs)


    # def _compiled_qualifier(self, func_qualifier: str, mapper: Dict[Any, Any], ndims: int, axis: int) -> str:
    #     """
    #     Generate a unique identifier for a function based on its characteristics.

    #     This internal method is used to create a unique string identifier for a function,
    #     which includes its name and other attributes like whether it's vectorized or
    #     designed for 3D to 2D conversion.

    #     Parameters
    #     ----------
    #     func_qualifier : str
    #         The qualifier of the function, typically its name.
    #     mapper : Dict[Any, Any]
    #         A mapping used in the function, influencing its behavior.
    #         Flag indicating if the function converts 3D arrays to 2D.
    #     no_vectorized : bool, default False
    #         Flag indicating if the function is non-vectorized.

    #     Returns
    #     -------
    #     str
    #         A unique string identifier for the function.

    #     Notes
    #     -----
    #     This method is crucial for managing the caching of various compiled versions of functions.
    #     """
    #     return 'fid'+ f'nvct{func_qualifier}' +f'axis{axis}' + str(functools.reduce(lambda x, y: x+y, mapper.keys())) + func_qualifier

    # def _build_apply_versions(self, func: Callable, mapping: Dict[Any, Any], func_int_identifier: str, ndims:int =2, axis: int = 0) -> Callable:
    #     """
    #     Build and cache different apply versions of a given function.

    #     This internal method compiles and caches various versions of a function for
    #     optimized execution, including vectorized and non-vectorized versions.

    #     Parameters
    #     ----------
    #     func : Callables
    #         The function to be compiled in different versions.
    #     mapping : Dict[Any, Any]
    #         A mapping dict influencing the behavior of the function.
    #     func_int_identifier : str
    #         The unique identifier for the function.
    #     D3_to_D2 : bool, default False
    #         Indicates if the function is used for 3D to 2D array conversion.
    #     no_vectorized : bool, default False
    #         Indicates if the function should not have a vectorized version.

    #     Returns
    #     -------
    #     Callable
    #         The wrapped function with fallback mechanism.

    #     Notes
    #     -----
    #     This method is central to the optimization capabilities of the custom DataFrame class, 
    #     as it prepares the function in various forms for efficient execution.
    #     """
    #     if func_int_identifier not in self.__class__._compiled_func:
    #         self.__class__._compiled_func[func_int_identifier] =  _prepare_funcs(func, mapping, str(func_int_identifier), ndims = ndims, axis = axis)
    #     return self._with_fallback_wrap(func_int_identifier)

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

    
    # def groupby(self, *args, **kwargs):
    #     return pandopt.GroupOpt(self, *args, **kwargs)

    # def resample(self, *args, **kwargs):
    #     # NotImplementedError('Expanding not yet implemented')
    #     return super().resample(*args, **kwargs)

    # def expanding(self, *args, **kwargs):
    #     # NotImplementedError('Expanding not yet implemented')
    #     return super().expanding(*args, **kwargs)


    def __getattr__(self, name):
        """
        Custom method to handle dynamic attribute access.

        This method is called if the requested attribute is not found in the usual places 
        (i.e., it is not an instance attribute nor is it found in the class tree for self).

        Parameters
        ----------
        name : str
            The name of the attribute being accessed.

        Returns
        -------
        Callable
            A function wrapper that when called will apply the dynamically loaded function.

        Raises
        ------
        AttributeError
            If the function name is not found or could not be loaded.

        Examples
        --------
        >>> import pandopt
        >>> custom_df = pandopt.DataFrame(data)
        >>> result = custom_df.some_custom_function()
        """
        def wrapper(*args, **kwargs):
            try:
                func = pandopt.load_function(name)
            except AttributeError:
                raise AttributeError(f"{name} not found or could not be loaded.")
            return self.apply(func=func, *args, **kwargs)
        return wrapper

