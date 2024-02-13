import pandas
import logging
import functools
import numpy as np
from typing import Callable, Type, Dict, Tuple, Any, Union 
logger = logging.getLogger()
logger.setLevel(0)
import pandopt
from .transfcoders import _prepare_funcs

class DataFrame(pandas.DataFrame):
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
    _compiled_func = {}
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

        data = kwargs["data"] if "data" in kwargs else args[0]
        if isinstance(data, np.ndarray) and not data.flags.f_contiguous:
            data = data.copy(order='F')
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

    def apply(self, func, axis = None, *args, pandas_fallback = False, data = None, new_index = None, from_rolling  = False, **kwargs):
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
        if pandas_fallback or args or kwargs: 
            logger.warning(f'{__class__} {"finish in pandas fallback for func " + str(func) if pandas_fallback else "apply only supports func and axis arguments, using default pandas apply"}')
            return super().apply(func, axis = axis, *args, **kwargs)
        if axis is None:
            self.apply(func, axis = 0, *args, pandas_fallback = False, data = None, new_index = None, from_rolling = from_rolling, **kwargs).apply(func, axis = 1, *args, pandas_fallback = False, data = None, new_index = None, from_rolling = from_rolling, **kwargs)
        new_columns = None
        mapping = self.colname_to_colnum if axis == 0 else {True: 0}
        if from_rolling:
            test_set = data[:min(data.shape[0] - 1, 10+ len(self.index) - len(new_index)),:,:]
            new_columns = self.columns if axis == 0 else None
        else:
            data =  (self.to_numpy() if axis else self.to_numpy().T)
            test_set = data[:min(data.shape[0] - 1, 10)]
            new_index = self.index if axis == 1 else self.columns

        func_int_identifier = self._compiled_qualifier(
            func_qualifier = func.__qualname__,
            mapper=mapping,
            D3_to_D2=from_rolling and axis==1,
            no_vectorized=from_rolling
        )
        prepared_func = self._build_apply_versions(func, mapping, func_int_identifier, D3_to_D2 = from_rolling and axis == 0, no_vectorized = from_rolling)
        test_res = prepared_func(test_set)
        result = prepared_func(data)
        return pandopt.DataFrame(result, index=new_index, columns=new_columns)


    def _with_fallback_wrap(self, func_int_identifier: str) -> Callable:
        """
        Wrap the function call with a fallback mechanism.

        This internal method is used to wrap the execution of a function with
        various compiled versions for optimization. If an optimized version fails,
        it falls back to the next available version or ultimately to the standard
        pandas apply method.

        Parameters
        ----------
        func_int_identifier : str
            The identifier of the function for which to retrieve compiled versions.

        Returns
        -------
        Callable
            A wrapped function that tries various compiled versions and falls back 
            to standard pandas apply if necessary.

        Notes
        -----
        This method is used internally for optimized function application in 
        the custom DataFrame class.
        """
        def _with_protects(*args, **kwargs):
            for key in ('_oc_vectorized_nb_compyled', '_oc_loop_nb_compyled', '_vectorized_nb_compyled', '_loop_nb_compyled', '_vectorized', '_loop'):
                if key not in self.__class__._compiled_func[func_int_identifier]:
                    logger.debug(f'No {key} for {func_int_identifier}')
                    continue
                try:
                    result = self.__class__._compiled_func[func_int_identifier][key](*args, **kwargs)
                    if np.shape(result):
                        logger.debug(f"returning {result} with {self.__class__._compiled_func[func_int_identifier][key]}")
                        return result
                    else:
                        logger.debug(f'Encountered {key} with not untented result {np.shape(result)} -> {result}')
                        self.__class__._compiled_func[func_int_identifier].pop(key)
                except Exception as e:
                    logger.warning('Encountered', e)
                    self.__class__._compiled_func[func_int_identifier].pop(key)
            return self.apply(func = self.__class__._compiled_func['_original'], *args, pandas_fallback = True, **kwargs)
        return _with_protects

    def _compiled_qualifier(self, func_qualifier: str, mapper: Dict[Any, Any], D3_to_D2: bool = False, no_vectorized: bool = False) -> str:
        """
        Generate a unique identifier for a function based on its characteristics.

        This internal method is used to create a unique string identifier for a function,
        which includes its name and other attributes like whether it's vectorized or
        designed for 3D to 2D conversion.

        Parameters
        ----------
        func_qualifier : str
            The qualifier of the function, typically its name.
        mapper : Dict[Any, Any]
            A mapping used in the function, influencing its behavior.
        D3_to_D2 : bool, default False
            Flag indicating if the function converts 3D arrays to 2D.
        no_vectorized : bool, default False
            Flag indicating if the function is non-vectorized.

        Returns
        -------
        str
            A unique string identifier for the function.

        Notes
        -----
        This method is crucial for managing the caching of various compiled versions of functions.
        """
        return 'fid'+ f'nvct{no_vectorized}' +f'D3_to_D2_{D3_to_D2}' + str(functools.reduce(lambda x, y: x+y, mapper.keys())) + func_qualifier

    def _build_apply_versions(self, func: Callable, mapping: Dict[Any, Any], func_int_identifier: str, D3_to_D2: bool = False, no_vectorized: bool = False) -> Callable:
        """
        Build and cache different apply versions of a given function.

        This internal method compiles and caches various versions of a function for
        optimized execution, including vectorized and non-vectorized versions.

        Parameters
        ----------
        func : Callables
            The function to be compiled in different versions.
        mapping : Dict[Any, Any]
            A mapping dict influencing the behavior of the function.
        func_int_identifier : str
            The unique identifier for the function.
        D3_to_D2 : bool, default False
            Indicates if the function is used for 3D to 2D array conversion.
        no_vectorized : bool, default False
            Indicates if the function should not have a vectorized version.

        Returns
        -------
        Callable
            The wrapped function with fallback mechanism.

        Notes
        -----
        This method is central to the optimization capabilities of the custom DataFrame class, 
        as it prepares the function in various forms for efficient execution.
        """
        if func_int_identifier not in self.__class__._compiled_func:
            self.__class__._compiled_func[func_int_identifier] =  _prepare_funcs(func, mapping, str(func_int_identifier), D3_to_D2 = D3_to_D2, no_vectorized = no_vectorized)
        return self._with_fallback_wrap(func_int_identifier)

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

