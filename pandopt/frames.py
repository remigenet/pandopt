import pandas
import logging
import functools
import numpy as np
from typing import Callable, Type, Dict, Tuple, Any
logger = logging.getLogger()
logger.setLevel(0)
import pandopt
from .transfcoders import _prepare_funcs

class DataFrame(pandas.DataFrame):
    _compiled_func = {}
    _outside_call = True

    def __init__(self, *args, **kwargs):
        data = kwargs["data"] if "data" in kwargs else args[0]
        if isinstance(data, np.ndarray) and not data.flags.f_contiguous:
            data = data.copy(order='F')
        super().__init__(*args, **kwargs)
        # if self.dtypes:

    def __getitem__(self, name):
        print(name)
        return super().__getitem__(name)

    def to_pandas(self):
        return pandas.DataFrame(self)

    @property
    def colname_to_colnum(self):
        return {k: i for i, k in enumerate(self.columns)}

    @property
    def rowname_to_rownum(self):
        return {k: i for i, k in enumerate(self.index)}

    def apply(self, func, axis = None, *args, pandas_fallback = False, data = None, new_index = None, from_rolling  = False, **kwargs):
        if pandas_fallback or args or kwargs: 
            logger.warning(f'{__class__} {"finish in pandas fallback for func "+func if pandas_fallback else "apply only supports func and axis arguments, using default pandas apply"}')
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


    def _with_fallback_wrap(self, func_int_identifier):
        def _with_protects(*args, **kwargs):
            for key in ('_vectorized_nb_compyled', '_loop_nb_compyled', '_vectorized', '_loop'):
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
            return self.apply(func = self.__class__._compiled_func['original'], *args, pandas_fallback = True, **kwargs)
        return _with_protects

    def _compiled_qualifier(self, func_qualifier, mapper, D3_to_D2: bool = False, no_vectorized = False):
        return 'fid'+ f'nvct{no_vectorized}' +f'D3_to_D2_{D3_to_D2}' + str(functools.reduce(lambda x, y: x+y, mapper.keys())) + func_qualifier

    def _build_apply_versions(self, func, mapping, func_int_identifier, D3_to_D2: bool = False, no_vectorized = False):
        if func_int_identifier not in self.__class__._compiled_func:
            self.__class__._compiled_func[func_int_identifier] =  _prepare_funcs(func, mapping, str(func_int_identifier), D3_to_D2 = D3_to_D2, no_vectorized = no_vectorized)
        return self._with_fallback_wrap(func_int_identifier)

    def rolling(self, window, *args, **kwargs):
        return pandopt.RollOpt(self, window=window, *args, **kwargs)

    
    def groupby(self, *args, **kwargs):
        return pandopt.GroupOpt(self, *args, **kwargs)

    def resample(self, *args, **kwargs):
        # NotImplementedError('Expanding not yet implemented')
        return super().resample(*args, **kwargs)

    def expanding(self, *args, **kwargs):
        # NotImplementedError('Expanding not yet implemented')
        return super().expanding(*args, **kwargs)


    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            try:
                func = pandopt.load_function(name)
            except AttributeError:
                raise AttributeError(f"{name} not found or could not be loaded.")
            return self.apply(func=func, *args, **kwargs)
        return wrapper

