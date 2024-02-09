import ast
import inspect
import types
import numpy as np
import numba as nb
import pandas as pd
import time
from enum import Enum
import logging
import functools
import copy
import sys
from typing import Callable, Type, Dict, Tuple, Any
from numpy.lib.stride_tricks import sliding_window_view
logger = logging.getLogger()
logger.setLevel(0)



def create_callmap_function_ast(mapping: Dict[str, int]) -> ast.FunctionDef:
    # Create the body of the callmap function
    body = []
    for key, value in mapping.items():
        try:
            key = int(key)
            continue
        except ValueError:
            key = ast.Constant(value=key)
        compare = ast.Compare(
            left=ast.Name(id='x', ctx=ast.Load()),
            ops=[ast.Eq()],
            comparators=[key]
        )
        body.append(
            ast.If(
                test=compare,
                body=[ast.Return(value=ast.Constant(value=value))],
                orelse=[]
            )
        )
    
    # Add a default return statement
    body.append(ast.Return(value=ast.Name(id='x', ctx=ast.Load())))

    # Create the function definition
    func_def = ast.FunctionDef(
        name='callmap',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='x')],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=body,
        decorator_list=[],
        returns=None
    )
    return func_def

class SubscriptReplacer(ast.NodeTransformer):
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == self.arg_name:
            # Check for Python version compatibility
            if sys.version_info >= (3, 9):
                # Python 3.9 and later
                old_slice = node.slice
            else:
                # Python 3.8 and earlier
                old_slice = node.slice.value if isinstance(node.slice, ast.Index) else node.slice

            # Wrap the subscript in a call to callmap
            node.slice = ast.Call(
                func=ast.Name(id='callmap', ctx=ast.Load()),
                args=[old_slice],
                keywords=[]
            )
        return self.generic_visit(node) 

def create_transformed_function_ast(original_func: Callable, mapping: Dict[str, int], func_int_identifier: str, D3_to_D2: bool = False, no_vectorized = False) -> Tuple[ast.AST, ast.AST, ast.AST]:
    # Parse the original function
    original_tree = ast.parse(inspect.getsource(original_func))
    arg_name = original_tree.body[0].args.args[0].arg
    
    original_tree.body[0].name = 'temporary'
    
    replacer = SubscriptReplacer(arg_name)
    original_tree = replacer.visit(original_tree)
    ast.fix_missing_locations(original_tree)

    if D3_to_D2:
        loop_base_func_str = f"""
def {func_int_identifier}_loop(Z):
    n, c, t = Z.shape
    res = np.zeros((n, c))
    for col in nb.prange(c):
        for row in nb.prange(n):
            res[row, col] = temporary(Z[row,col,:])
    return res
        """
    else:
        loop_base_func_str = f"""
def {func_int_identifier}_loop(Z):
    n = Z.shape[0]
    res = np.zeros((n, 1))
    for i in nb.prange(n):
        res[i, 0] = temporary(Z[i, :])
    return res
        """
    loop_func_tree = ast.parse(loop_base_func_str)
    vectorize_func_tree = None
    if not no_vectorized:
        vectorized_base_func_str = f"""
def {func_int_identifier}_vectorized(Z):
    return temporary(Z.T)
        """
        vectorize_func_tree = ast.parse(vectorized_base_func_str)

    return original_tree, loop_func_tree, vectorize_func_tree

def numba_decorate(func_tree: ast.AST, nopython: bool = True, nogil: bool = True, parallel: bool = True,
 fastmath: bool = True, forceinline: bool = True, looplift: bool = True, target_backend: bool = True, no_cfunc_wrapper: bool = True, cache: bool = True) -> ast.AST:
    # # Add Numba JIT decorator
    nb_compyled_func_tree = copy.deepcopy(ast.fix_missing_locations(func_tree))
    numba_decorator = ast.Call(
        func=ast.Attribute(value=ast.Name(id='nb', ctx=ast.Load()), attr='jit', ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg='nopython', value=ast.Constant(value=nopython)),
            ast.keyword(arg='nogil', value=ast.Constant(value=nogil)),
            ast.keyword(arg='parallel', value=ast.Constant(value=parallel)),
            ast.keyword(arg='fastmath', value=ast.Constant(value=fastmath)),
            ast.keyword(arg='forceinline', value=ast.Constant(value=forceinline)),
            ast.keyword(arg='looplift', value=ast.Constant(value=looplift)),
            ast.keyword(arg='target_backend', value=ast.Constant(value=target_backend)),
            ast.keyword(arg='no_cfunc_wrapper', value=ast.Constant(value=no_cfunc_wrapper)),
            # ast.keyword(arg='cache', value=ast.Constant(value=cache))
        ]
    )
    nb_compyled_func_tree.body[0].decorator_list.append(numba_decorator)
    nb_compyled_func_tree.body[0].name += '_nb_compyled'
    return ast.fix_missing_locations(nb_compyled_func_tree)

def encapulate(wrap_tree: ast.AST, callmap_tree: ast.AST, original_tree: ast.AST) -> ast.AST:
    wrap_tree.body[0].body.insert(0, callmap_tree.body[0])
    wrap_tree.body[0].body.insert(1, original_tree.body[0])
    return ast.fix_missing_locations(wrap_tree)

def compile_tree(built_func_tree: ast.AST, exec_globals: Dict[str, Any], qualname: str, build_qualifier: str) -> Dict:
    try:
        exec(compile(built_func_tree, filename="<ast>", mode="exec"), exec_globals)
        return {build_qualifier: exec_globals[qualname + build_qualifier]}
    except Exception as e:
        logger.warning(e)
    return {}


def _prepare_funcs(original_func: ast.AST, mapping: Dict[str, int], func_int_identifier: str, D3_to_D2: bool = False, no_vectorized = False) -> Dict[str, Callable]:
    available_funcs = {'original': original_func}
    
    exec_globals = globals().copy()
    exec_globals.update({'np': np, 'nb': nb})
    callmap_func_ast = create_callmap_function_ast(mapping)
    callmap_func_tree = ast.fix_missing_locations(ast.Module(body=[callmap_func_ast], type_ignores=[]))
    original_tree, loop_func_tree, vectorize_func_tree = create_transformed_function_ast(original_func, mapping, func_int_identifier, D3_to_D2 = D3_to_D2)

    loop_func_tree = encapulate(loop_func_tree, callmap_func_tree, original_tree)
    available_funcs.update(compile_tree(loop_func_tree, exec_globals, func_int_identifier, '_loop'))

    if not no_vectorized:
        vectorize_func_tree = encapulate(vectorize_func_tree, callmap_func_tree, original_tree)
        available_funcs.update(compile_tree(vectorize_func_tree, exec_globals, func_int_identifier, '_vectorized'))

    nb_compyled_loop_func_tree = numba_decorate(loop_func_tree)
    if not no_vectorized:
        nb_compyled_vectorize_func_tree = numba_decorate(vectorize_func_tree)
        available_funcs.update(compile_tree(nb_compyled_vectorize_func_tree, exec_globals, func_int_identifier, '_vectorized_nb_compyled'))


    available_funcs.update(compile_tree(nb_compyled_loop_func_tree, exec_globals, func_int_identifier, '_loop_nb_compyled'))

    return available_funcs


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

def make_class_decorator(function_decorator: Callable) -> Callable:
    """
    Creates a class decorator from a given function decorator.

    Args:
        function_decorator (Callable): A function decorator to be applied to class methods.

    Returns:
        Callable: A class decorator.
    """
    @functools.wraps(function_decorator)
    def class_decorator(cls: Type) -> Type:
        """
        The class decorator generated from the function decorator.

        Args:
            cls (Type): The class to which the decorator is applied.

        Returns:
            Type: The decorated class.
        """
        for attr_name, attr_value in cls.__bases__[0].__dict__.items():
            if callable(attr_value) and not attr_name.startswith('_') and attr_name not in cls.__dict__:
                setattr(cls, attr_name, function_decorator(attr_value))
        for attr_name, attr_value in cls.__dict__.items():
             if callable(attr_value) and not attr_name.startswith('_'):
                setattr(cls, attr_name, function_decorator(attr_value))
        return cls
    return class_decorator

def autowrap_pandas_return(fn: Callable) -> Callable:
    """
    Decorator to add validation and error handling to class methods.

    Args:
        fn (Callable): The original method of the class.

    Returns:
        Callable: The decorated method with added validation and error handling.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self._outside_call:
            self._outside_call = False
            res = fn(self, *args, **kwargs)
            if isinstance(res, pd.DataFrame):
                res = pandopt(res)
            elif isinstance(res, pd.core.window.rolling.Rolling):
                res = pandoptRoll(res)
            elif isinstance(res, pandas.core.groupby.generic.DataFrameGroupBy):
                res = pandoptGroup(res)
            self._outside_call = True
            return res
        return fn(self, *args, **kwargs)
    return wrapper

@make_class_decorator(autowrap_pandas_return)
class pandopt(pd.DataFrame):
    _compiled_func = {}
    _outside_call = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def to_pandas(self):
        return pd.DataFrame(self)

    @property
    def colname_to_colnum(self):
        return {k: i for i, k in enumerate(self.columns)}

    @property
    def rowname_to_rownum(self):
        return {k: i for i, k in enumerate(self.index)}

    @property
    def __name__(self):
        return 'fid' + functools.reduce(laapply_benchmarks-Copy1(lambda x, y: f'{x}&{y}', mapper) + func_qualifier)

    def apply(self, func, axis = None, *args, pandas_fallback = False, data = None, new_index = None, from_rolling  = False, **kwargs):
        if pandas_fallback or args or kwargs: 
            logger.warning(f'{__class__} {"finish in pandas fallback for func "+func if pandas_fallback else "apply only supports func and axis arguments, using default pandas apply"}')
            return super().apply(func, axis = axis, *args, **kwargs)
        if axis is None:
            self.apply(func, axis = 1, *args, pandas_fallback = False, data = None, new_index = None, from_rolling = from_rolling, **kwargs).apply(func, axis = 1, *args, pandas_fallback = False, data = None, new_index = None, from_rolling = from_rolling, **kwargs)
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
        return pandopt(result, index=new_index, columns=new_columns)


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
        return pandoptRoll(self, window=window, *args, **kwargs)

    
    def groupby(self, *args, **kwargs):
        return pandoptGroup(self, *args, **kwargs)

    def resample(self, *args, **kwargs):
        # NotImplementedError('Expanding not yet implemented')
        return super().resample(*args, **kwargs)

    def expanding(self, *args, **kwargs):
        # NotImplementedError('Expanding not yet implemented')
        return super().expanding(*args, **kwargs)

        
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            try:
                func = load_function(name)  
            except AttributeError:
                raise AttributeError(f"{name} not found or could not be loaded.")
            return self.apply(func=func, *args, **kwargs)
        return wrapper
        

@make_class_decorator(autowrap_pandas_return)
class pandoptRoll(pd.core.window.rolling.Rolling):
    _outside_call = True
    def __init__(self, df, window=10, *args, **kwargs):
        self._idf = pandopt(df) if not isinstance(df, pandopt) else df
        super().__init__(self._idf, window=window, *args, **kwargs)

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
    
    def aggregate(self, func, *args, **kwargs):
        if isinstance(func, dict):
            result = {}
            for column, func in func.items():
                func = load_function(func)  # CAN BE IMPROVED MUCH
                result[column] = self.apply(lambda x: x[column].pipe(func_name), *args, **kwargs)
            return pandopt(result)
        func = load_function(func)
        return self.apply(func, *args, **kwargs)

    def agg(self, func, *args, **kwargs):
        return self.aggregate(func, *args, **kwargs)
  
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            try:
                func = load_function(name)  
            except AttributeError:
                raise AttributeError(f"{name} not found or could not be loaded.")
            return self.apply(func=func, *args, **kwargs)
        return wrapper
    

@make_class_decorator(autowrap_pandas_return)
class pandoptGroup(pd.core.groupby.generic.DataFrameGroupBy):
    _outside_call = True
    def __init__(self, df, *args, **kwargs):
        self.index = np.range(df.shape[0])
        self._idf = pandopt(df) if not isinstance(df, pandopt) else df
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
    
    def aggregate(self, func, *args, **kwargs):
        if isinstance(func, dict):
            result = {}
            for column, func in func.items():
                func = load_function(func)  # CAN BE IMPROVED MUCH
                result[column] = self.apply(lambda x: x[column].pipe(func_name), *args, **kwargs)
            return pandopt(result)
        func = load_function(func)
        return self.apply(func, *args, **kwargs)

    def agg(self, func, *args, **kwargs):
        return self.aggregate(func, *args, **kwargs)
  
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            try:
                func = load_function(name)  
            except AttributeError:
                raise AttributeError(f"{name} not found or could not be loaded.")
            return self.apply(func=func, *args, **kwargs)
        return wrapper
    
