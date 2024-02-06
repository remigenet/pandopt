import ast
import inspect
import types
import numpy as np
import numba as nb
import pandas as pd
import time
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
        compare = ast.Compare(
            left=ast.Name(id='x', ctx=ast.Load()),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=key)]
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

def create_transformed_function_ast(original_func: Callable, mapping: Dict[str, int]) -> Tuple[ast.AST, ast.AST, ast.AST]:
    # Parse the original function
    original_tree = ast.parse(inspect.getsource(original_func))
    arg_name = original_tree.body[0].args.args[0].arg
    
    # Rename the original function
    original_tree.body[0].name = 'temporary'
    
    # Apply the AST transformation
    replacer = SubscriptReplacer(arg_name)
    original_tree = replacer.visit(original_tree)
    ast.fix_missing_locations(original_tree)

    # Replace dictionary accesses with callmap in the original function
    # This would be similar to the code in SubscriptReplacer

    # Create a new function that applies 'temporary' over an array
    loop_base_func_str = f"""
def {original_func.__qualname__}_loop(Z):
    n = Z.shape[0]
    res = np.zeros((n, 1))
    for i in nb.prange(n):
        res[i, 0] = temporary(Z[i, :])
    return res
    """
    vectorized_base_func_str = f"""
def {original_func.__qualname__}_vectorized(Z):
    return temporary(Z.T)
    """
    loop_func_tree = ast.parse(loop_base_func_str)
    vectorize_func_tree = ast.parse(vectorized_base_func_str)

    return original_tree, loop_func_tree, vectorize_func_tree

def numba_decorate(func_tree: ast.AST, nopython: bool = True, nogil: bool = True, parallel: bool = True) -> ast.AST:
    # # Add Numba JIT decorator
    nb_compyled_func_tree = copy.deepcopy(ast.fix_missing_locations(func_tree))
    numba_decorator = ast.Call(
        func=ast.Attribute(value=ast.Name(id='nb', ctx=ast.Load()), attr='jit', ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg='nopython', value=ast.Constant(value=nopython)),
            ast.keyword(arg='nogil', value=ast.Constant(value=nogil)),
            ast.keyword(arg='parallel', value=ast.Constant(value=parallel))
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


def _prepare_funcs(original_func: ast.AST, mapping: Dict[str, int]) -> Dict[str, Callable]:
    exec_globals = globals().copy()
    exec_globals.update({'np': np, 'nb': nb})
    callmap_func_ast = create_callmap_function_ast(mapping)
    callmap_func_tree = ast.fix_missing_locations(ast.Module(body=[callmap_func_ast], type_ignores=[]))
    original_tree, loop_func_tree, vectorize_func_tree = create_transformed_function_ast(original_func, mapping)

    loop_func_tree = encapulate(loop_func_tree, callmap_func_tree, original_tree)
    vectorize_func_tree = encapulate(vectorize_func_tree, callmap_func_tree, original_tree)
    
    available_funcs = {}
    available_funcs.update(compile_tree(vectorize_func_tree, exec_globals, original_func.__qualname__, '_vectorized'))
    available_funcs.update(compile_tree(loop_func_tree, exec_globals, original_func.__qualname__, '_loop'))
    
    nb_compyled_loop_func_tree = numba_decorate(loop_func_tree)
    nb_compyled_vectorize_func_tree = numba_decorate(vectorize_func_tree)

    available_funcs.update(compile_tree(nb_compyled_vectorize_func_tree, exec_globals, original_func.__qualname__, '_vectorized_nb_compyled'))
    available_funcs.update(compile_tree(nb_compyled_loop_func_tree, exec_globals, original_func.__qualname__, '_loop_nb_compyled'))

    return available_funcs


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
            self._outside_call = True
            return res
        return fn(self, *args, **kwargs)
    return wrapper

@make_class_decorator(autowrap_pandas_return)
class pandopt(pd.DataFrame):
    _compiled_func = None
    _compiled_func = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    @property
    def __name__(self):
        return functools.reduce(lambda x, y: x + y, self.name_to_index.keys())

    @property
    def colname_to_colnum(self):
        return {k: i for i, k in enumerate(self.columns)}

    @property
    def rowname_to_rownum(self):
        return {k: i for i, k in enumerate(self.index)}
    
    def _compiled_qualifier(self, func_qualifier, mapper):
        return hash(functools.reduce(lambda x, y: f'{x}&{y}', mapper) + func_qualifier)

    def apply(self, func, axis = 0, *args, pandas_fallback = False, **kwargs):
        if pandas_fallback: 
            logger.warning(f'{__class__} finish in pandas fallback for func {func}')
            return super().apply(func, axis = 0, *args, **kwargs)
        if args or kwargs:
            logger.warning(f'{__class__} apply only supports func and axis arguments, using default pandas apply')
            return super().apply(func, axis = 0, *args, **kwargs)
        return pandopt((self._compiled_func.get((name:=self._compiled_qualifier(func_qualifier = func.__qualname__, mapper=(mapper:=self.colname_to_colnum if axis else self.rowname_to_rownum)))) or self._build_apply_versions(func, mapper, name))(self.to_numpy() if axis else self.to_numpy().T), index = self.index if axis else self.columns)

    def _with_fallback_wrap(self, apply_func_dict):
        def _with_protects(*args, **kwargs):
            for key in ('_vectorized_nb_compyled', '_loop_compyled', '_vectorized', '_loop'):
                if key not in apply_func_dict:
                    continue
                try:
                    return apply_func_dict[key](*args, **kwargs)
                except:
                    apply_func_dict.pop(key)
            return self.apply(*args, pandas_fallback = True, **kwargs)
        return _with_protects
    
    
    def _build_apply_versions(self, func, map, name):
        self._compiled_func[name] = self._with_fallback_wrap(_prepare_funcs(func, map))
        return self._compiled_func[name]

    
 
@make_class_decorator(autowrap_pandas_return)
class pandopt(pd.DataFrame):
    _compiled_func = {}
    _outside_call = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._compiled_func = {}

    @property
    def __name__(self):
        return functools.reduce(lambda x, y: x + y, self.name_to_index.keys())

    @property
    def colname_to_colnum(self):
        return {k: i for i, k in enumerate(self.columns)}

    @property
    def rowname_to_rownum(self):
        return {k: i for i, k in enumerate(self.index)}
    
    def _compiled_qualifier(self, func_qualifier, mapper):
        return hash(functools.reduce(lambda x, y: f'{x}&{y}', mapper) + func_qualifier)

    def apply(self, func, axis = 0, *args, pandas_fallback = False, **kwargs):
        if pandas_fallback: 
            logger.warning(f'{__class__} finish in pandas fallback for func {func}')
            return super().apply(func, axis = 0, *args, **kwargs)
        if args or kwargs:
            logger.warning(f'{__class__} apply only supports func and axis arguments, using default pandas apply')
            return super().apply(func, axis = 0, *args, **kwargs)
        return pandopt((self._compiled_func.get((name:=self._compiled_qualifier(func_qualifier = func.__qualname__, mapper=(mapper:=self.colname_to_colnum if axis else self.rowname_to_rownum)))) or self._build_apply_versions(func, mapper, name))(self.to_numpy() if axis else self.to_numpy().T), index = self.index if axis else self.columns)

    def _with_fallback_wrap(self, apply_func_dict):
        def _with_protects(*args, **kwargs):
            for key in ('_vectorized_nb_compyled', '_loop_compyled', '_vectorized', '_loop'):
                if key not in apply_func_dict:
                    continue
                try:
                    return apply_func_dict[key](*args, **kwargs)
                except:
                    apply_func_dict.pop(key)
            return self.apply(*args, pandas_fallback = True, **kwargs)
        return _with_protects
    
    def _build_apply_versions(self, func, map, name):
        self._compiled_func[name] = self._with_fallback_wrap(_prepare_funcs(func, map))
        return self._compiled_func[name]

    def rolling(self, window, *args, **kwargs):
        return pandoptRoll(self, window=window, *args, **kwargs)

    
    def groupby(self, *args, **kwargs):
        return pandoptGroupBy(self, *args, **kwargs)

    def resample(self, *args, **kwargs):
        return pandoptResampler(self, *args, **kwargs)

    def expanding(self, *args, **kwargs):
        return pandoptExpanding(self, *args, **kwargs)


    def _apply_with_compiled_func(self, func, data, axis=0, *args, **kwargs):
        name = self._compiled_qualifier(func_qualifier=(func_name:=func.__qualname__), mapper=(mapper:=self.colname_to_colnum if axis else self.rowname_to_rownum))
        compiled_func = self._compiled_func.get(name) or self._build_apply_versions(func, mapper, name)
        return res.T if hasattr((res:=compiled_func(data, *args, **kwargs)), 'T') else res

@make_class_decorator(autowrap_pandas_return)
class pandoptRoll(pd.core.window.rolling.Rolling):
    _outside_call = True
    def __init__(self, df, *args, **kwargs):
        self._idf = df
        super().__init__(df, *args, **kwargs)

    def agg(self, func, *args, **kwargs):
        if isinstance(func, dict):
            # For dictionary input, apply each function to the respective column
            result = {}
            for column, func_name in func.items():
                if func_name in ['mean', 'std', 'sum', 'min', 'max', 'count', 'median']:
                    result[column] = super().agg({column: func_name})
                else:
                    result[column] = self.apply(lambda x: x[column].pipe(func_name), *args, **kwargs)
            return pandopt(result)
        elif func in ['mean', 'std', 'sum', 'min', 'max', 'count', 'median']:
            return super().agg(func, *args, **kwargs)
        else:
            return self.apply(func, *args, **kwargs)
        
    def apply(self, func, *args, pandas_fallback=False, **kwargs):
        try:
            return pandopt(self._idf._apply_with_compiled_func(func, sliding_window_view(self._idf.to_numpy(), self.window, axis=0), axis=2, *args, **kwargs), columns=self._idf.columns, index=self._idf.index[self.window-1:])
        except Exception as e:
            logger.warning(f'Error in pandoptRoll apply: {e}')
            if pandas_fallback:
                return super().apply(func, *args, **kwargs)
            raise
            
@make_class_decorator(autowrap_pandas_return)
class pandoptGroupBy(pd.core.groupby.DataFrameGroupBy):
    _outside_call = True
    def __init__(self, df, *args, **kwargs):
        self._idf = df
        super().__init__(df, *args, **kwargs)

    def agg(self, func, *args, **kwargs):
        if isinstance(func, dict):
            # For dictionary input, apply each function to the respective column
            result = {}
            for column, func_name in func.items():
                if func_name in ['mean', 'std', 'sum', 'min', 'max', 'count', 'median']:
                    result[column] = super().agg({column: func_name})
                else:
                    result[column] = self.apply(lambda x: x[column].pipe(func_name), *args, **kwargs)
            return pandopt(result)
        elif func in ['mean', 'std', 'sum', 'min', 'max', 'count', 'median']:
            return super().agg(func, *args, **kwargs)
        else:
            return self.apply(func, *args, **kwargs)

    def apply(self, func, *args, pandas_fallback=False, **kwargs):
        try:
            results = []
            for name, group in self:
                result_array = self._idf._apply_with_compiled_func(func, group.to_numpy(), axis=1, *args, **kwargs)
                
                # If the function returns a single value per group
                if result_array.ndim == 1 or result_array.shape[1] == 1:
                    result_df = pandopt(result_array, index=group.index, columns=[func.__name__])
                else:
                    result_df = pandopt(result_array, index=group.index, columns=group.columns)
                
                results.append(result_df)

            return pd.concat(results, axis=0)
        except Exception as e:
            logger.warning(f'Error in pandoptGroupBy apply: {e}')
            if pandas_fallback:
                return super().apply(func, *args, **kwargs)
            raise

@make_class_decorator(autowrap_pandas_return)
class pandoptResampler(pd.core.resample.Resampler):
    _outside_call = True
    def __init__(self, df, *args, **kwargs):
        self._idf = df
        super().__init__(df, *args, **kwargs)

    def agg(self, func, *args, **kwargs):
        if isinstance(func, dict):
            results = {col: (super().agg({col: f}) if f in ['mean', 'std', 'sum', 'min', 'max', 'count', 'median'] else self.apply(lambda x: x[col].pipe(f), *args, **kwargs)) for col, f in func.items()}
            return pandopt(results)
        elif func in ['mean', 'std', 'sum', 'min', 'max', 'count', 'median']:
            return super().agg(func, *args, **kwargs)
        else:
            return self.apply(func, *args, **kwargs)

    def apply(self, func, *args, pandas_fallback=False, **kwargs):
        try:
            if pandas_fallback:
                return super().apply(func, *args, **kwargs)
            return self._idf._apply_with_compiled_func(func, self._idf.to_numpy(), *args, **kwargs)
        except Exception as e:
            logger.warning(f'Error in pandoptResampler apply: {e}')
            return super().apply(func, *args, **kwargs)

@make_class_decorator(autowrap_pandas_return)
class pandoptExpanding(pd.core.window.expanding.Expanding):
    _outside_call = True
    def __init__(self, df, *args, **kwargs):
        self._idf = df
        super().__init__(df, *args, **kwargs)

    def agg(self, func, *args, **kwargs):
        if isinstance(func, dict):
            results = {col: (super().agg({col: f}) if f in ['mean', 'std', 'sum', 'min', 'max', 'count', 'median'] else self.apply(lambda x: x[col].pipe(f), *args, **kwargs)) for col, f in func.items()}
            return pandopt(results)
        elif func in ['mean', 'std', 'sum', 'min', 'max', 'count', 'median']:
            return super().agg(func, *args, **kwargs)
        else:
            return self.apply(func, *args, **kwargs)

    def apply(self, func, *args, pandas_fallback=False, **kwargs):
        try:
            if pandas_fallback:
                return super().apply(func, *args, **kwargs)
            return self._idf._apply_with_compiled_func(func, self._idf.to_numpy(), *args, **kwargs)
        except Exception as e:
            logger.warning(f'Error in pandoptExpanding apply: {e}')
            return super().apply(func, *args, **kwargs)

