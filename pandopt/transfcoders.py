import ast
import copy
import inspect
import logging
import functools
import sys
from typing import Callable, Type, Dict, Tuple, Any
import numpy as np
import numba as nb
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



#
# def make_class_decorator(function_decorator: Callable) -> Callable:
#     """
#     Creates a class decorator from a given function decorator.
#
#     Args:
#         function_decorator (Callable): A function decorator to be applied to class methods.
#
#     Returns:
#         Callable: A class decorator.
#     """
#     @functools.wraps(function_decorator)
#     def class_decorator(cls: Type) -> Type:
#         """
#         The class decorator generated from the function decorator.
#
#         Args:
#             cls (Type): The class to which the decorator is applied.
#
#         Returns:
#             Type: The decorated class.
#         """
#         for attr_name, attr_value in cls.__bases__[0].__dict__.items():
#             if callable(attr_value)  and attr_name not in cls.__dict__:
#                 setattr(cls, attr_name, function_decorator(attr_value))
#         for attr_name, attr_value in cls.__dict__.items():
#              if callable(attr_value) and not attr_name.startswith('_'):
#                 setattr(cls, attr_name, function_decorator(attr_value))
#         return cls
#     return class_decorator
#
# def autowrap_pandas_return(fn: Callable) -> Callable:
#     """and not attr_name.startswith('_')
#     Decorator to add validation and error handling to class methods.
#
#     Args:
#         fn (Callable): The original method of the class.
#
#     Returns:
#         Callable: The decorated method with added validation and error handling.
#     """
#
#     if isinstance(fn, types.FunctionType):
#         return fn
#     @functools.wraps(fn)
#     def wrapper(self, *args, **kwargs):
#         print('here', self._outside_call, type(self), args, kwargs, fn, type(fn))
#         if self._outside_call:
#             self._outside_call = False
#             if not hasattr(fn, '__self__'):
#                 res = fn(*args, **kwargs)
#             else:
#                 res = fn(self, *args, **kwargs)
#             print(type(res))
#             if isinstance(res, pandas.DataFrame) and not isinstance(res, pandopt.DataFrame):
#                 res = pandopt.DataFrame(res)
#             elif isinstance(res, pandas.core.window.rolling.Rolling) and not isinstance(res, pandopt.RollOpt):
#                 res = pandopt.RollOpt(res)
#             elif isinstance(res, pandas.core.groupby.generic.DataFrameGroupBy) and not isinstance(res, pandopt.GroupOpt):
#                 res = pandopt.GroupOpt(res)
#             self._outside_call = True
#             return res
#         if not hasattr(fn, '__self__'):
#             return fn(*args, **kwargs)
#         return fn(self, *args, **kwargs)
#     return wrapper