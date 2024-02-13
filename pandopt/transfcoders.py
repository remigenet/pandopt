import ast
import copy
import inspect
import logging
import functools
import sys
from typing import Callable, Type, Dict, Tuple, Any, Union
import numpy as np
import numba as nb
logger = logging.getLogger()
logger.setLevel(0)



def create_callmap_function_ast(mapping: Dict[str, int]) -> ast.FunctionDef:
    """
    Create an abstract syntax tree (AST) for a function that maps arguments based on a provided mapping.

    This function generates a FunctionDef node in an AST, representing a function
    that transforms its input based on a specified mapping dictionary.

    Parameters
    ----------
    mapping : Dict[str, int]
        A dictionary mapping strings to integers, used to transform the function's input.

    Returns
    -------
    ast.FunctionDef
        The AST node representing the defined function.

    Examples
    --------
    >>> mapping = {"a": 1, "b": 2}
    >>> func_ast = create_callmap_function_ast(mapping)
    >>> print(ast.dump(func_ast))
    """

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
    """
    AST Node Transformer to replace subscript expressions in a function's AST.

    This class is a custom AST NodeTransformer that traverses a function's AST and
    replaces subscript expressions (e.g., array[index]) based on a specified argument name.

    Attributes
    ----------
    arg_name : str
        The name of the argument whose subscript expressions are to be transformed.

    Methods
    -------
    visit_Subscript(node):
        Visit a Subscript node in the AST and replace it if it matches the specified argument.

    Examples
    --------
    >>> replacer = SubscriptReplacer('arg_name')
    >>> modified_tree = replacer.visit(original_tree)
    """
    
    def __init__(self, arg_name: str):
        """
        Initialize the SubscriptReplacer with the specified argument name.

        Parameters
        ----------
        arg_name : str
            The name of the argument to target for subscript replacement.

        Examples
        --------
        >>> replacer = SubscriptReplacer('data')
        """

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

def create_transformed_function_ast(original_func: Callable, mapping: Dict[str, int], func_int_identifier: str, D3_to_D2: bool = False, no_vectorized: bool = False) -> Tuple[ast.AST, ast.AST, ast.AST]:
    """
    Create transformed versions of a given function as abstract syntax trees (ASTs).

    This function generates ASTs for different versions of a given function, incorporating
    specific transformations and optimizations based on the provided parameters.

    Parameters
    ----------
    original_func : Callable
        The original function to transform.
    mapping : Dict[str, int]
        A mapping dictionary to be used in the transformed function.
    func_int_identifier : str
        A unique identifier for the function being transformed.
    D3_to_D2 : bool, default False
        Flag indicating if the function involves 3D to 2D array transformation.
    no_vectorized : bool, default False
        Flag indicating if the function should avoid vectorized implementation.

    Returns
    -------
    Tuple[ast.AST, ast.AST, ast.AST]
        A tuple containing ASTs for loop-based, vectorized, and loop-based compiled versions of the function.

    Examples
    --------
    >>> def original_function(x):
    >>>     return x + 1
    >>> mapping = {'x': 0}
    >>> func_ast, loop_ast, vector_ast = create_transformed_function_ast(original_function, mapping, 'func_id')
    """

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
        loop_func_str_oc = f"""
callmap = nb.jit(cache = True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True ,nopython=True, nogil=True)(callmap)
    
@nb.jit(cache = True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True ,nopython=True, nogil=True)
def {func_int_identifier}_compyle_compute(Z, res):
    for i in nb.prange(n):
        res[i] = {func_int_identifier}_comppyled_basefunc(Z[i, :])

    
@nb.jit(cache = True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True ,nopython=True, nogil=True)
def {_compyle_compute}(Z):
    z=np.zeros(len(Z), dtype=Z.dtype)
    fidnvctFalseD3_to_D2_FalseTrueagg_sum_loop_nb_compyledcompute(Z, z)
    return z
        """
    loop_func_tree = ast.parse(loop_base_func_str)
    loop_func_tree_oc = ast.parse(loop_base_func_str.replace('temporary', 'temporary_nb_compyled').replace('_loop', '_oc_loop'))
    vectorize_func_tree = None
    if not no_vectorized:
        vectorized_base_func_str = f"""
def {func_int_identifier}_vectorized(Z):
    return temporary(Z.T)
        """
        vectorize_func_tree = ast.parse(vectorized_base_func_str)
        vectorize_func_tree_oc = ast.parse(vectorized_base_func_str.replace('temporary', 'temporary_nb_compyled').replace('_vectorized', '_oc_vectorized'))
    
    return original_tree, loop_func_tree, vectorize_func_tree, loop_func_tree_oc, vectorize_func_tree_oc

def numba_decorate(func_tree: ast.AST, nopython: bool = True, nogil: bool = True, parallel: bool = True, fastmath: bool = True, forceinline: bool = True, looplift: bool = True, target_backend: bool = True, no_cfunc_wrapper: bool = True, cache: bool = True) -> ast.AST:
    """
    Apply Numba JIT decorators to a function's AST for optimized execution.

    This function modifies a given function's AST by adding Numba JIT decorators,
    which enable just-in-time compilation for improved performance.

    Parameters
    ----------
    func_tree : ast.AST
        The AST of the function to be decorated.
    nopython : bool, default True
        Flag to enable Numba's 'nopython' mode for more efficient code.
    nogil : bool, default True
        Flag to release the GIL (Global Interpreter Lock) within Numba-compiled functions.
    parallel : bool, default True
        Flag to enable automatic parallelization of loops in Numba-compiled functions.
    fastmath : bool, default True
        Flag to enable fast math operations within Numba-compiled functions.
    forceinline : bool, default True
        Flag to force inlining of functions within Numba-compiled functions.
    looplift : bool, default True
        Flag to enable loop lifting within Numba-compiled functions.
    target_backend : bool, default True
        Flag to set a specific target backend for Numba compilation.
    no_cfunc_wrapper : bool, default True
        Flag to avoid creating a C function wrapper.
    cache : bool, default True
        Flag to enable caching of the compiled function.

    Returns
    -------
    ast.AST
        The modified AST with Numba JIT decorators applied.

    Examples
    --------
    >>> func_tree = ast.parse('def func(x): return x + 1')
    >>> decorated_tree = numba_decorate(func_tree)
    """
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

def encapulate(compute_function: ast.AST, callmap_tree: ast.AST, original_tree: ast.AST) -> ast.AST:
    """
    Encapsulate a function's AST with additional functionality defined in another AST.

    This function combines ASTs to encapsulate the original function within additional
    logic defined in other ASTs. Typically used to add preprocessing or postprocessing steps.

    Parameters
    ----------
    wrap_tree : ast.AST
        The AST of the function that will encapsulate the original function.
    callmap_tree : ast.AST
        The AST of the function that provides additional preprocessing logic.
    original_tree : ast.AST
        The AST of the original function to be encapsulated.

    Returns
    -------
    ast.AST
        The combined AST with the original function encapsulated within the additional logic.

    Examples
    --------
    >>> wrap_tree = ast.parse('def wrapper(): pass')
    >>> callmap_tree = ast.parse('def callmap(x): return x')
    >>> original_tree = ast.parse('def original(x): return x + 1')
    >>> encapsulated_tree = encapulate(wrap_tree, callmap_tree, original_tree)
    """
    wrapper_str = f"""
def {compute_function.body[0].name}(Z):
    

    return compute(Z)
    """

    wrap_tree = ast.parse(wrapper_str)
    wrap_tree.body[0].body.insert(0, callmap_tree.body[0])
    wrap_tree.body[0].body.insert(1, original_tree.body[0])
    compute_function.body[0].name = 'compute'
    wrap_tree.body[0].body.insert(2, compute_function.body[0])
    return ast.fix_missing_locations(wrap_tree)


def compile_tree(built_func_tree: ast.AST, exec_globals: Dict[str, Any], qualname: str, build_qualifier: str) -> Dict:
    """
    Compile an AST into an executable function and store it in a dictionary.

    This function compiles an AST representing a Python function into an executable
    function object and stores it in a dictionary under a specified key.

    Parameters
    ----------
    built_func_tree : ast.AST
        The AST of the function to be compiled.
    exec_globals : Dict[str, Any]
        The global namespace in which the function should be compiled.
    qualname : str
        The qualified name to be used for the compiled function.
    build_qualifier : str
        A string qualifier to distinguish different versions of the compiled function.

    Returns
    -------
    Dict
        A dictionary containing the compiled function under the specified key.

    Examples
    --------
    >>> func_tree = ast.parse('def func(x): return x + 1')
    >>> exec_globals = globals()
    >>> compiled_funcs = compile_tree(func_tree, exec_globals, 'func', '_compiled')
    """
    try:
        compyled = None
        def wrapped(x):
            nonlocal compyled
            if compyled is None:
                exec(compile(built_func_tree, filename="<ast>", mode="exec"), exec_globals)
                compyled = exec_globals[qualname + build_qualifier]
            return compyled(x)
        return {build_qualifier: wrapped}
    except Exception as e:
        logger.warning(e)
        return {}
    

def _prepare_funcs(original_func: Callable, mapping: Dict[str, int], func_int_identifier: str, D3_to_D2: bool = False, no_vectorized: bool = False) -> Dict[str, Callable]:
    """
    Prepare and compile various versions of a function for optimized execution.

    This function takes an original function and prepares different versions of it, 
    including loop-based, vectorized, and compiled forms, based on the provided parameters 
    and mapping. These versions are used for optimized execution in different scenarios.

    Parameters
    ----------
    original_func : Callable
        The original function to be transformed and compiled.
    mapping : Dict[str, int]
        A mapping dict influencing the behavior of the function.
    func_int_identifier : str
        A unique identifier for the function.
    D3_to_D2 : bool, default False
        Indicates if the function is used for 3D to 2D array conversion.
    no_vectorized : bool, default False
        Indicates if the function should not have a vectorized version.

    Returns
    -------
    Dict[str, Callable]
        A dictionary containing different versions of the prepared function.

    Examples
    --------
    >>> def original_function(x):
    >>>     return x + 1
    >>> mapping = {'x': 0}
    >>> prepared_funcs = _prepare_funcs(original_function, mapping, 'func_id')
    """
    available_funcs = {'original': original_func}
    
    exec_globals = globals().copy()
    exec_globals.update({'np': np, 'nb': nb})
    callmap_func_ast = create_callmap_function_ast(mapping)
    callmap_func_tree = ast.fix_missing_locations(ast.Module(body=[callmap_func_ast], type_ignores=[]))
    original_tree, loop_func_tree, vectorize_func_tree, loop_func_tree_oc, vectorize_func_tree_oc = create_trans
    formed_function_ast(original_func, mapping, func_int_identifier, D3_to_D2 = D3_to_D2)
    available_funcs.update(compile_tree(original_tree, exec_globals, func_int_identifier, '_original'))
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

    nb_compyled_original_tree = numba_decorate(original_tree)
    nb_compyled_callmap_func_tree = numba_decorate(callmap_func_tree)
    if not no_vectorized:
        vectorize_func_tree_oc = encapulate(vectorize_func_tree_oc, nb_compyled_callmap_func_tree, nb_compyled_original_tree)
        nb_compyled_vectorize_func_oc_tree = numba_decorate(vectorize_func_tree_oc)
        available_funcs.update(compile_tree(nb_compyled_vectorize_func_oc_tree, exec_globals, func_int_identifier, '_oc_vectorized_nb_compyled'))

    nb_compyled_loop_func_tree_oc = encapulate(loop_func_tree_oc, nb_compyled_callmap_func_tree, nb_compyled_original_tree)
    nb_compyled_loop_func_tree_oc = numba_decorate(nb_compyled_loop_func_tree_oc)
    available_funcs.update(compile_tree(nb_compyled_loop_func_tree_oc, exec_globals, func_int_identifier, '_oc_loop_nb_compyled'))

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

# @decorator
# def {func_int_identifier}_loop(Z):
#     def callmap(x):
#         ....do stuff
#     def temporary(x):
#         do stuf...

#     n, c, t = Z.shape
#     res = np.zeros((n, c))
#     for col in nb.prange(c):
#         for row in nb.prange(n):
#             res[row, col] = temporary(Z[row,col,:])
#     return res


# but instead

# def {func_int_identifier}_loop():
#     @decorator
#     def callmap(x):
#         ....do stuff
#     @decorator
#     def temporary(x):
#         do stuf...
#     @decorator
#     def compute(Z)
#         n, c, t = Z.shape
#         res = np.zeros((n, c))
#         for col in nb.prange(c):
#             for row in nb.prange(n):
#                 res[row, col] = temporary(Z[row,col,:])
#         return res
#     return compute