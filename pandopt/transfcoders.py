import ast
import copy
import inspect
import logging
import functools
import sys
from typing import Callable, Type, Dict, Tuple, Any, Union
import numpy as np
from functools import reduce
import numba as nb
import hashlib
logger = logging.getLogger()
logger.setLevel(0)
import pandopt



def source_parser(f):
    source = inspect.getsource(f)
    if any(source.startswith(managed_method.__name__) for managed_method in [pandopt.DataFrame.apply, pandopt.DataFrame.aggregate, source_parser]):
        code = "    return " + source[source.find('x:') + 2:source.find('mapping')].strip().strip(',').strip(':').strip(')')
    elif f.__module__ != '__main__':
        code = "    import " + f.__module__ + '\n'
        code += "    return " + f.__module__ + '.' + f.__qualname__ +'(x)'
    else:
        return source
    uid = int(hashlib.sha1(code.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    return (f"""
def f{uid}(x):
{code} 
    """)

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

    # Create the Numba JIT decorator with specified options
    numba_decorator = ast.Call(
        func=ast.Attribute(value=ast.Name(id='nb', ctx=ast.Load()), attr='njit', ctx=ast.Load()),
        args=[
            ast.Call(
                func=ast.Attribute(value=ast.Name(id='nb', ctx=ast.Load()), attr='int8', ctx=ast.Load()),
                args=[ast.Attribute(value=ast.Name(id='nb', ctx=ast.Load()), attr='types.string', ctx=ast.Load())],
                keywords=[]
            )
        ],
        keywords=[
            ast.keyword(arg='cache', value=ast.Constant(value=False)),
            ast.keyword(arg='parallel', value=ast.Constant(value=True)),
            ast.keyword(arg='fastmath', value=ast.Constant(value=True)),
            ast.keyword(arg='forceinline', value=ast.Constant(value=True)),
            ast.keyword(arg='looplift', value=ast.Constant(value=True)),
            ast.keyword(arg='inline', value=ast.Constant(value='always')),
            ast.keyword(arg='target_backend', value=ast.Constant(value='host')),
            ast.keyword(arg='no_cfunc_wrapper', value=ast.Constant(value=True)),
            ast.keyword(arg='no_rewrites', value=ast.Constant(value=True)),
            ast.keyword(arg='nopython', value=ast.Constant(value=True)),
            ast.keyword(arg='nogil', value=ast.Constant(value=True)),
        ]
    )
    # Create the function definition
    func_def = ast.FunctionDef(
        name=(callmap_name:=f'callmap{hashlib(reduce(lambda x, y: x + y, mapping.keys()))}'),
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
        decorator_list=[numba_decorator],
        returns=None
    )
    return func_def, callmap_name


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
    
    def __init__(self, arg_name: str, callmap_name: str, axis: int, ndims : int):
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
        self.callmap_name = callmap_name
        self.axis = axis
        self.ndims = ndims
        self.used = False

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == self.arg_name:
            # Check for Python version compatibility
            if sys.version_info >= (3, 9):
                # Python 3.9 and later
                old_slice = node.slice
            else:
                # Python 3.8 and earlier
                old_slice = node.slice.value if isinstance(node.slice, ast.Index) else node.slice

            node.slice = ast.Tuple(
               elts=[
                  ast.Slice() if i != self.axis else
                  ast.Call(
                        func=ast.Name(id=self.callmap_name, ctx=ast.Load()),
                        args=[old_slice],
                        keywords=[]
                    ) for i in range(self.ndims)],
               ctx=ast.Load())
            self.used = True
            # Wrap the subscript in a call to callmap
        return self.generic_visit(node) 



def create_transformed_function_ast(original_func: Callable, mapping: Dict[str, np.int32], axis: int = 0, ndims: int = 2, dtype: str = 'float32') -> Tuple[ast.AST, ast.AST, ast.AST]:
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
    print(f'BUILDING WITH AXIS={axis} - ndims={ndims}')
    # Parse the original function
    source_code = source_parser(original_func)
    original_tree = ast.parse(source_code)
    hash_name = hashlib.sha1(str(reduce(lambda x, y: str(x) + str(y), mapping.keys())).encode('UTF-8')).digest().hex()
    if mapping:
        callmap_name=f'callmap{hash_name}'
    else:
        callmap_name = 'NoCallMap'
    uid = f'f{original_func.__qualname__}{hash_name}'
    arg_name = original_tree.body[0].args.args[0].arg
    original_tree.body[0].name = f'f{uid}'
    replacer = SubscriptReplacer(arg_name, callmap_name, axis = axis, ndims = ndims)
    
    results = []

    if "if" not in source_code:

        vectorize_tree = copy.deepcopy(original_tree)
        vectorize_tree = replacer.visit(vectorize_tree)
        ast.fix_missing_locations(vectorize_tree)
        
        vectorize_tree_opt = copy.deepcopy(vectorize_tree)
        vectorize_tree.body[0].name = 'vectorized'
        vectorize_tree_opt.body[0].name = 'vectorized_opt'
        vectorize_tree_opt = numba_decorate(vectorize_tree_opt)
        results += [vectorize_tree, vectorize_tree_opt]

    if ndims == 1:
        
        loop_tree = copy.deepcopy(original_tree)
        loop_tree_opt = copy.deepcopy(loop_tree)
        loop_tree.body[0].name = 'loop'
        loop_tree_opt.body[0].name = 'loop_opt'
        loop_tree_opt = numba_decorate(loop_tree_opt)
        return results + [loop_tree, loop_tree_opt]

    if ndims == 2:
        
        if axis == 0:
            loop_tree = copy.deepcopy(original_tree)
            loop_tree = replacer.visit(loop_tree)
            ast.fix_missing_locations(loop_tree)
            
            loop_tree_opt = copy.deepcopy(loop_tree)
            loop_tree.body[0].name += '_built__uloop_1'
            loop_tree_opt.body[0].name += '_built__uloop_opt_1'
            loop_tree_opt = numba_decorate(loop_tree_opt)
            loop_tree = ast.parse(ast.unparse(loop_tree) + f"""
def f{uid}_loop_1(Z, z, k):
    for i in nb.prange(k):
        z[i] = f{uid}_built__uloop_1(Z[i,:])

def loop(Z):
    k=Z.shape[0]
    z=np.zeros(k, dtype=np.{dtype})
    f{uid}_loop_1(Z, z, k)
    return z
        """)
            loop_tree_opt = ast.parse(ast.unparse(loop_tree_opt) + f"""
@nb.njit((nb.{dtype}[:,:], nb.{dtype}[:], nb.uint32), cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def f{uid}_loop_opt_1(Z, z, k):
    for i in nb.prange(k):
        z[i] = f{uid}_built__uloop_opt_1(Z[i,:])

@nb.njit(nb.{dtype}[:](nb.{dtype}[:,:]),cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def loop_opt(Z):
    k=np.uint32(len(Z))
    z=np.zeros(k , dtype=np.{dtype})
    f{uid}_loop_opt_1(Z, z, k)
    return z
            """)
            return results + [loop_tree, loop_tree_opt]
        if axis == 1:
            loop_tree = copy.deepcopy(original_tree)
            loop_tree = replacer.visit(loop_tree)
            ast.fix_missing_locations(loop_tree)
                
            loop_tree_opt = copy.deepcopy(loop_tree)
            loop_tree.body[0].name += '_built__uloop_1'
            loop_tree_opt.body[0].name += '_built__uloop_opt_1'
            loop_tree_opt = numba_decorate(loop_tree_opt)
            loop_tree = ast.parse(ast.unparse(loop_tree) + f"""
def f{uid}_loop_1(Z, z, k):
    for i in nb.prange(k):
        z[i] = f{uid}_built__uloop_1(Z[:,i])

@nb.njit(nb.{dtype}[:](nb.{dtype}[:,:]),cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def f{uid}_loop_opt_1(Z):
    z=np.zeros(Z.shape[0], dtype=np.{dtype})
    for i in nb.prange(k):
        z[i] = f{uid}_built__uloop_opt_1(Z[i,:])

@nb.njit(nb.{dtype}[:](nb.{dtype}[:,:]),cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def loop_opt(Z):
    k=np.uint32(len(Z))
    z=np.zeros(k , dtype=np.{dtype})
    f{uid}_loop_opt_1(Z, z, k)
    return z
        """)

            return results + [loop_tree, loop_tree_opt]
        elif axis is None:
            return results

    elif dim == 3:
            
        if axis == 0:
            loop_tree = copy.deepcopy(original_tree)
            loop_tree = replacer.visit(loop_tree)
            ast.fix_missing_locations(loop_tree)
            
            loop_tree_opt = copy.deepcopy(loop_tree)
            loop_tree.body[0].name += '_built__uloop_1'
            loop_tree_opt.body[0].name += '_built__uloop_opt_1'
            loop_tree_opt = numba_decorate(loop_tree_opt)
            loop_tree = ast.parse(ast.unparse(loop_tree) + f"""
def f{uid}_loop_1(Z, z, k):
    for i in nb.prange(k):
        z[i, :] = f{uid}_built__uloop_1(Z[i,:,:])

def loop(Z):
    z=np.zeros(Z.shape[1], dtype=np.{dtype})
    f{uid}_loop_1(Z, z, k)
    return z
        """)
        loop_tree_opt = ast.parse(ast.unparse(loop_tree_opt) + f"""
@nb.njit((nb.{dtype}[:,::], nb.{dtype}[:], nb.uint32), cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def f{uid}_loop_opt_1(Z, z, k):
    for i in nb.prange(k):
        z[i, :] = f{uid}_built__uloop_opt_1(Z[i,:,:])

@nb.njit(nb.{dtype}[:](nb.{dtype}[:,::]),cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def loop_opt(Z):
    k=np.uint32(len(Z))
    z=np.zeros(k , dtype=np.{dtype})
    f{uid}_loop_opt_1(Z, z, k)
    return z
            """)
        results += [loop_tree, loop_tree_opt]
    if axis == 1:
        loop_tree = copy.deepcopy(original_tree)
        loop_tree = replacer.visit(loop_tree)
        ast.fix_missing_locations(loop_tree)
        
        loop_tree_opt = copy.deepcopy(loop_tree)
        loop_tree.body[0].name += '_built__uloop_1'
        loop_tree_opt.body[0].name += '_built__uloop_opt_1'
        loop_tree_opt = numba_decorate(loop_tree_opt)
        loop_tree = ast.parse(ast.unparse(loop_tree) + f"""
def f{uid}_loop_1(Z, z, k):
    for i in nb.prange(k):
        z[i, :] = f{uid}_built__uloop_1(Z[:,i,:])

def loop(Z):
    z=np.zeros(Z.shape[1], dtype=np.{dtype})
    f{uid}_loop_1(Z, z, k)
    return z
        """)
        loop_tree_opt = ast.parse(ast.unparse(loop_tree_opt) + f"""
@nb.njit((nb.{dtype}[:,:], nb.{dtype}[:], nb.uint32), cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
    def f{uid}_loop_opt_1(Z, z, k):
    for i in nb.prange(k):
        z[i, :] = f{uid}_built__uloop_opt_1(Z[:,i,:])

@nb.njit(nb.{dtype}[:](nb.{dtype}[:,::,:]),cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
    def loop_opt(Z):
        k=np.uint32(len(Z))
        z=np.zeros(k , dtype=np.{dtype})
        f{uid}_loop_opt_1(Z, z, k)
        return z
            """)

        results += [loop_tree, loop_tree_opt]
    if axis == 2:
        loop_tree = copy.deepcopy(original_tree)
        loop_tree = replacer.visit(loop_tree)
        ast.fix_missing_locations(loop_tree)
        
        loop_tree_opt = copy.deepcopy(loop_tree)
        loop_tree.body[0].name += '_built__uloop_1'
        loop_tree_opt.body[0].name += '_built__uloop_opt_1'
        loop_tree_opt = numba_decorate(loop_tree_opt)
        loop_tree = ast.parse(ast.unparse(loop_tree) + f"""
def f{uid}_loop_1(Z, z, k):
    for i in nb.prange(k):
        z[i, :] = f{uid}_built__uloop_1(Z[:,:,i])

def loop(Z):
    z=np.zeros(Z.shape[2], dtype=np.{dtype})
    f{uid}_loop_1(Z, z, k)
    return z
        """)
        loop_tree_opt = ast.parse(ast.unparse(loop_tree_opt) + f"""
@nb.njit((nb.{dtype}[:,:], nb.{dtype}[:], nb.uint32), cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
    def f{uid}_loop_opt_1(Z, z, k):
    for i in nb.prange(k):
        z[i, :] = f{uid}_built__uloop_opt_1(Z[:,:,i])

@nb.njit(nb.{dtype}[:](nb.{dtype}[:,::,:]),cache = False, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
    def loop_opt(Z):
        k=np.uint32(len(Z))
        z=np.zeros(k , dtype=np.{dtype})
        f{uid}_loop_opt_1(Z, z, k)
        return z
            """)
        results += [loop_tree, loop_tree_opt]
    if replacer.used:
        callmap_func_tree, _ = create_callmap_function_ast(mapping)
        results.insert(0, callmap_func_tree)
    return results



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
        func=ast.Attribute(value=ast.Name(id='nb', ctx=ast.Load()), attr='njit', ctx=ast.Load()),
        args=[
            # ast.Call(
            #     func=ast.Attribute(value=ast.Name(id='nb', ctx=ast.Load()), attr='int8', ctx=ast.Load()),
            #     args=[ast.Attribute(value=ast.Name(id='nb', ctx=ast.Load()), attr='types.string', ctx=ast.Load())],
            #     keywords=[]
            # )
        ],
        keywords=[
            ast.keyword(arg='cache', value=ast.Constant(value=False)),
            ast.keyword(arg='parallel', value=ast.Constant(value=True)),
            ast.keyword(arg='fastmath', value=ast.Constant(value=True)),
            ast.keyword(arg='forceinline', value=ast.Constant(value=True)),
            ast.keyword(arg='looplift', value=ast.Constant(value=True)),
            ast.keyword(arg='inline', value=ast.Constant(value='always')),
            ast.keyword(arg='target_backend', value=ast.Constant(value='host')),
            ast.keyword(arg='no_cfunc_wrapper', value=ast.Constant(value=True)),
            ast.keyword(arg='no_rewrites', value=ast.Constant(value=True)),
            ast.keyword(arg='nogil', value=ast.Constant(value=True)),
        ]
    )
    nb_compyled_func_tree.body[0].decorator_list.append(numba_decorator)
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
        print(ast.unparse(built_func_tree))
        compyled = None
        def wrapped(x):
            nonlocal compyled
            if compyled is None:
                print("COMPILADO",  ast.unparse(built_func_tree))
                exec(compile(built_func_tree, filename="<ast>", mode="exec"), exec_globals)
                compyled = exec_globals[build_qualifier]
            return compyled(x)
        return {build_qualifier: wrapped}
    except Exception as e:
        logger.warning(e)
        return {}
    

def _prepare_funcs(original_func: Callable, mapping: Dict[str, int], func_int_identifier: str, ndims: int, axis: int, dtype:str = 'float32') -> Dict[str, Callable]:
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

    prepared_trees = create_transformed_function_ast(original_func, mapping, axis = axis, ndims = ndims, dtype = dtype)
    
    for tree in prepared_trees:
        build_qualifier = tree.body[0].name.split('_built__')[-1]
        if build_qualifier == "uloop_1":
            build_qualifier="loop"
        elif build_qualifier == "uloop_opt_1":
            build_qualifier="uloop_opt"
        available_funcs.update(compile_tree(tree, exec_globals, original_func.__qualname__, build_qualifier))
        print(original_func.__qualname__, build_qualifier)
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

