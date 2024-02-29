import ast
import copy
import inspect
import logging
import functools
import sys
from typing import Callable, Type, Dict, Tuple, Any, Union, Iterable, List, Optional
import numpy as np
from functools import reduce
import numba as nb
import hashlib
from numpy.lib.stride_tricks import sliding_window_view

def source_parser(f):
    source = inspect.getsource(f)
    if any(source.startswith(managed_method) for managed_method in ["apply", "aggregate", "source_parser"]):
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
    

def create_if_callmap_function_ast(index: Iterable[str], fuid: str) -> ast.FunctionDef:


    body = []
    for value, key in enumerate(index):
        compare = ast.Compare(
            left=ast.Name(id='x', ctx=ast.Load()),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value = key)]
        )
        body.append(
            ast.If(
                test=compare,
                body=[ast.Return(value=ast.Constant(value=value))],
                orelse=[]
            )
        )
    
    # Add a default return statement
    body.append(ast.Return(value=ast.Constant(value=value+1)))

    # Create the function definition
    func_def = ast.FunctionDef(
        name=(callmap_name:=f'if_callmap_{hashlib.sha1(fuid.encode("utf-8")).hexdigest()}'),
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
    
    callmap_str = ast.unparse(ast.fix_missing_locations(ast.Module(body=[func_def], type_ignores=[])))

    decorator = "@nb.njit(nb.uint8(nb.types.string), cache=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)\n"
    
    return decorator + callmap_str, callmap_name

def create_func_callmap_function_ast(funcs_list: Iterable[str]) -> ast.FunctionDef:

    
    body = []
    for idx, func in enumerate(funcs_list):
        compare = ast.Compare(
            left=ast.Name(id='idx', ctx=ast.Load()),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=idx)]
        )
        body.append(
            ast.If(
                test=compare,
                body=[ast.Return(value=ast.Call(func=ast.Name(id=func, ctx=ast.Load()),
                                               args=[ast.Name(id='x', ctx=ast.Load())], 
                                               keywords=[]))],
                orelse=[]
            )
        )
    # Add a default return statement
    body.append(ast.Return(value=ast.Call(func=ast.Name(id=func, ctx=ast.Load()),
                                               args=[ast.Name(id='x', ctx=ast.Load())], 
                                               keywords=[])))


    # Create the function definition
    func_def = ast.FunctionDef(
        name=(callmap_name:=f'func_callmap_{hashlib.sha1("".join((func for func in funcs_list)).encode("utf-8")).hexdigest()}'),
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='idx'), ast.arg(arg='x')],
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
    decorator = "@nb.njit(nb.PLACEHOLDERDTYPE(nb.uint8, nb.PLACEHOLDERDTYPE[:]), cache=True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)\n"
    
    callmap_str = ast.unparse(ast.fix_missing_locations(ast.Module(body=[func_def], type_ignores=[])))

    return decorator + callmap_str, callmap_name

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
    
    def __init__(self, arg_name: str, new_var_name: str, index: List[str], for_vectorize_form: bool, axis: int, ndims : int, callmap_name: Optional[str] = None, callmap_def: Optional[str] = None):
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
        self.new_var_name = new_var_name
        self.callmap_name = callmap_name
        self.callmap_def = callmap_def
        self.index = index
        self.for_vectorize_form = for_vectorize_form
        self.axis = axis
        self.ndims = ndims
        self.used = False
        self.vectorizable = True
        self.fuid = '_general_'

    
    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == self.arg_name:
            # Check for Python version compatibility
            if self.fuid == '_general_':
                self.fuid = hashlib.sha1("".join(self.index.astype(str)).encode("UTF-8")).digest().hex()
            if sys.version_info >= (3, 9):
                # Python 3.9 and later
                old_slice = node.slice
            else:
                # Python 3.8 and earlier
                old_slice = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            if isinstance(old_slice, ast.Constant) and len((idxs:=np.where(self.index==old_slice.value)[0]))==1:
                new_value = ast.Constant(value = idxs[0])
            else:
                if self.callmap_name is None:
                    self.callmap_def, self.callmap_name = create_if_callmap_function_ast(self.index, self.fuid)
                new_value = ast.Call(
                        func=ast.Name(id=self.callmap_name, ctx=ast.Load()),
                        args=[old_slice],
                        keywords=[]
                    )
                self.vectorizable = False
            if self.for_vectorize_form:
                node.slice = ast.Tuple(
                   elts=[
                       ast.Slice() if i != self.axis
                       else new_value
                       for i in range(self.ndims)],
                   ctx=ast.Load())
            else:
                node.slice = new_value
            node.value.id = self.new_var_name 
            # Wrap the subscript in a call to callmap
        return self.generic_visit(node) 

    def visit_If(self, node):
        self.vectorizable = False
        return self.generic_visit(node) 

    def visit_IfExp(self, node):
        self.vectorizable = False
        return self.generic_visit(node) 

    def visit_With(self, node):
        self.vectorizable = False
        return self.generic_visit(node) 

    def visit_While(self, node):
        self.vectorizable = False
        return self.generic_visit(node) 
    
    def visit_Name(self, node):
        if node.id == self.arg_name:
            self.vectorizable = False
            node.id = self.new_var_name
        return self.generic_visit(node) 

def AstModifier(original_funcs, index, axis, ndims):

    callmap_name, callmap_def = None, None
    modified_funcs, modified_names = [], []
    for original_func in original_funcs:
        modified_tree, vectorize_tree = None, None
        source_code = source_parser(original_func)
        original_tree = ast.parse(source_code)
    
        arg_name = original_tree.body[0].args.args[0].arg
        new_name = arg_name + "X"
        while new_name in source_code:
            new_name + "X"
            
        replacer = SubscriptReplacer(arg_name, new_name, index, for_vectorize_form=False, axis = axis, ndims = ndims, callmap_name = callmap_name, callmap_def = callmap_def)
        
        modified_tree = copy.deepcopy(original_tree)
        modified_tree = replacer.visit(modified_tree)
        callmap_name, callmap_def = replacer.callmap_name, replacer.callmap_def
        ast.fix_missing_locations(modified_tree)
        fuid = original_func.__name__ + replacer.fuid  + str(axis)
        modified_tree.body[0].args.args[0].arg = new_name
        modified_tree.body[0].name = f'f{hashlib.sha1((fuid).encode("utf-8")).hexdigest()}'
        ast.fix_missing_locations(modified_tree)
        modified_funcs.append(ast.unparse(modified_tree) + "\n\n" )
        modified_names.append(modified_tree.body[0].name)

    final_func_callmap, final_func_callmap_name = create_func_callmap_function_ast(modified_names)

    if ndims == 3 and axis == 1:
        decorator = "@nb.njit(nb.PLACEHOLDERDTYPE(nb.types.Array(nb.types.PLACEHOLDERDTYPE, 2, 'A', readonly=True)), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)\n"
    else:
        decorator = "@nb.njit(nb.PLACEHOLDERDTYPE(nb.PLACEHOLDERDTYPE[:]), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)\n"

    final_text = decorator + decorator.join(modified_funcs) + "\n"
    final_text += final_func_callmap
    if callmap_def is not None:
        final_text = callmap_def + "\n\n" + final_text
    
    return final_func_callmap_name, final_text


two_dim = """
PLACEHOLDERFUNC

@nb.njit((nb.PLACEHOLDERDTYPE[:,:], nb.PLACEHOLDERDTYPE[:, :],nb.types.uint32), cache=True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def cvlopt_PLACEHOLDERNAME(z, r, n):
    for i in nb.prange(n):
        for j in nb.prange(PLACEHOLDERNFUNC):
            r[i, j] = PLACEHOLDERNAME(j, zPLACEHOLDERAXISSELECT)

def vlopt_PLACEHOLDERNAME(z):
    n = np.uint32(z.shape[PLACEHOLDERAXIS])  # Determine the number of rows in z
    r = np.empty((n, PLACEHOLDERNFUNC), dtype=np.PLACEHOLDERDTYPE)  # Initialize the result array
    cvlopt_PLACEHOLDERNAME(z, r, n)
    return r
"""

three_dim_axis_0 = """
PLACEHOLDERFUNC

@nb.njit((nb.types.Array(nb.types.PLACEHOLDERDTYPE, 3, 'A', readonly=True), nb.PLACEHOLDERDTYPE[:, :],nb.types.uint32, nb.types.uint32), cache=True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def cvlopt_PLACEHOLDERNAME(z, r, n, m):
    for i in nb.prange(n):
        for j in nb.prange(PLACEHOLDERNFUNC):
            for k in nb.prange(m):
                r[i, j * k + k] = PLACEHOLDERNAME(j, z[i, k, :])

def vlopt_PLACEHOLDERNAME(arr, window):
    n = np.uint32(arr.shape[0]) - window + 1  
    m = np.uint32(arr.shape[1]) 
    r = np.zeros((n, PLACEHOLDERNFUNC * m), dtype=np.PLACEHOLDERDTYPE)  # Initialize the result array
    cvlopt_PLACEHOLDERNAME(sliding_window_view(arr, window, axis=0), r, n, m)
    return r
"""

three_dim_axis_1 = """
PLACEHOLDERFUNC

@nb.njit((nb.types.Array(nb.types.PLACEHOLDERDTYPE, 3, 'A', readonly=True), nb.PLACEHOLDERDTYPE[:, :],nb.types.uint32), cache=True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def cvlopt_PLACEHOLDERNAME(z, r, n):
    for i in nb.prange(n):
        for j in nb.prange(PLACEHOLDERNFUNC):
            r[i, j] = PLACEHOLDERNAME(j, z[i,:,:])

def vlopt_PLACEHOLDERNAME(arr, window):
    n = np.uint32(arr.shape[0]) - window + 1  
    r = np.zeros((n , PLACEHOLDERNFUNC), dtype=np.PLACEHOLDERDTYPE)  # Initialize the result array
    cvlopt_PLACEHOLDERNAME(sliding_window_view(arr, window, axis=0), r, n - 1)
    return r
"""

def _make_func(base_model: str, name: str, func: str, dtype: str, axis: str, axis_select: str, n_funcs: int):
    final = base_model.replace('PLACEHOLDERNAME', name)
    final = final.replace('PLACEHOLDERFUNC', func)
    final = final.replace('PLACEHOLDERDTYPE', dtype)
    final = final.replace('PLACEHOLDERAXISSELECT', axis_select)
    final = final.replace('PLACEHOLDERAXIS', axis)
    final = final.replace('PLACEHOLDERNFUNC', str(n_funcs))
    return final
    

def _compile_tree(func: Dict[str, str], exec_globals: Dict[str, Any]) -> Dict:
    def wrapped(*args):
        if func['name'] not in exec_globals:
            exec(compile(ast.parse(func['source_code']), filename="/tmp/numbacache", mode="exec"), exec_globals)
        return exec_globals[func['name']](*args)
    wrapped.__name__ = func['name']
    wrapped.__qualname__ = "__numba__." + func['name']
    return wrapped

def _prepare_func(original_funcs: Callable, index: Iterable[str], axis: int, ndims: int, dtype: str, globals: Dict[str, Any]) -> Dict[str, Any]:   

    if isinstance(original_funcs, Callable):
        original_funcs = [original_funcs]
    final_func_callmap_name, final_form = AstModifier(original_funcs = original_funcs, index = index, axis = axis, ndims = ndims)

    name_with_id = final_func_callmap_name

    if ndims == 2:
        final_name =  "vlopt_" + final_func_callmap_name 
        if axis > 1:
            raise ValueError    
        result = {"name": final_name, "source_code": _make_func(two_dim, final_func_callmap_name, final_form, dtype, "0" if axis else "1", "[i, :]" if axis else "[:, i]", len(original_funcs))}
        result["function"] = _compile_tree(result, globals)
        return result
    elif ndims == 3:
        final_name =  "vlopt_" + final_func_callmap_name 
        globals.update({'sliding_window_view': sliding_window_view})
        if axis > 1:
            raise ValueError    
        result = {"name": final_name, "source_code": _make_func(three_dim_axis_1 if axis else three_dim_axis_0, final_func_callmap_name, final_form, dtype, "", "", len(original_funcs))}
        result["function"] = _compile_tree(result, globals)
        return result


# def _compile_tree(func: Dict[str, str], globals: Dict[str, Any]) -> Dict:
#     def wrapped(x):
#         if func['name'] not in globals:
#             exec(compile(ast.parse(func['source_code']), filename=fp.name, mode="exec"), globals)
#         return globals[func['name']](x)
#     wrapped.__name__ = func['name']
#     wrapped.__qualname__ = "__numba__." + func['name']
#     return wrapped

# def _prepare_func(original_func: Callable, index: Iterable[str], axis: int, ndims: int, dtype: str, globals: Dict[str, Any]) -> Tuple[ast.AST, ast.AST, ast.AST]:
#     result = {}
#     fuid, original_name, original, modified, vectorized, callmap = AstModifier(original_func = original_func, index = index, axis = axis, ndims = ndims)
#     name_with_id = original_func.__name__ + fuid
#     result['original'] = {"name":  name_with_id + "_original", "source_code": original}
#     result["original"]["function"] = _compile_tree(result["original"], globals)
#     if ndims == 1:
#         decorator =  "@nb.njit(nb.{dtype}(nb.{dtype}[:]), cache = False, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)\n"
#         result['modified'] = decorator + modified
#         return result
#     elif ndims == 2:
#         final_name =  "vlopt_" + fuid 
#         modified_name = name_with_id + "_modified"
#         if axis > 1:
#             raise ValueError    
#         result['modified'] = {"name": final_name, "source_code": _make_func(two_dim_axis_0 if axis == 0 else two_dim_axis_1, modified_name, modified, dtype, fuid)}
#         result["modified"]["function"] = _compile_tree(result["modified"], globals)
#         if vectorized is not None:
#             vectorized_name = name_with_id + '_vectorized'
#             result['vectorized'] = {"name": vectorized_name, "source_code": vectorized}
#             result["vectorized"]["function"] = _compile_tree(result["vectorized"], globals)
#         return result
#     elif ndims == 3:
#         raise NotImplemented
