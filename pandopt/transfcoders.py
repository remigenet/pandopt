import ast
import copy
import inspect
import logging
import functools
import sys
from typing import Callable, Type, Dict, Tuple, Any, Union, Iterable, List
import numpy as np
from functools import reduce
import numba as nb
import hashlib
import tempfile
fp = tempfile.NamedTemporaryFile()

def source_parser(f):
    source = inspect.getsource(f)
    if any(source.startswith(managed_method) for managed_method in ['apply', 'aggregate', 'source_parser']):
        code = "def lamndaX(x):\n"
        code += "    return " + source[source.find('x:') + 2:source.find('mapping')].strip().strip(',').strip(':').strip(')'), 'lamndaX'
    elif f.__module__ != '__main__':
        code = "def "+ f.__module__ + '_' + f.__qualname__ + "(x):\n"
        code += "    return " + f.__module__ + '.' + f.__qualname__ +'(x)\n'
        code += "import " + f.__module__ + '\n'
        return  code, f.__module__ + '_' + f.__qualname__
    else:
        return source, f.__name__
    return 
    

def create_callmap_function_ast(index: Iterable[str], fuid: str) -> ast.FunctionDef:


    body = []
    for value, key in enumerate(index):
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
    body.append(ast.Return(value=ast.Constant(value=value+1)))

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
            ast.keyword(arg='cache', value=ast.Constant(value=True)),
            ast.keyword(arg='fastmath', value=ast.Constant(value=True)),
            ast.keyword(arg='looplift', value=ast.Constant(value=True)),
            ast.keyword(arg='inline', value=ast.Constant(value='always')),
            ast.keyword(arg='target_backend', value=ast.Constant(value='host')),
            ast.keyword(arg='no_rewrites', value=ast.Constant(value=True)),
            ast.keyword(arg='nogil', value=ast.Constant(value=True)),
        ]
    )
    
    # Create the function definition
    func_def = ast.FunctionDef(
        name=(callmap_name:=f'callmap{fuid}'),
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
    
    def __init__(self, arg_name: str, new_var_name: str, index: List[str], for_vectorize_form: bool, axis: int, ndims : int):
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
        self.callmap_name = None
        self.callmap_def = None
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
                self.index = self.index.astype(str)
                self.fuid = hashlib.sha1("".join(self.index).encode("UTF-8")).digest().hex()
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
                    self.callmap_def, self.callmap_name = create_callmap_function_ast(self.index, self.fuid)
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

def AstModifier(original_func, index, axis, ndims):
    modified_tree, vectorize_tree = None, None
    source_code, original_name = source_parser(original_func)
    original_tree = ast.parse(source_code)

    arg_name = original_tree.body[0].args.args[0].arg
    new_name = arg_name + "X"
    while new_name in source_code:
        new_name + "X"
        
    replacer = SubscriptReplacer(arg_name, new_name, index, for_vectorize_form=False, axis = axis, ndims = ndims)
    modified_tree = copy.deepcopy(original_tree)
    modified_tree = replacer.visit(modified_tree)
    ast.fix_missing_locations(modified_tree)
    fuid = original_name + replacer.fuid 
    modified_tree.body[0].args.args[0].arg = new_name
    modified_tree.body[0].name += fuid +"_modified" 
    ast.fix_missing_locations(modified_tree)

    if replacer.vectorizable:
        replacer = SubscriptReplacer(arg_name, new_name, index, for_vectorize_form=True, axis = axis, ndims = ndims)
        vectorize_tree = copy.deepcopy(original_tree)
        vectorize_tree = replacer.visit(vectorize_tree)
        ast.fix_missing_locations(vectorize_tree)
        vectorize_tree.body[0].args.args[0].arg = new_name
        vectorize_tree.body[0].name += fuid +"_vectorized"
        ast.fix_missing_locations(vectorize_tree)
    
    original_tree.body[0].name += fuid +"_original" 
    return fuid, original_name, ast.unparse(original_tree), ast.unparse(modified_tree), ast.unparse(vectorize_tree) if vectorize_tree is not None else None, ast.unparse(ast.fix_missing_locations(ast.Module(body=[replacer.callmap_def], type_ignores=[]))) if replacer.callmap_def is not None else ""

two_dim_axis_0 = """
PLACEHOLDERCALLMAP

@nb.njit(nb.PLACEHOLDERDTYPE(nb.PLACEHOLDERDTYPE[:]), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', nogil=True)
PLACEHOLDERFUNC

@nb.njit((nb.PLACEHOLDERDTYPE[:], nb.PLACEHOLDERDTYPE[:],nb.types.uint32), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', nogil=True)
def vl_PLACEHOLDERUID(z, r, i):
    r[i] = PLACEHOLDERNAME(z)

@nb.njit((nb.PLACEHOLDERDTYPE[:,:], nb.PLACEHOLDERDTYPE[:],nb.types.uint32), cache=True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def cvlopt_PLACEHOLDERUID(z, r, n):
    for i in nb.prange(n):
        vl_PLACEHOLDERUID(z[:,i], r, i)

def vlopt_PLACEHOLDERUID(z):
    n = np.uint32(z.shape[1])  # Determine the number of rows in z
    r = np.empty(n, dtype=np.PLACEHOLDERDTYPE)  # Initialize the result array
    cvlopt_PLACEHOLDERUID(z, r, n)
    return r
"""

two_dim_axis_1 = """
PLACEHOLDERCALLMAP

@nb.njit(nb.PLACEHOLDERDTYPE(nb.PLACEHOLDERDTYPE[:]), cache=True, fastmath=False, forceinline=True, looplift=True, inline='always', target_backend='host', nogil=True)
PLACEHOLDERFUNC

@nb.njit((nb.PLACEHOLDERDTYPE[:], nb.PLACEHOLDERDTYPE[:],nb.types.uint32), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', nogil=True)
def vl_PLACEHOLDERUID(z, r, i):
    r[i] = PLACEHOLDERNAME(z)

@nb.njit((nb.PLACEHOLDERDTYPE[:,:], nb.PLACEHOLDERDTYPE[:],nb.types.uint32), cache=True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def cvlopt_PLACEHOLDERUID(z, r, n):
    for i in nb.prange(n):
        vl_PLACEHOLDERUID(z[i,:], r, i)

def vlopt_PLACEHOLDERUID(z):
    n = np.uint32(z.shape[0])  # Determine the number of rows in z
    r = np.empty(n, dtype=np.PLACEHOLDERDTYPE)  # Initialize the result array
    cvlopt_PLACEHOLDERUID(z, r, n)
    return r
"""

def _make_func(base_model: str, name: str, func: str, dtype: str, uid: str, callmap: str):
    final = base_model.replace('PLACEHOLDERNAME', name)
    final = final.replace('PLACEHOLDERFUNC', func)
    final = final.replace('PLACEHOLDERDTYPE', dtype)
    final = final.replace('PLACEHOLDERUID', uid)
    final = final.replace('PLACEHOLDERCALLMAP', callmap)
    return final
    

def _compile_tree(func: Dict[str, str], globals: Dict[str, Any]) -> Dict:
    def wrapped(x):
        if func['name'] not in globals:
            exec(compile(ast.parse(func['source_code']), filename=fp.name, mode="exec"), globals)
        return globals[func['name']](x)
    wrapped.__name__ = func['name']
    wrapped.__qualname__ = "__numba__." + func['name']
    return wrapped

def _prepare_func(original_func: Callable, index: Iterable[str], axis: int, ndims: int, dtype: str, globals: Dict[str, Any]) -> Tuple[ast.AST, ast.AST, ast.AST]:
    result = {}
    fuid, original_name, original, modified, vectorized, callmap = AstModifier(original_func = original_func, index = index, axis = axis, ndims = ndims)
    name_with_id = original_func.__name__ + fuid
    result['original'] = {"name":  name_with_id + "_original", "source_code": original}
    result["original"]["function"] = _compile_tree(result["original"], globals)
    if ndims == 1:
        decorator =  "@nb.njit(nb.{dtype}(nb.{dtype}[:]), cache = False, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)\n"
        result['modified'] = decorator + modified
        return result
    elif ndims == 2:
        final_name =  "vlopt_" + fuid 
        modified_name = name_with_id + "_modified"
        if axis > 1:
            raise ValueError    
        result['modified'] = {"name": final_name, "source_code": _make_func(two_dim_axis_0 if axis == 0 else two_dim_axis_1, modified_name, modified, dtype, fuid, callmap)}
        result["modified"]["function"] = _compile_tree(result["modified"], globals)
        if vectorized is not None:
            vectorized_name = name_with_id + '_vectorized'
            result['vectorized'] = {"name": vectorized_name, "source_code": vectorized}
            result["vectorized"]["function"] = _compile_tree(result["vectorized"], globals)
        return result
    elif ndims == 3:
        raise NotImplemented
