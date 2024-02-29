# Pandopt
pandas light wrapper class for improved performance

## How to get it ?

Pandopt is available with pip command, you can install it with:
```bash
pip install pandopt
```

and then use it as simply as:
```python
import pandopt as pd
#here everything work as usual, pd.Timestamp will work as usual, pd.DataFrame too but this one will returns a pandopt defined DataFrame !
```
or
```python
import pandas as pd
import pandopt as pdo
...
df = pd.DataFrame(...) # doing your usual pandas stuff  
#then needing to use the apply method for example
pdo.DataFrame(df).apply(...).to_pandas() # to get back to normal pandas
```
## What it is ?
Pandopt is a lightweight library that only aims to improve pandas efficiency in certain methods, while keeping the user API totally the same.

Pandas success being it's resilience and ease of use, pandopt don't have the ambition to change this but only to offer a derived class on top of it, as certain methods, while being very practical to use, offer raw performance that can be improved by many multiples if adding some hidden specialisation in behing.

Pandopt targets a fully compatible API as it should fallback at worst on the pandas methods if it's specialisation do not work.

Being directly inherited from pandas, while having some mecanism to ensure not leaving the class to get back to pandas, it can be use as simply as import pandopt as pd to replace your usual import pandas as pd.

## Why ?

Pandas is a great library, and is widely use as it covers most of the use case for data manipulation. It also present a very user-friendly API that is pleasant to use, and that is very well documented. It's also very well maintained, and is a very active project.

But, pandas as a drawback, it's slow if you goes outside of the boundary of the standard use case a little bit, and want to use custom callable in the apply method, or if you want to use the rolling method on big data.

And in fact, you don't need to know well pandas if you just know what I would call the elementary pillars in it:
 - The apply method
 - The rolling method
 - The groupby method
and maybe also merges, concat and joins, but those are not the most used methods, and are not the most problematic in terms of performance.

I, for years, only do all with these 3 methods, and I'm sure that many people are in the same case, as why would you need to learn the other methods if you can do all with those 3 methods ? 

But those 3 methods are known to be slow, comparing DataFrame.apply(np.sum) versus DataFrame.sum() is a good example of the difference of performance that can be achieved, and the rolling method is known to be very slow too.

There are multiple libraries that are trying to solve this problem, like modin, dask, polars, etc. But they all have their own drawbacks, and are not as user-friendly as pandas. Modin is known to be very buggy, and to not be able to handle big data. Dask is known to be very complex to use, and to not be able to handle big data. Polars is known to be very fast, but totally diverge from the pandas API. 
And other issues can come along, as for example needs for serialisation, or to be able to use the library in a distributed way, that are not used by most of the users, and that are not needed for most of the use case.

Pandopt as a simple idea, to inherit directly from the pandas class, and to add some specialisation in the methods that are known to be slow, and to not be able to handle big data. It's not meant to be a replacement for pandas, but to be a drop-in replacement for pandas, that can be used as simply as importing pandas, and that can be used as simply as pandas.

It only aims to redefines the methods of thoose 3 pillars to improve the performance of pandas drastically, using the numpy and numba library, that are very well maintained, and that are very common, and that are not adding any dependencies to the library.

Pandopt just means pandas optimized - still it don't search to cover use cases like Big Data or distributed computing, but to be able to handle big data on a single machine with decent performance - still if you are searching something too manage memory it is not part of the project!

## Performance ?

Eard about the famous "apply" method that is so slow ? Pandopt can make it up to 14000 times faster, depending on the function you are applying. Yeah, you read it right. But that's not all, the rolling method can also be up to 1000 times faster, in fact all the basics unleash real potential !
You can find the benchmarks in the demos_n_examples folder, where pandopt library is able to bit even polars on some of the most basics tasks like summing, while maintaining a fully compatible API with pandas.

The other advantage is that if using apply with multiple functions the gains can be even higher, with seen benchmarks going to 25,000X quicker ! 

## What works now ?

Apply methods start to be well tested and the rolling window method is integration is done too. After multiple tests, either using numpy memory layout, pyarrow indexing, and numba, improving the groupby outside of the box seems much more complicated than expected and this is not planned for the moment.
However you can find tests and benchmarks in notebooks:
    - [Apply](demos_n_examples/Benchmarks/apply_benchmarks.ipynb)
    - [Rolling](demos_n_examples/Benchmarks/rolling_benchmarks.ipynb)
  

## Some helps to understand how it works ?

While creating I made several notebook to explain the ideas:
    - [Understand how even a simple sum can have drastic performance changes](demos_n_examples/UnderstandHowItWork/ManyWayToSum.ipynb)
    - [Apply intermediary version](demos_n_examples/UnderstandHowItWork/TheCodeTransformation.ipynb): showcase how to also implements with vectorization if possible, removed for simplicity when having multiple column due to time lack but more efficient than current implementation
    - [Apply Current explanation](demos_n_examples/UnderstandHowItWork/TheCodeTransformation_NewVersion.ipynb)
    - [How rolling is performed](demos_n_examples/UnderstandHowItWork/Rolling.ipynb)
  

## Versioning and dependencies ?

Pandopt idea is to be as general as posisible, and to be as compatible as possible with the pandas API. It's also to be as light as possible, and to not add any dependencies that are not strictly necessary. In that meaning it only requires the numpy and numba library for optimisation, that are not from standard library, but are very common and very well maintained. 
pandopt class directly inherit from pandas class, and is not meant to be used as a standalone library, but as a drop-in replacement for pandas, so it's versioning is directly linked to the pandas versioning, and it's compatibility is directly linked to the pandas compatibility.

## Bugs and current stability ?

It's a personal project for now on the side, it's for sure unsafe and there is still many bugs lying around !

## Ideas tested

Multiple ideas have been tested and compared to provide this package, here is a summary of them, that can be tricks you use for other purpose:
    - Memory layout:
        - Numpy strenght when it come to calculation comes from contiguity. What this means is that the datas inside an array are all "packed" together. 
        - However memory isn't 2 or 3 dimensional, it's 1 dimensional, and thus the way it is saved matters.
        - In numpy you can check any array contiguity with .flags which will tell you wether it is a pointer to raw datas, C-contiguous datas or F-contiguous datas
        - C-contiguous means for a 2D matrix that datas are saved in a row fashion, while F-contiguous (for Fortran) means in a columnar way. 
            - This can be change using array.copy(order='C') or .copy(order='F')
            - Over standards numpy arrays this matters a lot even for basic operations, you can find some examples [here](dev/apply_contiguity_tests.ipynb)
            - However, if you use your array inside numba, even if you can tell to numba the specific contiguity, the choice of it doens't have any real impact on performance
        - For rolling a known tricks used is the numpy strides tricks, which consists of using strides to obtain a view over the numpy arrays. Strides are a low-level way to create pointer over the raw array, but needs meticulous manipulation to work with. However numpy comes with sliding_window_view in  numpy.lib.stride_tricks that helps manipulate the arrays safely in this contexts. 

## How it works for apply ?

To achieve this the dataframe will create 4 options, two using direct vectorization to retrieve the datas in the array directly, the second using a numba loop, for each version having one compiled and one not compuled. To do so it's fully dynamic and the apply is creating this derivated function from the initial one using AST parsing and modifcation. 

For example _prepare_funcs will transform your functions here:

```python
def simple_start(z):
    x = (z['A'] + z['B']) / z['C']
    x += z['B'] * z['D']
    return x / z['B']

def harder_func(z):
    x = (z['A'] + z['B']) / z['C']
    if x > 0:
        return x / z['B']
    x += z['B'] * z['D']
    return x * z['B']

def harder2_func(z):
    x = (z['A'] + z['B']) / z['C']
    k=z['A']-z['C']
    j=z['B']/z['D']
    if k > j:
        return x / k
    x *= j
    return x - k if k > z['C'] else x + k


def harder3_func(z):
    g=lambda a, b: a if abs(a) > abs(b) else - 2 * (b**(-a))
    x = (z['A'] + z['B']) / z['C']
    k=z['A' if z['B'] > 0 else 'D']-z['C']
    j=g(z['B'],z['D'])
    if k > j:
        return j / k
    x *= j
    return x - k if k > z['C'] else x + k

def harder4_func(z):
    g=lambda a, b: a if abs(a) > abs(b) else - 2 * (b**(-int(a)))
    x = (z['A'] + z['B']) / z['C']
    k=z['A' if z['B'] > 0 else 'D']-z['C']
    j=g(z['B'],z['D'])
    if k > j:
        return j / k
    x *= j
    return x - k if k > z['C'] else x + k


df = pd.DataFrame(np.random.randn(int(1e4), 4), columns=['A', 'B', 'C', 'D']).astype(np.float32)

func_list = [simple_start, harder_func, harder2_func, harder3_func, harder4_func]

prepared = _prepare_func(func_list, df.columns.values.astype(str) if axis==1 else df.index.values.astype(str), axis = axis, ndims = 2, dtype = "float32")

print(prepared['source_code'])

```
So that the source code that will be compiled became:
```python
@nb.njit(nb.uint8(nb.types.string), cache=True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nopython=True, nogil=True)
def callmap_fb2f85c88567f3c8ce9b799c7c54642d0c7b41f6(x):
    if x == 'A':
        return 0
    if x == 'B':
        return 1
    if x == 'C':
        return 2
    if x == 'D':
        return 3
    return 4

@nb.njit(nb.float32(nb.float32[:]), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def simple_startfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(zX):
    x = (zX[0] + zX[1]) / zX[2]
    x += zX[1] * zX[3]
    return x / zX[1]

@nb.njit(nb.float32(nb.float32[:]), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def harder_funcfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(zX):
    x = (zX[0] + zX[1]) / zX[2]
    if x > 0:
        return x / zX[1]
    x += zX[1] * zX[3]
    return x * zX[1]

@nb.njit(nb.float32(nb.float32[:]), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def harder2_funcfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(zX):
    x = (zX[0] + zX[1]) / zX[2]
    k = zX[0] - zX[2]
    j = zX[1] / zX[3]
    if k > j:
        return x / k
    x *= j
    return x - k if k > zX[2] else x + k

@nb.njit(nb.float32(nb.float32[:]), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def harder3_funcfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(zX):
    g = lambda a, b: a if abs(a) > abs(b) else -2 * b ** (-a)
    x = (zX[0] + zX[1]) / zX[2]
    k = zX[callmap_fb2f85c88567f3c8ce9b799c7c54642d0c7b41f6('A' if zX[1] > 0 else 'D')] - zX[2]
    j = g(zX[1], zX[3])
    if k > j:
        return j / k
    x *= j
    return x - k if k > zX[2] else x + k

@nb.njit(nb.float32(nb.float32[:]), cache=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def harder4_funcfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(zX):
    g = lambda a, b: a if abs(a) > abs(b) else -2 * b ** (-int(a))
    x = (zX[0] + zX[1]) / zX[2]
    k = zX[callmap_fb2f85c88567f3c8ce9b799c7c54642d0c7b41f6('A' if zX[1] > 0 else 'D')] - zX[2]
    j = g(zX[1], zX[3])
    if k > j:
        return j / k
    x *= j
    return x - k if k > zX[2] else x + k


@nb.njit(nb.float32(nb.uint8, nb.float32[:]), cache=True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def func_callmap_simple_startfb2f85c88567f3c8ce9b799c7c54642d0c7b41f(idx, x):
    if idx == 0:
        return simple_startfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(x)
    if idx == 1:
        return harder_funcfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(x)
    if idx == 2:
        return harder2_funcfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(x)
    if idx == 3:
        return harder3_funcfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(x)
    if idx == 4:
        return harder4_funcfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(x)
    return harder4_funcfb2f85c88567f3c8ce9b799c7c54642d0c7b41f6_modified(x)

@nb.njit((nb.float32[:,:], nb.float32[:, :],nb.types.uint32), cache=True, parallel=True, fastmath=True, forceinline=True, looplift=True, inline='always', target_backend='host', no_cfunc_wrapper=True, no_rewrites=True, nogil=True)
def cvlopt_func_callmap_simple_startfb2f85c88567f3c8ce9b799c7c54642d0c7b41f(z, r, n):
    for i in nb.prange(n):
        for j in nb.prange(5):
            r[i, j] = func_callmap_simple_startfb2f85c88567f3c8ce9b799c7c54642d0c7b41f(j, z[i, :])

def vlopt_func_callmap_simple_startfb2f85c88567f3c8ce9b799c7c54642d0c7b41f(z):
    n = np.uint32(z.shape[0])  # Determine the number of rows in z
    r = np.empty((n, 5), dtype=np.float32)  # Initialize the result array
    cvlopt_func_callmap_simple_startfb2f85c88567f3c8ce9b799c7c54642d0c7b41f(z, r, n)
    return r


```

The compiled and non compiled version are created and the best one is used, if fails goes to next until finding a working one, and at worst get back to default behaviour.

## How it works for Rolling ?

For rolling Pandopt leverage the numpy API too with, using sliding_window_view from numpy.lib.stride_tricks to create a view of the array that is then used to apply the function on it. This is a very efficient way to do it, and the only one that is used in Pandopt. 
It use the same method as apply to create the function, and then use the view to apply it on the array.
It also add the axis parameter to the rolling method, that is not present in pandas, to allow to apply the rolling on the axis 1, and not only on the axis 0.


## Things to come:

For now nothing more is to except from pandopt in terms of implementations, but any improvments of the library, proposal or issue report is welcome.
You can look at the small tests notebooks that were just drafts of tests before implementation in dev.
- aggregate will probably be implemented (a part of the idea already done [here](dev/grouptest.ipynb))
- default numba engine use for groupby might come as in beginning of the same file too.