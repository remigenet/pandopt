# Pandopt
pandas light wrapper class for improved performance

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

## What works now ?

Apply methods start to be well tested and the rolling window method is seems to integrate well too. The groupby method is still in development and may change but are already very performant.
Tests and benchmarks are available at:
    - [Apply function in Notebook](demos_n_examples/apply_benchmarks.ipynb)
    - [Apply function in Notebook](demos_n_examples/rolling_benchmarks.ipynb)
  

## Versioning and dependencies ?

Pandopt idea is to be as general as posisible, and to be as compatible as possible with the pandas API. It's also to be as light as possible, and to not add any dependencies that are not strictly necessary. In that meaning it only requires the numpy and numba library for optimisation, that are not from standard library, but are very common and very well maintained. 
pandopt class directly inherit from pandas class, and is not meant to be used as a standalone library, but as a drop-in replacement for pandas, so it's versioning is directly linked to the pandas versioning, and it's compatibility is directly linked to the pandas compatibility.
In addition to inherit, it use decorators to avoid "leaving" the class, and to ensure that the methods are not overriden by the pandas methods, and that the class is not overriden by the pandas class.

## How it works for apply ?

To achieve this the dataframe will create 4 options, two using direct vectorization to retrieve the datas in the array directly, the second using a numba loop, for each version having one compiled and one not compuled. To do so it's fully dynamic and the apply is creating this derivated function from the initial one using AST parsing and modifcation. 

For example _prepare_funcs will transform your function test_func that is like:

```python
def test_func(z):
    x = (z['A']+z['B'])
    x = z['B']*z['D']
    return x / z['B']
```
Into this one for the loop method:
```python
@nb.jit(nopython=True, nogil=True, parallel=True, OTHER flags...)
def cdmtest_func(Z):
    def callmap(x):
        if x == 'A':
            return 0
        elif x == 'B':
            return 1
        elif x == 'C':
            return 2
        elif x == 'D':
            return 3
        return x
    def tmporary(x):
        x = (z['A']+z['B'])
        x = z['B']*z['D']
        return x / z['B']
    n = Z.shape[0]
    res = np.zeros((n, 1))
    for i in nb.prange(5, n):
        res[i,0] = np.var(Z[i,:])
    return res
```

The compiled and non compiled version are created and the best one is used, if fails goes to next until finding a working one, and at worst get back to default behaviour.

## How it works for Rolling ?

For rolling Pandopt leverage the numpy API too with, using sliding_window_view from numpy.lib.stride_tricks to create a view of the array that is then used to apply the function on it. This is a very efficient way to do it, and the only one that is used in Pandopt. 
It use the same method as apply to create the function, and then use the view to apply it on the array.
It also add the axis parameter to the rolling method, that is not present in pandas, to allow to apply the rolling on the axis 1, and not only on the axis 0.

## How it works for Groupby ?

Not implemented yet, coming soon

