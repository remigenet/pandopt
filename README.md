# pandopt
pandas light wrapper class for improved performance

## What it is ?
Pandopt is a lightweight library that only aims to improve pandas efficiency in certain methods, while keeping the user API totally the same.

Pandas success being it's resilience and ease of use, pandopt don't have the ambition to change this but only to offer a derived class on top of it, as certain methods, while being very practical to use, offer raw performance that can be improved by many multiples if adding some hidden specialisation in behing.

Pandopt targets a fully compatible API as it should fallback at worst on the pandas methods if it's specialisation do not work.

Being directly inherited from pandas, while having some mecanism to ensure not leaving the class to get back to pandas, it can be use as simply as import pandopt as pd to replace your usual import pandas as pd.

## What works now ?

The DataFrame.apply(callable, axis) is the first and working and most tested method, even if the first initialization of a func can be long, it's quickly forgetten with an average mulitple of the performance by 3x in most cases to 2000x in best scenarios

## How it works for apply ?

To achieve this the dataframe will create 4 options, two using direct vectorization to retrieve the datas in the array directly, the second using a numba loop, for each version having one compiled and one not compuled. To do so it's fully dynamic and the apply is creating this derivated function from the initial one using AST parsing and modifcation. 

For example _prepare_funcs will transform your function test_func that is like:

def test_func(z):
    x = (z['A']+z['B'])
    x = z['B']*z['D']
    return x / z['B']

Into this one for the loop method:

@nb.jit(nopython=True, nogil=True, parallel=True)
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