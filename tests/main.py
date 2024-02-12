import sys
import numpy as np
sys.path.append('/home/remi/PhDWork/pandopt')
import pandopt as pd

def agg_sum(x):
    return np.sum(x)

if __name__ == "__main__":
    df = pd.DataFrame(np.random.randn(10000, 4), columns=['A', 'B', 'C', 'D'])
    print(type(df))
    print('APPLY')
    print(df.apply(agg_sum, axis=1))
    print(type(df.apply(agg_sum, axis=1)))
    print(df.apply(agg_sum, axis=0))
    print(type(df.apply(agg_sum, axis=0)))
    print(df.apply(agg_sum))
    print(type(df.apply(agg_sum)))
    print(df.apply(agg_sum).apply(agg_sum))
    print(type(df.apply(agg_sum).apply(agg_sum)))
    print('\n\n')
    print('SUM')
    print(df.sum(axis=1))
    print(type(df.sum(axis=1)))
    print(df.sum(axis=0))
    print(type(df.sum(axis=0)))
    print(df.sum())
    print(type(df.sum()))
    print(df.sum().sum())
    print(type(df.sum().sum()))

    print(type(df))
    print(df.rolling(5).apply(agg_sum))
    print(df.rolling(5))