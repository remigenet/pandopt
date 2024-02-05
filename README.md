# pandopt
pandas light wrapper class for improved performance

## What it is ?
Pandopt is a lightweight library that only aims to improve pandas efficiency in certain methods, while keeping the user API totally the same.

Pandas success being it's resilience and ease of use, pandopt don't have the ambition to change this but only to offer a derived class on top of it, as certain methods, while being very practical to use, offer raw performance that can be improved by many multiples if adding some hidden specialisation in behing.

Pandopt targets a fully compatible API as it should fallback at worst on the pandas methods if it's specialisation do not work.

Being directly inherited from pandas, while having some mecanism to ensure not leaving the class to get back to pandas, it can be use as simply as import pandopt as pd to replace your usual import pandas as pd.
