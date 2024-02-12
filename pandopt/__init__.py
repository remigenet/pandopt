from pandas import *
from .frames import DataFrame
from .rolls import RollOpt
from .groups import GroupOpt
import pandas

pandas.core.frame.DataFrame = DataFrame
pandas.core.window.rolling.Rolling = RollOpt
# pandas.core.groupby.generic.DataFrameGroupBy = GroupOpt
