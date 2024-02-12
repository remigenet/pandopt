from pandas import *
from . import transfcoders
from . import math

from .frames import DataFrame
from .rolls import RollOpt
from .groups import GroupOpt

import pandas

pandas.core.frame.DataFrame = DataFrame
pandas.core.window.rolling.Rolling = RollOpt


