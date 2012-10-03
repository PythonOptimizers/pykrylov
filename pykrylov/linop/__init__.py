"""Linear Operator Type"""

from linop import *
from blkop import *

__all__ = filter(lambda s:not s.startswith('_'), dir())
