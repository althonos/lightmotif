__version__ = "0.1.0"

try:
    from .avx2 import lib
    from .avx2.lib import *
except ImportError:
    from . import lib
    from .lib import *

__build__ = lib.__build__
__doc__ = lib.__doc__
__author__ = lib.__author__
