__version__ = "0.1.0"


import contextlib
import archspec.cpu


_host = archspec.cpu.host()


if "avx2" in _host.features:
    try:
        from .avx2 import lib
        from .avx2.lib import *
    except ImportError:
        from . import lib
        from .lib import *
else:
    from . import lib
    from .lib import *
