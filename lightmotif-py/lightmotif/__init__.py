__version__ = "0.1.1"

from . import arch
from .lib import __doc__, __author__
from .lib import (
    EncodedSequence,
    StripedSequence,
    CountMatrix,
    WeightMatrix,
    ScoringMatrix,
    create,
)

if arch.AVX2_SUPPORTED:
    from .avx2.lib import (
        EncodedSequence,
        StripedSequence,
        CountMatrix,
        WeightMatrix,
        ScoringMatrix,
        create,
    )
