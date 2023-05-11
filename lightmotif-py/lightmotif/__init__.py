__version__ = "0.1.1"

from . import lib
from .lib import (
    EncodedSequence,
    StripedSequence,
    CountMatrix,
    WeightMatrix,
    ScoringMatrix,
    create,
    stripe,
)

__author__ = lib.__author__
__doc__ = lib.__doc__
