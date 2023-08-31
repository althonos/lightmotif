__version__ = "0.5.1"

from . import lib
from .lib import (
    EncodedSequence,
    StripedSequence,
    CountMatrix,
    WeightMatrix,
    ScoringMatrix,
    Motif,
    StripedScores,
    create,
    stripe,
)

__author__ = lib.__author__
__license__ = "MIT"
__doc__ = lib.__doc__
