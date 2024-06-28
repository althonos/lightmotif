__version__ = "0.8.0"

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
__license__ = "MIT OR GPL-3.0-or-later"
__doc__ = lib.__doc__
