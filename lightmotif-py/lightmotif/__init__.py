__version__ = "0.10.0"

from . import lib
from .lib import (
    EncodedSequence,
    StripedSequence,
    CountMatrix,
    WeightMatrix,
    ScoringMatrix,
    Motif,
    TransfacMotif,
    UniprobeMotif,
    JasparMotif,
    StripedScores,
    Scanner,
    Hit,
    Loader,
    create,
    stripe,
    scan,
    load
)

__author__ = lib.__author__
__license__ = "MIT OR GPL-3.0-or-later"
__doc__ = lib.__doc__
