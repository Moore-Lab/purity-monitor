#!/usr/bin/env python

"""Provides Gain Analysis Tools for both MCA and raw data.

Implements best-fit recognition and automated error detection for optimal usability.

Multi-core processing should be implemented when instantiating this class. Might be here in the future.
"""

# IMPORTS


__author__ = "Tiziano Buzzigoli"
__credits__ = ["Tiziano Buzzigoli"]
__version__ = "1.0.1"
#__maintainer__ = "Rob Knight"
#__email__ = "rob@spot.colorado.edu"
__status__ = "Development"


class PulseType: #8

    UNSET = 0
    LATE = 3
    EARLY = 4
    ALIGNED = 8
    FLAT = 5
    class Shape:
        STANDARD = 1
        SATURATED = 6
        NOISY = 7
        MULTIPLE_PEAKS = 2



class Status:

    NONE = 0
    INIT = 1
    ERROR = 2
    CLEARED = 3
    FAILED = 4
    FITTED = 5
    EXC_GAP = 6
