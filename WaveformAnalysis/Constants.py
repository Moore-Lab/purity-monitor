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


class PulseType:

    UNSET = 0
    STANDARD = 1
    MULTIPLE_PEAKS = 2
    LATE = 3
    EARLY = 4
    FLAT = 5
    SATURATED = 6
    NOISY = 7

class Status:

    NONE = 0
    INIT = 1
    ERROR = 2
