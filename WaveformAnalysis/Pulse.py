#!/usr/bin/env python

"""Provides Gain Analysis Tools for both MCA and raw data.

Implements best-fit recognition and automated error detection for optimal usability.

Multi-core processing should be implemented when instantiating this class. Might be here in the future.
"""

# IMPORTS
from Constants import PulseType
from Constants import Status
from scipy.signal import find_peaks

__author__ = "Tiziano Buzzigoli"
__credits__ = ["Tiziano Buzzigoli"]
__version__ = "1.0.1"
#__maintainer__ = "Rob Knight"
#__email__ = "rob@spot.colorado.edu"
__status__ = "Development"


class Pulse:

    status = Status.NONE

    time = None
    signal = None
    peaks = None
    shape = PulseType.UNSET # 1 = signle peak normal, 2 = multiple peaks, 3 = late/early, 4 = 

    baseline = None
    amplitude = None


    def __init__(self, waveform, time) -> None:
        if len(time) == len(waveform):
            self.signal = waveform
            self.time = time
            self.status = Status.INIT
        else:
            self.status = Status.ERROR
            return False

    def carachterize(self):
        if Status.ERROR: raise Exception('Some error occurred')

    

    def __has_peaks(self):
        # Will look for signle and multiple peaks with two different prominence values. Will assign a shape

        pks = find_peaks(self.signal,prominence=150,distance=30)[0]
        
        if len(pks) == 0:                   # The signal shows no significant peaks
            self.shape = PulseType.FLAT
            return False
        
        if len(pks) > 1:                    # The signal shows multiple prominent peaks --> correlated avalanches
            self.shape = PulseType.MULTIPLE_PEAKS
            return True

        if len(pks) == 1:                   # There is one big prominent peak, but there could be more smaller peaks.

            pks = find_peaks(self.signal, prominence=75,distance=20)[0]

            if len(pks) == 1:               # Same large peak --> regular pulse
                self.shape = PulseType.STANDARD
                return True

            if len(pks) > 1:                # There are smaller peaks (luckly we checked) --> correlated avalanches
                self.shape = PulseType.MULTIPLE_PEAKS
                return True



        if len(pks) > 1:
            #plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform)
            #for pk in pks:
                #plt.vlines([self.SiPM.Ch[self.CHANNEL].Time[pk]],0,waveform[pk],colors=['lime'],linestyles='dashed')
            #plt.show()
            #plt.savefig(f'{folder}/double1_{ii}.png',dpi=200,facecolor='white')
            return (False, 1)
        elif len(pks) == 1:
            if self.SiPM.Ch[self.CHANNEL].Time[pks] > 240 or self.SiPM.Ch[self.CHANNEL].Time[pks] < 180: #TODO CHECK THIS LIMIT
                #plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform,color='purple')
                #plt.show()
                return (False, 2)
                #plt.savefig(f'{folder}/timing_{ii}.png',dpi=200,facecolor='white')
            else:
                pks2 = find_peaks(waveform,prominence=80,distance=20)[0]
                if len(pks2) > 1:
                    return (False, 3)
                    #plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform,color='deepskyblue')
                    #for pk in pks2:
                        #plt.vlines([self.SiPM.Ch[self.CHANNEL].Time[pk]],0,waveform[pk],colors=['lime'],linestyles='dashed')
                    #plt.show()
                    #plt.title(str([round(self.SiPM.Ch[self.CHANNEL].Time[p],2) for p in pks2]))
                    #plt.savefig(f'{folder}/double2_{ii}.png',dpi=200,facecolor='white')
                else: return (True,True)
        return (True,True)    

    def __is_saturated(self):
        pass

    def __is_misaligned(self)
        pass

    def __has_correlated(self):
        pass