#!/usr/bin/env python

"""Provides Gain Analysis Tools for both MCA and raw data.

Implements best-fit recognition and automated error detection for optimal usability.

Multi-core processing should be implemented when instantiating this class. Might be here in the future.
"""

# IMPORTS
from ctypes import alignment
from Constants import PulseType
from Constants import Status
import numpy as np
from scipy.special import erfc
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import chisquare, gstd

__author__ = "Tiziano Buzzigoli"
__credits__ = ["Tiziano Buzzigoli"]
__version__ = "1.0.1"
#__maintainer__ = "TBD"
#__email__ = "TBD"
__status__ = "Development"


class Pulse:

    status = Status.NONE
    empty = None            # indicates that it is only used to store fit data, not the raw pulse data

    time = None
    signal = None
    peaks = None
    model = None
    amplitude = None
    sigma = None                # not acutal sigma --> see physics statistics on np.diag meaning and circle. (Dave's explanation) Will implement better sigma analysis some day (TODO)
    alignment = PulseType.UNSET
    shape = PulseType.UNSET # 1 = signle peak normal, 2 = multiple peaks, 3 = late/early, 4 = 
    
    selected = False

    baseline = None

    std_mu, std_sigma, std_tau, std_correction = None, None, None, None
    mu, tau, sigma = None, None, None   # used for estimation

    def __init__(self, waveform_or_amp, time_or_sigma) -> None:
        #WARNING: waveform must be baseline-adjusted! --> peak-finding might otherwise fail!

        if (isinstance(waveform_or_amp, float) or isinstance(waveform_or_amp, int)) and (isinstance(time_or_sigma, float) or isinstance(time_or_sigma, int)):
            # INITIALIZING AS FIT ONLY CONTAINER --> NOT A FULL WAVEFORM OBJECT
            self.empty = True
            self.status = Status.INIT

        elif len(time_or_sigma) == len(waveform_or_amp):    # otherwise it is being initalized as a full legit pulse
            self.signal = waveform_or_amp
            self.time = time_or_sigma
            self.status = Status.INIT
            self.empty = False
            self.amplitude = max(self.signal)

        else:
            print('error')
            self.status = Status.ERROR

    # --- CHARACTERIZATION ---

    def charachterize(self):
        if self.status == Status.ERROR: raise Exception('There was an error in the Pulse object you are trying to use')
        if self.empty: raise Exception('Attempting to characterize an empty pulse! (use only as a fit container)')

        if self.status is Status.CLEARED: return

        if self.__has_peaks():      # we have something to work one if there are peaks, let's do it under this "if"
            if self.__is_aligned(): # looking good so far
                if self.__is_unsaturated():     # strike! The pulse looks good and will be analyzed
                    self.status = Status.CLEARED       #NOTE: this loop is so that the pulse can be corrected before proceeding. No corrective actions implemented yet, so will just continue executing skipping other waves
    
    def __has_peaks(self):
        # Will look for signle and multiple peaks with two different prominence values. Will assign a shape

        pks = find_peaks(self.signal,prominence=(self.amplitude*0.8),distance=20)[0]
        
        if len(pks) == 0:                   # The signal shows no significant peaks
            self.shape = PulseType.FLAT
            return False
        
        if len(pks) > 1:                    # The signal shows multiple prominent peaks --> correlated avalanches
            self.shape = PulseType.Shape.MULTIPLE_PEAKS
            return True

        if len(pks) == 1:                   # There is one big prominent peak, but there could be more smaller peaks.

            pks = find_peaks(self.signal, prominence=(self.amplitude*0.2),distance=20)[0]

            if len(pks) == 1:               # Same large peak --> regular pulse
                self.shape = PulseType.Shape.STANDARD
                return True

            if len(pks) > 1:                # There are smaller peaks (luckly we checked) --> correlated avalanches
                self.shape = PulseType.Shape.MULTIPLE_PEAKS
                return True
  
    def __is_unsaturated(self):
        
        pks = find_peaks(self.signal,prominence=(self.amplitude*0.8),distance=20, plateau_size=5)[0]

        if len(pks) == 0: return True   # The wave looks normal, no plateau detected
        elif len(pks) > 0:
            self.shape = PulseType.Shape.SATURATED
            return False                # whoops, the signal saturated the device --> flat top

    def __is_aligned(self):
        pks = find_peaks(self.signal,prominence=(self.amplitude*0.8),distance=20)[0]
        
        if self.time[pks[0]] > 240:
            self.alignment = PulseType.LATE
            return False
        elif self.time[pks[0]] < 180:
            self.alignment = PulseType.EARLY
            return False
        else:
            self.alignment = PulseType.ALIGNED
            return True                              # the peak is aligned

    def __is_baselineAdjusted(self):
        raise NotImplementedError('Not yet! ;)')

    # --- ANALYSIS ---

    def __wave(self, x, A):
        tau = self.std_tau
        mu = self.std_mu
        sigma = self.std_sigma
        base = 0
        return self.std_correction * (base + A/2.0 * np.exp(0.5 * (sigma/tau)**2 - (x-mu)/tau) * erfc(1.0/np.sqrt(2.0) * (sigma/tau - (x-mu)/sigma)))

    def study(self, std_mu, std_sigma, std_tau, std_correction):    # requires approximate values for std_mu, std_sigma, std_tau and correction factor
        if self.status == Status.ERROR: raise Exception('There was an error in the Pulse object you are trying to use')
        if self.empty: raise Exception('Attempting to study an empty pulse! (use only as a fit container)')

        self.std_mu = std_mu
        self.std_sigma = std_sigma
        self.std_tau = std_tau
        self.std_correction = std_correction

        if self.shape is PulseType.Shape.STANDARD: self.__analyze_standard()

        if self.status is Status.FITTED: return self.amplitude

    def __analyze_standard(self):

        # Getting approximate values here
        pks, pdict = find_peaks(self.signal,prominence=(self.amplitude*0.8),distance=20)
        exp_main_A = self.signal[pks[0]]
        main_peak_X = pks[0]
        lx = pdict['left_bases'][0]
        rx = pdict['right_bases'][0]

        #we first need to try fitting a model
        try: popt,pcov = curve_fit(self.__wave, self.time, self.signal, p0=[exp_main_A], maxfev=10000000)
        except:
            #it could just be that noisy signal or correlated avalanches are distracting the fitting. Let's cut it and retry.

            time_cut = self.time[lx:rx]
            signal_cut = self.signal[lx:rx]

            try: popt,pcov = curve_fit(self.__wave, time_cut, signal_cut, p0=[exp_main_A], maxfev=100000000)
            except: self.status = Status.FAILED #sadly we cannot fit this pulse at this time :( --> set status accordingly
            
        else:

            real_A = float(popt[0])
            self.model = self.__wave(self.time,*popt)
            try: chi = chisquare(self.model,self.signal)
            except: chi = 'Undetermined'        #TODO: change to None when used to make decisions
            else:
                
                if chi > 10 and np.sqrt(np.abs(np.diag(pcov)))[1] < 10: #TODO RANDOM VALUE --> CHANGE TO SOMETHING THAT MAKES SENSE!!!
                    pass # something to do if it failed


            # now check how distant A is from the expected peak amplitude

            A_gap = np.abs(real_A - exp_main_A)
            if A_gap > exp_main_A*0.10:     # if gap is larger than 10% of estimate --> PROBLEM
                pass    #TODO: handle this case!!!

            self.amplitude = real_A         # if we made it here then chi is lower than the limit and the gap was handled (not yet, see above comment)
            self.sigma = np.sqrt(np.abs(np.diag(pcov)))[1]
            self.status = Status.FITTED

    def skim(self):
        pks, pdict = find_peaks(self.signal,prominence=(self.amplitude*0.8),distance=20)
        self.amplitude = self.signal[pks[0]]
        self.mu = pks[0]
        return self


