#!/usr/bin/env python

"""Provides Gain Analysis Tools for both MCA and raw data.

Implements best-fit recognition and automated error detection for optimal usability.

Multi-core processing should be implemented when instantiating this class. Might be here in the future.
"""

# IMPORTS
from argparse import ArgumentError
from hashlib import algorithms_guaranteed
import sys, glob,importlib, latex, itertools, os
sys.path.insert(0,'../../')
sys.path.insert(0,'/Library/TeX/texbin/')
sys.path.insert(0,'../../WaveformAnalysis')
sys.path.insert(0,'/home/tb829/project/purity-monitor/')
import numpy as np
import Dataset as Dataset
from Constants import PulseType
from Constants import Status
import Waveform as Waveform
import SiPM as SiPM
import matplotlib.pyplot as plt
import matplotlib as mpl
import pprint
from natsort import natsorted
from datetime import datetime
import inspect
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.optimize import curve_fit
from scipy.stats import chisquare, gstd
import requests
from scipy.special import erfc
from ipywidgets import IntProgress
from IPython.display import display
import scipy.special
from multiprocessing import Process, Manager, Pool
import multiprocessing
import tracemalloc
import h5py
from Pulse import Pulse

importlib.reload(Dataset)
importlib.reload(SiPM)
importlib.reload(Waveform)

__author__ = "Tiziano Buzzigoli"
__credits__ = ["Tiziano Buzzigoli", "Avinay Bhat"]
__version__ = "1.0.6"
#__maintainer__ = "TBD"
#__email__ = "t.buzzigoli@studenti.unipi.it"
__status__ = "Development"

# LOGGER SETTINGS AND VARIABLES
ANSI_RESET = '\u001b[0m'
ANSI_GREEN = '\u001b[32m'
ANSI_CYAN = '\u001b[36m'
ANSI_RED = '\u001b[31m'
ANSI_YELLOW = '\u001b[33m'
ANSI_BG_RED = '\u001b[41m'
ANSI_BG_GREEN = '\u001b[42m'

olderr = np.seterr(all='ignore')  # necessary to make the output cleaner. Turn off to debug
scipy.special.seterr(all='ignore') # necessary to make the output cleaner. Turn off to debug
import warnings # necessary to make the output cleaner. Turn off to debug
warnings.filterwarnings("ignore") # necessary to make the output cleaner. Turn off to debug

pp = pprint.PrettyPrinter(indent=4)

# MATPLOTLIB SETTINGS
#plt.style.use('../../style.mplstyle')
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams['figure.dpi']= 200

class SmartGAT:  #   GAT: Gain Analysis Tool

    CHANNEL = 0 # TODO SET A CHANNEL FROM WHICH TO TAKE WAVEFORM DATA
    source_files = []   #   stores the 
    max_bins = 10000
    voltage = None
    error = True
    estimate_parameters = True
    SiPM = None         # used to store SiPM object
    shaping_time = 1e-5 # currently not used. No shaping filter is applied. – DEPRECATED
    pulses = []      # array of Pulses (even empty ones representing just fits). Refer to the Pulse class
    waveform_fit_params = []    # TODO will become obsolete
    hist_params = []            # plotting parameters
    hist_data = ([],[])         # plotting data
    hist_fit_params = []        # ""
    peaks_final = None          # ""
    peaks_final_perr = None     # ""
    gain_evaluation = None      # gain evaluation
    gain_fit_params = []        # ""
    tau_est_final = None        # this will be set to the tau estimate when an estimate is requested
    data_type = 0               # 0 for RAW, 1 for MCA
    silenced = False

    mca_gain_eval = (None,None,None)    # gain from MCA functions response object

    MCA = 1
    RAW = 0

    def __init__(self, base_regex,filename_regex,voltage, shaping_time=None, silenced=False,debug=False,notify=False,channel=0,data_type=None):

        tracemalloc.start() # memory tracing begins
        self.debug = debug  # setting the specified log level. See the log method
        self.notifications = notify # setting the specified push notifications level. See the notify method
        self.silenced = silenced    # setting the notification level.
        self.update(60,'INIT')      # setting the monitoring website.
        self.CHANNEL = channel      # data channel in the hp5 files

        try: voltage = float(voltage)   # check if voltage can be a float value
        except:
            error = True
            raise(TypeError(f'Inappropriate argument provided for voltage: int or float expected but {voltage} ({type(voltage)}) was given.'))
        else:
            if not self.__check_available(voltage,base_regex,filename_regex):   # check if the requested voltage is among the files provided to the class
                error = True
                raise(ValueError(f'The voltage you requested ({voltage}) is not available. Try modifying the directory and/or filename you requested.'))
            else:
                self.voltage = float(voltage)   # save and store the voltage for this class.

        if self.source_files is None or len(self.source_files) == 0:    # no files loaded
            error = True
            raise(FileNotFoundError('Unable to locate source files with proper voltage information located at '+(base_regex+filename_regex)))
        elif not shaping_time is None and not type(shaping_time) is type(1e-1): # IF a shaping time is specified then it must be of that type otherwise an error will be raised
            self.error = True
            raise(ValueError('Shaping time must be a number (i.e. 1e-1). Only one shaping time per Gat instance is allowed'))
        else:
            self.error = False  # set the class in a non-error general state. Every method will check for the general health of the class and will not execute if this flag is True to prevent unexpected results and ensure data integrity.
            self.shaping_time = shaping_time    # none is default
            
            if data_type is None: self.data_type = self.__guess_type(base_regex,filename_regex) # guess data type (RAW or MCA) based on file structure. 
            
            if self.data_type == self.MCA: self.__load_files_MCA(base_regex,filename_regex) # load files from MCA
            elif self.data_type == self.RAW: self.__load_files(base_regex,filename_regex) # load files RAW
            
            if self.SiPM is None: self.log(f'FATAL ERROR: Could not load files for {base_regex+filename_regex} !',3)
            elif self.data_type == self.MCA: self.log(f'{len(self.source_files)} files of MCA data loaded for {self.voltage:.02f} OV',1)
            elif self.data_type == self.RAW: self.log(f'{len(self.source_files)} files of RAW data loaded for {self.voltage:.02f} OV',1)

    def __check_available(self, voltage, base_regex, filename_regex): 
        # checks if the voltage requested is available in the files provided. Returns false otherwise
        voltages = []
        if self.data_type == self.RAW:
            self.source_files = glob.glob(base_regex+filename_regex)
            voltages = [x.split('_')[-2] for x in self.source_files]
            voltages = np.array(sorted([float(x.split('OV')[0]) for x in voltages]))
            voltages = np.unique(voltages)
        else:
            self.source_files = glob.glob(base_regex+filename_regex)
            voltages = [x.split('_')[-2] for x in self.source_files]
            voltages = np.array(sorted([float(x.split('OV')[0]) for x in voltages]))
            voltages = np.unique(voltages)

        if float(voltage) in voltages: return True
        return False

    def __throw_error(self):
        raise(RuntimeError('The Gat object you are attempting to reach has experienced an unexpected error and is unable to perform the method you called at this time')) #TODO make error more descriptive so that you can reprint it here

    def __guess_type(self, base, filename):
        reg = base + filename+'_{:.2f}OV*.h5'.format(float(self.voltage))
        try:
            test_file = h5py.File(natsorted(glob.glob(reg))[0], 'r')
            if 'Time' in test_file.keys(): return self.RAW # the Time key is only found in raw waveform files (of course)
            else: return self.MCA
        except: return None

    def force_reload(self,base_regex,filename_regex):
        # the class will retain the loaded files. This method forces the reload of SiPM data and files. Useful when running long notebooks if you do not wish to reload everything to just reset an instance of this class.
        self.SiPM = None
        if self.data_type == self.RAW: self.__load_files(base_regex,filename_regex)
        else: self.log('Force reload only applies to RAW waveform data, but this SmartGAT instance is set to handle MCA data.',2)

    def __load_files(self,base_regex,filename_regex):
        if self.error: self.__throw_error() # status check

        #plt.figure(figsize=(10,6))
        #plt.rcParams.update({'axes.edgecolor': 'w','axes.labelcolor': 'w','xtick.color': 'w','ytick.color': 'w'})

        if not self.SiPM is None:
            self.log('WARNING: files already loaded! Call force_reload(base_regex, filename_regex) to force reload.' ,2)
            return
        
        self.SiPM = SiPM.SiPM(Path=base_regex, Selection=filename_regex+'_{:.2f}OV*.h5'.format(self.voltage))
        self.SiPM.Ch = [Waveform.Waveform(ID=x, Pol=1) for x in range(1,3)]

        self.log(f'Loading {filename_regex}_{self.voltage:.2f}OV*.h5 returned {len(self.SiPM.Files)} files',4)

        for file in natsorted(self.SiPM.Files):
            print(f'Memory usage (traced): {tracemalloc.get_traced_memory()[0]/1000000:.02f} MB',end='\r') # only traced memory will be available here.
            self.SiPM.ImportDataFromHDF5(file, self.SiPM.Ch, var=[])
            self.SiPM.get_sampling()
            if not self.shaping_time is None: self.SiPM.shaping_time=[self.shaping_time]
            self.SiPM.Ch[self.CHANNEL].SubtractBaseline(Data=self.SiPM.Ch[self.CHANNEL].Amp, cutoff=150)
            
            for waveform in self.SiPM.Ch[self.CHANNEL].Amp:
                """plt.figure(facecolor=(0,0,0,0))
                plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform,'white')
                plt.xlabel('Time [microseconds]')
                plt.ylabel('Amplitude [volts]')
                #plt.show()
                plt.savefig('/Users/tizi/Desktop/wave_.png', transparent=True)"""
                self.pulses.append(Pulse(waveform,self.SiPM.Ch[self.CHANNEL].Time)) # creates an array of Pulse instances, each one represetning a raw pulse class

        if len(self.SiPM.Files) == 0: self.SiPM = None
        print()
        print()
        if not self.__resolve_wf_fit_params(): self.log(f'WARNING: will proceed without fixed parameters!',2)

<<<<<<< HEAD
        self.eval_waveform_func_fit() # TODO here for testing purposes only
=======
        print(self.pulses)

        #self.eval_waveform_func_fit(fix_params=True)
>>>>>>> a1e5e451dfca3a1e80d256723b535b89dc690d45

        #self.eval_waveform_func_fit(fix_params=True)

    def __load_files_MCA(self,base_regex,filename_regex):
        if self.error: self.__throw_error() # status check
        
        reg = base_regex + filename_regex+'_{:.2f}OV*.h5'.format(float(self.voltage))
        files = natsorted(glob.glob(reg))
        self.source_files = files # storing list of files. This ONLY happens for MCA data!
        for file in files:
            f = h5py.File(file, 'r')  
            ch2 = f.get('ch2')
            for key in ch2.keys(): 
                df = np.array(ch2.get(key))
            h = df[250:]    # removing part of data which is useless due to it being part that leads to the pedestal.
            hx = np.arange(0,len(h),1)
            self.waveforms[file] = (hx,h)
            self.SiPM = 'Set'   # this prevents it from being set to none which would cause the program to raise an error when checking for the data integrity which is type-independent. Any value other than none would be acceptable.
        
    def eval_waveform_func_fit(self, bounds=None,reload=False,fix_params=False):
        # evaluate a waveform trying to fit a function to it
        if self.error: self.__throw_error() # status check

        if bounds is None: bounds = [0,len(self.SiPM.Ch[self.CHANNEL].Amp[0])] # allows to limit the signal analys segment with [start, end] format
        elif isinstance(bounds[0],int) and isinstance(bounds[1],int): bounds = bounds
        else:
            self.log('The bounds provided are not of the right format and WILL BE IGNORED. Try [int, int].',2)
            bounds = [0,len(self.SiPM.Ch[self.CHANNEL].Amp[0])]


        if not self.__open_coords():    # creates pulses (or fits) array otherwise returns false and the code below executes
            pass

        
        amp_tester = self.__wave_func2(self.SiPM.Ch[self.CHANNEL].Time,1000,0.5) # testing the amplitude of the wave with the estimated parameters (mu and tau) and an amplitude A = 1000
        plt.plot(self.SiPM.Ch[self.CHANNEL].Time, amp_tester) 
        plt.show()
        max_amp = max(amp_tester)   # get maximum of tester waveform.
        print(max(amp_tester))
        factor = 1000/max_amp   # factor to use to correct the fits
        print(factor)
        self.factor = factor    
        self.sigma_est_final = 0.5  # I could not find a way to estimate this.

        for pulse in self.pulses:   # plotting the pulses and fitting them using the func3 (USES THE ADJUSTMENT FACTOR CACLULATED ABOVE)
            waveform = pulse.signal
            plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform)
            popt,pcov = curve_fit(self.__wave_func3,self.SiPM.Ch[self.CHANNEL].Time,waveform,p0=[max(waveform),0.5],bounds=([max(waveform)-5,0.2],[max(waveform)+5,1.5]),maxfev=100000000)
            plt.plot(self.SiPM.Ch[self.CHANNEL].Time,self.__wave_func3(self.SiPM.Ch[self.CHANNEL].Time,*popt))
            plt.xlim(190,220)
            plt.show()
            print('popt ',popt)
            print('factored ', popt*self.factor)
            print('real max ',max(waveform))

     # ! ---- THE ABOVE WAY OF FITTING WAVEFORMS IS NOT WORKING PROPERLY AND THE ESTIMATES ARE NOT GOOD ENOUGH. FOR THIS REASON DEVELOPMENT OF THIS WAS INTERRUPTED. THE THINGS BELOW ARE THE OLD WAY OF FITING THE WAVEFORMS WHICH WORKED BUT IT WAS NOT PRECISE ENOUGH FOR DAVE --- !
        sys.exit()

        if not self.estimate_parameters or reload:

            coords = []     #   x,y coordinate arrays organized in one tuple per waveform
            pks_tally = 0   #   keep track of how many peaks are found
            lost = 0        #   keep track of how many waveforms are discarded

            self.log(f'Getting {len(self.SiPM.Ch[self.CHANNEL].Amp)} waveforms for {self.voltage:.2f} OV [{bounds[0]}:{bounds[1]}]',1)

            f = IntProgress(value=0,min=0,max=len(self.SiPM.Ch[self.CHANNEL].Amp),step=1,description='Fitting waveforms',bar_style='info',layout={"width": "100%"})
            display(f)

            intervals = []

            fit_params = []

            if fix_params:
                self.__resolve_wf_fit_params()
                if self.tau_est_final is None:
                    self.log(f'Failed to load tau estimate. Skipping waveform tau estimate.',2)
                    fix_params=False

            for waveform in self.SiPM.Ch[self.CHANNEL].Amp: #loop over the waveforms inside the file

                if not self.__check_waveform(waveform)[0]:
                    lost += 1
                    continue

                x = [] #    must be same lenght as y. NOTE: array of arrays due to some waveforms having multiple peaks (found with other methods).
                y = [] #    must be same length as x. NOTE: array of arrays due to some waveforms having multiple peaks (found with other methos).
                f.value += 1
                start = datetime.now()
                p0 = self.__p0estimate(waveform)
                if p0 is None:
                    self.log('Error loading p0',4)
                    lost += 1
                    continue
                if fix_params: p0 = [p0[0],p0[1],p0[3],p0[4]]
                
                popt,pcov = None, None
                try:
                    if not fix_params: popt,pcov = curve_fit(self.__wave_func,self.SiPM.Ch[self.CHANNEL].Time,waveform,p0=p0,maxfev=10000000)
                    if fix_params and not self.tau_est_final is None: popt,pcov = curve_fit(self.__wave_func2,self.SiPM.Ch[self.CHANNEL].Time,waveform,p0=p0,maxfev=10000000)
                except Exception as e:
                    self.log(e,3)
                else:
                    if popt is None or pcov is None:
                        lost += 1

                    else:
                        func = None
                        print(f'Will fit with {p0} and tau final {self.tau_est_final}')
                        if not fix_params: func = self.__wave_func(self.SiPM.Ch[self.CHANNEL].Time,*popt)
                        if fix_params: func = self.__wave_func2(self.SiPM.Ch[self.CHANNEL].Time,*popt)
                        peak_x = np.argmax(func)
                        #print(f'Max {np.asarray(func).max()} ({np.asarray(waveform).max()}) while {popt}')
                        max_x,max_y = self.SiPM.Ch[self.CHANNEL].Time[peak_x], func[peak_x]
                        """if max_x < bounds[0] or max_x > bounds[1]:
                            self.log(f'Skipping due to peak out of bounds at {max_x}x{max_y}',4)
                            continue"""
                        #print(f'Max {max_x}x{max_y} for {peak_x}')
                        self.log(f'Max {max_x}x{max_y}',4)
                        perr = np.sqrt(np.diag(pcov))[1]

                        calc_tau = round(popt[2],0)
                        calc_tau_ind = int(calc_tau/((1/self.SiPM.sampling_freq)*1000000))
                        print(f'Value at tau: {waveform[peak_x+calc_tau_ind]} (func: {func[peak_x+calc_tau_ind]})\nFor tau {calc_tau} (index: {calc_tau_ind})\nand function peak {func[peak_x]} (argmax of model) (-> {np.max(waveform)} argmax wave)\n')

                        try: print(chisquare(func,waveform))
                        except: print('Cannot get chisquare')

                        if perr > 0:
                            lost += 1
                            self.log(f'Perr {perr}',4)
                            possible_pks = find_peaks(waveform,prominence=40)[0]
                            plt.figure(figsize=(8,8))
                            plt.title(f'P_error: {perr:.5f}')
                            plt.xlim(180,400)
                            print(popt)
                            if len(possible_pks) > 1:
                                for p in possible_pks:
                                    ppk_x = self.SiPM.Ch[self.CHANNEL].Time[p]
                                    ppk_y = waveform[p]
                                    plt.scatter([ppk_x],[ppk_y],50,color='magenta',marker=11,zorder=4)
                                plt.scatter([max_x],[max_y],50,color='red',marker='+',label='Fit max',zorder=3)
                                plt.plot(self.SiPM.Ch[self.CHANNEL].Time,func,color='red',alpha=0.8,zorder=2)
                                plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform,zorder=1)
                                plt.show()
                            else:
                                
                                plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform,color='purple')
                                plt.plot(self.SiPM.Ch[self.CHANNEL].Time,func,color='lime',alpha=0.8)
                                plt.scatter([max_x],[max_y],50,color='blue',marker='+',label='Fit max')
                                plt.show()
                        else:
                            #plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform)
                            #plt.plot(self.SiPM.Ch[self.CHANNEL].Time,func)
                            #plt.show()
                            self.waveform_fit_params.append((popt,perr))
                            x.append([max_x])
                            y.append([max_y])
                            pks_tally += 1
                            coords.append((x,y))
                            fit_params.append({'x':self.SiPM.Ch[self.CHANNEL].Time,'popt':popt,'chi':None,'perr':perr})
                
                end = datetime.now()
                intervals.append((end-start).total_seconds())
                if f.value % 30 == 0: self.update(int(((np.mean(intervals)*(len(self.SiPM.Ch[self.CHANNEL].Amp)-f.value)))),'FUNC%20FIT')
                print(f'Estimated time to completion: {float(np.mean(intervals)*(len(self.SiPM.Ch[self.CHANNEL].Amp)-f.value)/60):.01f} minutes',end='\r')

                if len(intervals) == 5: self.notify('Initated fit calc.',f'ETC: {float(np.mean(intervals)*(len(self.SiPM.Ch[self.CHANNEL].Amp)-f.value)/60):.01f} minutes',0)
            
            if len(x) != len(y): raise(RuntimeError('Returning list of peak coordinates of different lengths. x: '+str(len(x))+ ' y: '+str(len(y))+' for '+str(len(self.SiPM.Ch[self.CHANNEL].Amp))+' waveforms'))

            if pks_tally > 0: self.log(f'{pks_tally} peaks found for {len(self.SiPM.Ch[self.CHANNEL].Amp)} waveforms. Loss: {lost} ({(float(lost/pks_tally))*100:.02f}%)',1)
            else: self.log('No peaks found. Lost: '+str(lost),2)

            self.__save_coords(fit_params)
            return (self.SiPM.Ch[self.CHANNEL].Amp, coords)
        elif not reload:
            self.log('Loading from backup file',1)
        
        self.log(f'{len(fit_params)} waveforms loaded.',1)

        if len(fit_params) > 0:

            for fit in fit_params:

                x_ax = fit['x']
                popt = fit['popt']
                chi = fit['chi']
                perr = fit['perr']

                model = self.__wave_func(x_ax,*popt)
                #plt.plot(x_ax,model)
                #plt.show()
                p_max = float(popt[1])
                p_x = float(popt[0])
                a_max_x = np.argmax(model)
                a_max = float(model[a_max_x])
                a_x = float(x_ax[a_max_x])
                print(np.abs(p_max-a_max))
                print(f'ARGMAX: {p_max} at {p_x}.')
                print(f'A VAL: {a_max} at {a_x}.')
                print()
                if (np.abs(p_max-a_max) > 20):
                    plt.plot(x_ax,model)
                    plt.show()
                #print('Max would be '+str(popt[1]))
                #print('Max would also be '+str(np.max(model)))
                #sys.exit()

        else:
            self.log('Something went wrong while loading fit_params.',3)
            self.error = True
            return False

    def resolve(self):
        self.__resolve_wf_fit_params()

    def __check_waveform(self, waveform):
        """ deprecated """

        pks = find_peaks(waveform,prominence=150,distance=30)[0]
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
                
    def __resolve_wf_fit_params(self):
        total = 0   # stats
        temp = [0]*len(self.pulses[0].time)    # necessary x-axis-long array to average values for each x value.
        correlated = 0 # stats
        misaligned = 0 # stats
        peaks_xs = []   # was used to estimate mu variance --> understand peak dispertion for avalanches

        to_plt = [] # array of pulses to plot

        for pulse in self.pulses:
            pulse.charachterize() # see Pulse class for reference. Requests an analysis of the pulse. After the analysis then pulse Shape and PulseType are available to be read as class attributes.

            if pulse.shape is PulseType.Shape.STANDARD and pulse.alignment is PulseType.ALIGNED: # only averages pulses that are aligned (on the x axis, so with the right timing) and shaped in the standard way.
                # TODO in the future you can make it better by letting it handle and fit double peaks and oddly shaped peaks.
                for ii, y in enumerate(pulse.signal):
                    temp[ii] = temp[ii] + y # adding to then average
                total += 1
                pulse.skim()    # see Pulse class for reference. This asks for a rough estimate of the amplitude and mu obtained by skimming the pulse and not actually by performing a long (and unecessary) analysis.

                peaks_xs.append(pulse.mu)   # used to verify variance of mu --> understand peak dispersion due to correlated avalanches. It has proven useful in before.

                to_plt.append(pulse.signal) # will be plotted

                pulse.selected = True   # for parameters estimation. as it is standard and aligned

            elif pulse.shape is PulseType.Shape.MULTIPLE_PEAKS: correlated += 1 # stats
            elif not pulse.alignment is PulseType.ALIGNED: misaligned += 1  # stats

        if total < 20:  # statistically not relevant under 20. Estimation error would be too high. TODO update to any number seems more suitable.
            self.log(f'No waveforms suitable for τ estimation -> will continue \u001b[0mwithout{ANSI_RESET}{ANSI_RED} fixed parameters!',3)
            return False

        # --- BELOW MU, TAU AND SIGMA ARE ESTIMATED AS WELL AS THE BASELINE ---
        # --- this estimation is based on argmax and other parameters that can be adjusted --- 
        avg = np.asarray([float(t/total) for t in temp])    # average the selected pulses
        peak_x = np.argmax(avg)     # getting the peak x value
        peak_A = avg[peak_x]        # getting the peak amplitude
        tau_end_val = peak_A*np.e**-1   # the y value at tau based on the equation
        tau_ind = np.where(avg[peak_x:] < tau_end_val)[0][0]    # index (array index) of tau
        tau_est = self.SiPM.Ch[self.CHANNEL].Time[tau_ind]   # x-value of tau

        baseline = np.average(avg[0:30])    # baseline estimation. All the selected pulses begin with a flatline
        
        sigma_ind = np.where(avg[:peak_x] < baseline*1.05)[0][0]    # index of sigma. TODO adjust the 1.05 parameter if necessary
        sigma_est = self.SiPM.Ch[self.CHANNEL].Time[sigma_ind]      # x-value for sigma

        temp_sum = 0    #   sum of he peaks' x-values
        for px in peaks_xs: temp_sum += px  # getting the sum
        mu_ind = int(temp_sum/len(peaks_xs))    # average of the mu of all the waveforms as estimate.
        mu_est = self.SiPM.Ch[self.CHANNEL].Time[mu_ind]    # x-value for mu
        

        # --- BELOW IS SOME PLOTTING TO SHOW THE WORK DONE ---

        plt.figure(figsize=(18,6))
        plt.suptitle(f'Estimating fit parameters for {self.voltage:.02f} OV', size=16)
        for wave in self.SiPM.Ch[self.CHANNEL].Amp:
            plt.subplot(1,2,1)
            plt.title('Unfiltered')
            plt.xlabel('[1 = 5 microseconds]')
            plt.ylabel('[mV]')
            plt.plot(self.SiPM.Ch[self.CHANNEL].Time,wave,color='navy')
        #plt.savefig(f'{folder}/all.png',dpi=200,facecolor='white')
        for tp in to_plt:
            plt.subplot(1,2,2)
            plt.title('Filtered')
            plt.xlabel('[1 = 5 microseconds]')
            plt.ylabel('[mV]')
            plt.plot(self.SiPM.Ch[self.CHANNEL].Time,tp,color='black')
        #plt.savefig(f'{folder}/cleaned.png',dpi=200,facecolor='white')
        plt.show()

        self.log(f'ESTIMATED PARAMETERS:\n  - τ: from {peak_A:.04f} mV to {tau_end_val:.04f} -> {tau_est:.04f} μs -> {tau_ind} indexes after the peak.\n - sigma: {sigma_est:.04f} μs -> {sigma_ind} indexes before the peak.\n - mu: {mu_est:.04f} μs -> {mu_ind} indexes to the peak.\n {total} out of {len(self.pulses)} waveforms recorded at {self.voltage:.02f} OV were used for this calculation',1)
        self.log(f'WAVEFORM FILTERING:\n {(correlated+misaligned)} ({((correlated+misaligned)/total*100):.03f}%) waveforms were not selected for τ estimation:\n  - {correlated} waveforms ({(correlated/total*100):.03f}%) excluded due to unwanted correlated avalanches. (multiple peaks)\n  - {misaligned} waveforms ({(misaligned/total*100):.03f}%) excluded due to timing issues.\n',2)

        plt.figure(figsize=(18,6))
        plt.suptitle(f'Estimating tau for {self.voltage:.02f} OV', size=16)
        plt.subplot(1,2,1)
        plt.plot(self.SiPM.Ch[self.CHANNEL].Time,avg,label='Averaged')
        plt.vlines(self.SiPM.Ch[self.CHANNEL].Time[tau_ind+peak_x],0,peak_A,linestyles='dashed',colors=['red'],label=f'tau at {self.SiPM.Ch[self.CHANNEL].Time[tau_ind+peak_x]}')
        plt.legend(loc=1)
        plt.subplot(1,2,2)
        plt.plot(self.SiPM.Ch[self.CHANNEL].Time,avg,label='Averaged')
        plt.vlines(self.SiPM.Ch[self.CHANNEL].Time[tau_ind+peak_x],0,peak_A,linestyles='dashed',colors=['red'],label='tau')
        plt.xlim(180,peak_x+tau_ind+50)
        plt.title('Zoomed in')
        plt.show()
        #plt.savefig(r'/users/Tizi/desktop/fig.png')

        # setting them as class variables below so they can be used by model functions

        self.tau_est_final = tau_est
        self.mu_est_final = mu_est
        self.sigma_est_final = sigma_est

        if self.__peak_dispersion(peaks_xs): return True    # if the peak are too dispersed then the data is considered not reliable as their average would result in an abnormal waveform. TODO can be remoed as the waveforms are filtered and only selected if aligned and standard-shaped.
        
        self.estimate_parameters = False    # the data is too diverse to allow for parameters estimation. The program will keep running without estimating the parameters before the fit.
        return False

    def __peak_dispersion(self, pks):
        # getting standard deviation of the peak x-value
        std_dv = gstd(pks)
        if std_dv < 1.1: self.log(f'Peak distribution shows a standard deviation of {std_dv} μs',4)
        elif std_dv < 1.5: self.log(f'WARNING: peak timing standard deviation is elevated: {std_dv} μs. More filtering is advised',2)
        else:
            self.log(f'SIGNAL QUALITY ERROR: the waveforms you are trying to analyze show very diverse peak timing. Filter the data before proceeding',3)
            return False
        return True
        """plt.figure(figsize=(10,8))
        plt.suptitle(f'Estimating timing sigma for peaks at {self.voltage:.02f} OV', size=16)
        plt.hist(pks,bins=np.arange(190,230,1),histtype='bar')
        plt.show()"""

    def __save_coords(self, fit_params):
        """ this method is used to save waveform fit parameters instead of of entire waveforms. Much less space, much more information in each file! File size is reduced by multiple orders of magnitude """

        # --- PLEASE NOTE: the np.savez function below is commented as I did not need to save the coordinates for testing purposes. Uncomment it for production!

        if not os.path.exists('./waveform_fits_data'): os.mkdir('waveform_fits_data')
        filename = f'./waveform_fits_data/_backup_{self.voltage:.01f}'
        #20.8 MB vs 216 KB --> compared to saving waveforms!!!

        x_axes = [] # they should all be an array of 2048 elements representing the time in us. Save it for every file just to be safe and because it will be useful later on.
        popts = []  # model parameters
        chis = []   # model quality parameter
        perrs = []  # model reliability parameter

        for fit in fit_params:
            x_axes.append(fit['x'])
            popts.append(fit['popt'])
            chis.append(fit['chi'])
            perrs.append(fit['perr'])
        
        if len(x_axes) == len(popts) == len(chis) == len(perrs):
            try: pass#np.savez_compressed(filename, x_axes=x_axes, popts=popts, chis=chis, perrs=perrs) # NOTE: uncomment for production
            except: pass
            else:
                if os.path.exists(filename+'.npz'):
                    self.log(f'{self.voltage:.02f} OV raw waveforms\' fit parameters saved successifully as {filename}',1)
                    return True
        
        self.log(f'Something went wrong while saving {filename}. Fit parameters will need to be reloaded at each run.',3)
        return False
    
    def __open_coords(self):
        """ open the files created above. """
        filename = f'./waveform_fits_data/_backup_{self.voltage:.01f}.npz'
        fit_params = []

        if os.path.exists(filename):

            file = None
            try: file = np.load(filename,allow_pickle=True)
            except: return False
            else:
                self.log(f'{filename} retreived successifully.',1)

                x_axes = file['x_axes'] 
                popts = file['popts']
                chis = file['chis']
                perrs = file['perrs']

                if len(x_axes) == len(popts) == len(chis) == len(perrs):
                    for ii in np.arange(0,len(x_axes),1):
                        fit_params.append({'x':x_axes[ii],'popt':popts[ii],'chi':chis[ii],'perr':perrs[ii]})
                    return fit_params
                else:
                    self.log(f'Error loading arrays from {filename}.',2)
                    return False

        else: return False
               
    def __p0estimate(self,waveform):
        """ useful to estimate mu, tau, and base to fit when the parameters have not been estimated and fixed. DEPRECATED due to a different approach being taken """

        temp_peaks_ind = np.asarray(find_peaks(waveform,prominence=30)[0])
        if len(temp_peaks_ind) == 0 or temp_peaks_ind is None:
            self.log('Error loading p0: could not find any peaks',4) 
            #plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform,color='magenta')
            #plt.show()
            return None
        
        LIMIT = 800
        temp_peaks_ind = temp_peaks_ind[(self.SiPM.Ch[self.CHANNEL].Time[temp_peaks_ind]<LIMIT)]
        if len(temp_peaks_ind) == 0:
            plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform,color='magenta')
            plt.show()
            self.log('Error loading p0: could not find any peaks below '+str(LIMIT),4) 
            return None
        tallest_peak_ind = temp_peaks_ind[np.argmax([waveform[p] for p in temp_peaks_ind])]
        A_est = waveform[tallest_peak_ind]
        mu_est = self.SiPM.Ch[self.CHANNEL].Time[tallest_peak_ind]-6
        base_est = float(np.mean(waveform[1:10]))
        try:
            tau_est = np.abs(self.SiPM.Ch[self.CHANNEL].Time[np.where(waveform[tallest_peak_ind:] < (A_est)*(np.e**-1)+base_est)[0][0]] - self.SiPM.Ch[self.CHANNEL].Time[tallest_peak_ind])
        except IndexError as e:
            self.log(e,4)
            return None
        except Exception as e: #when is index 0 out of bounds it is because the np.where function does not return any true values because that condition is never meth because it is not a regular wave. Probably just noise.
            self.log(f'Estimate error ({type(e)}: {e}',3)
            plt.plot(self.SiPM.Ch[self.CHANNEL].Time,waveform,color='brown')
            plt.show()
            return None
        sigma_est = 1

        if not self.tau_est_final is None: tau_est = self.tau_est_final

        self.log(f'Estimating A: {A_est}    mu: {mu_est}    tau: {tau_est}  base: {base_est}    sigma: {sigma_est}',4)

        return [mu_est,A_est,tau_est,base_est,sigma_est]
        popt,pcov = curve_fit(self.wave_func)

        for ii, val in enumerate(sample):
            if ii > find_peaks(sample,prominence=30)[0] and np.abs(val) < sample_max - sample_max**(np.e**-1):
                prob_return_x = (ii)
                break

        est_tau = prob_return_x-prob_rise_x
        est_tau2 = Ds[volt].Ch[0].Time[prob_return_x] - Ds[volt].Ch[0].Time[prob_rise_x]

    def __wave_func(self, x, mu, A, tau, base,sigma):
        """ NO PARAMETERS ESTIMATED IN ADVANCE EXPECT FOR TAU (OPTIONAL)"""

        if not self.tau_est_final is None: tau = self.tau_est_final #fixed if estimated
        return base + A/2.0 * np.exp(0.5 * (sigma/tau)**2 - (x-mu)/tau) * erfc(1.0/np.sqrt(2.0) * (sigma/tau - (x-mu)/sigma))

    def __wave_func2(self, x, A,sigma):
        """ TAU AND MU ESTIMATED """

        tau = self.tau_est_final
        #sigma = self.sigma_est_final   # TODO can use if trusted. It is not giving great results
        mu = self.mu_est_final
        return  A/2.0 * np.exp(0.5 * (sigma/tau)**2 - (x-mu)/tau) * erfc(1.0/np.sqrt(2.0) * (sigma/tau - (x-mu)/sigma))

    def __wave_func3(self, x, A,sigma):
        """ CORRECTION FACTOR INCLUDED ! AS WELL AS ESTIMATED MU AND TAU """

        tau = self.tau_est_final
        #sigma = self.sigma_est_final
        mu = self.mu_est_final
        return  self.factor * A/2.0 * np.exp(0.5 * (sigma/tau)**2 - (x-mu)/tau) * erfc(1.0/np.sqrt(2.0) * (sigma/tau - (x-mu)/sigma))

    def eval_waveform_argmax(self, voltage, bounds=None):
        """ initally used to get amplitude of pulses. Needs to be deprecated and removed for good as a different approach was deemed necessary. """

        if self.error: self.__throw_error()
        voltage = float(voltage)
        if type(voltage) == float: pass
        else:
            self.error = True
            raise ValueError('Parameter voltage must be of type float or int. '+str(type(voltage))+' found instead')

        if bounds is None: bounds = [0,len(self.SiPM.Ch[self.CHANNEL].Amp[0])]

        coords = [] #   x,y coordinate arrays organized in one tuple per waveform
        pks_tally = 0 #     keep track of how many peaks are found

        self.log(f'Gettig {len(self.SiPM.Ch[self.CHANNEL].Amp)} waveforms for {voltage:.2f}V [{bounds[0]}:{bounds[1]}]',1)
        
        for waveform in self.SiPM.Ch[self.CHANNEL].Amp: #loop over the waveforms inside the file
            filtered = self.SiPM.get_filtered_waveform(waveform) # return the filtered waveform
            x = [] #    must be same lenght as y. NOTE: array of arrays due to some waveforms having multiple peaks (found with other methods).
            y = [] #    must be same length as x. NOTE: array of arrays due to some waveforms having multiple peaks (found with other methos).
            
            for i,y1 in enumerate(filtered): #Iterates each wave filtered with a different shaping time
                peak_index = np.argmax(y1[bounds[0]:bounds[1]])+bounds[0] #Finds the index for max value in the waveform amplitude array
                peak_value = y1[peak_index] #Gets the max value from the array of amps
                pks_tally += 1
                x.append(peak_index)
                y.append(peak_value)
            
            coords.append((x,y))
        
        if len(x) != len(y): raise(RuntimeError('Returning list of peak coordinates of different lengths. x: '+str(len(x))+ ' y: '+str(len(y))+' for '+str(len(self.SiPM.Ch[self.CHANNEL].Amp))+' waveforms'))

        self.log(f'{pks_tally} peaks found for {len(self.SiPMs[voltage].Ch[self.CHANNEL].Amp)} waveforms.',0)

        return (self.SiPM.Ch[self.CHANNEL].Amp, coords)
  
    def peaks_test(self, voltage, maxfev=1000000):
        pass

    def __raw_eval_gain(self,voltage, waveform_fit=True,find_best=False,bin_width=5,distance=12,prominence=30,min=2,plot=False,fix_params=False, reload=False):
        """ this method allows for gain calculation after waveform analysis. Raw because it uses pulses compared to the MCA gain analysis """

        peaks_coords = None # it used to be (x,y) now it is onlt [y] as the x is always the same for aligned pulses
        if waveform_fit:    # this flag tells if the software should use model fitting or argmax to find pulse amplitude
            peaks_coords = self.eval_waveform_func_fit(voltage,fix_params=fix_params,reload=reload)[1]
        else: peaks_coords = self.eval_waveform_argmax(voltage)[1]  # no longer supported fully. Might fail.

        # --- GETTING THE PULSES' PEAKS COORDINATES ---

        popt_list, perr_list = None, None   # for all the waveforms

        for pulse in self.pulses:
            pulse.charachterize()
            if pulse.alignment == PulseType.ALIGNED and pulse.shape == PulseType.Shape.STANDARD:
                pulse.study(self.mu_est_final, self.sigma_est_final, self.tau_est_final, self.factor)
                peaks_coords.append(pulse.amplitude)

        #peaks_coords = self.__peak_filter(peaks_coords) # filters the waveforms and only returns the peaks

        if not find_best:   # this flag determines if the smart algorithm should be used for peak finding. Note: if the flag is set to false then bin_width, prominence, distance and min_peaks must be passed to this method.
            popt_list,perr_list = self.__get_peaks(peaks_coords,bin_width,prominence,distance,min,plot=plot)
        else:
            #   raise(NotImplementedError('Find best has not been implemented yet! Sorry ;)'))
            popt_list,perr_list = self.__find_best(peaks_coords)
        

        if popt_list is False:
            self.log(f'MAJOR ERROR: Cannot find best paramaters for find_peaks function! Cannot proceed. Review data and limits.',3)
            return None

        if popt_list is None:
            self.log(f'MAJOR ERROR: Cannot find any peaks! Cannot proceed. Review data and limits.',3)
            return None

        # --- PRINTING INFO --- 
        print()
        msg = ''
        print('Resulting peaks:')
        msg += 'Resulting peaks:\n'
        for ii,popt in enumerate(popt_list):
            perr = perr_list[ii]
            x_pos = popt[1]
            if perr == False:print(f'{ANSI_RED} * {x_pos:.16g} --> FIT FAILED{ANSI_RESET}')
            else:
                if perr == np.inf: print(f'{ANSI_BG_RED} * {x_pos:.16g} \u00b1 {perr:.16g}{ANSI_RESET}')
                elif perr < 1: print(f' * {x_pos:.16g} \u00b1 {ANSI_GREEN}{perr:.16g}{ANSI_RESET}')
                else: print(f' * {x_pos:.16g} \u00b1 {perr:.16g}')
                msg += f' * {x_pos:.10f} \u00b1 {perr:.10f}\n'
        print(popt_list)

        # --- getting the gain ---
        gain = self.__gain_from_peaks(popt_list,perr_list)
        self.gain_evaluation = gain

        print(f'Memory usage: {tracemalloc.get_traced_memory()[1]/1000000} MB peak')
        tracemalloc.stop()  # stop memory usage tracing. The memory-intensive analysis is over anyway.

        return gain

    def eval_gain(self, waveform_fit=True,find_best=False,bin_width=5,distance=12,prominence=30,min=2,plot=False,fix_params=False, reload=False,total_bins=None,min_bin=None, max_bin=None, min_prominence=None, max_prominence=None, min_distance=None, max_distance=None):
        """ called to get the gain out of both MCA and Raw data """

        if self.error: self.__throw_error()

        if self.data_type == self.RAW:
            return self.__raw_eval_gain(self.voltage,waveform_fit,find_best,bin_width,distance,prominence,min,plot,fix_params,reload)

        if self.data_type == self.MCA:
            if find_best:   # this flag determines if the smart algorithm should be used for peak finding. Note: if the flag is set to false then bin_width, prominence, distance and min_peaks must be passed to this method.
                return self.__mca_eval_gain(total_bins,min,best=True, min_bin=min_bin,max_bin=max_bin,min_distance=min_distance,max_distance=max_distance,min_prominence=min_prominence,max_prominence=max_prominence)
            return self.__mca_eval_gain(total_bins,min) # if find_best is True then all those parameters will be determined by the system to get the best fit

    def __gain_from_peaks(self, popt_list,perr_list):
        data = np.asarray([float(peak[1]) for peak in popt_list])
        errs = np.asarray([float(err) for err in perr_list])

        #plt.errorbar(np.arange(1,len(data)+1,1),data,yerr=errs,fmt='o',ms=5,color='red',label='Peaks')
        popt, pcov = curve_fit(self.line,np.arange(1,len(data)+1,1),data,sigma=errs)
        #plt.plot(np.arange(1,len(data)+1,1),self.line(np.arange(1,len(data)+1,1),*popt),color='black',ls="--",alpha=0.8)
        self.log(f'Gain: {popt[1]:.2f} \u00b1 {float(np.sqrt(np.diag(pcov)[1])):.2f}',1)
        self.gain_fit_params = (popt,float(np.sqrt(np.diag(pcov)[1])))
        return (float(popt[1]),float(np.sqrt(np.diag(pcov)[1])))

    def __find_best(self, peaks_coords,method=None):
        fits, errs = [], [] # used at the end to find the best fit.

        # -- TWEAK THE PARAMETERS BELOW TO YOUR LIKING. !!! NOTE THAT SOMETIMES THE BEST FIT IS A VERY PRECISE AND UNPREDICTABLE COMBINATION OF PARAMETERS !!! STATISTICAL EVIDENCE SUGGESTS THAT THE BEST WAY IS TO NOT INCRESE THE STEP SIZE OR THE MIMIMUM BIN WIDTH. I ADVISE AGAINST MODIFYING THESE PARAMETERS AS THEY ARE THE RESULT OF EXPERIENCE --- 

        # TODO it's easy and should be done. Make this like the find_best_MCA where the user can decide these bounds as parameters passed to the method

        bin_width_start = 1     # MINIMUM bin width
        bin_width_stop = 12     # MAXIMUM bin width
        bin_width_step = 0.5    # STEP in incresing bin width

        prominence_start = 10     # MINIMUM prominence
        prominence_stop = 100     # MAXIMUM prominence
        prominence_step = 5       # STEP in increasing prominence

        distance_start = 10      # MINIMUM distance
        distance_stop = 90       # MAXIMUM distance
        distance_step = 2        # STEP in increasing distance

        min_peaks = 5            # MINIMUM peaks
        max_error = 2.0          # MAXIMUM acceptable error 
        
        #trials = {}
        last = None
        best = None
        temp_errs = []  # used to evaluate and compare "combined values". See below.

        skipped = 0 # stats
        done = 0    # stats

        perms = [item for item in itertools.product(np.arange(bin_width_start, bin_width_stop,bin_width_step),np.arange(prominence_start, prominence_stop,prominence_step),np.arange(distance_start, distance_stop,distance_step))] # all the possibile parameters permutations allowed by our parameters set above.
        
        # progress bar below
        f = IntProgress(value=0,min=0,max=len(perms),step=1,description='Loading perms...',bar_style='info',layout={"width": "100%"})
        display(f)
        self.log('Multiprocessing does not support progress bars. Disregard it.',2)
        args = []   # used to call multiprocessing's pool method. Creating a different call for every single permutation so as to test them all.

        with Manager() as manager:  # necessary for multicore multiprocessing variable handling
            trials = manager.dict() # shared between different CPU cores. It slows everything down and can be made better.
            start = datetime.now()  # stats

            for perm in perms:
                args.append([perm, peaks_coords,min_peaks,trials,max_error])    # creating args array

            pool = Pool()   # creating processes
            pool.starmap(self.perm_run, args)   # activating multiprocessing
            pool.close()   # multiprocessing ends 

            # now multiple processes will have addedd all to the "trials" dictionary which is ready to be analyzed. The dictionary is organized as { permutation : data } where "permutation" is of type (##,##,##) and data is a tuple (popt_list, perr_list). popt_list is a list of optimal parameters for each peak found in the historgram that came up after using those particular distance, bins and prominence specified in the associated dictionary key. Perr_list is a list of curve_fit errors for the same thing.

            for perm in trials.keys():
                popt_list, perr_list = trials[perm]     # unpacking
                last = perm     # 
                num_peaks = len(popt_list)              # number of peaks detected and deemed suitable by the software
                combined = float(np.sum(perr_list))/num_peaks   # normalizing the error by the number of the peaks because we want the smallest error but also the highest possible number of peaks. This gives us the best combination of both since min_peaks is set we will not get too few peaks. This is part of the smart optimization of SmartGAT.

                if not True in np.asarray(perr_list) > max_error:   # check if all the peaks are within the max error range
                    if num_peaks >= min_peaks: temp_errs.append(combined)   # check that we have enough peaks and if so append the combined error/peaks value to a temporary array for further evaluation.
                else: self.log('Not counting for too big err'+str(perr_list),4)

                if True in np.asarray(temp_errs) < combined: continue   # if tthere are already "combined" values (from other permutations) in the temporary array smaller than the current one it makes no sense to keep analyzing it because we already have something better.
                else:
                    self.log(np.asarray(temp_errs) < combined,4)    # logging
                    self.log(temp_errs,4)                           # logging
                    self.log(f'Combined/peaks {combined}',4)        # logging
                    best = perm                                     # the best so far

                done += 1   # stats

            self.log(f'Skipped {skipped} of {done+skipped}',4)  # stats

            end = datetime.now()    # stats
            self.log(f'Runtime: {float(np.mean((end-start).total_seconds())):.01f} seconds',1)  # stats
            """ for p in perms:
                start = datetime.now()
                f.value += 1

                
                popt_list,perr_list = self.__perm_run(p,peaks_coords,min_peaks)
                self.__get_peaks(peaks_coords,p[0],p[1],p[2],peaks_min=min_peaks,plot=False)


                if False in perr_list or np.inf in perr_list: skipped += 1
                else:
                    trials[p] = (popt_list,perr_list)
                    last = p
                    combined = float(np.sum(perr_list))
                    if not True in np.asarray(perr_list) > max_error:
                        temp_errs.append(combined)
                    else: self.log('Not counting for too big err'+str(perr_list),4)
                    if True in np.asarray(temp_errs) < combined: continue
                    else:
                        self.log(np.asarray(temp_errs) < combined,4)
                        self.log(temp_errs,4)
                        self.log(f'Combined {combined}',4)
                        best = p
                    done += 1
                    self.log(f'Skipped {skipped} of {done+skipped}',4)
                
                end = datetime.now()
                diffs.append((end-start).total_seconds())
                if f.value % 30 == 0: self.update(int(((np.mean(diffs)*(len(perms)-f.value)))),'BEST%20FIT')
                print(f'Estimated time to completion: {float(np.mean(diffs)*(len(perms)-f.value)/60):.01f} minutes',end='\r')
                if len(diffs) == 5: self.notify('Initated gain calc.',f'ETC: {float(np.mean(diffs)*(len(perms)-f.value)/60):.01f} minutes',1) """


            if not best is None: # a best fit was found for the parameters in the best variable stored as (bin width, prominence, distance). 
                self.log(f'Best found at prominence {best[1]}, distance {best[2]}, bin width {best[0]} for {len(trials)} trials',1)

            else:   # the bounds set above are too narrow or the data is too diverse to find a best fit
                self.log('Fatal error: could not load any peaks!',3)
                return (False,False)

            return self.__get_peaks(peaks_coords,*best,min_peaks,plot=True, is_best=True)   # return the popt_list and perr_list for the optimal parameters so the optimal list of optimal peak parameters and their errors (to display for the user)
    
    def perm_run(self, perm, peaks_coords, min_peaks,trials, max_error):
        """ running a permutation to see if the historgram made with these parameters is good or not. Unless it fails, it adds to the trials dictionary for later analysis of best fit. """

        popt_list,perr_list = self.__get_peaks(peaks_coords,perm[0],perm[1],perm[2],peaks_min=min_peaks,plot=False)
        if False in perr_list or np.inf in perr_list: pass#skipped += 1
        else:
            trials[perm] = (popt_list,perr_list)
            """last = perm
            combined = float(np.sum(perr_list))
            if not True in np.asarray(perr_list) > max_error:
                temp_errs.append(combined)
            else: self.log('Not counting for too big err'+str(perr_list),4)
            if True in np.asarray(temp_errs) < combined: continue
            else:
                self.log(np.asarray(temp_errs) < combined,4)
                self.log(temp_errs,4)
                self.log(f'Combined {combined}',4)
                best = p
            done += 1
            self.log(f'Skipped {skipped} of {done+skipped}',4) """            

    def __find_best_MCA(self, file, min_peaks, min_bin, max_bin, min_prominence, max_prominence, min_distance, max_distance):
        """ the same sa the find_best above but for MCA data """

        fits, errs = [], []
        x, y = self.waveforms[file] # take MCA data
        x = x[500:] # there is nothing interesting 0:500. I know because I looked at tons of files
        y = y[500:] # there is nothing interesting 0:500. I know because I looked at tons of files

        bin_start = min_bin
        bin_stop = max_bin
        bin_step = 1 # NOTE DO NOT SET THIS TO FLOAT VALUES --> REBIN ONLY TAKES INTS AND CONVERTS TO INT ANYWAY!!!!!!
        bins = np.arange(bin_start,bin_stop, bin_step)

        prominence_start = min_prominence
        prominence_stop = max_prominence
        prominence_step = 5

        distance_start = min_distance
        distance_stop = max_distance
        distance_step = 2

        min_peaks = min_peaks
        max_error = 5.0
        
        #trials = {}
        last = None
        
        best = None
        temp_errs = []

        skipped = 0
        done = 0

        perms = [item for item in itertools.product(np.arange(bin_start, bin_stop,bin_step),np.arange(prominence_start, prominence_stop,prominence_step),np.arange(distance_start, distance_stop,distance_step))]
        
        self.log(f'Best fit for {file} using {len(perms)} permutations.', 4)
        
        f = IntProgress(value=0,min=0,max=len(perms),step=1,description='Loading MCA perms...',bar_style='info',layout={"width": "100%"})
        display(f)
        self.log('Multiprocessing does not support progress bars. Disregard it.',2)
        args = []


        with Manager() as manager:
            trials = manager.dict()
            start = datetime.now()

            for perm in perms:
                args.append([perm, x,y,min_peaks, max_error, trials])

            self.log(f'Started multiprocessing on {multiprocessing.cpu_count()} CPUs',4)
            pool = Pool()
            pool.starmap(self.perm_run_MCA, args)
            pool.close()
            #self.perm_run_MCA((13,80,3),x,y,min_peaks,max_error,trials)

            self.log('Best fit analysis for MCA data completed. Now running on single thread',4)

            for perm in trials.keys():
                peaks, perr_list = trials[perm]     # Go through all the permutations with less than max_err error
                last = perm
                num_peaks = len(peaks)
                combined = float(np.sum(perr_list))/num_peaks   # Get best combination of num_peaks and min error

                if not True in np.asarray(perr_list) > max_error:       # Discard if any peak has error > max error
                    if num_peaks >= min_peaks: temp_errs.append(combined)       # Discard if num peaks < min peaks
                    else: self.log('Not counting for not enough peaks',4)
                else: self.log('Not counting for too big err'+str(perr_list),4)
                if True in np.asarray(temp_errs) < combined: continue           # It is not the best one in the array of combined values
                else:                                                           # Best one yet (combined value)
                    #self.log(np.asarray(temp_errs) < combined,4)
                    #self.log(temp_errs,4)
                    #self.log(f'Combined/peaks: {combined}',4)
                    best = perm
                done += 1
                #self.log(f'Skipped {skipped} of {done+skipped}',4)

            end = datetime.now()
            self.log(f'Runtime: {float(np.mean((end-start).total_seconds())):.01f} seconds',1)

            if not best is None: 
                self.log(f'Best found at prominence {best[1]}, distance {best[2]}, bins {best[0]} for {len(trials)} trials',1)
                #self.gauss_peaks_MCA(self.perm_run_MCA_2(best,x,y,min_peaks,max_error,trials),None,x,y,max_error,plot=True)
            else:
                self.log('Fatal error: could not load any peaks!',3)
                return None

            #self.__mca_fit_peaks2(min_peaks,best[0],x,y,prominence=best[1],distance=best[2],plot=True)
            return best # returns the optimal permutation

    def perm_run_MCA_2(self, perm, x,y, min_peaks, max_error, trials):
        """ same as perm_run but for MCA data and v2.0 """

        x, y = self.rebin(x,y,perm[0])
        peaks,pdict = find_peaks(y,prominence=perm[1],distance=perm[2])
        
        if len(peaks) == 0: return False  
        if len(peaks) >= min_peaks:     # we have enough peaks
            # now it is time to fit these peaks individually with gaussians. Finding them was not enough like it was for raw data. NOTE More on this in the documentation.

            perrs, pks = self.gauss_peaks_MCA(peaks,pdict,x,y,max_error,plot=False)
            if perrs is False: return False # unable to fit them properly
            if not True in (np.asarray(perrs) > max_error) and len(pks) >= min_peaks:   # enough peaks fitted and all with an acceptable error
               trials[perm] = (pks,perrs)   # adds to the trials dictionary for later best fit eval.
               return pks
            
 
    def gauss_peaks_MCA(self,peaks,pdict,x,y,max_error,plot=False):
        """ fitting all the peaks from MCA data. More on this in the documentation provided with this code. """
        """ DISCLAIMER: part of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """
        """ ONLY comments necessary to understand this code will be made by me. The rest was copied. """

        err_temp = []
        gain_temp = []
        
        fit_peak_x = x[peaks[0]]
        fit_peak_amp = y[peaks[0]]
        x_idx_array=(y<0.5*fit_peak_amp) & (x>fit_peak_x)   # returns a boolean array where both conditions are true
        right_side_x= x[np.where(x_idx_array)[0][0]] #finding the first time where x_idx_array is True
        sigma_guess=np.abs(fit_peak_x-right_side_x) # We need this to fit the width of the Gaussian peaks

        cut= (x < fit_peak_x+sigma_guess) & (x > fit_peak_x-sigma_guess)
        try: popt,pcov=curve_fit(self.gauss,x[cut],y[cut],p0=[fit_peak_amp,fit_peak_x,sigma_guess],maxfev=1000000)
        except: return False, False
        
        #err_temp.append(np.sqrt(np.diag(pcov))[1])

        if plot:
            plt.figure(figsize=(12,2)) # Call the figure here
            plt.subplot(1,3,1) #This subplot will plot the position of the peaks and also the data
            # plt.ylim(0,50)
            plt.yscale('log')
            plt.plot(x[peaks],y[peaks],'*') # plot the peak markers
            plt.plot(x,y,lw=1) #plot the signal
            plt.plot(x[cut],self.gauss(x[cut],*popt),color='green',label='Fit',lw=2,alpha=0.5) # Here we plot the fit on the 2nd peak to see if everything looks ok.
    
        for i,peak in enumerate(peaks): #here we ignore the first peak because it could be the pedestal
            new_first_pe_max=x[peak] #x-value of the peak
            new_max_value=y[peak] #height of the peak
            new_x_idx_array=(y<0.5*new_max_value) & (x>new_first_pe_max) # returns a boolean array where both conditions are true
            new_right_side_x= x[np.where(new_x_idx_array)[0][0]] #finding the first time where x_idx_array is True
            new_sigma_guess=np.abs(new_first_pe_max-new_right_side_x) #We need this to fit the width of the Gaussian peaks


            new_cut= (x < new_first_pe_max+new_sigma_guess) & (x > new_first_pe_max-new_sigma_guess) # This cut helps to fix the width of the peak-fit
            popt_new,pcov_new=curve_fit(self.gauss,x[new_cut],y[new_cut],p0=[new_max_value,new_first_pe_max,new_sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix
            if plot: plt.plot(x[new_cut],self.gauss(x[new_cut],*popt_new),color='r',label='Fit',lw=3) # Here we plot the fit on all the peaks
            gain_temp.append(popt_new[1]) #Here we append the value of the peak fit mean
            err_temp.append(np.sqrt(np.diag(pcov_new))[1])

        if plot: plt.show()
        #print(f'Returning {gain_temp}')
        #weird = y[peaks[2]]/total_bins<90
        return (err_temp, peaks)
        

    def perm_run_MCA(self, perm, x,y, min_peaks, max_error, trials):
        """ DEPRECATED """

        #x,y = self.rebin(x,y, perm[0]) #1 (prom) 2(dist)
        try:
            err_list, peaks = self.__mca_fit_peaks2(min_peaks,perm[0],x,y,perm[1],perm[2],plot=False)
            #print(np.asarray(err_list))
            if peaks is False: return False
            elif True in (np.asarray(err_list) > max_error):
                return False
        except Exception as e:
            return False
        else:
            distances = []
            sorted_peaks = np.sort(peaks)
            for i in np.arange(4,len(sorted_peaks),1):
                prev = np.abs(sorted_peaks[i-1]-sorted_peaks[i-2])
                curr = np.abs(sorted_peaks[i]-sorted_peaks[i-1])
                if prev > curr*1.5 or curr > prev*1.5 or curr > 12:
                    if i > 6:
                        #print(f'Letting {sorted_peaks} pass because {i}>6')
                        trials[perm] = (peaks, err_list)
                        return
                    else: return False
                distances.append(np.abs(sorted_peaks[i]-sorted_peaks[i-1]))
            trials[perm] = (peaks, err_list)

    def __peak_filter(self, peaks_coords): #TODO CHECK THAT PEAK IS EXTRACTED FROM FUNCTION AND NOT ARG_MAX!!!
        """ get just pulse peaks that are aligned. This is now obsolete. It DOES NOT have to do with MCA """
        temp = []
        avg = 0
        for wvf_tuple in peaks_coords:
            x,y = wvf_tuple
            max_index = np.argmax(y)
            if isinstance(x[max_index], float) or isinstance(x[max_index], int): avg = avg + float(x[max_index])
            else: avg = avg + float(x[max_index][0]) #TODO UNDERSTAND WHY THIS HAPPENS
            temp.append((x[max_index],y[max_index]))
        self.log(f'Average peak mu is {avg/len(peaks_coords)} for {avg} and {len(peaks_coords)}',2)
        return temp

    def __get_peaks(self, peaks_filtered, bin_width, prominence, distance, peaks_min,max_bins=1000,plot=False,is_best=False):
        """ DEPRECATED """

        y_data = np.asarray([])
        x_data = np.asarray([])
        for xy_val in peaks_filtered:
            y_val = xy_val[1]
            #x_val = xy_val[0]
            y_data = np.append(y_data,y_val) #TODO GET RID OF NEGATIVE PEAKS
            #x_data = np.append(x_data,x_val) #TODO GET RID OF NEGATIVE PEAKS
            #TODO GET SCALES
 
        #plt.hist(data,bins=np.arange(0,max_bins,bin_width))
        y,x = np.histogram(y_data,bins=np.arange(y_data.min(),y_data.max(),bin_width))
        #plt.hist(data,bins=np.arange(0,max_bins,bin_width))
        #x = np.asarray([x_data[val] for val in x]).reshape(-1)
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        x = x[:-1]  #   get rid of last bin end value. It makes x > y otherwise.
        if plot: pass
            #plt.bar(x,y,color='royalblue',width=bin_width)
            #plt.xlim(0,1000)
        if is_best: self.hist_data = (x,y)
        if is_best: self.hist_params = (bin_width, prominence, distance)

        peaks_indexes = find_peaks(y, prominence=prominence,distance=distance)[0]

        self.log(f'{len(peaks_indexes)} peaks found',4)
        if len(peaks_indexes) < peaks_min:
            #raise(BaseException('Peaks not found'))
            return (None,[False])

        gauss_fits_temp = []
        gauss_err_temp = []

        peaks_final = ([],[])   #   format is ( [estimated], [gaussian fitted] ). Each one is an array of tuples of coordinates.
        peaks_final_perr = []

        for ii, peak_index in enumerate(peaks_indexes):
            pk_Y = y[peak_index]
            x_idx = (y < 0.5 * pk_Y) & (x > x[peak_index])
            rs_x = x[np.where(x_idx)[0][0]]
            sigma_guess = np.abs(x[peak_index]-rs_x)

            cut = (x > x[peak_index]-sigma_guess) & (x < x[peak_index]+sigma_guess)

            try: popt,pcov = curve_fit(self.gauss, x[cut], y[cut], p0=[pk_Y,x[peak_index],sigma_guess])
            except Exception as e:
                if 'must not exceed func output vector' in str(e): self.log('Sigma too small. Could not fit gaussian!',4)
                else: self.log(str(e),3)
                #plt.close()
                gauss_fits_temp.append((pk_Y,x[peak_index]))
                gauss_err_temp.append(False)
                #if is_best: plt.scatter([x[peak_index]],[pk_Y],50,color='red',marker='+',label='No fit')
                #if is_best: plt.hlines([pk_Y],0,x[peak_index],linestyles="dotted",colors=['red'],linewidths=0.8)

                #peaks_final[0].append((x[peak_index], pk_Y))
                #peaks_final[1].append((False, False))
                continue
            else:
                real_peak_index = float(popt[1])
                if is_best: self.hist_fit_params.append(((popt,np.sqrt(np.diag(pcov))[1]), (x[cut],self.gauss(x[cut],*popt))))
                self.log(popt,4)
                gauss_fits_temp.append(popt)
                gauss_err_temp.append(np.sqrt(np.diag(pcov))[1])
                if np.sqrt(np.diag(pcov))[1] == np.inf and plot: plt.plot(x[cut],self.gauss(x[cut],*popt),'m',label='Bad fit')
                elif plot: pass#plt.plot(x[cut],self.gauss(x[cut],*popt),'lime',label='Good fit')

                #if plot:plt.scatter([real_peak_index],[pk_Y],50,color='green',marker='+')
                #if plot:plt.hlines([pk_Y],0,real_peak_index,linestyles="--",colors=['magenta'],linewidths=0.8)
                #if plot:plt.hlines([popt[0]],0,popt[1],linestyles="--",colors=['green'],linewidths=0.8)

                peaks_final[0].append((x[peak_index], pk_Y))
                peaks_final[1].append((real_peak_index, popt[0]))
                peaks_final_perr.append(np.sqrt(np.diag(pcov))[1])

        #if plot: plt.show()
        if is_best: self.peaks_final = peaks_final
        if is_best: self.peaks_final_perr = peaks_final_perr
        return (gauss_fits_temp, gauss_err_temp)

    def _plot_hist(self):   # just plotting
        mpl.rcParams['figure.dpi']= 200
        plt.bar(*self.hist_data,color='royalblue',width=self.hist_params[0])
        plt.xlabel('peak amplitude [mV]')
        plt.ylabel('peaks [counts]')
        #plt.xlim(0,1000)

    def _plot_hist_fit(self):   # just more plotting
        for val in self.hist_fit_params:
            params, xy_tuple = val
            popt, perr = params
            plt.plot(*xy_tuple,'lime',label='Good fit',linewidth=1)
        
        for ((est_x, est_y), (real_x, real_y)) in zip(self.peaks_final[0],self.peaks_final[1]):
            plt.hlines([est_y], 0, est_x,linestyles="--",colors=['magenta'],linewidths=0.8)
            plt.hlines([real_y],0,real_x,linestyles="--",colors=['lime'],linewidths=1)
            plt.vlines([est_x], 0, est_y,linestyles="--",colors=['magenta'],linewidths=0.8)
            plt.vlines([real_x],0,real_y,linestyles="--",colors=['lime'],linewidths=1)
            plt.scatter([real_x],[real_y],50,color='green',marker='+')

    def _plot_gain(self):   # even more plotting
        data = []
        errs = []
        for (peak,err) in zip(self.peaks_final[1], self.peaks_final_perr):
            data.append(float(peak[0]))
            errs.append(float(err))
        plt.errorbar(np.arange(1,len(data)+1,1),data,yerr=errs,fmt='o',ms=5,color='red',label='Peaks')
        plt.xticks(np.arange(1,len(data)+1,1))
        plt.ylabel('peak amplitude [mV]')
        popt, perr = self.gain_fit_params
        plt.plot(np.arange(1,len(data)+1,1),self.line(np.arange(1,len(data)+1,1),*popt),color='black',ls="--",alpha=0.8)
        
    def plot(self, hist=False,hist_fit=False, gain=False):  # anyone said plotting?
        if self.error: self.__throw_error()
        plt.clf()
        mpl.rcParams['figure.dpi']= 200

        if hist and not gain:
            plt.figure(figsize=(10,10))
            plt.title('Peaks histogram')
            self._plot_hist()
            if hist_fit: self._plot_hist_fit()
        if gain and not hist:
            plt.figure(figsize=(10,10))
            plt.title(f'Gain: {self.gain_evaluation[0]:.2f} \u00b1 {float(self.gain_evaluation[1]):.2f}')
            self._plot_gain()
        if hist and gain:
            plt.figure(figsize=(10,5))
            plt.suptitle(f'Overvoltage: {float(self.voltages[0]):.02f}V',fontsize=20)
            plt.subplot(1,2,1)
            plt.title('Histogram')
            self._plot_hist()
            if hist_fit: self._plot_hist_fit()
            plt.subplot(1,2,2)
            plt.title(f'Gain: {self.gain_evaluation[0]:.2f} \u00b1 {float(self.gain_evaluation[1]):.2f}')
            self._plot_gain()

        plt.show()
            
    def gauss(self,x,a,mu,sigma): return (a*np.exp(-0.5*((x-mu)/sigma)**2))     # gaussian function

    def line(self,x,m,c): return (m*x)+c    # line function

    def line_mca(self,x,a,b): return a*(x-b)    # line function for MCA

    def rebin(self,hx,h,bins):
        """ DISCLAIMER: part or all of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """

        bins = int(bins)
        h_rebin=[]
        for i in range(int(len(h)/bins)):
            start_idx=i*bins
            end_idx=(i+1)*bins
            h_rebin.append(np.sum(h[start_idx:end_idx]))
        hx_rebin=range(len(h_rebin))
        return np.array(hx_rebin), np.array(h_rebin)

    def rebin_center(self,hx,h,bins):
        """ DISCLAIMER: part or all of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """

        h_rebin=[]
        hx_rebin=[]
        for i in range(int(len(h)/bins)):
            start_idx=i*bins
            end_idx=(i+1)*bins
            h_rebin.append(np.sum(h[start_idx:end_idx]))
            hx_rebin.append(np.mean(hx[start_idx:end_idx]))
        # hx_rebin=range(len(h_rebin))
        return np.array(hx_rebin), np.array(h_rebin)

    def __mca_fit_peaks2(self,min_peaks,total_bins, x,y, prominence=None, distance=None, plot=True):
        """ DISCLAIMER: part of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """

        x,y = self.rebin(x,y, total_bins)
        gain_temp=[]#reset the gain temp list here to store gain values for one file
        err_temp = []

        PROMINENCE = 1E3
        DISTANCE = 80/total_bins    # introduced to estimate a distance and save processing power! So set the distance to none and let this estimate it. You can try but sometimes it won't work. It depends on the time of acquisition too. NOTE change this as you wish.
        if not prominence is None: PROMINENCE= prominence 
        if not distance is None: DISTANCE=distance

        peaks,pdict=find_peaks(y,prominence=PROMINENCE,distance=DISTANCE)
        if len(peaks) < min_peaks: return False,False   # these parameters will not get us enough peaks
            
        #To avoid fitting the pedestal, we ignore the first peak. In case the pedestal isn't there, then first peak gets ignored. This shouldn't change gain or BV calculation
        first_pe_max=x[peaks[0]] # The x-value of the 3rd peak.Index=1 means the second peak will be used for getting fit parameters
        max_value=y[peaks[0]] # The height of the 3rd peak
        x_idx_array=(y<0.5*max_value) & (x>first_pe_max)# returns a boolean array where both conditions are true
        right_side_x= x[np.where(x_idx_array)[0][0]] #finding the first time where x_idx_array is True
        sigma_guess=np.abs(first_pe_max-right_side_x) #We need this to fit the width of the Gaussian peaks

        cut= (x < first_pe_max+sigma_guess) & (x > first_pe_max-sigma_guess) # This cut helps to fix the width of the peak-fit
        popt,pcov=curve_fit(self.gauss,x[cut],y[cut],p0=[max_value,first_pe_max,sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix
        #err_temp.append(np.sqrt(np.diag(pcov))[1])

        if plot:
            plt.figure(figsize=(12,2)) # Call the figure here
            plt.subplot(1,3,1) #This subplot will plot the position of the peaks and also the data
            plt.xlim(0,total_bins*3)
            plt.suptitle(f'P: {PROMINENCE}, D: {DISTANCE}, MIN: {min_peaks}, BINS: {total_bins}')
            # plt.ylim(0,50)
            plt.yscale('log')
            plt.plot(x[peaks],y[peaks],'*') # plot the peak markers
            plt.plot(x,y,lw=1) #plot the signal
            plt.plot(x[cut],self.gauss(x[cut],*popt),color='green',label='Fit',lw=2,alpha=0.5) # Here we plot the fit on the 2nd peak to see if everything looks ok.
    
        for i,peak in enumerate(peaks[2:]): #here we ignore the first peak because it could be the pedestal
            new_first_pe_max=x[peak] #x-value of the peak
            new_max_value=y[peak] #height of the peak
            new_x_idx_array=(y<0.5*new_max_value) & (x>new_first_pe_max) # returns a boolean array where both conditions are true
            new_right_side_x= x[np.where(new_x_idx_array)[0][0]] #finding the first time where x_idx_array is True
            new_sigma_guess=np.abs(new_first_pe_max-new_right_side_x) #We need this to fit the width of the Gaussian peaks


            new_cut= (x < new_first_pe_max+new_sigma_guess) & (x > new_first_pe_max-new_sigma_guess) # This cut helps to fix the width of the peak-fit
            popt_new,pcov_new=curve_fit(self.gauss,x[new_cut],y[new_cut],p0=[new_max_value,new_first_pe_max,new_sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix
            if plot: plt.plot(x[new_cut],self.gauss(x[new_cut],*popt_new),color='r',label='Fit',lw=3) # Here we plot the fit on all the peaks
            gain_temp.append(popt_new[1]) #Here we append the value of the peak fit mean
            err_temp.append(np.sqrt(np.diag(pcov_new))[1])

        if plot: plt.show()
        #print(f'Returning {gain_temp}')
        weird = y[peaks[2]]/total_bins<90
        return (err_temp, peaks)

    def __mca_fit_peaks(self,min_peaks,total_bins, file, prominence=None, distance=None):
        """ DISCLAIMER: part of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """
        print(f'MCA data from {file}')

        gain_list=[] #empty list to fill in the values of gain, returned at the end of this function
        gain_err=[] #empty list to fill in the values of gain fit error, returned at the end of this function
        calib_pe=[]#empty list to fill in the values for calibrated PE 
        calib_count=[]

        x, y = self.waveforms[file]
        x = x[500:]
        y = y[500:]
        x,y = self.rebin(x,y, total_bins)

        gain_temp=[]#reset the gain temp list here to store gain values for one file
        #Use scipy find_peaks to find peaks starting with a very high prominence 
        PROMINENCE = 1E3
        DISTANCE = 80/total_bins    # this was introduced 
        if not prominence is None: PROMINENCE = prominence 
        if not distance is None: DISTANCE =distance

        peaks,pdict=find_peaks(y,prominence=PROMINENCE,distance=DISTANCE)
        """peak_length=len(peaks)
        #We want to ensure that using a high prominence gives us at least N_peaks peaks to fit a straight line to. If it doesn't we reduce prominence till we get at least 3 peaks. N_peaks is set above
        while (peak_length < min_peaks+1) and prominence is None:
            PROMINENCE=PROMINENCE-1
            
            peaks,pdict=find_peaks(y,prominence=PROMINENCE,distance=DISTANCE)
            peak_length=len(peaks)"""
            
        #To avoid fitting the pedestal, we ignore the first peak. In case the pedestal isn't there, then first peak gets ignored. This shouldn't change gain or BV calculation
        first_pe_max=x[peaks[0]] # The x-value of the 3rd peak.Index=1 means the second peak will be used for getting fit parameters
        max_value=y[peaks[0]] # The height of the 3rd peak
        x_idx_array=(y<0.5*max_value) & (x>first_pe_max)# returns a boolean array where both conditions are true
        right_side_x= x[np.where(x_idx_array)[0][0]] #finding the first time where x_idx_array is True
        sigma_guess=np.abs(first_pe_max-right_side_x) #We need this to fit the width of the Gaussian peaks

    
        plt.figure(figsize=(12,2)) # Call the figure here
        plt.subplot(1,3,1) #This subplot will plot the position of the peaks and also the data
        #plt.xlim(0,1000/total_bins)
        plt.xlim(0,total_bins*3)
        # plt.ylim(0,50)
        plt.yscale('log')
        plt.plot(x[peaks],y[peaks],'*') # plot the peak markers
        plt.plot(x,y,lw=1) #plot the signal
        cut= (x < first_pe_max+sigma_guess) & (x > first_pe_max-sigma_guess) # This cut helps to fix the width of the peak-fit
        popt,pcov=curve_fit(self.gauss,x[cut],y[cut],p0=[max_value,first_pe_max,sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix

        plt.plot(x[cut],self.gauss(x[cut],*popt),color='green',label='Fit',lw=2,alpha=0.5) # Here we plot the fit on the 2nd peak to see if everything looks ok.
    
        
        for i,peak in enumerate(peaks[2:]): #here we ignore the first peak because it could be the pedestal
            new_first_pe_max=x[peak] #x-value of the peak
            new_max_value=y[peak] #height of the peak
            new_x_idx_array=(y<0.5*new_max_value) & (x>new_first_pe_max) # returns a boolean array where both conditions are true
            new_right_side_x= x[np.where(new_x_idx_array)[0][0]] #finding the first time where x_idx_array is True
            new_sigma_guess=np.abs(new_first_pe_max-new_right_side_x) #We need this to fit the width of the Gaussian peaks


            new_cut= (x < new_first_pe_max+new_sigma_guess) & (x > new_first_pe_max-new_sigma_guess) # This cut helps to fix the width of the peak-fit
            popt_new,pcov_new=curve_fit(self.gauss,x[new_cut],y[new_cut],p0=[new_max_value,new_first_pe_max,new_sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix
            plt.plot(x[new_cut],self.gauss(x[new_cut],*popt_new),color='r',label='Fit',lw=3) # Here we plot the fit on all the peaks
            gain_temp.append(popt_new[1]) #Here we append the value of the peak fit mean 
                
        #print(f'Returning {gain_temp}')
        weird = y[peaks[2]]/total_bins<90
        return (gain_temp,x,y, weird, peaks)

    def __mca_gain(self,vals, weird, total_bins, peaks, multiplication=None): #TODO understand why Y and total_bins are important here
        """ DISCLAIMER: part of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """

        if not multiplication is None: vals = np.asarray(vals)*multiplication
        plt.subplot(1,3,2) #This subplot shows the straight line fit to the peak means to obtain the slope/gain
        if weird: #TODO hande that sometimes this gives error!!!!
            popt_temp,pcov_temp=curve_fit(self.line_mca,np.arange(1,len(vals)+1),vals,p0=[10,0],maxfev=10000) #Use the straight line fit here
            plt.plot(np.arange(1,len(vals)+1),self.line_mca(np.arange(1,len(vals)+1),*popt_temp),color='k',label=(str(np.round(popt_temp[0],2)))+'$\pm$'+str(np.round(np.sqrt(np.diag(pcov_temp))[0],2))+' ADC/PE') # plot the straight line fit

            plt.scatter(np.arange(1,len(vals)+1),vals,color='r') #plot the values of the peak means
            plt.legend(loc=2)
        else:
            other = np.arange(2,len(vals)+1)
            popt_temp,pcov_temp=curve_fit(self.line_mca,np.arange(2,len(vals)),vals,p0=[10,0],maxfev=10000) #Use the straight line fit here
            print()
            plt.plot(np.arange(2,len(peaks)),self.line_mca(np.arange(2,len(vals)),*popt_temp),color='k',label=(str(np.round(popt_temp[0],2)))+'$\pm$'+str(np.round(np.sqrt(np.diag(pcov_temp))[0],2))+' ADC/PE') # plot the straight line fit
            plt.scatter(np.arange(2,len(vals)),vals,color='r') #plot the values of the peak means
            plt.legend(loc=2)
        
        gain = popt_temp[0] #append the gain values to obtain BV later
        gain_err = np.sqrt(np.diag(pcov_temp))[0] #append the straight line error fit

        #print(f'Returning {gain} and {gain_err}')
        return (gain,gain_err, popt_temp)

    def __calibrate_gain(self, popt_temp, x, y): #TODO understand why x is important and what popt[0]+[1] is

        calib_pe = x/popt_temp[0]+popt_temp[1]
        calib_count = y

        plt.subplot(1,3,3)#This subplot shows the calibrated PE spectra
        plt.plot(x/popt_temp[0]+popt_temp[1],y)
        plt.yscale('log')
        plt.xlim(0,5)
        plt.xticks(np.arange(0,5))
        plt.grid()
        #plt.suptitle(file, y=1.12)
        plt.show() #show the plot

        return (calib_pe, calib_count)

    def __mca_eval_gain(self,total_bins,min_peaks, best=False, min_bin=None, max_bin=None, min_prominence=None, max_prominence=None, min_distance=None, max_distance=None):
        """ DISCLAIMER: part of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """

        if total_bins is None: raise(ArgumentError('Specify a number as the total number of bins flag for gat.eval_gain method!'))

        print(f'MCA data -> peaks: {min_peaks}, bins: {total_bins}')
        
        calib_pe, calib_count_arr, gains, perrs = [], [], [], []

        for file in self.source_files:
            if best:
                best_fit = self.__find_best_MCA(file,min_peaks,min_bin,max_bin,min_prominence,max_prominence,min_distance,max_distance)
                if best_fit is None:
                    self.log(f'Skipping {file} --> COULD NOT LOAD ANY PEAKS!',3)
                    continue

                bins, prominence, distance = best_fit

                vals, x, y,weird, peaks = self.__mca_fit_peaks(min_peaks,bins,file,prominence=prominence,distance=distance)
                distances = []
                sorted_peaks = np.sort(peaks)
                for i in np.arange(4,len(sorted_peaks),1):
                    prev = np.abs(sorted_peaks[i-1]-sorted_peaks[i-2])
                    curr = np.abs(sorted_peaks[i]-sorted_peaks[i-1])
                    if prev > curr*1.4 or curr > prev*1.4 or curr > 14:
                        if i > 6:
                            print(f'Prev {prev}, curr {curr} -> trimming to {i}:')
                            #peaks = np.split(sorted_peaks, [i-2])
                            vals = np.split(vals, [i-2])[0]
                        else: continue
                gain, perr, gain_popt = self.__mca_gain(vals,weird, bins,peaks, multiplication=bins)
                calib_peak, calib_count = self.__calibrate_gain(gain_popt,x,y)

                calib_pe.append(calib_peak)
                calib_count_arr.append(calib_count)
                gains.append(gain)
                perrs.append(perr)
            else:
                vals, x, y,weird, peaks = self.__mca_fit_peaks(min_peaks,total_bins,file)
                gain, perr, gain_popt = self.__mca_gain(vals,weird, total_bins,peaks)
                calib_peak, calib_count = self.__calibrate_gain(gain_popt,x,y)

                calib_pe.append(calib_peak)
                calib_count_arr.append(calib_count)
                gains.append(gain)
                perrs.append(perr)

        return calib_pe, calib_count_arr, gains, perrs
        gain_list=[] #empty list to fill in the values of gain, returned at the end of this function
        gain_err=[] #empty list to fill in the values of gain fit error, returned at the end of this function
        calib_pe=[]#empty list to fill in the values for calibrated PE 
        calib_count=[]

        for file in self.source_files:
            x, y = self.waveforms[file]
            x,y = self.rebin(x,y, total_bins)

            gain_temp=[]#reset the gain temp list here to store gain values for one file
            #Use scipy find_peaks to find peaks starting with a very high prominence 
            PROMINENCE=1E3 #This prominence is re-set here to ensure that every file starts out with a high prominence
        
            peaks,pdict=find_peaks(y,prominence=PROMINENCE,distance=80/total_bins)
            peak_length=len(peaks)
            #We want to ensure that using a high prominence gives us at least N_peaks peaks to fit a straight line to. If it doesn't we reduce prominence till we get at least 3 peaks. N_peaks is set above
            while (peak_length < min_peaks+1):
                PROMINENCE=PROMINENCE-1
                
                peaks,pdict=find_peaks(y,prominence=PROMINENCE,distance=80/total_bins)
                peak_length=len(peaks)
                
            #To avoid fitting the pedestal, we ignore the first peak. In case the pedestal isn't there, then first peak gets ignored. This shouldn't change gain or BV calculation
            first_pe_max=x[peaks[0]] # The x-value of the 3rd peak.Index=1 means the second peak will be used for getting fit parameters
            max_value=y[peaks[0]] # The height of the 3rd peak
            x_idx_array=(y<0.5*max_value) & (x>first_pe_max)# returns a boolean array where both conditions are true
            right_side_x= x[np.where(x_idx_array)[0][0]] #finding the first time where x_idx_array is True
            sigma_guess=np.abs(first_pe_max-right_side_x) #We need this to fit the width of the Gaussian peaks

        
            plt.figure(figsize=(12,2)) # Call the figure here
            plt.subplot(1,3,1) #This subplot will plot the position of the peaks and also the data
            plt.xlim(0,1000/total_bins)
            # plt.ylim(0,50)
            plt.yscale('log')
            plt.plot(x[peaks],y[peaks],'*') # plot the peak markers
            plt.plot(x,y,lw=1) #plot the signal
            cut= (x < first_pe_max+sigma_guess) & (x > first_pe_max-sigma_guess) # This cut helps to fix the width of the peak-fit
            popt,pcov=curve_fit(self.gauss,x[cut],y[cut],p0=[max_value,first_pe_max,sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix

            plt.plot(x[cut],self.gauss(x[cut],*popt),color='green',label='Fit',lw=2,alpha=0.5) # Here we plot the fit on the 2nd peak to see if everything looks ok.
        
            
            for i,peak in enumerate(peaks[2:]): #here we ignore the first peak because it could be the pedestal
                new_first_pe_max=x[peak] #x-value of the peak
                new_max_value=y[peak] #height of the peak
                new_x_idx_array=(y<0.5*new_max_value) & (x>new_first_pe_max) # returns a boolean array where both conditions are true
                new_right_side_x= x[np.where(new_x_idx_array)[0][0]] #finding the first time where x_idx_array is True
                new_sigma_guess=np.abs(new_first_pe_max-new_right_side_x) #We need this to fit the width of the Gaussian peaks


                new_cut= (x < new_first_pe_max+new_sigma_guess) & (x > new_first_pe_max-new_sigma_guess) # This cut helps to fix the width of the peak-fit
                popt_new,pcov_new=curve_fit(self.gauss,x[new_cut],y[new_cut],p0=[new_max_value,new_first_pe_max,new_sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix
                plt.plot(x[new_cut],self.gauss(x[new_cut],*popt_new),color='r',label='Fit',lw=3) # Here we plot the fit on all the peaks
                gain_temp.append(popt_new[1]) #Here we append the value of the peak fit mean 

            plt.subplot(1,3,2) #This subplot shows the straight line fit to the peak means to obtain the slope/gain
            if (y[peaks[2]]/total_bins<90): 
                popt_temp,pcov_temp=curve_fit(self.line_mca,np.arange(3,len(peaks)+1),gain_temp,p0=[10,0],maxfev=10000) #Use the straight line fit here
                plt.plot(np.arange(3,len(peaks)+1),self.line_mca(np.arange(3,len(peaks)+1),*popt_temp),color='k',label=(str(np.round(popt_temp[0],2)))+'$\pm$'+str(np.round(np.sqrt(np.diag(pcov_temp))[0],2))+' ADC/PE') # plot the straight line fit

                plt.scatter(np.arange(3,len(peaks)+1),gain_temp,color='r') #plot the values of the peak means
                plt.legend(loc=2)
            else:
                popt_temp,pcov_temp=curve_fit(self.line_mca,np.arange(2,len(peaks)),gain_temp,p0=[10,0],maxfev=10000) #Use the straight line fit here
                plt.plot(np.arange(2,len(peaks)),self.line_mca(np.arange(2,len(peaks)),*popt_temp),color='k',label=(str(np.round(popt_temp[0],2)))+'$\pm$'+str(np.round(np.sqrt(np.diag(pcov_temp))[0],2))+' ADC/PE') # plot the straight line fit
                plt.scatter(np.arange(2,len(peaks)),gain_temp,color='r') #plot the values of the peak means
                plt.legend(loc=2)
            
            gain_list.append(popt_temp[0]) #append the gain values to obtain BV later
            gain_err.append(np.sqrt(np.diag(pcov_temp))[0]) #append the straight line error fit

            calib_pe.append(x/popt_temp[0]+popt_temp[1])
            calib_count.append(y)

            plt.subplot(1,3,3)#This subplot shows the calibrated PE spectra
            plt.plot(x/popt_temp[0]+popt_temp[1],y)
            plt.yscale('log')
            plt.xlim(0,5)
            plt.xticks(np.arange(0,5))
            plt.grid()
            plt.suptitle(file, y=1.12)
            plt.show() #show the plot

        self.mca_gain_eval = (np.array(calib_pe),np.array(calib_count),np.array(gain_list),np.array(gain_err))

    def mca_gain_corrected(self):
        """ DISCLAIMER: part of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """
        for x,y,z,r in zip(*self.mca_gain_eval): #z,r are useless and nothing but necessary to unpack with the star
            plt.figure(figsize=(12,2))
            plt.subplot(1,3,1)
            plt.plot(x,y)
            plt.yscale('log')
            plt.xlim(0,10)
            plt.grid()
            plt.xlabel('PE')
            plt.ylabel('Count')
            
            
            N_peaks=4
            PROMINENCE=1E3 #This prominence is re-set here to ensure that every file starts out with a high prominence
            
            peaks,pdict=find_peaks(y,prominence=PROMINENCE,distance=5)
            peak_length=len(peaks)
            
        #     We want to ensure that using a high prominence gives us at least N_peaks peaks to fit a straight line to. If it doesn't we reduce prominence till we get at least 3 peaks. N_peaks is set above
            while (peak_length<N_peaks+1):
                PROMINENCE=PROMINENCE-1
                    
                peaks,pdict=find_peaks(y,prominence=PROMINENCE,distance=5)
                peak_length=len(peaks)
                
            plt.plot(x[peaks],y[peaks],'*',ms=5) # plot the peak markers
            first_pe_max=x[peaks[2]] # The x-value of the 3rd peak.Index=1 means the second peak will be used for getting fit parameters
            max_value=y[peaks[2]] # The height of the 3rd peak
            x_idx_array=(y<0.5*max_value) & (x>first_pe_max)# returns a boolean array where both conditions are true
            right_side_x= x[np.where(x_idx_array)[0][0]] #finding the first time where x_idx_array is True
            sigma_guess=np.abs(first_pe_max-right_side_x) #We need this to fit the width of the Gaussian peaks
            cut= (x < first_pe_max+sigma_guess) & (x > first_pe_max-sigma_guess) # This cut helps to fix the width of the peak-fit
            popt,pcov=curve_fit(self.gauss,x[cut],y[cut],p0=[max_value,first_pe_max,sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix
            plt.plot(x[cut],self.gauss(x[cut],*popt),color='r',label='Fit',lw=2) # Here we plot the fit on the 2nd peak to see if everything looks ok.
            
            params_pe = []
            params_pe_count=[]
            for i,peak in enumerate(peaks[2:]): #here we ignore the first peak because it could be the pedestal
                new_first_pe_max=x[peak] #x-value of the peak
                new_max_value=y[peak] #height of the peak
                new_x_idx_array=(y<0.5*new_max_value) & (x>new_first_pe_max) # returns a boolean array where both conditions are true
                new_right_side_x= x[np.where(new_x_idx_array)[0][0]] #finding the first time where x_idx_array is True
                new_sigma_guess=np.abs(new_first_pe_max-new_right_side_x) #We need this to fit the width of the Gaussian peaks


                new_cut= (x < new_first_pe_max+new_sigma_guess) & (x > new_first_pe_max-new_sigma_guess) # This cut helps to fix the width of the peak-fit
                popt_new,pcov_new=curve_fit(self.gauss,x[new_cut],y[new_cut],p0=[new_max_value,new_first_pe_max,new_sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix
                plt.plot(x[new_cut],self.gauss(x[new_cut],*popt_new),color='r',label='Fit',lw=3) # Here we plot the fit on all the peaks
                params_pe.append(popt_new[1]) #Here we append the value of the peak fit mean 
                params_pe_count.append((x[peak]))

            plt.subplot(1,3,2) 
            plt.scatter(params_pe_count, np.array(params_pe), s=10, zorder=10)
            popt_line,pcov_line = curve_fit(self.line_mca, params_pe_count, np.array(params_pe), p0=[10,0])
            plt.plot(params_pe_count, self.line_mca(params_pe_count, *popt_line), color=colors[1], label='Gain: {:.2f} ADC/p.e.'.format(popt_line[0]))
            #NO_FIELD_pe_corrected_gain.append(popt_line[0])
            #NO_FIELD_pe_corrected_gain_err.append(np.sqrt(np.diag(pcov_line)[0]))
            plt.legend(loc=2)
            plt.grid()
            plt.xlabel('PE')
            plt.ylabel('Counts')



            plt.subplot(1,3,3)
            #NO_FIELD_calib_pe_updated.append(x/popt_line[0])
            plt.step(x/popt_line[0],y)
            plt.yscale('log')
            plt.xlim(0,10)
            plt.grid()
            plt.xlabel('PE')
            plt.ylabel('Count')



            plt.tight_layout()
            plt.show()

            #self.other()

    def other(self,data):
        """ DISCLAIMER: part of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """
        NO_FIELD_calib_pe,NO_FIELD_calib_count,NO_FIELD_gain_list,NO_FIELD_gain_err = data
        plt.figure(figsize=(6,4))

        time_NO_FIELD=np.arange(0,len(NO_FIELD_calib_pe))
        plt.errorbar(time_NO_FIELD,NO_FIELD_gain_list*9,yerr=NO_FIELD_gain_err,fmt='o',ms=7,label='Uncalibrated Gain')
        plt.errorbar(time_NO_FIELD,NO_FIELD_pe_corrected_gain,yerr=NO_FIELD_pe_corrected_gain_err,fmt='o',ms=7,label='PE Calibrated Gain')

        plt.ylabel('[ADC or PE]/PE')
        plt.xlabel('Time [minutes]')
        plt.title('NO_FIELD')
        plt.grid()
        plt.legend(loc='best')
        #plt.savefig('NO_FIELD_gain_comparison.pdf')
        plt.show()

    def gain_calculator(self,PATH,N_BINS,N_PEAKS):
        """ DISCLAIMER: part of this code was written and invented by Dr. Avinay Bhat to whom I am very thankful. """
        BINS=N_BINS #Number of bins to rebin the MCA data with
        N_peaks= N_PEAKS# Number o peaks to use for calculating the gain
        gain_list=[] #empty list to fill in the values of gain, returned at the end of this function
        gain_err=[] #empty list to fill in the values of gain fit error, returned at the end of this function
        calib_pe=[]#empty list to fill in the values for calibrated PE 
        calib_count=[]
        Files = glob.glob(PATH+'*mca*.h5')
        #for loop to loop over all the files
        for i,file in enumerate(natsorted(Files)):
            print(file) 
        
        
            f = h5py.File(file, 'r')  
            ch2 = f.get('ch2')
            for key in ch2.keys(): 
                df = np.array(ch2.get(key))
            h = df[250:]
            hx = np.arange(0,len(h),1)
            hx,h = self.rebin(hx,h, BINS)
            

            gain_temp=[]#reset the gain temp list here to store gain values for one file
            #Use scipy find_peaks to find peaks starting with a very high prominence 
            PROMINENCE=1E3 #This prominence is re-set here to ensure that every file starts out with a high prominence
        
            peaks,pdict=find_peaks(h,prominence=PROMINENCE,distance=80/BINS)
            peak_length=len(peaks)
            #We want to ensure that using a high prominence gives us at least N_peaks peaks to fit a straight line to. If it doesn't we reduce prominence till we get at least 3 peaks. N_peaks is set above
            while (peak_length<N_peaks+1):
                PROMINENCE=PROMINENCE-1
                
                peaks,pdict=find_peaks(h,prominence=PROMINENCE,distance=80/BINS)
                peak_length=len(peaks)

            #To avoid fitting the pedestal, we ignore the first peak. In case the pedestal isn't there, then first peak gets ignored. This shouldn't change gain or BV calculation
            first_pe_max=hx[peaks[0]] # The x-value of the 3rd peak.Index=1 means the second peak will be used for getting fit parameters
            max_value=h[peaks[0]] # The height of the 3rd peak
            x_idx_array=(h<0.5*max_value) & (hx>first_pe_max)# returns a boolean array where both conditions are true
            right_side_x= hx[np.where(x_idx_array)[0][0]] #finding the first time where x_idx_array is True
            sigma_guess=np.abs(first_pe_max-right_side_x) #We need this to fit the width of the Gaussian peaks

        
            plt.figure(figsize=(12,2)) # Call the figure here
            plt.subplot(1,3,1) #This subplot will plot the position of the peaks and also the data
            plt.xlim(0,1000/BINS)
            # plt.ylim(0,50)
            plt.yscale('log')
            plt.plot(hx[peaks],h[peaks],'*') # plot the peak markers
            plt.plot(hx,h,lw=1) #plot the signal
            cut= (hx < first_pe_max+sigma_guess) & (hx > first_pe_max-sigma_guess) # This cut helps to fix the width of the peak-fit
            popt,pcov=curve_fit(self.gauss,hx[cut],h[cut],p0=[max_value,first_pe_max,sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix
            plt.plot(hx[cut],self.gauss(hx[cut],*popt),color='green',label='Fit',lw=2,alpha=0.5) # Here we plot the fit on the 2nd peak to see if everything looks ok.
        
            
            for i,peak in enumerate(peaks[2:]): #here we ignore the first peak because it could be the pedestal
                new_first_pe_max=hx[peak] #x-value of the peak
                new_max_value=h[peak] #height of the peak
                new_x_idx_array=(h<0.5*new_max_value) & (hx>new_first_pe_max) # returns a boolean array where both conditions are true
                new_right_side_x= hx[np.where(new_x_idx_array)[0][0]] #finding the first time where x_idx_array is True
                new_sigma_guess=np.abs(new_first_pe_max-new_right_side_x) #We need this to fit the width of the Gaussian peaks


                new_cut= (hx < new_first_pe_max+new_sigma_guess) & (hx > new_first_pe_max-new_sigma_guess) # This cut helps to fix the width of the peak-fit
                popt_new,pcov_new=curve_fit(self.gauss,hx[new_cut],h[new_cut],p0=[new_max_value,new_first_pe_max,new_sigma_guess],maxfev=100000) # We use curve_fit to return the optimal parameters and the covariance matrix
                plt.plot(hx[new_cut],self.gauss(hx[new_cut],*popt_new),color='r',label='Fit',lw=3) # Here we plot the fit on all the peaks
                gain_temp.append(popt_new[1]) #Here we append the value of the peak fit mean 

            plt.subplot(1,3,2) #This subplot shows the straight line fit to the peak means to obtain the slope/gain
            if (h[peaks[2]]/BINS<90): 
                popt_temp,pcov_temp=curve_fit(self.line,np.arange(3,len(peaks)+1),gain_temp,p0=[10,0],maxfev=10000) #Use the straight line fit here
                plt.plot(np.arange(3,len(peaks)+1),self.line(np.arange(3,len(peaks)+1),*popt_temp),color='k',label=(str(np.round(popt_temp[0],2)))+'$\pm$'+str(np.round(np.sqrt(np.diag(pcov_temp))[0],2))+' ADC/PE') # plot the straight line fit

                plt.scatter(np.arange(3,len(peaks)+1),gain_temp,color='r') #plot the values of the peak means
                plt.legend(loc=2)
            else:
                popt_temp,pcov_temp=curve_fit(self.line,np.arange(2,len(peaks)),gain_temp,p0=[10,0],maxfev=10000) #Use the straight line fit here
                plt.plot(np.arange(2,len(peaks)),self.line(np.arange(2,len(peaks)),*popt_temp),color='k',label=(str(np.round(popt_temp[0],2)))+'$\pm$'+str(np.round(np.sqrt(np.diag(pcov_temp))[0],2))+' ADC/PE') # plot the straight line fit
                plt.scatter(np.arange(2,len(peaks)),gain_temp,color='r') #plot the values of the peak means
                plt.legend(loc=2)
    
            
            gain_list.append(popt_temp[0]) #append the gain values to obtain BV later
            gain_err.append(np.sqrt(np.diag(pcov_temp))[0]) #append the straight line error fit

            calib_pe.append(hx/popt_temp[0]+popt_temp[1])
            calib_count.append(h)
            
            plt.subplot(1,3,3)#This subplot shows the calibrated PE spectra
            plt.plot(hx/popt_temp[0]+popt_temp[1],h)
            plt.yscale('log')
            plt.xlim(0,5)
            plt.xticks(np.arange(0,5))
            plt.grid()
            plt.show() #show the plot

        return(np.array(calib_pe),np.array(calib_count),np.array(gain_list),np.array(gain_err))

    def log(self, message, level, notify=0):
        """ Logging class. please refrain from using print statements in the code. """
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)

        if self.silenced: return False

        time_format = "%I:%M:%S%p"

        if level == 0:
            print(ANSI_RESET,f'[{datetime.now().strftime(time_format)} - {calframe[1][3]}]: ',message,ANSI_RESET)
        elif level == 1:
            print(ANSI_GREEN,f'[{datetime.now().strftime(time_format)} - {calframe[1][3]}]: ',message,ANSI_RESET)
        elif level == 2:
            print(ANSI_YELLOW,f'[{datetime.now().strftime(time_format)} - {calframe[1][3]}]: ',message,ANSI_RESET)
        elif level == 3:
            print(ANSI_RED,f'[{datetime.now().strftime(time_format)} - {calframe[1][3]}]: ',message,ANSI_RESET)
        elif level == 4 and self.debug:
            print(ANSI_CYAN,f'[{datetime.now().strftime(time_format)} - {calframe[1][3]}]: ',message,ANSI_RESET)

        return True

    def update(self, etc,task):
        """ This would update a website in real time from which I could check how the code was running while away from my desk. """
        try: requests.get('http://tizianobuzz.pythonanywhere.com/logger/update/'+str(etc)+'/'+str(task))
        except Exception: pass

    def notify(self,title,message,priority,users=['Tiziano']):
        """ This code is able to send push notifications to my phone to notify me of the execution of the code. """
        if not self.notifications: return False
        accounts = {'Tiziano': "hwr043mln2xft1y"}
        url = "https://alertzy.app/send"
        status = True
        for user in users:
            try: params = {'accountKey': accounts[user], 'title' : title, 'message': message, 'priority': priority}
            except: status = False
            else: requests.post(url, data=params)
        return status

