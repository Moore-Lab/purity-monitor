#!/usr/bin/env python

"""Provides Gain Analysis Tools for both MCA and raw data.

Implements best-fit recognition and automated error detection for optimal usability.

Multi-core processing should be implemented when instantiating this class. Might be here in the future.
"""

# IMPORTS
import sys, glob,importlib, latex, itertools, os
sys.path.insert(0,'../../')
sys.path.insert(0,'/Library/TeX/texbin/')
sys.path.insert(0,'../../WaveformAnalysis')
import numpy as np
import Dataset as Dataset
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
import requests
from scipy.special import erfc
from ipywidgets import IntProgress
from IPython.display import display
import scipy.special
from multiprocessing import Process, Manager, Pool
import tracemalloc

importlib.reload(Dataset)
importlib.reload(SiPM)
importlib.reload(Waveform)

__author__ = "Tiziano Buzzigoli"
__credits__ = ["Tiziano Buzzigoli", "Peter Maxwell", "Gavin Huttley",
                    "Matthew Wakefield"]
__version__ = "1.0.1"
#__maintainer__ = "Rob Knight"
#__email__ = "rob@spot.colorado.edu"
__status__ = "Development"

# LOGGER SETTINGS AND VARIABLES
ANSI_RESET = '\u001b[0m'
ANSI_GREEN = '\u001b[32m'
ANSI_CYAN = '\u001b[36m'
ANSI_RED = '\u001b[31m'
ANSI_YELLOW = '\u001b[33m'
ANSI_BG_RED = '\u001b[41m'
ANSI_BG_GREEN = '\u001b[42m'

olderr = np.seterr(all='ignore') 
scipy.special.seterr(all='ignore')
import warnings
warnings.filterwarnings("ignore")

pp = pprint.PrettyPrinter(indent=4)

# MATPLOTLIB SETTINGS
plt.style.use('../../style.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams['figure.dpi']= 200

class Gat_HPC:  #   Gain Analysis Tool

    CHANNEL = 0 # TODO SET A CHANNEL FROM WHICH TO TAKE WAVEFORM DATA
    source_files = []
    max_bins = 10000
    voltages = []
    error = True
    SiPMs = {}
    shaping_time = 1e-5
    waveforms = {}
    waveform_fit_params = []
    hist_params = []
    hist_data = ([],[])
    hist_fit_params = []
    peaks_final = None
    peaks_final_perr = None
    gain_evaluation = None
    gain_fit_params = []
    tau_est_final = None

    def __init__(self, base_regex,filename_regex,voltage, shaping_time=None, silenced=False,debug=False,force=False,notify=False):
        tracemalloc.start()
        self.update(60,'INIT')
        self.source_files = glob.glob(base_regex+filename_regex)
        self.voltages = [x.split('_')[-2] for x in self.source_files]
        self.voltages = np.array(sorted([float(x.split('OV')[0]) for x in self.voltages]))
        self.voltages = np.unique(self.voltages)

        try: voltage = float(voltage)
        except:
            error = True
            raise(TypeError(f'Inappropriate argument provided for voltage: int or float expected but {voltage} ({type(voltage)}) was given.'))
        else:
            if not voltage in self.voltages:
                error = True
                raise(ValueError(f'The voltage you requested ({voltage}) is not available among the files you selected.'))
            else:
                self.voltages = [voltage]
        self.notifications = notify

        if self.source_files is None or len(self.source_files) == 0:
            error = True
            raise(FileNotFoundError('Unable to locate source files located at '+(base_regex+filename_regex)))
        elif self.voltages is None or len(self.voltages) == 0:
            self.error = True
            raise(FileNotFoundError('The files you are trying to load are missing voltage information in their names i.e.'+self.source_files[0]))
        elif not shaping_time is None and not type(shaping_time) is type(1e-1):
            self.error = True
            raise(ValueError('Shaping time must be a number. Only one shaping time per Gat object is allowed'))
        else:
            self.error = False
            self.silenced = silenced
            self.debug = debug
            self.shaping_time = shaping_time
            self.load_files(base_regex,filename_regex,force=force)
            self.log(f'{len(self.source_files)} files and {len(self.voltages)} voltages loaded',1)

    def __throw_error(self):
        raise(RuntimeError('The Gat object you are attempting to call has experienced an unexpected error and is unable to perform the method you requried at this time'))

    def load_files(self,base_regex,filename_regex,force=False):
        if self.error: self.__throw_error()
        if len(self.SiPMs.keys()) > 0 and not force:
            self.log('WARNING: files already loaded! Use the force optional argument to force load.',2)
            return
        if len(self.SiPMs.keys()) > 0 and force:
            self.log('Forcing to reload files....',2)

        for volt in self.voltages:
            self.SiPMs[volt] = SiPM.SiPM(Path=base_regex, Selection=filename_regex+'_{:.2f}OV*.h5'.format(volt))
            self.log(f'Loading {filename_regex}_{volt:.2f}OV*.h5 returned {len(self.SiPMs[volt].Files)} files',4)
            self.SiPMs[volt].Ch = [Waveform.Waveform(ID=x, Pol=1) for x in range(1,3)]

            for file in natsorted(self.SiPMs[volt].Files):
                print(f'Memory usage (traced): {tracemalloc.get_traced_memory()[0]/1000000:.02f} MB',end='\r')
                self.SiPMs[volt].ImportDataFromHDF5(file, self.SiPMs[volt].Ch, var=[])
                self.SiPMs[volt].get_sampling()
                if not self.shaping_time is None: self.SiPMs[volt].shaping_time=[self.shaping_time]
                self.SiPMs[volt].Ch[self.CHANNEL].SubtractBaseline(Data=self.SiPMs[volt].Ch[self.CHANNEL].Amp, cutoff=150)
                #self.SiPMs[volt].setup_butter_filter() # calculate the butterworth filter coefficients
        
    def eval_waveform_func_fit(self, voltage, bounds=None,reload=False,fix_params=False):
        #raise(NotImplementedError('Not ready yet!'))
        if self.error: self.__throw_error()
        voltage = float(voltage)
        if type(voltage) == float: pass
        else:
            self.error = True
            raise ValueError('Parameter voltage must be of type float or int. '+str(type(voltage))+' found instead')

        if bounds is None: bounds = [0,len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp[0])]

        try_loading = self.__open_coords(voltage)
        if not try_loading or reload:

            coords = [] #   x,y coordinate arrays organized in one tuple per waveform
            pks_tally = 0 #     keep track of how many peaks are found
            lost = 0

            self.log(f'Getting {len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp)} waveforms for {voltage:.2f}V [{bounds[0]}:{bounds[1]}]',1)

            f = IntProgress(value=0,min=0,max=len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp),step=1,description='Loading fit...',bar_style='info',layout={"width": "100%"})
            display(f)

            diffs = []

            if fix_params:
                self.__resolve_wf_fit_params(voltage)

            for waveform in self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp: #loop over the waveforms inside the file
                x = [] #    must be same lenght as y. NOTE: array of arrays due to some waveforms having multiple peaks (found with other methods).
                y = [] #    must be same length as x. NOTE: array of arrays due to some waveforms having multiple peaks (found with other methos).
                f.value += 1
                start = datetime.now()
                p0 = self.__p0estimate(waveform,voltage)
                
                if p0 is None:
                    self.log('Error loading p0',4)
                    lost += 1
                    continue
                
                popt,pcov = None, None
                try:
                    popt,pcov = curve_fit(self.__wave_func,self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,waveform,p0=p0,maxfev=1000000)
                except Exception as e:
                    self.log(e,3)
                else:
                    if popt is None or pcov is None:
                        lost += 1
                    else:
                        func = self.__wave_func(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,*popt)
                        peak_x = 0
                        print(f'Max {np.asarray(func).max()} ({np.asarray(waveform).max()}) while {popt}')
                        print(peak_x)
                        max_x,max_y = self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[peak_x], func[peak_x]
                        if max_x < bounds[0] or max_x > bounds[1]: self.log(f'Skipping due to peak out of bounds at {max_x}x{max_y}',4)
                        self.log(f'Max {max_x}x{max_y}',4)
                        perr = np.sqrt(np.diag(pcov))[1]

                        if perr > 1.0:
                            lost += 1
                            self.log(f'Perr {perr}',4)
                            possible_pks = find_peaks(waveform,prominence=40)[0]
                            """ plt.title(f'P_error: {perr:.5f}')
                            if len(possible_pks) > 1:
                                for p in possible_pks:
                                    ppk_x = self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[p]
                                    ppk_y = waveform[p]
                                    ax.scatter([ppk_x],[ppk_y],50,color='magenta',marker=11,zorder=4)
                                ax.scatter([max_x],[max_y],50,color='blue',marker='+',label='Fit max',zorder=3)
                                ax.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,func,color='red',alpha=0.8,zorder=2)
                                ax.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,waveform,zorder=1)
                                fig.canvas.draw()
                                fig.canvas.flush_events()
                            else:
                                continue
                                plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,waveform,color='purple')
                                plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,func,color='lime',alpha=0.8)
                                plt.scatter([max_x],[max_y],50,color='red',marker='+',label='Fit max')
                                plt.close() """
                        else:
                            #plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,waveform)
                            #plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,func)
                            #plt.show()
                            self.waveform_fit_params.append((popt,perr))
                            x.append([max_x])
                            y.append([max_y])
                            pks_tally += 1
                            coords.append((x,y))
                
                end = datetime.now()
                diffs.append((end-start).total_seconds())
                if f.value % 30 == 0: self.update(int(((np.mean(diffs)*(len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp)-f.value)))),'FUNC%20FIT')
                print(f'Estimated time to completion: {float(np.mean(diffs)*(len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp)-f.value)/60):.01f} minutes',end='\r')

                if len(diffs) == 5: self.notify('Initated fit calc.',f'ETC: {float(np.mean(diffs)*(len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp)-f.value)/60):.01f} minutes',0)
            
            if len(x) != len(y): raise(RuntimeError('Returning list of peak coordinates of different lengths. x: '+str(len(x))+ ' y: '+str(len(y))+' for '+str(len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp))+' waveforms'))

            self.log(f'{pks_tally} peaks found for {len(self.SiPMs[voltage].Ch[self.CHANNEL].Amp)} waveforms. Loss: {lost} ({(float(lost/pks_tally))*100:.02f}%)',1)

            self.__save_coords(coords,voltage)
            return (self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp, coords)
        elif not reload:
            self.log('Loading from backup file',1)
            return (self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp, try_loading)

    def __resolve_wf_fit_params(self, voltage):
        total = len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp)
        temp = self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp[1]
        double = 0
        late = 0

        

        for waveform in self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp[1:]:
            pks = find_peaks(waveform,prominence=150,distance=30)[0]
            if len(pks) > 1:
                #plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,waveform)
                for pk in pks:
                    plt.vlines([self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[pk]],0,waveform[pk],colors=['lime'],linestyles='dashed')
                #plt.show()
                double += 1
            elif len(pks) == 1:
                if self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[pks] > 240: #TODO CHECK THIS LIMIT
                    #plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,waveform,color='magenta')
                    #plt.show()
                    late += 1
            for ii, y in enumerate(waveform): temp[ii] = temp[ii] + y
        avg = [float(t/total) for t in temp]
        avg = np.asarray(avg)
        peak_x = np.argmax(avg)
        peak_A = avg[peak_x]
        tau_ind = np.where(avg[peak_x:] < (peak_A)*(np.e**-1))[0][0]
        tau_est = np.abs(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[np.where(avg[peak_x:] < (peak_A)*(np.e**-1))[0][0]] - self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[peak_x])
        tau_ind_est = tau_ind-peak_x

        #plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,avg)
        #plt.xlim(180,tau_ind+20)
        #plt.show()
        #plt.plot(np.arange(0,len(avg),1),avg)
        #plt.xlim(np.arange(0,len(avg),1)[180],np.arange(0,len(avg),1)[tau_ind+20])
        #plt.show()
        self.log(f'Found tau {tau_est} us to get to {(peak_A)*(np.e**-1)}. Using {total} waveforms for {voltage:.02f} V. Tau-ind-est is {tau_ind_est} waveform indexes',1)
        self.log(f'Double {double} ({double/total*100}%), late {late} ({late/total*100}%) out of {total} waveforms.',2)
        
        self.tau_est_final = tau_est
        return tau_est  

    def __save_coords(self, coords,voltage):
        filename = f'_backup_{voltage:.01f}.bd'
        with open(filename,'w') as file:
            for coordinates in coords:
                x = coordinates[0][0]
                y = coordinates[1][0]
                str_x = ''
                str_y = ''
                for x_coord in x: str_x += str(float(x_coord))+'%'
                for y_coord in y: str_y += str(float(y_coord))+'%'
                file.write(str_x[:-1]+'\n'+str_y[:-1]+'\n'+'@@@@'+'\n')
            file.close()
        return True
    
    def __open_coords(self,voltage):
        filename = f'_backup_{voltage:.01f}.bd'
        coords = []
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                lines = file.readlines()
                lines = [line.replace('\n','') for line in lines]
                #pp.pprint(lines)
                for ii, line in enumerate(lines):
                    if str(line) == '@@@@':
                        str_x = lines[ii-2]
                        str_y = lines[ii-1]

                        x_vals = str_x.split('%')
                        y_vals = str_y.split('%')

                        x = [float(val) for val in x_vals]
                        y = [float(val) for val in y_vals]

                        coords.append((x,y))

                file.close()
                #pp.pprint(coords)
                return coords
                #for line in file.readlines():
        else: return False
                    
    def __p0estimate(self,waveform,voltage):
        temp_peaks_ind = np.asarray(find_peaks(waveform,prominence=30)[0])
        if len(temp_peaks_ind) == 0 or temp_peaks_ind is None:
            self.log('Error loading p0: could not find any peaks',4) 
            #plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,waveform,color='magenta')
            #plt.show()
            return None
        
        LIMIT = 800
        temp_peaks_ind = temp_peaks_ind[(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[temp_peaks_ind]<LIMIT)]
        if len(temp_peaks_ind) == 0:
            plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,waveform,color='magenta')
            plt.show()
            self.log('Error loading p0: could not find any peaks below '+str(LIMIT),4) 
            return None
        tallest_peak_ind = temp_peaks_ind[np.argmax([waveform[p] for p in temp_peaks_ind])]
        A_est = waveform[tallest_peak_ind]
        mu_est = self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[tallest_peak_ind]-6
        base_est = float(np.mean(waveform[1:10]))
        try:
            tau_est = np.abs(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[np.where(waveform[tallest_peak_ind:] < (A_est)*(np.e**-1)+base_est)[0][0]] - self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time[tallest_peak_ind])
        except IndexError as e:
            self.log(e,4)
            return None
        except Exception as e: #when is index 0 out of bounds it is because the np.where function does not return any true values because that condition is never meth because it is not a regular wave. Probably just noise.
            self.log(f'Estimate error ({type(e)}: {e}',3)
            plt.plot(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Time,waveform,color='brown')
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
        return base + A/2.0 * np.exp(0.5 * (sigma/tau)**2 - (x-mu)/tau) * erfc(1.0/np.sqrt(2.0) * (sigma/tau - (x-mu)/sigma))

    def eval_waveform_argmax(self, voltage, bounds=None):
        if self.error: self.__throw_error()
        voltage = float(voltage)
        if type(voltage) == float: pass
        else:
            self.error = True
            raise ValueError('Parameter voltage must be of type float or int. '+str(type(voltage))+' found instead')

        if bounds is None: bounds = [0,len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp[0])]

        coords = [] #   x,y coordinate arrays organized in one tuple per waveform
        pks_tally = 0 #     keep track of how many peaks are found

        self.log(f'Gettig {len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp)} waveforms for {voltage:.2f}V [{bounds[0]}:{bounds[1]}]',1)
        
        for waveform in self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp: #loop over the waveforms inside the file
            filtered = self.SiPMs[float(voltage)].get_filtered_waveform(waveform) # return the filtered waveform
            x = [] #    must be same lenght as y. NOTE: array of arrays due to some waveforms having multiple peaks (found with other methods).
            y = [] #    must be same length as x. NOTE: array of arrays due to some waveforms having multiple peaks (found with other methos).
            
            for i,y1 in enumerate(filtered): #Iterates each wave filtered with a different shaping time
                peak_index = np.argmax(y1[bounds[0]:bounds[1]])+bounds[0] #Finds the index for max value in the waveform amplitude array
                peak_value = y1[peak_index] #Gets the max value from the array of amps
                pks_tally += 1
                x.append(peak_index)
                y.append(peak_value)
            
            coords.append((x,y))
        
        if len(x) != len(y): raise(RuntimeError('Returning list of peak coordinates of different lengths. x: '+str(len(x))+ ' y: '+str(len(y))+' for '+str(len(self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp))+' waveforms'))

        self.log(f'{pks_tally} peaks found for {len(self.SiPMs[voltage].Ch[self.CHANNEL].Amp)} waveforms.',0)

        return (self.SiPMs[float(voltage)].Ch[self.CHANNEL].Amp, coords)
  
    def peaks_test(self, voltage, maxfev=1000000):
        pass

    def eval_gain(self, voltage, waveform_fit=True,find_best=False,bin_width=5,distance=12,prominence=30,min=2,plot=False,fix_params=False):
        if self.error: self.__throw_error()
        if not float(voltage) in self.voltages:
            self.log('Voltage not available, asked for: '+str(voltage),3) #TODO IMPLEMENT A METHOD THAT CHECKS AND HANDLES VOLTAGES AND CONVERTS THEM
            return False

        peaks_coords = None
        if waveform_fit:
            peaks_coords = self.eval_waveform_func_fit(voltage,fix_params=fix_params)[1]
        else: peaks_coords = self.eval_waveform_argmax(voltage)[1]

        popt_list, perr_list = None, None
        peaks_coords = self.__peak_filter(peaks_coords)

        if not find_best:
            popt_list,perr_list = self.__get_peaks(peaks_coords,bin_width,prominence,distance,min,plot=plot)
        else:
            #   raise(NotImplementedError('Find best has not been implemented yet! Sorry ;)'))
            popt_list,perr_list = self.__find_best(peaks_coords)
        

        if popt_list is False:
            self.log(f'MAJOR ERROR: Cannot find best paramaters for find_peaks function! Cannot proceed. Review the bound.',3)
            return None

        if popt_list is None:
            self.log(f'MAJOR ERROR: Cannot find any peaks! Cannot proceed. Review the bound.',3)
            return None

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
        gain = self.__gain_from_peaks(popt_list,perr_list)
        self.gain_evaluation = gain
        print(f'Memory usage: {tracemalloc.get_traced_memory()[1]/1000000} MB peak')
        tracemalloc.stop()
        return gain

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
        fits, errs = [], []

        bin_width_start = 1
        bin_width_stop = 12
        bin_width_step = 0.5

        prominence_start = 10
        prominence_stop = 100
        prominence_step = 5

        distance_start = 10
        distance_stop = 90
        distance_step = 2

        min_peaks = 5
        max_error = 2.0
        
        #trials = {}
        last = None
        
        best = None
        temp_errs = []

        skipped = 0
        done = 0

        perms = [item for item in itertools.product(np.arange(bin_width_start, bin_width_stop,bin_width_step),np.arange(prominence_start, prominence_stop,prominence_step),np.arange(distance_start, distance_stop,distance_step))]
        
        f = IntProgress(value=0,min=0,max=len(perms),step=1,description='Loading perms...',bar_style='info',layout={"width": "100%"})
        display(f)
        self.log('Multiprocessing does not support progress bars. Disregard it.',2)
        args = []


        with Manager() as manager:
            trials = manager.dict()
            start = datetime.now()

            for perm in perms:
                args.append([perm, peaks_coords,min_peaks,trials,max_error])

            pool = Pool()
            pool.starmap(self.perm_run, args)
            pool.close()

            for perm in trials.keys():
                popt_list, perr_list = trials[perm]
                last = perm
                num_peaks = len(popt_list)
                combined = float(np.sum(perr_list))/num_peaks
                if not True in np.asarray(perr_list) > max_error:
                    if num_peaks >= min_peaks: temp_errs.append(combined)
                else: self.log('Not counting for too big err'+str(perr_list),4)
                if True in np.asarray(temp_errs) < combined: continue
                else:
                    self.log(np.asarray(temp_errs) < combined,4)
                    self.log(temp_errs,4)
                    self.log(f'Combined/peaks {combined}',4)
                    best = perm
                done += 1
                self.log(f'Skipped {skipped} of {done+skipped}',4)

            end = datetime.now()
            self.log(f'Runtime: {float(np.mean((end-start).total_seconds())):.01f} seconds',1)
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

            if not best is None: self.log(f'Best found at prominence {best[1]}, distance {best[2]}, bin width {best[0]} for {len(trials)} trials',1)
            else:
                self.log('Fatal error: could not load any peaks!',3)
                return (False,False)

            return self.__get_peaks(peaks_coords,*best,min_peaks,plot=True, is_best=True)
    
    def perm_run(self, perm, peaks_coords, min_peaks,trials, max_error):
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

    def __peak_filter(self, peaks_coords): #TODO CHECK THAT PEAK IS EXTRACTED FROM FUNCTIONA AND NOT ARG_MAX!!!
        temp = []
        avg = 0
        for wvf_tuple in peaks_coords:
            x,y = wvf_tuple
            max_index = np.argmax(y)
            avg = avg + float(x[max_index])
            temp.append((x[max_index],y[max_index]))
        self.log(f'Average peak mu is {avg/len(peaks_coords)} for {avg} and {len(peaks_coords)}',2)
        return temp

    def __get_peaks(self, peaks_filtered, bin_width, prominence, distance, peaks_min,max_bins=1000,plot=False,is_best=False):

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

    def _plot_hist(self):
        mpl.rcParams['figure.dpi']= 200
        plt.bar(*self.hist_data,color='royalblue',width=self.hist_params[0])
        plt.xlabel('peak amplitude [mV]')
        plt.ylabel('peaks [counts]')
        #plt.xlim(0,1000)

    def _plot_hist_fit(self):
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

    def _plot_gain(self):
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
        
    def plot(self, hist=False,hist_fit=False, gain=False):
        if self.error: self.__throw_error()
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
            
    def gauss(self,x,a,mu,sigma): return (a*np.exp(-0.5*((x-mu)/sigma)**2))

    def line(self,x,m,c): return (m*x)+c

    def log(self, message, level, notify=0):
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
        try: requests.get('http://tizianobuzz.pythonanywhere.com/logger/update/'+str(etc)+'/'+str(task))
        except Exception: pass

    def notify(self,title,message,priority,users=['Tiziano']):
        if not self.notifications: return False
        accounts = {'Tiziano': "hwr043mln2xft1y"}
        url = "https://alertzy.app/send"
        status = True
        for user in users:
            try: params = {'accountKey': accounts[user], 'title' : title, 'message': message, 'priority': priority}
            except: status = False
            else: requests.post(url, data=params)
        return status