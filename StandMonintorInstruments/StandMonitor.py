import numpy as np
import time, datetime, sys, os, glob, struct
import matplotlib
import matplotlib.pyplot as plt
from optparse import OptionParser
import h5py
import datetime
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MultipleLocator

colors = ['#1f78b4', '#e66101', '#d62728', '#33a02c']

class SensorData:
    def __init__(self, Filepath='', PlotTime=0):
        self.Filepath = Filepath
        
        self.Data = {}
        self.Index = np.arange(0,22,1) # creating a list with indices with respect to the labels
        self.PlotTime = PlotTime
        self.StartTime = datetime.datetime.now()
        self.Labels = ['Gas System', 'Chamber', 'Stainless-steel Cylinder 1', 'Stainless-steel Cylinder 2', 'LN Dewar 1', 'LN Dewar 2', 'Xenon Pump', 'Flow Meter', 'Back Pump', 'Cold Head', 'Copper Ring', 'Copper Jacket', 'TPC Bottom', 'dummy', 'dummy', 'dummy','Compressor', 'Inlet', 'Outlet', 'dummy', 'dummy', 'dummy', 'Time']
        # ['Gas System', 'Chamber', 'Stainless-steel Cylinder 1 (R)', 'Stainless-steel Cylinder 2 (L)', 'LN Dewar 1 (R)', 'LN Dewar 2 (L)', 'Flow Meter', '?', 'Circulation Pump', 'Cold Head', 'Copper Ring', 'Copper Jacket', 'TPC Bottom', '?', '?', '?','Compressor', 'Inlet', 'Outlet', 'dummy', 'dummy', 'dummy', '?']
        
    def GetData(self, Selection=None, Return=None):
        self.File = h5py.File(self.Filepath, 'r')
        self.Key = list(self.File.keys())[0]
        self.RawData = np.array(self.File[self.Key])
        self.Date = os.path.split(self.Filepath)[1][:-3]
        self.RefTime, self.DateTime = self.GetDateFromInput(self.Date)

        if Selection is None: 
            for ii,Label in enumerate(self.Labels):
                if Label == 'Time':
                    self.Seconds = [(x - self.RefTime - 14400 - 3600) for x in self.RawData[:,15]]
                    self.Time = np.asarray([self.DateTime + datetime.timedelta(seconds=x) for x in self.Seconds])
                else:
                    self.Data[Label] = self.RawData[:,self.Index[ii]]
        else: 
            SelectionIndex = np.where(self.Labels == Selection)[0][0]
            self.Data[Selection] = self.RawData[:,self.Index[SelectionIndex]]
        self.File.close()

        if Return is not None:
            self.Data['Time'] = self.Time
            return self.Data
    
    def GetColdData(self):
        '''Returning the files that have recorded data when the compressor is on'''
        Compressor = self.Data['Compressor']
        if (Compressor > 50).any(): # when compressor temperature is above 50 C
            return self.Filepath

    def GetXenonData(self):
        '''Returning the files that have recorded data when the xenon is by the pressure reader'''
        XenonPressure = np.concatenate((self.Data['Stainless-steel Cylinder 1'], self.Data['Stainless-steel Cylinder 2']))
        if (XenonPressure > 300).any(): # when xenon pressure is above 300 PSI
            return self.Filepath

    def Combine(self, Sensors): 
        Temp = tuple([np.array(Sensor.ReturnData('Temperature')) for Sensor in Sensors])
        self.Temp = np.concatenate(Temp,axis=1)

        XPressure = tuple([np.array(Sensor.ReturnData('Xenon Pressure')) for Sensor in Sensors])
        self.XPressure = np.concatenate(XPressure,axis=1)

        SPressure = tuple([np.array(Sensor.ReturnData('System Pressure')) for Sensor in Sensors])
        self.SPressure = np.concatenate(SPressure,axis=1)

        Compressor = tuple([np.array(Sensor.ReturnData('Compressor')) for Sensor in Sensors])
        self.Compressor = np.concatenate(Compressor,axis=1)

        self.Time = np.concatenate(tuple([Sensor.Time for Sensor in Sensors]), axis=0)
    
        self.RefTime, self.DateTime = Sensors[0].RefTime, Sensors[0].DateTime
        return

    def GetDateFromInput(self, Date):
        dd = list(Date)
        year = int(dd[0]+dd[1]+dd[2]+dd[3])
        month = int(dd[4]+dd[5])
        day = int(dd[6]+dd[7])
        dt = datetime.datetime(year,month,day,00,00,00)
        tmp = datetime.datetime(1904,1,1,0,0)
        at = int((dt - tmp).total_seconds())
        return at, dt
    
    def ReturnData(self, Selection='Temperature'):
        if Selection == 'Temperature': 
            Tags = self.Labels[9:13]
            Data = [self.Data[x] for x in Tags]
        elif Selection == 'Xenon Pressure': 
            Tags = self.Labels[2:4]
            Data = [self.Data[x] for x in Tags]
        elif Selection == 'System Pressure': 
            Tags = self.Labels[0:2]
            Data = [self.Data[x] for x in Tags]
        elif Selection == 'Compressor': 
            Tags = self.Labels[16:19]
            Data = [self.Data[x] for x in Tags]
        return Data

    def PlotData(self, Data, Selection='Temperature', Time=None, XYLabels=None, Labels=None, Tags=None, XRange=0, YRange=[1,1], YTicks=10, XTicks=2, Bin=1):
        if Selection == 'Temperature': 
            XYLabels = ['Time [hh:mm]', 'Temperature [C]']
            Tags = self.Labels[9:13] # labels of the different temperature sensors ('Cold Head', 'Copper Ring', 'Copper Jacket', 'TPC Bottom')
            # Data = [self.Data[x] for x in Tags]
        elif Selection == 'Xenon Pressure': 
            XYLabels = ['Time [hh:mm]', 'Pressure [PSIG]']
            Tags = self.Labels[2:4] # labels of the two xenon cylinders ('Stainless-steel Cylinder 1', 'Stainless-steel Cylinder 2')
            # Data = [self.Data[x] for x in Tags]
        elif Selection == 'System Pressure':  
            XYLabels = ['Time [hh:mm]', 'Pressure [PSIG]']
            Tags = self.Labels[0:2] # labels of the two pressure sensors ('Gas System', 'Chamber')
            # Data = [self.Data[x] for x in Tags]
        elif Selection == 'Compressor': 
            XYLabels = ['Time [hh:mm]', 'Temperature [C]']
            Tags = self.Labels[16:19] # labels of the different temperatures ('Compressor', 'Inlet', 'Outlet')

        fig = plt.figure(figsize=(20,5))
        ax = fig.gca()
        if(YRange[0]!=1 or YRange[1]!=1):
            plt.ylim(YRange[0], YRange[1])
        
        ax.minorticks_on()
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval=XTicks)) # number of minutes in between each x tick 
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(YTicks))
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))

        ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.8)
        ax.grid(b=True, which='minor', color='grey', linestyle=':')

        plt.xlabel(XYLabels[0])
        plt.ylabel(XYLabels[1])

        plt.gcf().autofmt_xdate()
        formatter = DateFormatter('%H:%M')
        plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

        for ii,(X,Tag) in enumerate(zip(Data,Tags)):
            plt.plot(self.Time[::Bin], X[::Bin], label=Tag, linewidth=2, color=colors[ii])
        plt.legend(loc='best')

        if(XRange != 0):
            xlim1 = XRange[0]
            xlim2 = XRange[1]
        else:
            xlim1 = self.DateTime + datetime.timedelta(seconds=3600*0)
            xlim2 = self.DateTime + datetime.timedelta(seconds=3600*24)
            # xlim2 = self.Time[-1]
        plt.xlim(xlim1, xlim2)
        plt.savefig('StandStatus.pdf')
        plt.show()

    def MonitorData(self, Data, Selection, Time=None, XYLabels=None, Labels=None, Tags=None, XRange=0, YRange=[1,1], YTicks=[10, 10, 10], XTicks=2, Bin=1):
        '''Plotting data over the span of a few months for the purpose of monitoring levels
        Similar to Plot data, but assuming the x-axis is over months instead of a few hours'''
        if Selection == 'Temperature': 
            XYLabels = ['Time [yyyy-mm]', 'Temperature [C]']
            Tags = self.Labels[9:13] # labels of the different temperature sensors ('Cold Head', 'Copper Ring', 'Copper Jacket', 'TPC Bottom')
            # Data = [self.Data[x] for x in Tags]
        elif Selection == 'Xenon Pressure': 
            XYLabels = ['Time [yyyy-mm]', 'Pressure [PSIG]']
            Tags = self.Labels[2:4] # labels of the two xenon cylinders ('Stainless-steel Cylinder 1', 'Stainless-steel Cylinder 2')
            # Data = [self.Data[x] for x in Tags]
        elif Selection == 'System Pressure':  
            XYLabels = ['Time [yyyy-mm]', 'Pressure [PSIG]']
            Tags = self.Labels[0:2] # labels of the two pressure sensors ('Gas System', 'Chamber')
            # Data = [self.Data[x] for x in Tags]
        elif Selection == 'Compressor': 
            XYLabels = ['Time [yyyy-mm]', 'Temperature [C]']
            Tags = self.Labels[16:19] # labels of the different temperatures ('Compressor', 'Inlet', 'Outlet')
        
        fig, axes = plt.subplots(1, len(Tags), figsize=(6*len(Tags), 5))
        for ax, ytick in zip(axes, YTicks):
            if(YRange[0]!=1 or YRange[1]!=1):
                plt.ylim(YRange[0], YRange[1])
            
            ax.minorticks_on()
            ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=XTicks)) # number of months in between each x tick 
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
            ax.yaxis.set_major_locator(MultipleLocator(ytick))
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

            ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.8)
            ax.grid(b=True, which='minor', color='grey', linestyle=':')

            ax.set_xlabel(XYLabels[0])
            ax.set_ylabel(XYLabels[1])

            plt.gcf().autofmt_xdate()
            formatter = DateFormatter('%Y-%m')
            ax.xaxis.set_major_formatter(formatter)
        
        if Selection == 'Compressor': # selecting only the data points when the compressor is on
            AverageData = {Tag: [] for Tag in Tags} # taking an average of all the data points each day
            AverageTime = []
            # finding and separating the data when the compressor is hot to take the average temperature
            threshold = 70
            for X in Data:
                compressor_hot = np.where(X['Compressor'] >= threshold)[0] # finding all the instances where the compressor is above 70C
                time = X['Time'][compressor_hot]
                AverageTime.append(time[0])
                for Tag in Tags:
                    average = np.average(X[Tag][compressor_hot])
                    AverageData[Tag].append(average)

            for i, (ax, Tag) in enumerate(zip(axes, Tags)):
                ax.scatter(AverageTime, AverageData[Tag], color=colors[i], label=Tag, s=10)
                ax.legend(loc='best')
            
            axes[0].set_ylim(70, 80)
            axes[1].set_ylim(22, 25)
            axes[2].set_ylim(30, 33)
        elif Selection == 'Xenon Pressure': # selecting only the data points when there is xenon by the pressure gauge
            AverageData = {Tag:[] for Tag in Tags} # taking an average of all the data points each day
            AverageTime = {Tag:[] for Tag in Tags}

            # finding and separating the data when the xenon is in the pressure reading tube
            Lthreshold = 350
            Rthreshold = 550
            for X in Data:
                Lcylinder_on = np.where(X['Stainless-steel Cylinder 2'] > Lthreshold)[0] # finding all the instances where the pressure is above 350 PSI
                Rcylinder_on = np.where(X['Stainless-steel Cylinder 1'] > Rthreshold)[0] # finding all the instances where the pressure is above 550 PSI
                Ltime = X['Time'][Lcylinder_on]
                if len(Ltime) > 0:
                    AverageTime['Stainless-steel Cylinder 2'].append(Ltime[0])
                Rtime = X['Time'][Rcylinder_on]
                if len(Rtime) > 0:
                    AverageTime['Stainless-steel Cylinder 1'].append(Rtime[0])
                for Tag, pressure in zip(Tags, [Rcylinder_on, Lcylinder_on]):
                    if len(pressure) > 0:
                        average = np.average(X[Tag][pressure])
                        AverageData[Tag].append(average)
            for i, (ax, Tag) in enumerate(zip(axes, Tags)):
                ax.scatter(AverageTime[Tag], AverageData[Tag], color=colors[i], label=Tag, s=10)
                ax.legend(loc='best')
            axes[0].set_ylim(500, 700)
            axes[1].set_ylim(300, 500)

# add more options here as needed



def ComparePlots(SensorDataList, DataList, StartTimeList, Selection='Temperature', Time=None, XYLabels=None, Labels=None, Tags=None, XRange=0, YRange=[1,1], YTicks=10, XTicks=2, Bin=1):
    '''Plotting multiple datasets against each other for comparison'''
        
    if Selection == 'Temperature': 
        XYLabels = ['Time [hh:mm]', 'Temperature [C]']
        Tags = SensorDataList[0].Labels[9:13] # labels of the different temperature sensors ('Cold Head', 'Copper Ring', 'Copper Jacket', 'TPC Bottom')
        # Data = [self.Data[x] for x in Tags]
    elif Selection == 'Xenon Pressure': 
        XYLabels = ['Time [hh:mm]', 'Pressure [PSIG]']
        Tags = SensorDataList[0].Labels[2:4] # labels of the two xenon cylinders ('Stainless-steel Cylinder 1', 'Stainless-steel Cylinder 2')
        # Data = [self.Data[x] for x in Tags]
    elif Selection == 'System Pressure':  
        XYLabels = ['Time [hh:mm]', 'Pressure [PSIG]']
        Tags = SensorDataList[0].Labels[0:2] # labels of the two pressure sensors ('Gas System', 'Chamber')
        # Data = [self.Data[x] for x in Tags]
    elif Selection == 'Compressor': 
        XYLabels = ['Time [hh:mm]', 'Temperature [C]']
        Tags = SensorDataList[0].Labels[16:19] # labels of the different temperatures ('Compressor', 'Inlet', 'Outlet')


    if YRange == [1,1]:
        YRange = [[1,1] for Tag in Tags]
    for i, (Tag, Range) in enumerate(zip(Tags, YRange)):

        fig = plt.figure(figsize=(20,5))
        ax = fig.gca()
        if(Range[0]!=1 or Range[1]!=1):
            plt.ylim(Range[0], Range[1])

        ax.minorticks_on()
        # ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval=XTicks))
        # ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(YTicks))
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))

        ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.8)
        ax.grid(b=True, which='minor', color='grey', linestyle=':') 

        plt.xlabel(XYLabels[0])
        plt.ylabel(XYLabels[1])

        for j, (Sensor, StartTime, Data) in enumerate(zip(SensorDataList, StartTimeList, DataList)):
            plt.gcf().autofmt_xdate() # formatting x axis ticks to better display dates and times
            formatter = DateFormatter('%H:%M')
            plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

            Time = Sensor.Time[::Bin] # getting the "absolute" time of the measurements
            # RelativeTime = Time - StartTime 
            RelativeTime = [(T - StartTime).total_seconds()/3600 for T in Time] # getting the time passed since some given start time
            if j > 0:
                plt.plot(RelativeTime, Data[i][::Bin], label=Tag, linewidth=2, color=colors[i], alpha=0.7)
            else:
                plt.plot(RelativeTime, Data[i][::Bin], label=Tag, linewidth=2, color=colors[i])
            plt.legend(loc='best')
        
        print(RelativeTime)

        print(XRange.total_seconds()/3600)
        if(XRange != 0):
            xlim1 = 0
            xlim2 = XRange.total_seconds()/3600
        else:
            xlim1 = StartTime + datetime.timedelta(seconds=3600*0)
            xlim2 = StartTime + datetime.timedelta(seconds=3600*24)
            # xlim2 = self.Time[-1]
        plt.xlim(xlim1, xlim2)
        plt.show()