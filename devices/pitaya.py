import scpi
import struct
import numpy as np
import h5py
import time, datetime


class RedPitaya():

    def __init__(self,ip_address):
        self.inst = scpi.scpi(ip_address)
        self.max_buffer_size = 16384
    
    def __del__(self):
        print('Destructor called, Employee deleted.')
    
    def configure(self, dec_fac):
        self.dec_fac = dec_fac
        self.inst.tx_txt('ACQ:DATA:FORMAT ASCII')
        self.inst.tx_txt('ACQ:DATA:UNITS VOLT')
        self.inst.tx_txt('ACQ:DEC {}'.format(dec_fac))

    def gain(self, ch=1, gain='LV'):
        self.inst.tx_txt('ACQ:SOUR{}:GAIN {}'.format(ch, gain))

    def trigger(self, ch, level, delay, edge='P'):
        self.inst.tx_txt('ACQ:TRIG CH{}_{}E'.format(ch,edge))
        self.inst.tx_txt('ACQ:TRIG:LEV {} mV'.format(level))
        self.inst.tx_txt('ACQ:TRIG:DLY {}'.format(delay))

    def start(self):
        self.inst.tx_txt('ACQ:START')

    def acquire(self, ch, size, ascii=False):
        while 1:
            self.inst.tx_txt('ACQ:TRIG:STAT?')
            if self.inst.rx_txt() == 'TD':
                break
        self.inst.tx_txt('ACQ:SOUR{}:DATA:OLD:N? {}'.format(ch,size))
        self.inst.tx_txt('ACQ:START')

        buff_string = self.inst.rx_txt()
        buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
        buff = list(map(float, buff_string))
        t_wvf = np.linspace(0, size*8E-9*self.dec_fac, size)

        return t_wvf, np.array(buff)

    def convert(self, data):
        return np.array([struct.unpack('!f',bytearray(data[i:i+4]))[0] for i in range(0, len(data), 4)])

    def save(self, t_wvf, data, ch, tag, path):
        f = h5py.File("{}/{}.h5".format(path, tag), "w")
        grp1=f.create_group("ch1")
        grp2=f.create_group("ch2")
       
        for i,waveforms in enumerate(data):
            if ch == 1:
                
                # now = datetime.datetime.now()
                # seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                # s_m=str(seconds_since_midnight)
                index=str(i)        
                grp1.create_dataset(index, data=waveforms)
                # print(s_m)
            elif ch ==2:
                
                # now = datetime.datetime.now()
                # print('now: ',now)
                # seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                # s_m=str(seconds_since_midnight)
                # print(s_m)
                index=str(i)        
                grp2.create_dataset(index, data=waveforms)
                
        f.create_dataset('Time', data=t_wvf)
        f.close()
