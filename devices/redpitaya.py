import scpi
import struct
import numpy as np
import h5py
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
        t_wvf = np.linspace(0, size*8E-9*self.dec_fac, size)*1e6

        return t_wvf, np.array(buff)

    def convert(self, data):
        return np.array([struct.unpack('!f',bytearray(data[i:i+4]))[0] for i in range(0, len(data), 4)])

    def save(self, t_wvf, data, ch, tag, path):
        f = h5py.File("{}/{}.h5".format(path, tag), "w")
        if ch == 1:
            f.create_dataset('Ch1', data=data)
        elif ch ==2:
            f.create_dataset('Ch2', data=data)
        f.create_dataset('Time', data=t_wvf)
        f.close()
