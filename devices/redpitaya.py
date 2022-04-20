import scpi
import struct
import numpy as np

class RedPitaya():

    def __init__(self,ip_address):
        self.inst = scpi.scpi(ip_address)
        self.max_buffer_size = 16384
    
    def configure(self, dec_fac):
        self.dec_fac = dec_fac
        self.inst.tx_txt('ACQ:DATA:FORMAT BIN')
        self.inst.tx_txt('ACQ:DATA:UNITS RAW')
        self.inst.tx_txt('ACQ:DEC {}'.format(dec_fac))
        self.inst.tx_txt('ACQ:AVG OFF')

    def trigger(self, ch, level, delay):
        self.inst.tx_txt('ACQ:TRIG CH{}_PE'.format(ch))
        self.inst.tx_txt('ACQ:TRIG:LEV {} mV'.format(level))
        self.inst.tx_txt('ACQ:TRIG:DLY {}'.format(delay))
        self.inst.tx_txt('ACQ:TRIG:DLY?')

    def acquire(self, ch, size, ascii=False):
        self.inst.tx_txt('ACQ:START')
        while 1:
            self.inst.tx_txt('ACQ:TRIG:STAT?')
            if self.inst.rx_txt() == 'TD':
                self.inst.tx_txt('ACQ:SOUR{}:DATA:OLD:N? {}'.format(ch, size))
                buff_byte = self.inst.rx_arb()
                break
        t_wvf = np.linspace(0, size*8E-9*self.dec_fac, size)*1e6
        if ascii:
            return t_wvf,[struct.unpack('!h',bytearray(buff_byte[i:i+2]))[0] for i in range(0, len(buff_byte), 2)]
        else:
            return t_wvf,buff_byte