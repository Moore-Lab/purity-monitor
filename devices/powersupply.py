from RsInstrument import *

class NGE100:
    def __init__(self, address='ASRL9::INSTR'):
        RsInstrument.assert_minimum_version('1.21.0.78')
        self.inst = RsInstrument('USB0::0x0AAD::0x0197::5601.1414k03-100771::INSTR', True, True, "SelectVisa='rs',")

    def voltage(self, ch, volt):
        self.inst.write('INSTrument:NSELect {}'.format(ch))
        self.inst.write('SOURce:VOLTage:LEVel:IMMediate:AMPLitude {}'.format(volt))

    def output(self, ch, state):
        self.inst.write('INSTrument:NSELect {}'.format(ch))
        self.inst.write('OUTPut:STATe {}'.format([0,1][state]))