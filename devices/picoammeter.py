import time
import pyvisa as visa


class Keithley6487:
    def __init__(self, address='ASRL9::INSTR'):
        rm = visa.ResourceManager()
        self.inst = rm.open_resource(address)
        self.inst.timeout = 10000

    def current(self):
        """Measure current and parse the reading, return only current value.
        Example format:
        -6.329391E-13A,+1.261521E+03,+0.000000E+00
        will return float("-6.329391E-13")
        """
        readout = self.query("READ?")
        return float(readout.split(',')[0][:-1])

    def voltage(self, volts):
        """Set the voltage level of the voltage source."""
        self.write("SOUR:VOLT " + str(volts))

    def voltage_source_state(self, on):
        """Turn voltage source on or off. Send "False" to turn off."""
        self.write("SOUR:VOLT:STAT " + ["OFF", "ON"][on])

    def __enter__(self):
        return self

    def query(self, cmd):
        """Send arbitrary command to instrument."""
        time.sleep(0.1)
        return self.inst.query(cmd).strip()

    def write(self, cmd):
        """Send arbitrary command to instrument."""
        time.sleep(0.1)
        return self.inst.write(cmd)

    def __exit__(self):
        # Disable the voltage output
        self.voltage_source_state(False)
        self.inst.close()