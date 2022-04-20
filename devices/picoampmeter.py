import time
import pyvisa as visa


class Keithley6487:
    """Interact with the Keithley 6487 picoammeter.
    Documentation can be found at
    http://www.physics.fsu.edu/courses/Spring02/phy3802L/intlabdoc
    /instruments/keithley/6485_901_01A.pdf
    or
    http://www.testequity.com/documents/pdf/keithley/manuals/6485-6487-m.pdf
    """

    def __init__(self, gpib_address='GPIB0::22::INSTR'):
        rm = visa.ResourceManager()
#        resources = rm.list_resources()
        self.inst = rm.open_resource(gpib_address)
        print(self.query("*IDN?"))
        print("Configuring for single-point measurements...")
        self.write("CONF")
        # Enable the bias voltage output

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Disable the voltage output
        self.voltage_source_state(False)
        self.inst.close()