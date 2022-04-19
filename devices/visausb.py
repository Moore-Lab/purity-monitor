""" Defines a superclass for communicating with usb instruments through pyvisa
The pyvisa package provides a powerful framework for communication with
devices conforming to the VISA communication standards.  The pyvisa package
provides a link between python an the VISA instance installed on the local
computer.  The most common installation is NIVISA, which is maintained by
National Instruments and is freely available online.
"""

import logging
from time import time, sleep
from difflib import get_close_matches
import pyvisa as visa

__email__ = "chrisark7@gmail.com"
__status__ = "Development"

logger = logging.getLogger(__name__)


class VisaUsbInstrument(object):
    """ A class for interacting with USB instruments through pyvisa
    This class relies on pyvisa for communication which, in turn, relies on
    a third party VISA installation such as NIVISA.
    The class implements low-level communication appropriate for all
    USB-based instruments and is intended primarily as a superclass for more
    detailed instrument-specific implementations.
    """
    def __init__(self, device_id=0, timeout=0.5):
        """ The constructor for the VisaUsbInstrument class
        This function searches the devices connected to the computer and
        initializes the VisaUsbInstrument instance with one of them.  Note that
        it is still necessary to open the connection before being able to
        interact with the instrument.
        The list of devices connected to the computer will be printed to the log.
        :param device_id: Integer or string which describes the device to connect to
        :param timeout: The timeout value to use with the instrument in seconds
        :type device_id: int or str
        :type timeout: int or float
        :return: An instance of the VisaUsbInstrument class
        :rtype: VisaUsbInstrument
        """
        # Get list of devices connected to the computer
        rm = visa.ResourceManager()
        devices = rm.list_resources()

        # Check device list
        if not devices:
            raise LookupError('no devices are connected to the computer')
        else:
            logger.info("=== Connected Devices ===")
            for i, device in enumerate(devices):
                logger.info("{0}: ".format(i) + device)
        # Parse device_id and assign
        if type(device_id) is str:
            if device_id not in devices:
                raise ValueError('device_id is not in list of devices')
            else:
                inst_id = device_id
        elif type(device_id) is not int:
            try:
                device_id = int(device_id)
            except:
                raise ValueError('device_id should be a string or an integer')
            if device_id > len(devices) - 1:
                raise ValueError('device_id is larger than the number of '
                                 'devices')
            else:
                inst_id = devices[device_id]
        elif device_id > len(devices) - 1:
            raise ValueError('device_id is larger than the number of devices')
        else:
            inst_id = devices[device_id]
        # Print chosen device
        logger.info('Initializing device: {0}'.format(inst_id))
        # Set parameters
        self.timeout = timeout*1e3
        self.inst_id = inst_id
        self.resource_manager = rm
        self.is_open = False
        self.device = None

    ###########################################################################
    # Helper Functions
    ###########################################################################
    @staticmethod
    def _get_close_string(target_string, string_list):
        """ A helper function to find the most similar string in a list
        This function is designed to be used with text-based inputs where the
        user might misspell or otherwise make a mistake when entering a string
        input.  It compares the `target_string` to a list of strings in
        `string_list` and returns the most probable match.
        The metric used to determine the distance between two string
        distinguishes between caps and non-caps letters so 'apple' and 'APPLE'
        are very dissimilar.
        The threshold for a
        :param string_list: A list of strings to be compared to target_string
        :param target_string: The target string to match
        :type string_list: list of str, tuple of str, dict_keys
        :type target_string: str
        :return: most likely string
        :rtype: str
        """
        out = get_close_matches(target_string, string_list, n=1)
        if out:
            return out[0]
        else:
            return None

    ###########################################################################
    # Communication Commands
    ###########################################################################
    def open(self):
        """ Opens the connection to the instrument
        """
        if self.is_open:
            raise IOError('Communication to instrument is already open')
        try:
            self.device = self.resource_manager.open_resource(self.inst_id,
                                                              open_timeout=10e3)
        except:
            raise IOError('Unable to open connection to instrument')
        self.is_open = True
        self.device.timeout = self.timeout

    def close(self):
        """ Closes the connection to the instrument
        """
        if not self.is_open:
            raise IOError('Communication to instrument is already closed')
        self.device.close()
        self.is_open = False

    def flush(self):
        """ Flushes the input and output buffers
        """
        if not self.is_open:
            raise IOError('Communication to function generator is closed')
        self.device.flush(mask=64)
        self.device.flush(mask=128)

    def write(self, command):
        """ Writes a command to the instrument
        :param command: A valid command to the instrument
        :type command: str
        :return: the output of the write command
        :rtype: str
        """
        if not self.is_open:
            raise IOError('Communication to the instrument is closed')
        try:
            out = self.device.write(command)
        except visa.VisaIOError:
            raise ValueError('Command timed out; most likely it is not a valid '
                             'command')
        return out

    def read(self):
        """ Reads the most recent output from the instrument
        :param timeout: The timeout length in seconds
        :type timeout: float
        :return: The output of the read command
        :rtype: str
        """
        pause_between_tries = 100e-3
        if not self.is_open:
            raise IOError('Communication to instrument is closed')
        try:
            out = self.device.read()
        except visa.VisaIOError:
            logger.debug('Instrument did not return anything, sleeping for '
                         '{0} seconds and trying again'.format(pause_between_tries))
            sleep(pause_between_tries)
            try:
                out = self.device.read()
            except visa.VisaIOError:
                out = ''
                logger.warning('Instrument did not return anything when trying '
                               'to read')
        return out.rstrip()

    def query(self, command):
        """ Queries a value from the instrument
        This method is equivalent to writing a query command and then reading
        the instrument's output.
        :param command: A valid command to the function generator
        :type command: str
        :return: the output of the query command
        :rtype: str
        """
        self.write(command)
        out = self.read()
        return out

    ###########################################################################
    # Get/Set Routines
    ###########################################################################
    def _set_with_check(self, command, query, result, transform=None, timeout=5):
        """ Issues the command and checks that the query gives the result
        This method is designed to ease the implementation of positive feedback
        routines in the subclasses.  It issues the `command` and then checks
        that the `query` returns the `result` until either: 1) `query` returns
        `result` or 2) the timeout is reached.  It returns `True` if the
        `query` returns the result or `False` if it times out.
        Note that the check between the output of `query` and `result` is a
        check for hard equality (i.e. `==`).
        The `transform` variable can be used to apply a function to the result
        before comparing for equality.  This is helpful for e.g. converting
        the readback to an integer before comparing for equality.
        :param command: The full command to issue to the device
        :param query: The full query command
        :param result: The expected result
        :param transform: A function applied to readback before comparision
        :param timeout: The length of time to continue trying to set in seconds
        :type command: str
        :type query: str
        :type result: str
        :type timeout: float or int
        :return: True or False
        :rtype: bool
        """
        pause_between_loops = 100e-3
        pause_between_set_and_query = 25e-3
        set_bool = False
        t0 = time()
        while not set_bool:
            # Set and query
            self.write(command)
            sleep(pause_between_set_and_query)
            out = self.query(query)
            # Try the query command again if out is an empty string
            if not out:
                sleep(pause_between_loops)
                out = self.query(query)
            # Transform the result
            if transform is not None:
                out = transform(out)
            # Check the result
            if out == result:
                return True
            elif time() - t0 > timeout:
                return False
            else:
                sleep(pause_between_loops)