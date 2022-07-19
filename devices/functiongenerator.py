import logging
from visausb import VisaUsbInstrument

__email__ = "chrisark7@gmail.com"
__status__ = "Development"

logger = logging.getLogger(__name__)


class AFG2225(VisaUsbInstrument):
    """ A class for communicating with the GW Instek AFG-2225
    This class is likely generic enough to work with other models in the GW
    Instek line but has not been tested.
    """
    def __init__(self, device_id=0, timeout=0.5):
        """ The constructor for the AFG2225 class
        This function searches the devices connected to the computer and
        initializes the AFG2225 instance with one of them.  Note that
        it is still necessary to open the connection before being able to
        interact with the instrument.
        The list of devices connected to the computer will be printed to the
        log.
        :param device_id: Integer or string which describes the device to connect to
        :param timeout: The timeout value to use with the instrument in seconds
        :type device_id: int or str
        :type timeout: int or float
        :return: An instance of the AFG2225 class
        :rtype: AFG2225
        """
        super(AFG2225, self).__init__(device_id=device_id, timeout=timeout)
        self.open()

    ###########################################################################
    # Helper Methods
    ###########################################################################
    @staticmethod
    def _check_channel(channel):
        """ Checks to make sure that channel is an integer in [1, 2]
        :param channel: channel to check
        :type channel: int
        :return: channel integer in [1, 2]
        :rtype: int
        """
        try:
            channel = int(channel)
        except:
            raise TypeError("channel should be an integer")
        if channel not in [1, 2]:
            raise ValueError("channel should be 1 or 2")
        return channel

    ###########################################################################
    # System Methods
    ###########################################################################
    def system_read_error_queue(self):
        """ Reads all messages stored in the error queue
        This method reads all messages stored in the AFG2225's internal error
        queue and returns them to the user as a list of 2-element tuples where
        the first element is the error code and the second is the message.
        Note that the error queue reads out in a strange way from this device
        where the separation between successive reads from the error queue is
        broken at the end of the error code.  This means that the messages
        returned have the last error's message and the next error's code.
        However, this quirk is handled internally so it should not affect the
        user.
        :return: list of tuples with (error code, error message)
        :rtype: list of tuple
        """
        # Read the error queue until 'No error.' is returned
        last_message = ''
        messages = []
        i = 0
        self.flush()
        while i < 20:
            last_message = self.query("SYSTEM:ERROR?")
            if last_message == "No error.":
                break
            else:
                messages.append(last_message)
        self.flush()
        print(messages)
        # Parse messages
        out = []
        next_code = None
        last_code = None
        last_message = None
        for message in messages:
            # Interpret message
            if not message:
                pass
            elif '.' in message:
                split_messages = message.split('.')
                last_message = '.'.join(split_messages[:-1])
                if split_messages[-1]:
                    next_code = split_messages[-1]
            elif next_code is None:
                next_code = message
            else:
                last_message = message
            # Check that error code is an integer
            if next_code is not None:
                try:
                    next_code = int(next_code)
                except (ValueError, TypeError):
                    logger.warning("Unable to interpret code as an integer; "
                                   "something may be wrong with error readback")
            # Save to output
            if last_message is not None:
                out.append((last_code, last_message))
            last_code = next_code
        # Return
        return out

    ###########################################################################
    # Set/Get Waveform Properties Individually
    ###########################################################################
    def set_wavetype(self, channel, wavetype):
        """ Sets the channel's function
        Possible channel wavetypes are: SINUSOID, SQUARE, RAMP, PULSE, NOISE,
        USER
        :param channel: The channel to set (1 or 2)
        :param wavetype: A string corresponding to the function
        :type channel: int
        :type wavetype: str
        :return: True if set, False if not
        :rtype: bool
        """
        assert type(wavetype) is str
        # Check channel
        channel = self._check_channel(channel)
        # Define possibilities
        inputs = {"SINUSOID": "SIN",
                  "SQUARE": "SQU",
                  "RAMP": "RAMP",
                  "PULSE": "PULS",
                  "NOISE": "NOIS",
                  "USER": "ARB",
                  "SIN": "SIN",
                  "SQU": "SQU",
                  "PULS": "PULS",
                  "NOIS": "NOIS"}
        # Check for string match
        wavetype = wavetype.upper()
        if wavetype not in inputs.keys():
            best_match = self._get_close_string(wavetype, inputs.keys())
            if best_match is None:
                raise ValueError("wavetype does not match any possible wavetypes")
            else:
                logger.warning("{0} does not match any wavetype; using {1} "
                               "instead".format(wavetype, best_match))
            wavetype = best_match
        # Form command
        command = "SOURCE{0}:FUNCTION {1}".format(channel, wavetype)
        query = "SOURCE{0}:FUNCTION?".format(channel)
        result = inputs[wavetype]
        # Issue command
        return self._set_with_check(command, query, result)

    def get_wavetype(self, channel):
        """ Returns the channel's current function setting
        :param channel: The channel to query
        :type channel: int
        :return: The current function setting for the specified channel
        :rtype: str
        """
        # Check channel
        channel = self._check_channel(channel)
        # Get Current Function
        return self.query("SOURCE{0}:FUNCTION?".format(channel))

    def set_frequency(self, channel, frequency):
        """ Sets the channel's frequency
        The AG2225 can be set down to uHz even when the frequency is in the MHz
        regime.  So, 10.000000000001e6 will be set even though it is highly
        unlikely that the internal oscillator has this kind of precision.
        Note that the positive feedback on this routine depends on comparing
        two floats which is never a safe operation.  If it returns False, it
        is a good idea to double check that floating point precision is not
        causing the issue.
        :param channel: The channel to adjust (1 or 2)
        :param frequency: The frequency in Hz
        :type channel: int
        :type frequeny: float or int
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Check
        # Query the min and max possible frequencies
        min_freq = float(self.query("SOURCE{0}:FREQUENCY? MIN".format(channel)))
        max_freq = float(self.query("SOURCE{0}:FREQUENCY? MAX".format(channel)))
        # Check that frequency is in the range
        in_range = True
        if frequency > max_freq:
            logger.warning("Frequency is greater than the max for this function; "
                           "setting to {0} instead".format(max_freq))
            frequency = max_freq
            in_range = False
        elif frequency < min_freq:
            logger.warning("Frequency is less than the min for this function; "
                           "setting to {0} instead".format(min_freq))
            frequency = min_freq
            in_range = False
        # Build commands
        command = "SOURCE{0}:FREQUENCY {1}".format(channel, frequency)
        query = "SOURCE{0}:FREQUENCY?".format(channel)
        result = frequency
        out =  self._set_with_check(command=command,
                                    query=query,
                                    result=result,
                                    transform=float)
        return in_range and out

    def get_frequency(self, channel):
        """ Returns the channel's current frequency in Hz
        :param channel: The channel to query
        :type channel: int
        :return: The channel's current frequency
        :rtpye: float
        """
        # Check channel
        channel = self._check_channel(channel)
        # Return
        return float(self.query("SOURCE{0}:FREQUENCY?".format(channel)))

    def set_amplitude(self, channel, amplitude):
        """ Sets the channel's amplitude
        Units are Vpp
        The min and max of the possible amplitude settings depend on the
        current function and output load settings on the function generator
        This method will warn you if the value is below the min or above the
        max and set the amplitude appropriately.
        Note that the feedback on this routine depends on comparing two floats
        which is never a safe operation.  If it returns False, it is a good
        idea to double check that floating point precision is not causing the
        issue.  In addition, a False will be returned if the specified
        amplitude is below the min or above the max, but the amplitude will
        still be set to the min or max respectively.
        :param channel: The channel who's amplitude will be set
        :param amplitude: The amplitude value to set in units given by `unit`
        :type channel: int
        :type amplitude: float
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel=channel)
        # Get min and max
        min_amp = float(self.query("SOURCE{0}:AMPLITUDE? MIN".format(channel)))
        max_amp = float(self.query("SOURCE{0}:AMPLITUDE? MAX".format(channel)))
        # Check the amplitude and set
        in_range = True
        if amplitude > max_amp:
            logger.warning("amplitude is greater than the max for the current "
                           "settings; setting to {0}".format(max_amp))
            amplitude = max_amp
            in_range = False
        elif amplitude < min_amp:
            logger.warning("amplitude is less than the min for the current "
                           "settings; setting to {0}".format(min_amp))
            amplitude = min_amp
            in_range = False
        # Issue command
        command = "SOURCE{0}:AMPLITUDE {1}".format(channel, amplitude)
        query = "SOURCE{0}:AMPLITUDE?".format(channel)
        result = amplitude
        out = self._set_with_check(command=command,
                                   query=query,
                                   result=result,
                                   transform=float)
        return in_range and out

    def get_amplitude(self, channel):
        """ Query the channel's amplitude in units of Vpp
        :param channel: The chanel whose amplitude will be queried
        :type channel: int
        :return: The current amplitude in units of Vpp
        :rtype: float
        """
        # Check channel
        channel = self._check_channel(channel)
        # Query
        return float(self.query("SOURCE{0}:AMPLITUDE?".format(channel)))

    def set_offset(self, channel, offset):
        """ Sets the channel's DC offset
        The min and max of the DC offset depends on the current amplitude
        (among other things).  This method will query the current max and min
        and warn the user if the specified value is out of range.  If the
        specified value is out of range, then the offset will be set to the
        min/max.
        Note that the feedback on this routine depends on comparing two floats
        which is never a safe operation.  If it returns False, it is a good
        idea to double check that floating point precision is not causing the
        issue.  In addition, a False will be returned if the specified
        offset is below the min or above the max, but the offset will
        still be set to the min or max respectively.
        :param channel: The channel whose offset will be set
        :param offset: The offset in Volts
        :type channel: int
        :type offset: float
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Get min and max
        min_off = float(self.query("SOURCE{0}:DCOFFSET? MIN".format(channel)))
        max_off = float(self.query("SOURCE{0}:DCOFFSET? MAX".format(channel)))
        # Check the offset and set
        in_range = True
        if offset > max_off:
            logger.warning("offset is greater than the max for the current "
                           "settings; setting to {0}".format(max_off))
            offset = max_off
            in_range = False
        elif offset < min_off:
            logger.warning("offset is less than the min for the current "
                           "settings; setting to {0}".format(min_off))
            offset = min_off
            in_range = False
        command = "SOURCE{0}:DCOFFSET {1}".format(channel, offset)
        query = "SOURCE{0}:DCOFFSET?".format(channel)
        result = offset
        out = self._set_with_check(command=command,
                                   query=query,
                                   result=result,
                                   transform=float)
        return in_range and out

    def get_offset(self, channel):
        """ Queries the channel's current DC offset
        :param channel: The chanel whose offset will be queried
        :type channel: int
        :return: The current offset in units of Volts
        :rtype: float
        """
        # Check channel
        channel = self._check_channel(channel)
        # Query channel
        return float(self.query("SOURCE{0}:DCOFFSET?".format(channel)))

    def set_square_duty(self, channel, duty):
        """ Sets the duty cycle for the square waveform
        The duty cycle which the AFG 2225 can generate depends on the
        frequency of the square wave.  This method will query the max and min
        and warn the user if the specified duty is outside of that range.  It
        will also set the duty to the max or min in that situation.
        Note that the feedback on this routine depends on comparing two floats
        which is never a safe operation.  If it returns False, it is a good
        idea to double check that floating point precision is not causing the
        issue.  In addition, a False will be returned if the specified
        duty cycle is below the min or above the max, but the duty cycle will
        still be set to the min or max respectively.
        :param channel: The channel whose duty cycle will be set
        :param duty: Duty cycle in percent: 1-99
        :type channel: int
        :type duty: float
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Get min and max
        min_duty = float(self.query("SOURCE{0}:SQUARE:DCYCLE? MIN".format(channel)))
        max_duty = float(self.query("SOURCE{0}:SQUARE:DCYCLE? MAX".format(channel)))
        # Check the offset and set
        in_range = True
        if duty > max_duty:
            logger.warning("duty is greater than the max for the current "
                           "settings; setting to {0}".format(max_duty))
            duty = max_duty
            in_range = False
        elif duty < min_duty:
            logger.warning("duty is less than the min for the current "
                           "settings; setting to {0}".format(min_duty))
            duty = min_duty
            in_range = False
        command = "SOURCE{0}:SQUARE:DCYCLE {1}".format(channel, duty)
        query = "SOURCE{0}:SQUARE:DCYCLE?".format(channel)
        result = duty
        out = self._set_with_check(command=command,
                                   query=query,
                                   result=result,
                                   transform=float)
        return in_range and out

    def get_square_duty(self, channel):
        """ Queries the current square wave duty cycle for the specified channel
        :param channel: The channel whose duty cycle will be queried
        :type channel: int
        :return: The current duty cycle
        :rtype: float
        """
        # Check channel
        channel = self._check_channel(channel)
        # Query
        return float(self.query("SOURCE{0}:SQUARE:DCYCLE?".format(channel)))

    def set_ramp_symmetry(self, channel, symmetry):
        """ Sets the symmetry parameter for the ramp waveform
        Note that the feedback on this routine depends on comparing two floats
        which is never a safe operation.  If it returns False, it is a good
        idea to double check that floating point precision is not causing the
        issue.  In addition, a False will be returned if the specified
        symmetry is below the min or above the max, but the symmetry will
        still be set to the min or max respectively.
        :param channel: The channel whose duty cycle will be set
        :param symmetry: Symmetry in percent: 0-100
        :type channel: int
        :type symmetry: float
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Get min and max
        min_sym = float(self.query("SOURCE{0}:RAMP:SYMMETRY? MIN".format(channel)))
        max_sym = float(self.query("SOURCE{0}:RAMP:SYMMETRY? MAX".format(channel)))
        # Check the offset and set
        in_range = True
        if symmetry > max_sym:
            logger.warning("symmetry is greater than the max for the current "
                           "settings; setting to {0}".format(max_sym))
            symmetry = max_sym
            in_range = False
        elif symmetry < min_sym:
            logger.warning("symmetry is less than the min for the current "
                           "settings; setting to {0}".format(min_sym))
            symmetry = min_sym
            in_range = False
        command = "SOURCE{0}:RAMP:SYMMETRY {1}".format(channel, symmetry)
        query = "SOURCE{0}:RAMP:SYMMETRY?".format(channel)
        result = symmetry
        out =  self._set_with_check(command=command,
                                    query=query,
                                    result=result,
                                    transform=float)
        return in_range and out

    def get_ramp_symmetry(self, channel):
        """ Queries the current ramp symmetry for the specified channel
        :param channel: The channel whose symmetry will be queried
        :type channel: int
        :return: The current symmetry setting
        :rtype: float
        """
        # Check channel
        channel = self._check_channel(channel)
        # Query
        return float(self.query("SOURCE{0}:RAMP:SYMMETRY?".format(channel)))

    def set_phase(self, channel, phase):
        """ Sets the phase of the specified channel
        The phase is specified in degrees, and the valid range goes from -180
        to 180.
        :param channel: The channel whose phase to set
        :param phase: The pahse [-180, 180]
        :type channel: int
        :type phase: float or int
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Get min and max
        min_phase = float(self.query("SOURCE{0}:PHASE? MIN".format(channel)))
        max_phase = float(self.query("SOURCE{0}:PHASE? MAX".format(channel)))
        # Check the phase and set
        in_range = True
        if phase > max_phase:
            logger.warning("phase is greater than the max for the current "
                           "settings; setting to {0}".format(max_phase))
            phase = max_phase
            in_range = False
        elif phase < min_phase:
            logger.warning("phase is less than the min for the current "
                           "settings; setting to {0}".format(min_phase))
            phase = min_phase
            in_range = False
        command = "SOURCE{0}:PHASE {1}".format(channel, phase)
        query = "SOURCE{0}:PHASE?".format(channel)
        result = phase
        out =  self._set_with_check(command=command,
                                    query=query,
                                    result=result,
                                    transform=float)
        return in_range and out

    def get_phase(self, channel):
        """ Queries the phase setting of the specified channel
        :param channel: The channel whose phase to query
        :type channel: int
        :return: The current phase of the specified channel
        :rtype: float
        """
        # Check channel
        channel = self._check_channel(channel)
        # Query
        return float(self.query("SOURCE{0}:PHASE?".format(channel)))

    ###########################################################################
    # Set/Get Interface Properties
    ###########################################################################
    def set_output_onoff(self, channel, on_off):
        """ Switches the output on or off
        `on_off` can be the strings "ON" or "OFF" or a 0 or 1.  Any other
        value will be evaluated with bool(), used to set the output to on for
        true/off for false, and raise a warning.
        :param channel: The channel whose output will be switched
        :param on_off: "ON" or "OFF", 0 or 1
        :type channel: int
        :type on_off: str or int
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Parse on_off
        if type(on_off) is str:
            on_off = on_off.upper()
            if on_off == "ON":
                on_off = True
            elif on_off == "OFF":
                on_off = False
            else:
                logger.error("on_off not understood")
                raise ValueError("on_off should be /'ON/' or /'OFF/'")
        elif on_off == 0:
            on_off = False
        elif on_off == 1:
            on_off = True
        else:
            on_off = bool(on_off)
            if on_off:
                logger.warning("on_off is not a standard input; using ON")
            else:
                logger.warning("on_off is not a standard input; using OFF")
        if on_off:
            command = "OUTPUT{0} ON".format(channel)
            query = "OUTPUT{0}?".format(channel)
            result = '1'
        else:
            command = "OUTPUT{0} OFF".format(channel)
            query = "OUTPUT{0}?".format(channel)
            result = '0'
        return self._set_with_check(command=command,
                                    query=query,
                                    result=result)

    def get_output_onoff(self, channel):
        """ Queries the output state of the specified channel
        :param channel: The channel whose output will be queried
        :type channel: int
        :return: True if output is on, False otherwise
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Get value and parse
        return bool(int(self.query("OUTPUT{0}?".format(channel))))

    def set_output_load(self, channel, load):
        """ Sets the output load to 50 Ohm or infinite for the specified channel
        Valid inputs for load are
          - 50 Ohm (Low Z): '50', 'FIFTY', 'DEF', 'DEFAULT', 'LOWZ', 'LZ'
          - High Z: 'HZ', 'HIGHZ', 'INF', 'INFINITE'
        :param channel: The channel whose load will be set
        :param load: The desired load (see above for valid inputs)
        :type channel: int
        :type load: str or int
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Possible inputs
        inputs = {"HZ": "INF",
                  "HIGHZ": "INF",
                  "INF": "INF",
                  "INFINITE": "INF",
                  "50": "DEF",
                  "FIFTY": "DEF",
                  "LZ": "DEF",
                  "LOWZ": "DEF",
                  "DEF": "DEF",
                  "DEFAULT": "DEF",
                  50: "DEF"}
        # Parse `load`
        set_load = False
        if type(load) is str:
            load = load.upper().replace(" ", "")
            if load not in inputs.keys():
                best_match = self._get_close_string(load, inputs.keys() - [50])
                if best_match is None:
                    raise ValueError("load does not match any possible options")
                else:
                    logger.warning("{0} does not match any valid load; using "
                                   "{1} instead".format(load, best_match))
                load = best_match
            if inputs[load] == "INF":
                set_load = True
            elif inputs[load] == "DEF":
                set_load = False
        elif load == 50:
            set_load = False
        else:
            logger.error("load was not understood")
            raise ValueError("`load` was not understood")
        if set_load:
            command = "OUTPUT{0}:LOAD INFINITY".format(channel)
            query = "OUTPUT{0}:LOAD?".format(channel)
            result = inputs[load]
        else:
            command = "OUTPUT{0}:LOAD DEFAULT".format(channel)
            query = "OUTPUT{0}:LOAD?".format(channel)
            result = inputs[load]
        return self._set_with_check(command=command,
                                    query=query,
                                    result=result)

    def get_output_load(self, channel):
        """ Queries the current output load
        The output returned to the user is directly from the function generator
        which uses 'DEF' (i.e. default) to mean 50 Ohms and 'INF' (i.e.
        infinite) to mean high Z of infinite impedance.
        :param channel: The channel whose load to query
        :type channel: int
        :return: Either 'DEF' or 'INF'
        :rytpe: str
        """
        # Check channel
        channel = self._check_channel(channel)
        # Query
        return self.query("OUTPUT{0}:LOAD?".format(channel))

    def set_voltageunits(self, channel, unit='VPP'):
        """ Sets the current units used to specify the voltage
        :param channel: The channel whose units will be set
        :param unit: One of ['VPP', 'VRMS', 'DBM']
        :type channel: int
        :type unit: str
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Check unit
        unit = unit.upper()
        unit_list = ['VPP', 'VRMS', 'DBM']
        if unit not in unit_list:
            best_match = self._get_close_string(unit, unit_list)
            if best_match is None:
                raise ValueError("unit does not match any possible units")
            else:
                logger.warning("{0} does not match any function; using {1} "
                               "instead".format(unit, best_match))
            unit = best_match
        # Set unit
        command = "SOURCE{0}:VOLTAGE:UNIT {1}".format(channel, unit)
        query = "SOURCE{0}:VOLTAGE:UNIT?".format(channel)
        result = unit
        return self._set_with_check(command=command,
                                    query=query,
                                    result=result)

    def get_voltageunits(self, channel):
        """ Queries the channel's current voltage unit
        :param channel: The channel whose units will be queried
        :type channel: int
        :return: The current unit of the channel
        :rtype: str
        """
        # Check channel
        channel = self._check_channel(channel)
        # Query channel
        return self.query("SOURCE{0}:VOLTAGE:UNIT?".format(channel))

    ###########################################################################
    # Composite Methods
    ###########################################################################
    def set_output(self, channel, on_off=None, load=None):
        """ Sets the output settings of the AFG2225
        Any parameter left as ``None`` will not be changed from its current
        state.
        The ``channel`` parameter must be either ``1`` or ``2`` for the two
        output channels of the function generator.
        The ``on_off`` parameter determines whether to set the output to be on
        or off.  Valid settings are those accepted by the `set_output_onoff`
        method.
        The ``load`` parameter determines whether the output impedance is set
        to 50 Ohms or High Z. Valid inputs are those accepted by the
        `set_output_load` method.
        :param channel: 1 or 2
        :param on_off: 'ON' of 'OFF'
        :param load: 50 or 'HZ'
        :type channel: int
        :type on_off: str
        :type load: int or str
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Issue commands
        it_worked = True
        if load is not None:
            out = self.set_output_load(channel=channel, load=load)
            it_worked = it_worked and out
        if on_off is not None:
            out = self.set_output_onoff(channel=channel, on_off=on_off)
            it_worked = it_worked and out
        return it_worked

    def set_wave(self, channel, wavetype=None, frequency=None, amplitude=None,
                 offset=None, symmetry=None, duty=None, phase=None):
        """ Composite function to set all key output function parameters
        Any parameters left as `None` will not be changed from the current
        state.  Note that many of the parameters are only valid for some
        wavetypes.
        :param channel: 1 or 2
        :param wavetype: 'SIN', 'SQUARE', 'RAMP', 'PULSE', 'NOISE', 'USER'
        :param frequency: unit is Hertz, minimum is 1e-6, max is 25e6
        :param amplitude: unit is Volts
        :param offset: unit is Volts
        :param symmetry: unit is percent, only for ramp wave, 0-100
        :param duty: unit is percent, only for square wave, 1-99
        :param phase: unit is degrees, -180 - 180
        :type channel: int
        :type wavetype: str
        :type frequency: float
        :type amplitude: float
        :type offset: float
        :type symmetry: float
        :type duty: float
        :type phase: float
        :return: True if set, False if not
        :rtype: bool
        """
        # Check channel
        channel = self._check_channel(channel)
        # Issue commands
        it_worked = True
        if wavetype is not None:
            out = self.set_wavetype(channel, wavetype)
            it_worked = it_worked and out
        if frequency is not None:
            out = self.set_frequency(channel, frequency)
            it_worked = it_worked and out
        if amplitude is not None:
            out = self.set_amplitude(channel, amplitude)
            it_worked = it_worked and out
        if offset is not None:
            out = self.set_offset(channel, offset)
            it_worked = it_worked and out
        if symmetry is not None:
            out = self.set_ramp_symmetry(channel, symmetry)
            it_worked = it_worked and out
        if duty is not None:
            out = self.set_square_duty(channel, duty)
            it_worked = it_worked and out
        if phase is not None:
            out = self.set_phase(channel, phase)
            it_worked = it_worked and out
        return it_worked

    def couple_channel(self, channel, mode='on'):
        """ Couple channel to the other (useful as a trigger)
        mode must be "Off" or "on"
        """

        if mode == 'on':
            command = "SOURCE{0}:FREQ:COUP:MODE Ratio".format(channel)
            self.write(command)

            command = "SOURCE{0}:FREQ:COUP:RAT 1".format(channel)
            self.write(command)
        else:
            command = "SOURCE{0}:FREQ:COUP:MODE Off".format(channel)
            self.write(command)