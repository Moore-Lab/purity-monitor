import socket

class mca (object):
    """MCA class used to access Red Pitaya over an IP network."""
    delimiter = '' #'\r\n'

    def __init__(self, host='172.28.175.57', timeout=2, port=1001):
        """Initialize object and open IP connection.
        Host IP should be a string in parentheses, like '192.168.1.100'.
        """
        self.host    = host
        self.port    = port
        self.timeout = timeout

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            if timeout is not None:
                self._socket.settimeout(timeout)

            self._socket.connect((host, port))

        except socket.error as e:
            print('SCPI >> connect({:s}:{:d}) failed: {:s}'.format(host, port, e))

    def __del__(self):
        if self._socket is not None:
            self._socket.close()
        self._socket = None

    def close(self):
        """Close IP connection."""
        self.__del__()

    
    def rx_txt(self, chunksize = 4096):
        """Receive text string and return it after removing the delimiter."""
        return self._socket.recv(chunksize) #.decode('utf-8') # Receive chunk size of 2^n 

    def command(self, code, chan, data=0):
        code_str = chr(code) # "%#02x"%code ## two byte hex string for the code (see server code)
        if(chan == 0):
            chan_str = '\x00'
        else:
            chan_str = '\x10' #"%#01x"%chan ## 1 byte for the channel (0 or 1)
        data_str = "\x00\x00\x00\x00\x00\x00" #this is a dummy string with right number of bits for now... fix later
        buffer = data_str + chan_str + code_str
        self._socket.send(buffer.encode('utf-8'))
        

    def read_histo_data(self, chan, size=65536):
        self.command(14, chan)
        data=self._socket.recv(size)
        print(data)
        return(list(data))