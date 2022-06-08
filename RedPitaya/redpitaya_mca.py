import socket
import paramiko
import h5py, datetime

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
        self.timer_multiple_seconds = 125000000 ## seconds to timer clicks
        self.un = 'root'
        self.pw = 'root'

        ## make sure the server code is running
        #self.start_mca()

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            if timeout is not None:
                self._socket.settimeout(timeout)

            self._socket.connect((host, port))

        except socket.error as e:
            print('SCPI >> connect({:s}:{:d}) failed: {:s}'.format(host, port, e))

    def __del__(self):

        #self.stop_mca()

        if self._socket is not None:
            self._socket.close()
        self._socket = None

    def close(self):
        """Close IP connection."""
        self.__del__()

    def ssh_connect(self):
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(self.host, username=self.un, password=self.pw)
        return ssh

    def start_mca(self):
        ssh = self.ssh_connect()
        ssh.exec_command('cat /opt/redpitaya/www/apps/mcpha/mcpha.bit > /dev/xdevcfg')
        ssh.exec_command('/opt/redpitaya/www/apps/mcpha/start.sh')

        ssh.close()

    def stop_mca(self):
        ssh = self.ssh_connect()
        ssh.exec_command('/opt/redpitaya/www/apps/mcpha/stop.sh')

        ssh.close()

    def connect(self):
        self.command(4,0,4)
        self.command(5,0,0)
        self.command(5,1,0)

    def setup_mca(self, chan=0, dec=4, negative=False, baseline_mode='none', baseline_level=0,
                        min_thresh=0, max_thresh=16380, trig_source=0, trig_slope=0, integ_time=100, delay=100 ):
        
        ## set decimation factor
        self.command(4,0,dec)

        ## set the channel polarity
        if(negative):
            self.command(5,chan,1)
        else:
           self.command(5,chan,0) 

        ## baseline mode (auto subtract or none)
        if baseline_mode == 'none':
            self.command(6,chan,0)
        elif baseline_mode == 'auto':
            self.command(6,chan,1)
        else:
            print("Failed to set baseline mode, must be '''none''' or '''auto''' ")

        ## baseline level (in bins)
        self.command(7,chan,baseline_level)

        ## PHA delay
        self.command(8,chan,delay)

        ## min/max threshold (in bins)
        self.command(9,chan,min_thresh)
        self.command(10,chan,max_thresh)


        ## integration time
        self.command(11, chan, integ_time*self.timer_multiple_seconds)
        self.command(0, chan, 0)

        ## PHA delay
        self.command(8,chan,delay)

        ## trigger
        #self.command(15, trig_source, 0) 
        #self.command(16, trig_source, trig_slope)

    def read_timer(self, chan=0):

        self.command(13, chan)
        t=self._socket.recv(8)
        return t
    
    def reset_histo(self, chan=0):
        
        self.command(12, chan, 0)
        self.command(0, chan, 0) ## reset timer
        self.command(1, chan, 0) ## reset histo
        
        self.read_timer(chan)
        self.read_histo_data(chan)

    def start_histo(self, chan=0):
        self.command(12, chan, 1)

    def pause_histo(self, chan=0):
        self.command(12, chan, 0)

    def command(self, code, chan, data=0):
 
        code_bytes = code.to_bytes(length=1, byteorder='little')

        if(chan == 0): ## 1 byte for the channel (0 or 1)
            chan_bytes = '\x00'.encode()
        else:
            chan_bytes = '\x10'.encode() 

        data_bytes = data.to_bytes(length=6, byteorder='little')

        buffer = data_bytes + chan_bytes + code_bytes

        self._socket.send(buffer)
        

    def read_histo_data(self, chan, size=65536):
        ## read the current histogram data from the MCA for a given channel and return a list of the
        ## histogram values
        ## chan -- channel number (must be 0 or 1)
        ## size -- should be left to default of 16384 bins x 4 bytes per bin
        
        read_enough_data = False ## loop until we get the right amount of data
        max_reads = 1
        for n in range(max_reads):
            self.command(14, chan) ## ask MCA for data
            data=self._socket.recv(size)
            converted_data = []
            nbytes = 4 ## for uint32s from the MCA code
            for i in range(int(len(data)/nbytes)):
                converted_data.append(int.from_bytes(data[(i*nbytes):((i+1)*nbytes)], byteorder='little', signed=False))
            
            if(len(converted_data) == int(size/nbytes)):
                read_enough_data = True
                break

        if read_enough_data:
            return(converted_data)
        else:
            return([]) ## return empty if we didn't get the data we expected

    def save(self, data, ch, tag, path):
        f = h5py.File("{}/{}.h5".format(path, tag), "w")
        grp1=f.create_group("ch1")
        grp2=f.create_group("ch2")
        date = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        grp1.attrs['time'] = date
        grp2.attrs['time'] = date
        if ch == 1:
            index='0'        
            grp1.create_dataset(index, data=data)

        elif ch ==2:
            index='0'        
            grp2.create_dataset(index, data=data)
                
        f.close()  