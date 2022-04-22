import numpy as np 
import scpi

class MCA():

    def __init__(self, address, port=1001):
        self.address = address
        self.inst = scpi.scpi(self.address, port=port)