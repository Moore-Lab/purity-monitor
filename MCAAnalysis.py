import numpy as np
import glob

class MCAAnalysis():
    
    def __init__(self, Path, Selection='*'):
        self.Path = Path
        self.Selection = Selection
        self.Files = glob.glob(self.Path+self.Selection)
        print("Found %d files in path %s"%(len(self.Files), self.Path))