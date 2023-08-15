import pickle

class AmpHandler():
    def __init__(self,storedInfo,T2infile,T1infile=None,T3infile=None):
        self.t1=None
        self.t2=None
        self.t3=None
        self.getAmps()
        self.g=storedInfo.integralInfo["tei"]
        self.denoms=storedInfo.denoms



    def getAmps(self):
        with open(T2infile,'rb') as t2handle:
            self.t2=pickle.load(t2handle)

        if T1infile:
            with open(T1infile,'rb') as t1handle:
                self.t1=pickle.load(t1handle)

        if T3infile:
            with open(t3infile,'rb') as t3handle:
                self.t3=pickle.load(t3handle)

        
