import pickle

class AmpHandler():
    def __init__(T1infile,T2infile,T3infile=None):
        with open(T1infile,'rb') as t1handle:
            t1=pickle.load(t1handle)
        with open(T2infile,'rb') as t2handle:
            t2=pickle.load(t2handle)

        if T3infile:
            with open(t3infile,'rb') as t3handle:
                t3=pickle.load(t3handle)
        else:
            t3=None

        self.t1=t1
        
