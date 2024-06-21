import numpy as np
import pickle

def set_tampsSLOW(cc_calc,no,nv,t2ampFile=None):
    tamps={}
    if "S" in cc_calc:
        t1=np.zeros((no,nv))
        tamps.update({"t1aa":t1})

    if "D" in cc_calc:
        if t2ampFile:
            with open(t2ampFile,'rb') as t2handle:
                t2=pickle.load(t2handle)
        else:
            t2=np.zeros((no,no,nv,nv))#self.ints["tei"][va,va,oa,oa]*self.denom["D2aa"]

        tamps.update({"t2aa":t2})

    if "T" in cc_calc:
        t3=np.zeros((no,no,no,nv,nv,nv))
        tamps.update({"t3aa":t3})

    return tamps


def get_oldvec(tamps,cc_calc):
    if "DQ" in cc_calc:
        t2=tamps["t2aa"].flatten()
        t4=tamps["t4aa"].flatten()
        oldvec=np.hstack((t2,t4))
    elif "SDT" in cc_calc: 
        t1=tamps["t1aa"].flatten()
        t2=tamps["t2aa"].flatten()
        t3=tamps["t3aa"].flatten()
        oldvec=np.hstack((t1,t2,t3))
    elif "SD" in cc_calc:
        t1=tamps["t1aa"].flatten()
        t2=tamps["t2aa"].flatten()
        oldvec=np.hstack((t1,t2))
    elif "D" in cc_calc:
        t2=tamps["t2aa"].flatten()
        oldvec=np.hstack((t2))

    return oldvec


