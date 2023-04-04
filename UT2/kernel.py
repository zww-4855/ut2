import numpy as np
import pyscf
from pyscf import ao2mo
from pyscf import cc
import UT2.t2energy as t2energy
import UT2.t2residEqns as t2residEqns
import sys

import UT2.modify_T2resid_T4Qf1 as qf1
import UT2.modify_T2resid_T4Qf2 as qf2
import UT2.modify_T2energy_pertQf as pertQf
from numpy import linalg

    
class UltT2CC():
    def __init__(self,storedInfo):
        self.tamps={}
        self.resid={}
        self.max_iter=storedInfo.get_cc_runtype("max_iter")
        self.stopping_eps=storedInfo.get_cc_runtype("stopping_eps")
        self.diis_size=storedInfo.get_cc_runtype("diis_size")
        self.diis_start_cycle=storedInfo.get_cc_runtype("diis_start_cycle")

        self.cc_type=storedInfo.get_cc_runtype("ccdType")
    
        self.nocca=storedInfo.get_occInfo("nocc_aa")
        self.noccb=storedInfo.get_occInfo("nocc_bb")
        self.nvrta=storedInfo.get_occInfo("nvirt_aa")
        self.nvrtb=storedInfo.get_occInfo("nvirt_bb")

        self.sliceInfo=storedInfo.get_occSliceInfo()
        self.ints=storedInfo.get_integralInfo()
        if "ccdType" in storedInfo.get_cc_runtype(None):
            t2aa,t2bb,t2ab,resT2aa,resT2bb,resT2ab=np.zeros((nvrta,nvrta,nocca,nocca))
            self.tamps = {"t2aa":t2aa,"t2bb":t2bb,"t2ab",t2ab}
            self.resid{"resT2aa":resT2aa,"resT2bb",resT2bb,"resT2ab":resT2ab}

            if self.diis_size is not None:
                from UT2.diis import DIIS
        
                self.diis_update = DIIS(self.diis_size, start_iter=self.diis_start_cycle)
                self.old_vec = np.hstack((self.tamps["t2aa"].flatten(), self.tamps["t2bb"].flatten(), self.tamps["t2ab"].flatten()))
     
        elif "fullCCtype" in storedInfo.get_cc_runtype(None):
            t1aa,t1bb,resT1aa,resT1bb=np.zeros((nvrta,nocca)) 
            t2aa,t2bb,t2ab,resT2aa,resT2bb,resT2ab=np.zeros((nvrta,nvrta,nocca,nocca))
            t3aa,t3bb,t3ab,resT3aa,resT3bb,resT3ab=np.zeros((nvrta,nvrta,nvrta,nocca,nocca,nocca))
            tamps={"t1aa":t1aa,"t1bb":t1bb,
                   "t2aa":t2aa,"t2bb":t2bb,"t2ab":t2ab,
                   "t3aa":t3aa,"t3bb":t3bb,"t3ab":t3ab}

            resid={"resT1aa":resT1aa,"resT1bb":resT1bb,
                   "resT2aa":resT2aa,"resT2bb":resT2bb,"resT2ab":resT2ab,
                   "resT3aa":resT3aa,"resT3bb":resT3bb,"resT3ab":resT3ab}

            if storedInfo.get_cc_runtype("fullCCtype") == "CCSDTQ":
                t4aa,t4bb,t4ab,resT4aa,resT4bb,resT4ab=np.zeros((nvrta,nvrta,nvrta,nvrta,nocca,nocca,nocca,nocca))
                tamps.update({"t4aa":t4aa,"t4bb":t4bb,"t4ab":t4ab})       
                resid.update({"resT4aa":resT4aa,"resT4bb":resT4bb,"resT4ab":resT4ab})

            self.tamps=tamps
            self.resid=resid


    def set_tamps(self,tamps_spin,label):
        self.tamps[str(label)]=tamps_spin

    def set_resid(self, resid_spin,label):
        self.resid[str(label)]=resid_spin

    def kernel():
        print("    ==> ", self.cc_runtype["ccdType"], " amplitude equations <==")
        print("")
        print("     Iter              Corr. Energy                 |dE|    ")
        print(flush=True)



        for idx in range(self.max_iter):
            self.l2{"l2aa":self.tamps["t2aa"].transpose(2,3,0,1),"l2bb":self.tamps["t2bb"].transpose(2,3,0,1),"l2ab":self.tamps["t2ab"].transpose(2,3,0,1)}

            # Updates all of aaaa,abab,bbbb spin residuals
            t2residEqns.residMain(self,self.tamps,self.ints,self.sliceInfo,self.cc_runtype)

