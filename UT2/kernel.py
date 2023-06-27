"""
Handles the iterative solution of the CC residual equations
"""

import numpy as np
import pyscf
from pyscf import ao2mo
from pyscf import cc
import UT2.t2energy as t2energy
import UT2.t2residEqns as t2residEqns
import UT2.fullCCenergy as fullCCenergy

import UT2.t2energySlow as t2energySlow
import UT2.t2residEqnsSlow as t2residEqnsSlow
import UT2.fullCCresidEqns as fullCCresidEqns


import sys

import UT2.modify_T2resid_T4Qf1 as qf1
import UT2.modify_T2resid_T4Qf2 as qf2
import UT2.modify_T2energy_pertQf as pertQf
from numpy import linalg



def get_calc(storedInfo,calc_list):
    """
    Determines which among the "ccdType", "ccdTypeSlow", and "fullCCType" tags are being used in the UT2 call. 
    Parameters
    ----------
    storedInfo : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    calc_list : list of calculation types

    Returns
    -------
    calcType : str
        Compiled string including quote and optional attribution.
    cc_runtype : str, runtype of coupled cluster calc 

    :param storedInfo:
    :param calc_list: list of possible calculation labels
    
    :return: Returns the calculation label, and CC calculation type
    
    """
    for calcType in calc_list:
        try:
            return calcType, storedInfo.get_cc_runtype(calcType)
        except:
            continue


class UltT2CC():
    """
    UltT2CC class contains all the necessary routines to setup and run the various CC implementations. Current implementation is tested for spin-integrated and spin-orbital T2 methods only.
    
    :param storedInfo: A dictionary-based Class containing all the pertinent background information required to setup the CC calculation. See StoredInfo() class in run_ccd.py for further information.
    
    :return: Returns the correlation and total energy. If needed, the converged
    set of T2 amplitudes can also be returned
    """
    def __init__(self,storedInfo):
        self.calc_list=["ccdType","ccdTypeSlow","fullCCType"]
        self.tamps={}
        self.resid={}
        self.max_iter=storedInfo.get_cc_runtype("max_iter")
        self.stopping_eps=storedInfo.get_cc_runtype("stopping_eps")
        self.diis_size=storedInfo.get_cc_runtype("diis_size")
        self.diis_start_cycle=storedInfo.get_cc_runtype("diis_start_cycle")
        self.dump_tamps=storedInfo.get_cc_runtype("dump_tamps")
        self.pert_wvfxn_corr=storedInfo.get_cc_runtype("pert_wvfxn_corr")
        self.pert_E_corr = storedInfo.get_cc_runtype("pert_E_corr")
 
 
        self.cc_label, self.cc_type=get_calc(storedInfo,self.calc_list) #storedInfo.get_cc_runtype("ccdType")
        self.nucE=storedInfo.get_cc_runtype("nuclear_energy")
        self.hf_energy=storedInfo.get_cc_runtype("hf_energy") 

        self.nocca=storedInfo.get_occInfo("nocc_aa")
        self.noccb=storedInfo.get_occInfo("nocc_bb")
        self.nvrta=storedInfo.get_occInfo("nvirt_aa")
        self.nvrtb=storedInfo.get_occInfo("nvirt_bb")
        self.denom=storedInfo.get_denoms()
        self.sliceInfo=storedInfo.get_occSliceInfo()
        self.ints=storedInfo.get_integralInfo()
        self.l2={}
        self.t_base={}
        print(self.nvrta)
        self.contractInfo={"nocc":self.nocca,"nvir":self.nvrta,"tamps":self.tamps["t2aa"],"ints":self.ints["tei"],"oa":self.sliceInfo["occ_aa"],"va":self.sliceInfo["virt_aa"]}


        if "ccdType" in storedInfo.get_cc_runtype(None) or "ccdTypeSlow" in storedInfo.get_cc_runtype(None):
            t2aa=t2bb=t2ab=resT2aa=resT2bb=resT2ab=np.zeros((self.nvrta,self.nvrta,self.nocca,self.nocca))
            self.tamps = {"t2aa":t2aa,"t2bb":t2bb,"t2ab":t2ab}
            self.resid = {"resT2aa":resT2aa,"resT2bb":resT2bb,"resT2ab":resT2ab}

            if self.diis_size is not None:
                from UT2.diis import DIIS
        
                self.diis_update = DIIS(self.diis_size, start_iter=self.diis_start_cycle)
                self.old_vec = np.hstack((self.tamps["t2aa"].flatten(), self.tamps["t2bb"].flatten(), self.tamps["t2ab"].flatten()))
     
        elif "fullCCType" in storedInfo.get_cc_runtype(None):
            nvrta=self.nvrta
            nocca=self.nocca
            t1aa=t1bb=resT1aa=resT1bb=np.zeros((nvrta,nocca)) 
            t2aa=t2bb=t2ab=resT2aa=resT2bb=resT2ab=np.zeros((nvrta,nvrta,nocca,nocca))
            t3aaa=t3bbb=t3aab=t3abb=resT3aaa=resT3bbb=resT3aab=resT3abb=np.zeros((nvrta,nvrta,nvrta,nocca,nocca,nocca))
            

            tamps={"t1aa":t1aa,"t1bb":t1bb,
                   "t2aa":t2aa,"t2bb":t2bb,"t2ab":t2ab,
                   "t3aaa":t3aaa,"t3bbb":t3bbb,"t3aab":t3aab,"t3abb":t3abb}

            resid={"resT1aa":resT1aa,"resT1bb":resT1bb,
                   "resT2aa":resT2aa,"resT2bb":resT2bb,"resT2ab":resT2ab,
                   "resT3aaa":resT3aaa,"resT3bbb":resT3bbb,"resT3aab":resT3aab,"resT3abb":resT3abb}

            if storedInfo.get_cc_runtype("fullCCType") == "CCSDTQ":
                t4aa=t4bb=t4ab=resT4aa=resT4bb=resT4ab=np.zeros((nvrta,nvrta,nvrta,nvrta,nocca,nocca,nocca,nocca))
                tamps.update({"t4aa":t4aa,"t4bb":t4bb,"t4ab":t4ab})       
                resid.update({"resT4aa":resT4aa,"resT4bb":resT4bb,"resT4ab":resT4ab})

            self.tamps=tamps
            self.resid=resid

        #elif "ccdTypeSlow" in storedInfo.get_cc_runtype(None):
        

    
    def set_tamps(self,tamps_spin,label=None):
        """
        Updates the set of T amplitudes. Options for specific amplitudes (ie T2aa) or simply stores a dictionary of T amplitude information.
        
        :param tamps_spin: dictionary that stores requiste T amplitudes
        :param label: Optional argument that, if present, dictates which specific ampltiudes are going to be reset
        
        """
        if label==None:
            self.tamps=tamps_spin
        else:
            self.tamps[str(label)]=tamps_spin

    def set_resid(self, resid_spin,label=None):
        """
        Updates dictionary of residual equations using dictionary using same philosophy as set_tamps method
        """
        if label==None:
            self.resid=resid_spin
        else:
            self.resid[str(label)]=resid_spin

    def set_l2amps(self,firstOrderT2=None):
        if firstOrderT2 == None:
            self.l2={"l2aa":self.tamps["t2aa"].transpose(2,3,0,1),"l2bb":self.tamps["t2bb"].transpose(2,3,0,1),"l2ab":self.tamps["t2ab"].transpose(2,3,0,1)}
        else:
            sliceInfo=self.sliceInfo
            o=sliceInfo["occ_aa"]
            v=sliceInfo["virt_aa"]
            print('running first order T2 for L2')
            tei=self.ints["tei"]
            g_aaaa=tei["g_aaaa"][v,v,o,o]*self.denom["D2aa"]
            g_bbbb=tei["g_bbbb"][v,v,o,o]*self.denom["D2bb"]
            g_abab=tei["g_abab"][v,v,o,o]*self.denom["D2ab"]
          
            t2aaDag=g_aaaa.transpose(2,3,0,1)
            t2bbDag=g_bbbb.transpose(2,3,0,1)
            t2abDag=g_abab.transpose(2,3,0,1)
 
            self.l2={"l2aa":t2aaDag,"l2bb":t2bbDag,"l2ab":t2abDag}

    def get_l2amps(self):
        return self.l2

    def finalize(self,nucE,current_energy,hf_energy):
        """
        Handles final printing when CC iterations are converged. Also extract perturbative corrections based on converged T ampltitudes (ie CCD(Qf) ) if necessary
        
        """
        print("\n\n\n")
        print("************************************************************")
        print("************************************************************\n\n")
        print('Total SCF energy: \t {: 20.12f}'.format(hf_energy))
        print('Nuclear repulsion energy: \t {: 20.12f}'.format(nucE))
        print(f"\n \t**** Results for {self.cc_type}: ****")

        UT2_run=t2energySlow.extract_UT2(self.cc_type)
        print('Is this a UT2 run?', UT2_run) 
        if self.cc_type != "CCD(Qf)":
            corrE=nucE+current_energy-hf_energy
            print('Correlation energy, without perturbative effects: \t {: 20.12f}'.format(corrE))
            corrE=nucE+current_energy
            tfinalEnergy=current_energy+nucE

        if self.cc_type == "CCD(Qf)" or self.cc_type == "CCD(Qf*)" or self.cc_type == "CCSDT(Qf)" or self.cc_type == "CCSDT(Qf*)":
            corrE=nucE+current_energy-hf_energy

            if self.cc_label == "ccdType":
                qf_corr = t2energy.ccd_energyMain(self,get_perturbCorr=True)

            elif self.cc_label == "ccdTypeSlow":
                qf_corr = t2energySlow.ccd_energyMain(self, get_perturbCorr=True)

            elif self.cc_label == "fullCCType":
                qf_corr = fullCCenergy.fullCC_energyMain(self, get_perturbCorr=True)


            print("CCD correlation contribution: \t {: 20.12f}".format(corrE))
            print("(Qf) perturbative energy correction: \t {: 20.12f}".format(qf_corr)) 
            tfinalEnergy=current_energy+nucE+qf_corr
            corrE=qf_corr

        if UT2_run:
            corrE=t2energySlow.ccd_energyMain(self,get_perturbCorr=True)
            print('ut2 corrE:',corrE)
            tfinalEnergy=nucE+corrE
            perturbE=corrE-current_energy
            print("UT2-CCD perturbative energy correction: \t {: 20.12f}".format(perturbE))

        self.tfinalEnergy=tfinalEnergy
        self.corrE=corrE
        print('Final CC energy: \t {: 20.12f}'.format(self.tfinalEnergy))
        print("\n\n\n")
        return self

    def kernel(self):
        """
        Driver routine that iterates the CC equations. Handles all background tasks such as setting intermediate T amplitudes, etc, for the spin-integrated and spin-orbital, T2-based methods. No functionality built for higher order methods that incorporate amplitudes other than T2. 
        
        :return: Returns set of converged T amplitudes, as well as the correlation and final total energy
        
        """
        print("    ==> ", self.cc_type, " amplitude equations <==")
        print("")
        print("     Iter              Corr. Energy                 |dE|    ")
        print(flush=True)

        print(self.cc_label)
        if self.cc_label == "ccdType":
            old_energy = t2energy.ccd_energyMain(self)
        elif self.cc_label =="ccdTypeSlow":
            old_energy = t2energySlow.ccd_energyMain(self) 

        elif self.cc_label == "fullCCType":
            old_energy = fullCCenergy.fullCC_energyMain(self) 


        for idx in range(self.max_iter):
            self.set_l2amps()
            #self.l2={"l2aa":self.tamps["t2aa"].transpose(2,3,0,1),"l2bb":self.tamps["t2bb"].transpose(2,3,0,1),"l2ab":self.tamps["t2ab"].transpose(2,3,0,1)}

            # Updates all of aaaa,abab,bbbb spin residuals
            if self.cc_label == "ccdType":
                t2residEqns.residMain(self)
            elif self.cc_label == "ccdTypeSlow":
                t2residEqnsSlow.residMain(self)

            elif self.cc_label == "fullCCType":
                fullCCresidEqns.residMain(self)


        # diis update
            if self.diis_size is not None:
                vectorized_iterate = np.hstack(
                    (
                        self.tamps["t2aa"].flatten(),
                        self.tamps["t2bb"].flatten(),
                        self.tamps["t2ab"].flatten(),
                    )
                )
                error_vec = self.old_vec - vectorized_iterate
                new_vectorized_iterate = self.diis_update.compute_new_vec(
                    vectorized_iterate, error_vec
                )
                t2aaaa=self.tamps["t2aa"]
                t2bbbb=self.tamps["t2bb"]
                t2abab=self.tamps["t2ab"]
                t2aaaa_dim=self.tamps["t2aa"].size
                t2bbbb_dim=self.tamps["t2bb"].size
                t2abab_dim=self.tamps["t2ab"].size
                self.tamps["t2aa"] = new_vectorized_iterate[:t2aaaa_dim].reshape(t2aaaa.shape)
                self.tamps["t2bb"] = new_vectorized_iterate[
                    t2aaaa_dim : t2aaaa_dim + t2bbbb_dim
                ].reshape(t2bbbb.shape)
                self.tamps["t2ab"] = new_vectorized_iterate[
                    t2aaaa_dim + t2bbbb_dim :
                ].reshape(t2abab.shape)
                self.old_vec = new_vectorized_iterate


            if self.cc_label == "ccdType":
                current_energy = t2energy.ccd_energyMain(self)
            elif self.cc_label =="ccdTypeSlow":
                current_energy = t2energySlow.ccd_energyMain(self)
            elif self.cc_label == "fullCCType":
                current_energy = fullCCenergy.fullCC_energyMain(self)

#            current_energy = t2energy.ccd_energyMain(self)
            delta_e = np.abs(old_energy - current_energy)

            print(
                "    {: 5d} {: 20.12f} {: 20.12f} ".format(
                    idx, self.nucE + current_energy - self.hf_energy, delta_e
                )
            )
            print(flush=True)
            if delta_e < self.stopping_eps:  # and res_norm < stopping_eps:
                break
            else:
                old_energy = current_energy
            if idx > self.max_iter: 
                raise ValueError("CC iterations did not converge")

        if self.dump_tamps == True:
            import pickle
            with open('tamps.pickle', 'wb') as handle:
                pickle.dump(self.tamps, handle)
 
        self.finalize(self.nucE,current_energy,self.hf_energy)
 
        return self.tamps,self.tfinalEnergy, self.corrE
