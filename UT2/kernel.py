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
import re
from numpy import einsum
import numpy as np
import UT2.XCCDbasebuilder as XCCDbasebuilder
import UT2.drive_ucc4 as drive_ucc4

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


def extract_integer(string):
    """
    Parses the method name string (ie XCCD(5) ) and returns a list of sequential integers up to the maximum order.
    :param string: Method being currently run
    :return: Either a list of sequential integers between [5-n] for maximum n=9, or None if a CCD calculation is specified.
    """
    tmp=string.split()
    store=[]
    print(tmp)
    for val in string:
        print(val)
        try:
            store.append(int(val))
        except:
            pass
    print(store)
    if not store:
        return None
    pertOrder=[]
    for i in range(5,store[-1:][0]+1):
        pertOrder.append(i)
    return pertOrder


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
        self.XCCDflag="XCCD" in self.cc_type
        self.xccd_resids="X" in self.cc_type # Boolean that when turned on, modifies the CCD T2 residual equations according to the order of XCCD
        if "ccdType" in storedInfo.get_cc_runtype(None) or "ccdTypeSlow" in storedInfo.get_cc_runtype(None):
            t2aa=t2bb=t2ab=resT2aa=resT2bb=resT2ab=np.zeros((self.nvrta,self.nvrta,self.nocca,self.nocca))
            self.tamps = {"t2aa":t2aa,"t2bb":t2bb,"t2ab":t2ab,'t1aa':None}
            self.resid = {"resT2aa":resT2aa,"resT2bb":resT2bb,"resT2ab":resT2ab}
            nvrta=self.nvrta
            nocca=self.nocca
            if self.diis_size is not None:
                from UT2.diis import DIIS
        
                self.diis_update = DIIS(self.diis_size, start_iter=self.diis_start_cycle)

                if "UCC(4)" in self.cc_type:
                    added_amps={'t1aa':np.zeros((nvrta,nocca)),'t3aa':np.zeros((nvrta,nvrta,nvrta,nocca,nocca,nocca))}
                    self.tamps.update(added_amps)

                    self.old_vec= np.hstack((self.tamps["t1aa"].flatten(), self.tamps["t2aa"].flatten(), self.tamps["t3aa"].flatten()))
                else:
                    self.old_vec = np.hstack((self.tamps["t2aa"].flatten(), self.tamps["t2bb"].flatten(), self.tamps["t2ab"].flatten()))
     
        elif "fullCCType" in storedInfo.get_cc_runtype(None):
            nvrta=self.nvrta
            nocca=self.nocca
            t1aa=t1bb=resT1aa=resT1bb=np.zeros((nvrta,nocca)) 
            t2aa=t2bb=t2ab=resT2aa=resT2bb=resT2ab=np.zeros((nvrta,nvrta,nocca,nocca))
            t3aaa=t3bbb=t3aab=t3abb=resT3aaa=resT3bbb=resT3aab=resT3abb=np.zeros((nvrta,nvrta,nvrta,nocca,nocca,nocca))
            t4aaaa=t4bbbb=t4aaab=t4aabb=t4abbb=np.zeros((nvrta,nvrta,nvrta,nvrta,nocca,nocca,nocca,nocca)) 

            tamps={"t1aa":t1aa,"t1bb":t1bb,
                   "t2aa":t2aa,"t2bb":t2bb,"t2ab":t2ab,
                   "t3aaa":t3aaa,"t3bbb":t3bbb,"t3aab":t3aab,"t3abb":t3abb,
                   "t4aaaa":t4aaaa,"t4bbbb":t4bbbb,"t4aaab":t4aaab,"t4aabb":t4aabb,"t4abbb":t4abbb}

            resid={"resT1aa":resT1aa,"resT1bb":resT1bb,
                   "resT2aa":resT2aa,"resT2bb":resT2bb,"resT2ab":resT2ab,
                   "resT3aaa":resT3aaa,"resT3bbb":resT3bbb,"resT3aab":resT3aab,"resT3abb":resT3abb}

            self.tamps=tamps
            if self.diis_size is not None:
                from UT2.diis import DIIS

                self.diis_update = DIIS(self.diis_size, start_iter=self.diis_start_cycle)
                self.old_vec= np.hstack((self.tamps["t2aa"].flatten(), self.tamps["t2bb"].flatten(), self.tamps["t2ab"].flatten(), self.tamps["t4aaaa"].flatten(),self.tamps["t4aaab"].flatten(),self.tamps["t4aabb"].flatten(),self.tamps["t4abbb"].flatten(),self.tamps["t4bbbb"].flatten()))

            #if storedInfo.get_cc_runtype("fullCCType") == "CCSDTQ" or storedInfo.get_cc_runtype("fullCCType")=='CCDQ':
            #    t4aaaa=t4bbbb=t4aaab=t4aabb=t4abbb==np.zeros((nvrta,nvrta,nvrta,nvrta,nocca,nocca,nocca,nocca))
            #    tamps.update({"t4aaaa":t4aaaa,"t4bbbb":t4bbbb,"t4aaab":t4aaab,"t4aabb":t4aabb,"t4abbb":t4abbb})       
            #    #resid.update({"resT4aa":resT4aa,"resT4bb":resT4bb,"resT4ab":resT4ab})

            self.tamps=tamps
            self.resid=resid

        self.contractInfo={"nocc":self.nocca,"nvir":self.nvrta,"tamps":self.tamps["t2aa"],"ints":self.ints["tei"],"oa":self.sliceInfo["occ_aa"],"va":self.sliceInfo["virt_aa"]}
        self.pertOrder=extract_integer(self.cc_type)
        print(self.pertOrder,self.cc_type)

    
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
        
        :param nucE: nuclear repulsion energy
        :param current_energy: Current value for the (CCD-like) correlation energy ONLY
        :param hf_energy: Energy associated with SCF

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
            #corrE=nucE+current_energy
            tfinalEnergy=current_energy+nucE

        if self.cc_type == "CCD(Qf)" or self.cc_type == "CCD(Qf*)" or self.cc_type == "CCSDT(Qf)" or self.cc_type == "CCSDT(Qf*)" or self.cc_type == "CCSD(Qf)":
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
            # Total correlation energy is the base CCD energy plus the perturbative correction to the energy
            ut2_energy=t2energySlow.ccd_energyMain(self,get_perturbCorr=True)
            print('Perturbative energy correction associated with UT2 method: \t {: 20.12f}'.format(ut2_energy))
            corrE+=ut2_energy
            print('Total correlation energy (CCD E + UT2-CCD(n) E): \t {: 20.12f}'.format(corrE))
            tfinalEnergy=current_energy+nucE+ut2_energy

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
            sliceInfo=self.sliceInfo
            oa=sliceInfo["occ_aa"]
            va=sliceInfo["virt_aa"]
            self.tamps["t2aa"]=0.0*self.ints["tei"][va,va,oa,oa]*self.denom["D2aa"]
            if "UCC(4)" in self.cc_type:
                old_energy=drive_ucc4.ucc4_energy_simplified(self)
            else:
                old_energy = t2energySlow.ccd_energyMain(self) 
            print('mp2 energy:', 0.25*np.einsum('jiab,abji',self.ints["tei"][oa,oa,va,va],self.tamps["t2aa"]))
            print('initial CC energy:',old_energy)
        elif self.cc_label == "fullCCType":
            old_energy = fullCCenergy.fullCC_energyMain(self) 


        for idx in range(self.max_iter):
            self.set_l2amps()

            # Updates all of aaaa,abab,bbbb spin residuals
            if self.cc_label == "ccdType":
                t2residEqns.residMain(self)
            elif self.cc_label == "ccdTypeSlow":
                if "UCC(4)" in self.cc_type:
                    drive_ucc4.resid_main(self)
                else:
                    t2residEqnsSlow.residMain(self)

            elif self.cc_label == "fullCCType":
                fullCCresidEqns.residMain(self)


        # diis update
            if self.diis_size is not None:
                if "UCC(4)" in self.cc_type:
                    vectorized_iterate = np.hstack(
                        (
                            self.tamps["t1aa"].flatten(),
                            self.tamps["t2aa"].flatten(),
                            self.tamps["t3aa"].flatten(),
                        )
                    )
                elif "CCDQ" in self.cc_type:
                    vectorized_iterate = np.hstack(
                        (
                            self.tamps["t2aa"].flatten(),
                            self.tamps["t2bb"].flatten(),
                            self.tamps["t2ab"].flatten(),
                            self.tamps["t4aaaa"].flatten(),
                            self.tamps["t4aaab"].flatten(),
                            self.tamps["t4aabb"].flatten(),
                            self.tamps["t4abbb"].flatten(),
                            self.tamps["t4bbbb"].flatten(),
                        )
                    )
                else:
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

                if "UCC(4)" in self.cc_type:
                    t1=self.tamps["t1aa"]
                    t2=self.tamps["t2aa"]
                    t3=self.tamps["t3aa"]
                    t1_dim=self.tamps["t1aa"].size
                    t2_dim=self.tamps["t2aa"].size
                    t3_dim=self.tamps["t3aa"].size
                    self.tamps["t1aa"] = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
                    self.tamps["t2aa"] = new_vectorized_iterate[
                        t1_dim : t1_dim + t2_dim
                    ].reshape(t2.shape)
                    self.tamps["t3aa"] = new_vectorized_iterate[
                        t1_dim + t2_dim :
                    ].reshape(t3.shape)
                    self.old_vec = new_vectorized_iterate
                elif "CCDQ" in self.cc_type:
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
                        t2aaaa_dim + t2bbbb_dim : t2aaaa_dim + t2bbbb_dim + t2abab_dim
                    ].reshape(t2abab.shape)

                    start=t2aaaa_dim + t2bbbb_dim + t2abab_dim
                    t4_dim1=t2aaaa_dim + t2bbbb_dim + t2abab_dim + self.tamps["t4aaaa"].size
                    t4_dim2=t4_dim1 + self.tamps["t4aaab"].size
                    t4_dim3=t4_dim2 + self.tamps["t4aabb"].size
                    t4_dim4=t4_dim3 + self.tamps["t4abbb"].size
                    t4_dim5=t4_dim4 + self.tamps["t4bbbb"].size

                    t4aaaa_shape=self.tamps["t4aaaa"].shape
                    t4aaab_shape=self.tamps["t4aaab"].shape
                    t4aabb_shape=self.tamps["t4aabb"].shape
                    t4abbb_shape=self.tamps["t4abbb"].shape
                    t4bbbb_shape=self.tamps["t4bbbb"].shape
                    self.tamps["t4aaaa"] = new_vectorized_iterate[start:t4_dim1].reshape(t4aaaa_shape)
                    self.tamps["t4aaab"] = new_vectorized_iterate[t4_dim1:t4_dim2].reshape(t4aaab_shape)
                    self.tamps["t4aabb"] = new_vectorized_iterate[t4_dim2:t4_dim3].reshape(t4aabb_shape)
                    self.tamps["t4abbb"] = new_vectorized_iterate[t4_dim3:t4_dim4].reshape(t4abbb_shape)
                    self.tamps["t4bbbb"] = new_vectorized_iterate[t4_dim4:].reshape(t4bbbb_shape)

                    self.old_vec = new_vectorized_iterate
                else:
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
                if "UCC(4)" in self.cc_type:
                    current_energy=drive_ucc4.ucc4_energy_simplified(self)
                else:
                    current_energy = t2energySlow.ccd_energyMain(self)
            elif self.cc_label == "fullCCType":
                current_energy = fullCCenergy.fullCC_energyMain(self)

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
                pickle.dump(self.tamps["t2aa"], handle)


        print(self.tamps["t2aa"][0,0,0,:],'\n',self.tamps["t2aa"][0,:,0,0]) 
        self.finalize(self.nucE,current_energy,self.hf_energy)
 
        return self.tamps,self.tfinalEnergy, self.corrE



class BuildBaseAmps():
    """
    Constructs the base amplitudes for a given method (ie XCCD(n), UCCD(n), or perturbatively corrected CC theory

    :param UltT2CC: The CC object containing all pertinent data structures such as T amplitudes, energy denominators, etc 
    :return: Sets the  t_base parameters in parent UltT2CC class
    """
    def __init__(self,contractInfo):
        self.t_base={}
        #self.UltT2CC=UltT2CC
        self.contractInfo=contractInfo #UltT2CC.contractInfo
        #self.t_amps=None

    def buildXCCDbase(self, t2, order=5):
        """
    Constructs the base T2 amplitudes for XCCD(5-9) and sets them to the UltT2CC class parameter for later use

    :param order: XCCD order
        """
        #t2 = self.UltT2CC.tamps["t2aa"]
        resid = XCCDbasebuilder.build_XCCDbase(t2,order,self.contractInfo)
        self.t_base.update({order:resid})


    def buildUCCDbase(self,order=5):
        pass




class ContractAdjointAmps():
    """
    Contracts the base amplitudes of a given method (ie XCCD(n), UCCD(n) with either T^daggers or 2e- integrals. One choice of capping the base amplitudes is made after it has been determined whether to rigorously invoke the factorization theorem

    :param UltT2CC: The CC object containing all pertinent data structures such as T amplitudes, energy denominators, etc
    :return: Returns a calculated energy, or a dictionary of T amplitudes, depending on whether we are trying to modify the residual eqns or simply calculate an energy.  
    """
    def __init__(self,UltT2CC):
        self.result={}
        self.UltT2CC=UltT2CC
        self.contractInfo=UltT2CC.contractInfo
        self.sliceInfo=UltT2CC.sliceInfo
        self.g=UltT2CC.contractInfo["ints"]


    def buildXCCD_T2energy(self,order=5, factorization=True):
        """
    Returns XCCD-like corrections to the energy. Note, that this can be in the style of CCSDT(Qf) where we cap with T2^\dag and one W-2, or it can be in style of XCCD where we simply cap with all T2^\dag

    :param order: order of XCCD
    :factorization: Boolean variable that determines if we cap with a final W-2 (True) or a T2^\dag (False). Default is False.

    :return: XCCD-like energy correction at some order
        """
        t2=self.UltT2CC.tamps["t2aa"]
        #follow CCSDT(Qf) route to calculate energy 
        if factorization == True:
            g=self.g #UltT2CC.ints["tei"]
            D2=self.UltT2CC.denom["D2aa"]
            oa=self.sliceInfo["occ_aa"]
            va=self.sliceInfo["virt_aa"]
            t2_dag = g[va,va,oa,oa] * D2
            t2_dag1 = t2_dag.transpose(2,3,0,1)

        else:# Do XCCD-like correction
            t2_dag1=t2.transpose(2,3,0,1)

        t2_order=XCCDbasebuilder.build_XCCDbase(t2,order,self.contractInfo)
        t2energy_mod = (1.0/4.0)*einsum("ijab,abij",t2_dag1,t2_order)

        self.result.update({order:t2energy_mod})
        return t2energy_mod

