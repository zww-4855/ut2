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

import UT2.kernel as kernel
import pickle
import UT2.amphandler as ampHandles
import UT2.xacc as xacc

class StoredInfo():
    """ StoredInfo() is a class that holds all data necessary for a CC calculation. Similar in design to C-struct
    
    
    :param occupationInfo: A dictionary containing information regarding the occupation of occupied/virtual orbitals
    
    :param occupationSliceInfo: A dictionary containing information regarding the slicing structure of occupied/virtual orbitals
    
    :param denomInfo: A dictionary containing information on the relevant denominators for any CC
    
    :param cc_runtype: Dictionary specifying information regarding the correlation calculation that is requested. Has two callable options; "ccdType" which refers to any of the T2-based methods that are implemented, and "fullCCType" which calls one of CCSDT, CCSDTQ, or CCSDTQf calculations. A third option, "ccdTypeSlow" will lead to spin-orbital formulations of the T2-based methods. Also contains information such as convergence criteria, number of diis vectors & length, type of CC calculation, and max. number of iterations allowed in a given CC calculation
    
    
    :param tamps: A dictionary containing the current iterations T-amplitude
    
    :param resid: A Dictionary containing the current iterations residuals
    
    :return: An archive object that can be queried for the various quantities in question
    """



    def __init__(self,occupationInfo={},occupationSliceInfo={},denomInfo={},cc_runtype={},tamps={},resid={},integralInfo={}):
        self.occInfo=occupationInfo
        self.occSliceInfo=occupationSliceInfo
        self.denomInfo=denomInfo
        self.cc_runtype=cc_runtype
        self.integralInfo=integralInfo

    def set_occInfo(self,occInfo):
        ''' Sets the occupation for occupied/virtual orbitals'''
        self.occInfo=occInfo

    def set_occSliceInfo(self,occSliceInfo):
        ''' Sets the occupation slice information to be used in subsequent CC tensor contractions'''
        self.occSliceInfo=occSliceInfo

    def set_denoms(self,denomInfo):
        ''' Sets the energy denominators'''
        self.denomInfo=denomInfo

    def set_cc_runtype(self,cc_runtype):
        ''' Sets the cc_runtype dictionary that dictates specifics of the CC calculation'''
        self.cc_runtype=cc_runtype

    def set_integralInfo(self,integralInfo):
        ''' Sets the fock matrix and set of 2electron integrals'''
        self.integralInfo=integralInfo

    def get_denoms(self,dataString=None):
        ''' Returns set of energy denominators to turn residual into T amplitudes '''
        if dataString==None:
            return self.denomInfo
        else:
            return self.denomInfo[str(dataString)]

    def get_occInfo(self,dataString):
        ''' Returns occupation info of the occupied/virtual orbitals '''
        try:
            return self.occInfo[str(dataString)]
        except:
            return None

    def get_occSliceInfo(self,dataString=None):
        ''' Returns occupational slice info '''
        if dataString==None:
            return self.occSliceInfo
        else:
            return self.occSliceInfo[str(dataString)]

    def get_cc_runtype(self,dataString=None):
        ''' Returns cc_runtype dictionary '''
        if dataString==None:
            return self.cc_runtype
        else:
            return self.cc_runtype[str(dataString)]

    def get_integralInfo(self,elec_spec=None,dataString=None):
        ''' Returns set of fock and two electron integrals in dictionary '''
        if elec_spec==None:
            return self.integralInfo

        dic=self.integralInfo[str(elec_spec)]
        if dataString==None:
            return dic
        else:
            return dic[str(dataString)]



def ccd_main(mf, mol, orb, cc_runtype):
    """ ccd_main() is a general function that performs the background tasks necessary for a general CC calculation to take place, based on either RHF or UHF orbitals. Serves as a driver for both the spin-integrated and spin-orbital based CC codes.
    
    :param mf: A PySCF SCF object, containing pertinent info regarding SCF calculation
    :param mol: PySCF object containing info regarding the info regarding the molecule/system in question
    :param orb: Set of - presumably converged - SCF coefficients
    :param cc_runtype: Dictionary specifying information regarding the correlation calculation that is requested. Has two callable options; "ccdType" which refers to any of the T2-based methods that are implemented, and "fullCCType" which calls one of CCSDT, CCSDTQ, or CCSDTQf calculations. A third option, "ccdTypeSlow" will lead to spin-orbital formulations of the T2-based methods. Also contains such as convergence criteria, number of diis vectors & length, type of CC calculation, and max. number of iterations allowed in a given CC calculation
    
    
    :return: Returns the correlation and full (combined correlation + mean field) energy
    
    """



    print('starting Ut2...')
    print(flush=True)
    storedInfo=StoredInfo()
    cc_runtype.update({"hf_energy":mf.e_tot,"nuclear_energy":mf.energy_nuc()})
    if "stopping_eps" not in cc_runtype:
        cc_runtype.update({"stopping_eps":10**-10})

    if "max_iter" not in cc_runtype:
        cc_runtype.update({"max_iter":75})

    if "dump_tamps" not in cc_runtype:
        cc_runtype.update({"dump_tamps":False})


    if "pert_wvfxn_corr" not in cc_runtype:
        cc_runtype.update({"pert_wvfxn_corr":"0"})
 
    if "pert_E_corr" not in cc_runtype:
        cc_runtype.update({"pert_E_corr":"0"})

    if "pertCorr" in cc_runtype: # keyword used to jumpstart the perturbative correction-only logic
        # rely solely on xacc-piped information like oei,tei,tamps,denoms,etc
        if cc_runtype["pertOrderSoftware"]=='xacc':
            infiles=cc_runtype["xaccfiles"]
            xaccObj=xacc.run_xacc(infiles["fock"],infiles["tamps"],infiles["ints"])

            occupationInfo={'nocc_aa':xaccObj.nocc,'nvirt_aa':xaccObj.nvirt}
            occupationSliceInfo={'occ_aa':xaccObj.o,'virt_aa':xaccObj.v}
            integralInfo={'oei':xaccObj.mo_energies,'tei':xaccObj.tei}
            denomInfo={"D1aa":xaccObj.e_ai, "D2aa":xaccObj.e_abij, "D3aa":xaccObj.e_abcijk}
            storedInfo=StoredInfo(occupationInfo,occupationSliceInfo,denomInfo,cc_runtype,None,None,integralInfo)
            ampObj=ampHandles.AmpHandler(xaccObj.o,xaccObj.v,storedInfo,None,None,xaccObj)
        else:#otherwise,read external T1/T2 amplitude files and run pyscf for ints
            storedInfo=convertSCFinfo_tmpslow(mf,mol,orb,cc_runtype,storedInfo)
            # overwrite/add denominator information for T1/T3 amplitudes to storedInfo    
            n = np.newaxis
            o=storedInfo.occSliceInfo["occ_aa"]
            v=storedInfo.occSliceInfo["virt_aa"]
            nocc=storedInfo.occInfo["nocc_aa"]
            nvirt=storedInfo.occInfo["nvirt_aa"]
            eps=np.kron(mf.mo_energy,np.ones(2))
            eps = np.sort(eps)
            print('eps:',eps)
            print('o',o,'v',v)
            D1_aa=1.0/(-eps[v,n]+eps[n,o])
            D2_aa=1.0/(-eps[v,n,n,n]-eps[n,v,n,n]+eps[n,n,o,n]+eps[n,n,n,o])
            D3_aa=1.0/(-eps[v,n,n,n,n,n]-eps[n,v,n,n,n,n]-eps[n,n,v,n,n,n]+
                       eps[n,n,n,o,n,n]+eps[n,n,n,n,o,n]+eps[n,n,n,n,n,o])
     
            storedInfo.denomInfo.update({'D1aa':D1_aa,'D2aa':D2_aa,'D3aa':D3_aa})
            # call AmpHandler class to harvest amps, extract perturbative correction
            ampObj = ampHandles.AmpHandler(o,v,storedInfo,cc_runtype["pertCorr"]["T2infile"],cc_runtype["pertCorr"]["T1infile"])


        if cc_runtype["pertCorrOrders"] == "pdagq_parT":
            currentE, corrE=ampObj.run_pdagq()
        elif cc_runtype["pertCorrOrders"] == "wicked_parT":
            ampObj.run_wickedTest()
        else: # run UCC(5)-based energy corrections for triples
            pass
#        sys.exit()

#        iterateOver=cc_runtype["pertCorr"]
#        factorization=False
#        for correction in iterateOver:# iterates over "T" or "Qf" keys
#            corrOrder=iterateOver[correction]
#            for order in corrOrder: # iterates over list of values specifying pert order
#               #Build the T3/T4 base, then contract it to form the perturbative correction
#               if corrOrder == "T":
#                   energy=ampObj.build_T3energy(order,factorization)
#               elif corrOrder == "Qf":
#                   energy=ampObj.build_T4energy(order,factorization)


    if "ccdType" in cc_runtype: # can run all T2 spin-integrt methods

        storedInfo = convertSCFinfo(mf, mol, orb, cc_runtype, storedInfo)
        cc_runtype.update({"max_iter":75,"stopping_eps":10**-10, "diis_size":10, "diis_start_cycle":1})

        CCDobj=kernel.UltT2CC(storedInfo)
        t2, currentE, corrE = CCDobj.kernel()


    elif "fullCCType" in cc_runtype: # running >T2 spin-integrated code
        storedInfo = convertSCFinfo(mf, mol, orb, cc_runtype, storedInfo)

        if "stopping_eps" not in cc_runtype:
            cc_runtype.update({"stopping_eps":10**-10})

        if "max_iter" not in cc_runtype:
            cc_runtype.update({"max_iter":75})
        
        cc_runtype.update({"diis_size":None, "diis_start_cycle":None})
        CCDobj=kernel.UltT2CC(storedInfo)
        t2, currentE, corrE = CCDobj.kernel()

    elif "ccdTypeSlow" in cc_runtype: # can run all T2 spin-orb methods
        storedInfo=convertSCFinfo_tmpslow(mf, mol, orb,cc_runtype,storedInfo) #convertSCFinfo_slow(mf, mol, orb,cc_runtype,storedInfo)
        if "diis_size" in cc_runtype:
            ds=cc_runtype["diis_size"]
            dstart=cc_runtype["diis_start_cycle"]
            #cc_runtype.update({"diis_size":10, "diis_start_cycle":1})
            cc_runtype.update({"diis_size":ds, "diis_start_cycle":dstart})
        else:
            cc_runtype.update({"diis_size":None, "diis_start_cycle":None})
        CCDobj=kernel.UltT2CC(storedInfo)
        t2, currentE, corrE = CCDobj.kernel()


    return currentE, corrE


def ccd_kernel(
    na,
    nb,
    nvirta,
    nvirtb,
    occaa,
    virtaa,
    occbb,
    virtbb,
    faa,
    fbb,
    gaaaa,
    gbbbb,
    gabab,
    eabij_aa,
    eabij_bb,
    eabij_ab,
    hf_energy,
    nucE,
    diis_size=None,
    diis_start_cycle=4,
    cc_runtype=None,
):
    fock_e_abij_aa = np.reciprocal(eabij_aa)
    fock_e_abij_bb = np.reciprocal(eabij_bb)
    fock_e_abij_ab = np.reciprocal(eabij_ab)

    t2aaaa = np.zeros((nvirta, nvirta, na, na))
    t2bbbb = np.zeros((nvirtb, nvirtb, nb, nb))
    t2abab = np.zeros((nvirta, nvirtb, na, nb))

    print(t2aaaa.shape, t2bbbb.shape, t2abab.shape)
    # Initialize t2 amplitudes to 2e- integrals
    #t2aaaa=gaaaa.transpose(2,3,0,1)[:nvirta,:nvirta,:na,:na]
    #t2bbbb=gbbbb.transpose(2,3,0,1)[:nvirtb,:nvirtb,:nb,:nb]
    #t2abab=gabab.transpose(2,3,0,1)[:nvirta,:nvirtb,:na,:nb]


    if diis_size is not None:
        from UT2.diis import DIIS

        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t2aaaa_dim = t2aaaa.size
        t2bbbb_dim = t2bbbb.size
        t2abab_dim = t2abab.size

        old_vec = np.hstack((t2aaaa.flatten(), t2bbbb.flatten(), t2abab.flatten()))

    old_energy = t2energy.ccd_energy_with_spin(
        t2aaaa,
        t2bbbb,
        t2abab,
        faa,
        fbb,
        gaaaa,
        gbbbb,
        gabab,
        occaa,
        occbb,
        virtaa,
        virtbb,
    )

    print("initial energy:", old_energy)
    max_iter = 75
    stopping_eps = 1e-12
    print("    ==> ", cc_runtype["ccdType"], " amplitude equations <==")
    print("")
    print("     Iter              Corr. Energy                 |dE|    ")
    print(flush=True)
    g = {"aaaa": gaaaa, "bbbb": gbbbb, "abab": gabab}
    for idx in range(max_iter):
        t2 = {"aaaa": t2aaaa, "bbbb": t2bbbb, "abab": t2abab}
        l2 = {
            "aaaa": t2aaaa.transpose(2, 3, 0, 1),
            "bbbb": t2bbbb.transpose(2, 3, 0, 1),
            "abab": t2abab.transpose(2, 3, 0, 1),
        }



        resid_aaaa = (
            t2residEqns.ccd_t2_aaaa_residual(
                t2aaaa,
                t2bbbb,
                t2abab,
                faa,
                fbb,
                gaaaa,
                gbbbb,
                gabab,
                occaa,
                occbb,
                virtaa,
                virtbb,
                cc_runtype,
            )
            + fock_e_abij_aa * t2aaaa
        )

        resid_bbbb = (
            t2residEqns.ccd_t2_bbbb_residual(
                t2aaaa,
                t2bbbb,
                t2abab,
                faa,
                fbb,
                gaaaa,
                gbbbb,
                gabab,
                occaa,
                occbb,
                virtaa,
                virtbb,
                cc_runtype,
            )
            + fock_e_abij_bb * t2bbbb
        )
        resid_abab = (
            t2residEqns.ccd_t2_abab_residual(
                t2aaaa,
                t2bbbb,
                t2abab,
                faa,
                fbb,
                gaaaa,
                gbbbb,
                gabab,
                occaa,
                occbb,
                virtaa,
                virtbb,
                cc_runtype,
            )
            + fock_e_abij_ab * t2abab
        )

        # ***I DONT KNOW IF THE PREFACTOR OF 0.5 IS RIGHT
        if cc_runtype["ccdType"] == "CCDQf-1":

            qf1_aaaa = qf1.residQf1_aaaa(g, l2, t2, occaa, virtaa)
            qf1_bbbb = qf1.residQf1_bbbb(g, l2, t2, occaa, virtaa)
            qf1_abab = qf1.residQf1_abab(g, l2, t2, occaa, virtaa)
            resid_aaaa += 0.5 * qf1_aaaa
            resid_bbbb += 0.5 * qf1_bbbb
            resid_abab += 0.5 * qf1_abab

        elif cc_runtype["ccdType"] == "CCDQf-2":

            qf1_aaaa = qf1.residQf1_aaaa(g, l2, t2, occaa, virtaa)
            qf1_bbbb = qf1.residQf1_bbbb(g, l2, t2, occaa, virtaa)
            qf1_abab = qf1.residQf1_abab(g, l2, t2, occaa, virtaa)

            qf2_aaaa = qf2.residQf2_aaaa(g, l2, t2, occaa, virtaa)
            qf2_bbbb = qf2.residQf2_bbbb(g, l2, t2, occaa, virtaa)
            qf2_abab = qf2.residQf2_abab(g, l2, t2, occaa, virtaa)

            resid_aaaa += 0.5 * qf1_aaaa + (1.0 / 6.0) * qf2_aaaa
            resid_bbbb += 0.5 * qf1_bbbb + (1.0 / 6.0) * qf2_bbbb
            resid_abab += 0.5 * qf1_abab + (1.0 / 6.0) * qf2_abab

#        elif cc_runtype["ccdType"] == "CCDQfHf-1":


        #new_doubles_aaaa = resid_aaaa * eabij_aa  # doubles_res_aaaa * eabij_aa
        #new_doubles_bbbb = resid_bbbb * eabij_bb  # doubles_res_bbbb * eabij_bb
        #new_doubles_abab = resid_abab * eabij_ab  # doubles_res_abab * eabij_ab
        elif cc_runtype["ccdType"] == "pCCD":
            #new_doubles_aaaa = new_doubles_aaaa * 0.0
            #new_doubles_bbbb = new_doubles_aaaa
            tmpT2aa = np.zeros((nvirta, nvirta, na, na))
            tmpT2bb = np.zeros((nvirta, nvirta, na, na))
            tmpT2ab = np.zeros((nvirta, nvirta, na, na))

            for a in range(nvirta):
                for i in range(na):
                    tmpT2ab[a, a, i, i] = resid_abab[a, a, i, i]
                    tmpT2aa[a, a, i, i] = resid_aaaa[a, a, i, i]
                    tmpT2bb[a, a, i, i] = resid_bbbb[a, a, i, i]

            resid_aaaa=0.0*resid_aaaa
            resid_bbbb=0.0*resid_bbbb
            resid_abab=0.0*resid_abab
            resid_abab=tmpT2ab
        new_doubles_aaaa = resid_aaaa * eabij_aa  # doubles_res_aaaa * eabij_aa
        new_doubles_bbbb = resid_bbbb * eabij_bb  # doubles_res_bbbb * eabij_bb
        new_doubles_abab = resid_abab * eabij_ab  # doubles_res_abab * eabij_ab
#        elif cc_runtype["ccdType"] == "DiagCCD":
#            matDim = nvirta * na
#
#            def reshape(new_doubles_abab, new_doubles_aaaa, new_doubles_bbbb, matDim):
#                abab = new_doubles_abab.transpose(0, 2, 1, 3)
#                aaaa = new_doubles_aaaa.transpose(0, 2, 1, 3)
#                bbbb = new_doubles_bbbb.transpose(0, 2, 1, 3)
#                t2abab = np.reshape(abab, (matDim, matDim), order="F")
#                t2aaaa = np.reshape(aaaa, (matDim, matDim), order="F")
#                t2bbbb = np.reshape(bbbb, (matDim, matDim), order="F")
#                return t2abab, t2aaaa, t2bbbb
#
#            t2abab, t2aaaa, t2bbbb = reshape(
#                new_doubles_abab, new_doubles_aaaa, new_doubles_bbbb, matDim
#            )
#
#
#            def place_tensorDiag(eps, nv, no):
#                t2 = np.zeros((nv, nv, no, no))
#                count = 0
#                for a in range(nv):
#                    for i in range(no):
#                        t2[a][a][i][i] = eps[count]
#                        count += 1
#                return t2
#
#            def diag_t2matrix(t2, nv, no):
#                roots, vec = linalg.eig(t2)
#                indx = roots.argsort()
#                roots = roots[indx]
#                print("roots", roots)
#                newt2 = place_tensorDiag(roots, nv, no)
#
#                return newt2
#
#            t2abab = diag_t2matrix(t2abab, nvirta, na)
#            t2aaaa = diag_t2matrix(t2aaaa, nvirta, na)
#            t2bbbb = diag_t2matrix(t2bbbb, nvirta, na)
        if cc_runtype["ccdType"] == "pCCD":
            new_doubles_aaaa = new_doubles_aaaa * 0.0
            new_doubles_bbbb = new_doubles_aaaa
            tmpT2 = np.zeros((nvirta, nvirta, na, na))
            tmpResids=resid_abab
            for a in range(nvirta):
                for i in range(na):
                    tmpT2[a, a, i, i] = resid_abab[a,a,i,i]#new_doubles_abab[a, a, i, i]

            new_doubles_abab = new_doubles_abab * 0.0
            new_doubles_abab = tmpT2* eabij_ab
        elif cc_runtype["ccdType"] == "DiagCCD":
            matDim = nvirta * na

            def reshape(new_doubles_abab, new_doubles_aaaa, new_doubles_bbbb, matDim):
                abab = new_doubles_abab.transpose(0, 2, 1, 3)
                aaaa = new_doubles_aaaa.transpose(0, 2, 1, 3)
                bbbb = new_doubles_bbbb.transpose(0, 2, 1, 3)
                t2abab = np.reshape(abab, (matDim, matDim), order="F")
                t2aaaa = np.reshape(aaaa, (matDim, matDim), order="F")
                t2bbbb = np.reshape(bbbb, (matDim, matDim), order="F")
                return t2abab, t2aaaa, t2bbbb

            t2abab, t2aaaa, t2bbbb = reshape(
                new_doubles_abab, new_doubles_aaaa, new_doubles_bbbb, matDim
            )


            def place_tensorDiag(eps, nv, no):
                t2 = np.zeros((nv, nv, no, no))
                count = 0
                for a in range(nv):
                    for i in range(no):
                        t2[a][a][i][i] = eps[count]
                        count += 1
                return t2

            def diag_t2matrix(t2, nv, no):
                roots, vec = linalg.eig(t2)
                indx = roots.argsort()
                roots = roots[indx]
                print("roots", roots)
                newt2 = place_tensorDiag(roots, nv, no)

                return newt2

            t2abab = diag_t2matrix(t2abab, nvirta, na)
            t2aaaa = diag_t2matrix(t2aaaa, nvirta, na)
            t2bbbb = diag_t2matrix(t2bbbb, nvirta, na)

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (
                    new_doubles_aaaa.flatten(),
                    new_doubles_bbbb.flatten(),
                    new_doubles_abab.flatten(),
                )
            )
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(
                vectorized_iterate, error_vec
            )
            new_doubles_aaaa = new_vectorized_iterate[:t2aaaa_dim].reshape(t2aaaa.shape)
            new_doubles_bbbb = new_vectorized_iterate[
                t2aaaa_dim : t2aaaa_dim + t2bbbb_dim
            ].reshape(t2bbbb.shape)
            new_doubles_abab = new_vectorized_iterate[
                t2aaaa_dim + t2bbbb_dim :
            ].reshape(t2abab.shape)
            old_vec = new_vectorized_iterate

        current_energy = t2energy.ccd_energy_with_spin(
            new_doubles_aaaa,
            new_doubles_bbbb,
            new_doubles_abab,
            faa,
            fbb,
            gaaaa,
            gbbbb,
            gabab,
            occaa,
            occbb,
            virtaa,
            virtbb,
        )
        delta_e = np.abs(old_energy - current_energy)

        print(
            "    {: 5d} {: 20.12f} {: 20.12f} ".format(
                idx, nucE + current_energy - hf_energy, delta_e
            )
        )
        print(flush=True)
        if delta_e < stopping_eps:  # and res_norm < stopping_eps:
            # assign t1 and t2 variables for future use before breaking
            t2aaaa = new_doubles_aaaa
            t2bbbb = new_doubles_bbbb
            t2abab = new_doubles_abab
            break
        else:
            # assign t1 and t2 and old_energy for next iteration
            t2aaaa = new_doubles_aaaa
            t2bbbb = new_doubles_bbbb
            t2abab = new_doubles_abab
            old_energy = current_energy
    else:
        raise ValueError("CC iterations did not converge")

    print("\n\n\n")
    if cc_runtype["ccdType"] != "CCD(Qf)":
        print(
            cc_runtype["ccdType"],
            " correlation contribution:",
            nucE + current_energy - hf_energy,
        )
        corrE=nucE+current_energy
        print(cc_runtype["ccdType"], " energy:", nucE + current_energy)
        tfinalEnergy=current_energy+nucE
    if cc_runtype["ccdType"] == "CCD(Qf)":

        qf_corr = pertQf.energy_pertQf(g, l2, t2, occaa, virtaa)
        print("CCD correlation contribution: ", nucE + current_energy - hf_energy)
        print("(Qf) perturbative energy correction: ", qf_corr)
        print(cc_runtype["ccdType"], " energy:", nucE + current_energy + qf_corr)
        tfinalEnergy=current_energy+nucE+qf_corr
        corrE=qf_corr
    print("\n\n\n")

    return t2aaaa, t2bbbb, t2abab, tfinalEnergy, corrE 



def convertSCFinfo(mf, mol, orb,cc_runtype,storedInfo):
    """ convertSCFinfo() is a general function that performs the background tasks necessary for a general CC calculation to take place, based on either RHF, UHF, or spin-orbital formalisms. Harvests relevant information, such as number of alpha/beta electrons, from PySCF mf and mol objects.
    
    :param mf: A PySCF SCF object, containing pertinent info regarding SCF calculation
    :param mol: PySCF object containing info regarding the info regarding the molecule/system in question
    :param orb: Set of - presumably converged - SCF coefficients
    :param cc_runtype: Dictionary specifying information regarding the correlation calculation that is requested. Has two callable options; "ccdType" which refers to any of the T2-based methods that are implemented, and "fullCCType" which calls one of CCSDT, CCSDTQ, or CCSDTQf calculations. A third option, "ccdTypeSlow" will lead to spin-orbital formulations of the T2-based methods
    
    :param occupationInfo: Dict of number of alpha/beta occupied/virtual orbitals. Callable with keys "nocc_aa", "nocc_bb", "nvirt_aa", and "nirt_bb".
    :param integralInfo: Dict of (transformed) integral information, indexed by "faa", "fbb", "g_aaaa", "g_bbbb", and "g_abab"
    :param occupationSliceInfo: Dict containing slices for the occupied/virtual alpha/beta orbitals, indexed by "occ_aa", "virt_aa", "occ_bb", and "virt_bb"
    :param eps: Dict containing alpha/beta molecular orbital energies, indexed by "eps_aa" or "eps_bb"
    :param denomInfo: Dict containing denominators for all CC methods, indexed by "D2aa", "D2ab", "D2bb", and so on for arbitrary CC method.
    
    :return: Returns dictionaries of occupationInfo,integralInfo, eps, denomInfo, occupationSliceInfo
    
    """

    # Means we are running RHF; must generalize data structs for use in UHF code
    if orb.ndim <= 2:
        h1e = np.array((mf.get_hcore(), mf.get_hcore()))
        f = np.array((mf.get_fock(), mf.get_fock()))
        na = mol.nelectron // 2
        nb = na
        nvirta = f[0].shape[0] - na
        nvirtb = nvirta
        orb = np.array((orb, orb))
        print("shape of numpy coeff rhf:", np.shape(orb))
        moE_aa = mf.mo_energy
        moE_bb = moE_aa
    elif orb.ndim > 2:  # MEANS IM RUNNING UHF CALC
        h1e = np.array((mf.get_hcore(), mf.get_hcore()))
        f = mf.get_fock()
        na, nb = mf.nelec
        nvirta = f[0].shape[0] - na
        nvirtb = f[1].shape[0] - nb
        moE_aa = mf.mo_energy[0]
        moE_bb = mf.mo_energy[1]
        print("mo energy:", np.shape(moE_aa))

    faa = f[0]
    fbb = f[1]

    occupationInfo={"nocc_aa":na, "nocc_bb":nb,"nvirt_aa":nvirta,"nvirt_bb":nvirtb}
    integralInfo = generalUHF(mf, mol, h1e, f, na, nb, orb)

    #n = np.newaxis
    occ_aa = slice(None, na)
    virt_aa = slice(na, None)
    occ_bb = slice(None, nb)
    virt_bb = slice(nb, None)
    #epsaa = moE_aa
    #epsbb = moE_bb
    eps={'eps_aa':moE_aa,'eps_bb':moE_bb}
    occupationSliceInfo={"occ_aa":  occ_aa, "virt_aa":virt_aa, 
                     "occ_bb": occ_bb, "virt_bb":virt_bb}

    denomInfo=get_denoms(cc_runtype,occupationSliceInfo,eps) 
    storedInfo.set_denoms(denomInfo)
    storedInfo.set_occInfo(occupationInfo)
    storedInfo.set_occSliceInfo(occupationSliceInfo)
    storedInfo.set_cc_runtype(cc_runtype)
    storedInfo.set_integralInfo(integralInfo)
    return storedInfo #occupationInfo, integralInfo, eps, denomInfo, occupationSliceInfo


def spin_block_tei(I):
    """
    Function that spin blocks two-electron integrals
    Using np.kron, we project I into the space of the 2x2 identity, tranpose the result
    and project into the space of the 2x2 identity again. This doubles the size of each axis.
    The result is our two electron integral tensor in the spin orbital form.
    """
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)


def convertSCFinfo_tmpslow(mf,mol,orb,cc_runtype,storedInfo):
    if 'RHF' in str(type(mf)): # running RHF calculation
        occ = mf.mo_occ
        nele = int(sum(occ))
        nocc = nele // 2
        norbs = mf.get_fock().shape[0] #oei.shape[0]
        nsvirt = 2 * (norbs - nocc)
        nsocc = 2 * nocc
        occupationInfo={"nocc_aa":nsocc,"nvirt_aa":nsvirt}
        storedInfo.set_occInfo(occupationInfo)

        Ca = Cb = np.asarray(mf.mo_coeff)
        eps_a = eps_b = np.asarray(mf.mo_energy)

    elif 'UHF' in str(type(mf)): # running UHF calculation
        norbs=mf.get_fock().shape[0] + mf.get_fock().shape[1]
        na,nb=mf.nelec
        nele=na+nb
        nsvirt = 2 * (mf.get_fock()[0].shape[0] - na)#(norbs - nocc)
        occupationInfo={"nocc_aa":nele,"nvirt_aa":nsvirt}
        storedInfo.set_occInfo(occupationInfo)
        Ca = np.asarray(mf.mo_coeff[0])
        Cb = np.asarray(mf.mo_coeff[1])
        eps_a = np.asarray(mf.mo_energy[0])
        eps_b = np.asarray(mf.mo_energy[1])



    # default is to try and use PySCF object to harvest AO 2eints
    eri = mol.intor('int2e',aosym='s1')
    if np.shape(eri)==(0,0,0,0):# otherwise,
        eri=np.zeros((norbs,norbs,norbs,norbs))
        print(np.shape(eri))
        with open('ao_tei.pickle', 'rb') as handle:
            eri=pickle.load(handle)


    C = np.block([
             [      Ca           ,   np.zeros_like(Cb) ],
             [np.zeros_like(Ca)  ,          Cb         ]
            ])

    n = np.newaxis
    o = slice(None, nele)
    v = slice(nele, None)
    occupationSliceInfo={"occ_aa":o,"virt_aa":v}
    storedInfo.set_occSliceInfo(occupationSliceInfo)



    I = np.asarray(eri)
    I_spinblock = spin_block_tei(I)
    # Converts chemist's notation to physicist's notation, and antisymmetrize
    # (pq | rs) ---> <pr | qs>
    # Physicist's notation
    tmp = I_spinblock.transpose(0, 2, 1, 3)
    # Antisymmetrize:
    # <pr||qs> = <pr | qs> - <pr | sq>
    gao = tmp - tmp.transpose(0, 1, 3, 2)
    eps = np.append(eps_a, eps_b)


    # Sort the columns of C according to the order of increasing orbital energies
    C = C[:, eps.argsort()]

# Sort orbital energies in increasing order
    eps = np.sort(eps)

    print(np.shape(gao),np.shape(C))
    # Transform gao, which is the spin-blocked 4d array of physicist's notation,
    # antisymmetric two-electron integrals, into the MO basis using MO coefficients
    gmo = np.einsum('pQRS, pP -> PQRS',
          np.einsum('pqRS, qQ -> pQRS',
          np.einsum('pqrS, rR -> pqRS',
          np.einsum('pqrs, sS -> pqrS', gao, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)

    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])

    t2=e_abij*gmo[v,v,o,o]

    mp2E=0.250000000000000 * np.einsum('jiab,abji',gmo[o, o, v, v], t2)

    print('mp2E:', mp2E)

    eps2={"eps_aa":eps}
    if 'pertCorr' not in cc_runtype:
        denomInfo=get_denoms(cc_runtype,occupationSliceInfo,eps2)
        storedInfo.set_denoms(denomInfo)

    #print(np.shape(eps),eps,np.diag(eps),np.diag(np.diag(eps)))
    fock=np.diag(eps)
    integralInfo={"oei":fock,"tei":gmo}
    storedInfo.set_cc_runtype(cc_runtype)
    storedInfo.set_integralInfo(integralInfo)

    return storedInfo

def convertSCFinfo_slow(mf, mol, orb,cc_runtype,storedInfo):
    """
    Transforms relevant integrals into MO framework. Used specifically in the spin-orbital based code.
    
    :param mf: PySCF SCF object
    :param mol: PySCF Molecule object
    :param orb: SCF coefficients, obtain from PySCF
    :param cc_runtype: Dictionary containing calculation-specific information on the CC calculation
    :param storedInfo: Object storing relevant information, by way of dictionaries, to the CC calculation.
    
    :return: Returns an updated storedInfo object, where the integrals and denominator information has been updated. 
    """
    orb=np.asarray(orb)
    print('orb:', np.shape(orb), orb.ndim)
    if orb.ndim <= 2:
        occ = mf.mo_occ
        print('occ:',occ)
        nele = int(sum(occ))
        nocc = nele // 2
        norbs = mf.get_fock().shape[0] #oei.shape[0]
        global nsvirt, nsocc
        
        nsvirt = 2 * (norbs - nocc)
        nsocc = 2 * nocc
        
        occupationInfo={"nocc_aa":nsocc,"nvirt_aa":nsvirt}
        storedInfo.set_occInfo(occupationInfo)
        
        n = np.newaxis
        o = slice(None, nsocc)
        v = slice(nsocc, None)
        occupationSliceInfo={"occ_aa":o,"virt_aa":v}
        storedInfo.set_occSliceInfo(occupationSliceInfo)
        
        moE_aa = mf.mo_energy
        eps_aa=np.kron(moE_aa,np.ones(2))
        eps={"eps_aa":eps_aa}
        denomInfo=get_denoms(cc_runtype,occupationSliceInfo,eps)
        storedInfo.set_denoms(denomInfo)
        
        
        hcore=mf.get_hcore()
        hcoreMO=(orb.T@hcore)@orb
        
        try:
            twoEints=ao2mo.kernel(mol,mf.mo_coeff)
            two_electron_integrals = ao2mo.restore(
                1, # no permutation symmetry
                twoEints, orb.shape[0])

        except:
            #t4=np.zeros((nvir,nvir,nocc,nocc))
            with open('ao_tei.pickle', 'rb') as handle:
                twoEints=pickle.load(handle)

        print('final tei load')
        print(two_electron_integrals)
        #sys.exit()
#        two_electron_integrals = ao2mo.restore(
#                1, # no permutation symmetry
#                twoEints, orb.shape[0])
            # See PQRS convention in OpenFermion.hamiltonians._molecular_data
            # h[p,q,r,s] = (ps|qr)
        two_electron_integrals = np.asarray(
                two_electron_integrals.transpose(0, 2, 3, 1), order='C')
        
        soei, stei = spinorb_from_spatial(hcoreMO,two_electron_integrals) #oei, tei)
        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
        gtei = astei.transpose(0, 1, 3, 2)
        
        fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])


        integralInfo={"oei":fock,"tei":gtei}
        storedInfo.set_cc_runtype(cc_runtype)
        storedInfo.set_integralInfo(integralInfo)
    elif orb.ndim > 2:  # MEANS IM RUNNING UHF CALC
        print('iNSIDE UHF')
        norbs=mf.get_fock().shape[0] + mf.get_fock().shape[1]
        na,nb=mf.nelec
        nele=na+nb
        nsvirt = 2 * (mf.get_fock()[0].shape[0] - na)#(norbs - nocc)
        occupationInfo={"nocc_aa":nele,"nvirt_aa":nsvirt}
        storedInfo.set_occInfo(occupationInfo)

        n = np.newaxis
        o = slice(None, nele)
        v = slice(nele, None)
        occupationSliceInfo={"occ_aa":o,"virt_aa":v}
        storedInfo.set_occSliceInfo(occupationSliceInfo)


        eri = mol.intor('int2e', aosym='s1') 
    
        Ca = np.asarray(mf.mo_coeff[0])
        Cb = np.asarray(mf.mo_coeff[1])
        C = np.block([
                 [      Ca           ,   np.zeros_like(Cb) ],
                 [np.zeros_like(Ca)  ,          Cb         ]
                ])
    
        I = np.asarray(eri)
        I_spinblock = spin_block_tei(I)


        # Converts chemist's notation to physicist's notation, and antisymmetrize
        # (pq | rs) ---> <pr | qs>
        # Physicist's notation
        tmp = I_spinblock.transpose(0, 2, 1, 3)
        # Antisymmetrize:
        # <pr||qs> = <pr | qs> - <pr | sq>
        gao = tmp - tmp.transpose(0, 1, 3, 2)
    
        eps_a = np.asarray(mf.mo_energy[0])
        eps_b = np.asarray(mf.mo_energy[1])
        eps = np.append(eps_a, eps_b)


        # Sort the columns of C according to the order of increasing orbital energies
        C = C[:, eps.argsort()]
    
    # Sort orbital energies in increasing order
        eps = np.sort(eps)


#        nalpha = mf.nelec[0]
#        nbeta = mf.nelec[1]
#        nocc = nalpha + nbeta
    
    
#        n = np.newaxis
#        o = slice(None, nocc)
#        v = slice(nocc, None)

    # Transform gao, which is the spin-blocked 4d array of physicist's notation,
    # antisymmetric two-electron integrals, into the MO basis using MO coefficients
        gmo = np.einsum('pQRS, pP -> PQRS',
              np.einsum('pqRS, qQ -> pQRS',
              np.einsum('pqrS, rR -> pqRS',
              np.einsum('pqrs, sS -> pqrS', gao, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)

        e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])

        t2=e_abij*gmo[v,v,o,o]
    
        mp2E=0.250000000000000 * np.einsum('jiab,abji',gmo[o, o, v, v], t2)
    
        print('mp2E:', mp2E)

        eps2={"eps_aa":eps}
        denomInfo=get_denoms(cc_runtype,occupationSliceInfo,eps2)
        storedInfo.set_denoms(denomInfo)

        #print(np.shape(eps),eps,np.diag(eps),np.diag(np.diag(eps)))
        fock=np.diag(eps)
        integralInfo={"oei":fock,"tei":gmo}
        storedInfo.set_cc_runtype(cc_runtype)
        storedInfo.set_integralInfo(integralInfo) 
    



    return storedInfo

def get_denoms(cc_runtype,occupationSliceInfo,eps):
    """ get_denoms() is a general function that performs the background tasks necessary for a general CC calculation to take place, based on either RHF, UHF, or spin-orbital formalisms. Constructs the denominators based on the CC theory being used.
    
    :param cc_runtype: Dictionary specifying information regarding the correlation calculation that is requested. Has two callable options; "ccdType" which refers to any of the T2-based methods that are implemented, and "fullCCType" which calls one of CCSDT, CCSDTQ, or CCSDTQf calculations. A third option, "ccdTypeSlow" will lead to spin-orbital formulations of the T2-based methods
    
    :param occupationSliceInfo: Dict containing slices for the occupied/virtual alpha/beta orbitals
    :param eps: Dict containing alpha/beta molecular orbital energies
    
    
    :return: Dictionary of denominator info accessible using keys that looks like "D2aa" for T2 alpha/alpha relevant denoms. This naming nomenclature continues beyond T2 methods as well ie "D4aa" for T4, alpha/alpha specific denoms. 
    
    """


    set_spinIntegrt=set(["ccdType", "fullCCType"])
    denomInfo={}
    n = np.newaxis
    if "ccdTypeSlow" in cc_runtype: # spin-orb formalism
        virt_aa=occupationSliceInfo["virt_aa"]
        occ_aa=occupationSliceInfo["occ_aa"]
        epsaa=eps['eps_aa']
        print('inside two')
        

    elif "ccdType" in cc_runtype or "fullCCType" in cc_runtype: # spin-integrated formalisms
        print('inside one')
        virt_aa=occupationSliceInfo["virt_aa"]
        virt_bb=occupationSliceInfo["virt_bb"]
        occ_aa=occupationSliceInfo["occ_aa"]
        occ_bb=occupationSliceInfo["occ_bb"]
        epsaa=eps['eps_aa']
        epsbb=eps['eps_bb']

        eabij_bb = 1.0 / (
            -epsbb[virt_bb, n, n, n]
            - epsbb[n, virt_bb, n, n]
            + epsbb[n, n, occ_bb, n]
            + epsbb[n, n, n, occ_bb]
        )
        eabij_ab = 1.0 / (
            -epsaa[virt_aa, n, n, n]
            - epsbb[n, virt_bb, n, n]
            + epsaa[n, n, occ_aa, n]
            + epsbb[n, n, n, occ_bb]
        )
        denomInfo = {"D2ab":eabij_ab,"D2bb":eabij_bb}

    eabij_aa = 1.0 / (
        -epsaa[virt_aa, n, n, n]
        - epsaa[n, virt_aa, n, n]
        + epsaa[n, n, occ_aa, n]
        + epsaa[n, n, n, occ_aa]
    )
    denomInfo.update({"D2aa":eabij_aa})
    if "ccdTypeSlow" in cc_runtype:
        hgherO=cc_runtype["ccdTypeSlow"]
    else:
        hgherO=0
    if hgherO == "UT2-CCD(7)" or hgherO == "UT2-CCD(8)" or hgherO == "UT2-CCD(9)":
        virt_aa=occupationSliceInfo["virt_aa"]
        occ_aa=occupationSliceInfo["occ_aa"]
        epsaa=eps['eps_aa']
        
        D4=1.0/(-epsaa[virt_aa, n, n, n, n, n, n, n]
                -epsaa[n,      virt_aa, n, n, n, n, n, n]
                -epsaa[n, n,           virt_aa, n, n, n, n, n]
                -epsaa[n, n, n,                virt_aa, n, n, n, n]
                +epsaa[n, n, n, n, occ_aa, n, n, n]
                +epsaa[n, n, n, n, n,       occ_aa, n, n]
                +epsaa[n, n, n, n, n, n,            occ_aa, n]
                +epsaa[n, n, n, n, n, n, n,                 occ_aa])


#        D6=1.0/(-epsaa[virt_aa, n, n, n, n, n, n, n, n, n, n, n]
#                -epsaa[n,      virt_aa, n, n, n, n, n, n, n, n, n, n]
#                -epsaa[n, n,           virt_aa, n, n, n, n, n, n, n, n, n]
#                -epsaa[n, n, n,                virt_aa, n, n, n, n, n, n, n, n]
#                -epsaa[n, n, n, n,                      virt_aa, n, n, n, n, n, n, n]
#                -epsaa[n, n, n, n, n,                           virt_aa, n, n, n, n, n, n]
#                +epsaa[n, n, n, n, n, n, virt_aa, n, n, n, n, n]
#                +epsaa[n, n, n, n, n, n, n,       virt_aa, n, n, n, n]
#                +epsaa[n, n, n, n, n, n, n, n,            virt_aa, n, n, n]
#                +epsaa[n, n, n, n, n, n, n, n, n,                 virt_aa, n, n]
#                +epsaa[n, n, n, n, n, n, n, n, n, n,                      virt_aa, n]
#                +epsaa[n, n, n, n, n, n, n, n, n, n, n, virt_aa])
        denomInfo.update({"D4aa":D4})

    if "fullCCType" in cc_runtype: 
   # Singles Denom
        eai_aa = 1.0/ (-epsaa[virt_aa,n] + epsaa[n,occ_aa])
        eai_bb = 1.0/ (-epsbb[virt_bb,n] + epsaa[n,occ_bb])
        eai_ab = 0.0 #1.0/ (-epsab[virt_bb,n] + epsaa[n,occ_ab])


        # Triples Denom
        eabcijk_aa = 1.0/ (
            -epsaa[virt_aa, n, n, n, n, n]
            - epsaa[n, virt_aa, n, n, n, n]
            - epsaa[n, n,virt_aa, n, n, n]
            + epsaa[n, n, n, occ_aa, n, n]
            + epsaa[n, n, n, n, occ_aa, n]
            + epsaa[n, n, n, n, n, occ_aa]
        )
    
    
        eabcijk_bb =1.0/ (
            -epsbb[virt_bb, n, n, n, n, n]
            - epsbb[n, virt_bb, n, n, n, n]
            - epsbb[n, n,virt_bb, n, n, n]
            + epsbb[n, n, n, occ_bb, n, n]
            + epsbb[n, n, n, n, occ_bb, n]
            + epsbb[n, n, n, n, n, occ_bb]
        )
    
        eabcijk_aab =1.0/ (
            -epsaa[virt_aa, n, n, n, n, n]
            - epsbb[n, virt_aa, n, n, n, n]
            - epsaa[n, n,virt_bb, n, n, n]
            + epsaa[n, n, n, occ_aa, n, n]
            + epsbb[n, n, n, n, occ_aa, n]
            + epsaa[n, n, n, n, n, occ_bb]
        )

        eabcijk_abb =1.0/ (
            -epsaa[virt_aa, n, n, n, n, n]
            - epsbb[n, virt_bb, n, n, n, n]
            - epsaa[n, n,virt_bb, n, n, n]
            + epsaa[n, n, n, occ_aa, n, n]
            + epsbb[n, n, n, n, occ_bb, n]
            + epsaa[n, n, n, n, n, occ_bb]
        )
        denomInfo.update({"D1aa":eai_aa,"D1bb":eai_bb,"D1ab":eai_ab,
                          "D3aaa":eabcijk_aa,"D3bbb":eabcijk_bb,"D3aab":eabcijk_aab,
                          "D3abb":eabcijk_abb})
        if "CCSDTQ" in cc_runtype.values():
            eabcdijkl_aa = 1.0/ (
                -epsaa[virt_aa,n,n, n, n, n, n, n]
                - epsaa[n, virt_aa,n,n, n, n, n, n]
                - epsaa[n, n,virt_aa, n,n,n, n, n]
                - epsaa[n, n,n, virt_aa, n,n, n, n]
                + epsaa[n, n, n, n, occ_aa,n, n, n]
                + epsaa[n, n, n, n, n,occ_aa, n,n]
                + epsaa[n, n, n, n, n, n,occ_aa,n]
                + epsaa[n, n, n, n, n, n,n,occ_aa]
            )
            print('quads alla ')
            eabcdijkl_bb = 1.0/ (
                -epsbb[virt_bb,n,n, n, n, n, n, n]
                - epsbb[n, virt_bb,n,n, n, n, n, n]
                - epsbb[n, n,virt_bb, n,n,n, n, n]
                - epsbb[n, n,n, virt_bb, n,n, n, n]
                + epsbb[n, n, n, n, occ_bb,n, n, n]
                + epsbb[n, n, n, n, n,occ_bb, n,n]
                + epsbb[n, n, n, n, n, n,occ_bb,n]
                + epsbb[n, n, n, n, n, n,n,occ_bb]
            )
            print('quads allb')
            eabcdijkl_ab = 1.0/ (
                -epsaa[virt_aa,n,n, n, n, n, n, n]
                - epsbb[n, virt_bb,n,n, n, n, n, n]
                - epsaa[n, n,virt_aa, n,n,n, n, n]
                - epsbb[n, n,n, virt_bb, n,n, n, n]
                + epsaa[n, n, n, n, occ_aa,n, n, n]
                + epsbb[n, n, n, n, n,occ_bb, n,n]
                + epsaa[n, n, n, n, n, n,occ_aa,n]
                + epsbb[n, n, n, n, n, n,n,occ_bb]
            )


            denomInfo.update({"D4aa":eabcdijkl_aa,"D4bb":eabcdijkl_bb,
                           "D4ab":eabcdijkl_ab})

    return denomInfo



def generalUHF(mf, mol, h1e, f, na, nb, orb):
    """
    Converts UHF/RHF 1 and 2e- integrals into the MO framework. For use in RHF/UHF codes ONLY.
    
    :param mf: PySCF SCF object
    :param mol: PySCF Molecule object
    :param h1e: Core Hamiltonian
    :param f: Fock Matrix
    :param na: Number of alpha occupied orbitals
    :param nb: Number of beta occupied orbitals
    :param orb: SCF coefficients
     
    :return: Returns MO-transformed fock and two electron integrals in a dictionary that is accessible using keys "oei" and "tei", respectively. 
    
    """

    # h1e = mf.get_hcore()
    h1aa = orb[0].T @ h1e[0] @ orb[0]
    h1bb = orb[1].T @ h1e[1] @ orb[1]

    # f=mf.get_fock()
    faa = orb[0].T @ f[0] @ orb[0]
    fbb = orb[1].T @ f[1] @ orb[1]

    # nelec=mol.nelectron
    # na, nb = mf.nelec
    eri = mol.intor("int2e", aosym="s1")
    g_aaaa = ao2mo.incore.general(eri, (orb[0], orb[0], orb[0], orb[0]))
    g_bbbb = ao2mo.incore.general(eri, (orb[1], orb[1], orb[1], orb[1]))
    g_abab = ao2mo.incore.general(eri, (orb[0], orb[0], orb[1], orb[1]))

    # Verify the 2e- integral coulomb energy
    ga = g_aaaa.transpose(0, 2, 1, 3)
    gb = g_bbbb.transpose(0, 2, 1, 3)
    e_coul = np.einsum("ijij", ga[:na, :na, :na, :na]) + np.einsum(
        "ijij", gb[:nb, :nb, :nb, :nb]
    )
    e_exch = 0.5 * np.einsum("ijji", ga[:na, :na, :na, :na]) + 0.5 * np.einsum(
        "ijji", gb[:nb, :nb, :nb, :nb]
    )

    print("total 2e- integral energy:", e_coul - e_exch)

    # Now, convert to Dirac notation, and antisymmetrize g_aaaa/g_bbbb
    g_aaaa = g_aaaa.transpose(0, 2, 1, 3) - g_aaaa.transpose(0, 3, 2, 1)  # (0,3,1,2)
    g_bbbb = g_bbbb.transpose(0, 2, 1, 3) - g_bbbb.transpose(0, 3, 2, 1)
    g_abab = g_abab.transpose(0, 2, 1, 3)
    import sys

    print(np.shape(g_aaaa))
    # sys.exit()

    # Now, verify the UHF energy
    e1 = 0.5 * np.einsum("ii", h1aa[:na, :na]) + 0.5 * np.einsum("ii", h1bb[:nb, :nb])
    e2 = 0.5 * np.einsum("ii", faa[:na, :na]) + 0.5 * np.einsum("ii", fbb[:nb, :nb])
    totSCFenergy = e1 + e2 + mf.energy_nuc()
    print("final rhf/uhf energy:", totSCFenergy)
    fock={"faa":faa,"fbb":fbb}
    tei={"g_aaaa":g_aaaa,"g_bbbb":g_bbbb,"g_abab":g_abab}
    integralInfo={"oei":fock,"tei":tei}#{"faa":faa,"fbb":fbb,"g_aaaa":g_aaaa,"g_bbbb":g_bbbb,"g_abab":g_abab}
    return integralInfo #faa, fbb, g_aaaa, g_bbbb, g_abab

def spinorb_from_spatial(one_body_integrals, two_body_integrals):
    """ Converts one and two electron integrals from spatial MOs to spin orbitals. For use in
    the Slow CC codes 
   
    :param one_body_integrals: List of one body (fock) matrices. 
    :param two_body_integrals: List of two body (2e- integral) tensors.

    :return: one and two body integrals in spin-orbital framework. """

    n_qubits = 2 * one_body_integrals.shape[0]

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):

                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (two_body_integrals[p, q, r, s])
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s])

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                          s] = (two_body_integrals[p, q, r, s])
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                          1, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s])

    return one_body_coefficients, two_body_coefficients

def spinorb_from_UHFspatial(oei_alpha,oei_beta,tei_alpha,tei_beta,tei_mixed):
    import numpy
    n_qubits = 2 * oei_alpha.shape[0]

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = numpy.zeros((n_qubits, n_qubits))
    two_body_coefficients = numpy.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = oei_alpha[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = oei_beta[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):

                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (tei_mixed[p, q, r, s])
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (tei_mixed[p, q, r, s])


                 #   two_body_coefficients[2*p,2*q+1,2*r,2*s+1]=(-tei_mixed[p,q,s,r])
                 #   two_body_coefficients[2*p+1,2*q,2*r+1,2*s]=(-tei_mixed[p,q,s,r])
                    #print('verify antisym')
                    #print(tei_mixed[p,q,r,s],tei_mixed[p,q,s,r])
                    # Same spin

                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                          s] = (tei_alpha[p, q, r, s])#-tei_alpha[p,q,s,r])
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                          1, 2 * s +
                                          1] = (tei_beta[p, q, r, s])#-tei_beta[p,q,s,r])

                    #print('verify same spin')
                    #print(tei_alpha[p, q, r, s]-tei_alpha[p, q, s,r])
    #sys.exit()
    return one_body_coefficients, two_body_coefficients


#def test_rhf_energy(mol, mf, orb):
#    eri = ao2mo.full(mol, orb, verbose=0)
#    print("eri:", eri, np.shape(eri))
#    eriFull = ao2mo.restore("s1", eri, orb.shape[1])
#    print("full", eriFull, np.shape(eriFull))
#    eriFull = eriFull.transpose(0, 2, 1, 3)
#
#    hcore = mf.get_hcore()
#    hcoreMO = orb.T @ hcore @ orb
#
#    f = mf.get_fock()
#    fock = orb.T @ f @ orb
#
#    nelec = mol.nelectron
#    nocc = nelec // 2
#
#    test_e = np.einsum("ii", hcoreMO[:nocc, :nocc]) + np.einsum(
#        "ii", fock[:nocc, :nocc]
#    )
#
#    teint_energy = 2.0 * np.einsum(
#        "ijij", eriFull[:nocc, :nocc, :nocc, :nocc]
#    ) - np.einsum("ijji", eriFull[:nocc, :nocc, :nocc, :nocc])
#    test_e2 = np.einsum("ii", hcoreMO[:nocc, :nocc]) * 2.0 + teint_energy
#
#    print(mf.e_tot, test_e + mf.energy_nuc(), test_e2 + mf.energy_nuc())


## TODO:
## Construct general code for both RHF and UHF
## Write test to verify I get same HF SCF energy using 1 and 2 e- ints
