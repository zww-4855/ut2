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

""" StoredInfo() is a class that holds all data necessary for a CC calculation. Similar in design to C-struct


:param occupationInfo: A dictionary containing information regarding the occupation of occupied/virtual orbitals

:param occupationSliceInfo: A dictionary containing information regarding the slicing structure of occupied/virtual orbitals

:param denomInfo: A dictionary containing information on the relevant denominators for any CC

:param cc_runtype: Dictionary specifying information regarding the correlation calculation that is requested. Has two callable options; "ccdType" which refers to any of the T2-based methods that are implemented, and "fullCCtype" which calls one of CCSDT, CCSDTQ, or CCSDTQf calculations. A third option, "SOccdtype" will lead to spin-orbital formulations of the T2-based methods. Also contains information such as convergence criteria, number of diis vectors & length, type of CC calculation, and max. number of iterations allowed in a given CC calculation


:param tamps: A dictionary containing the current iterations T-amplitude

:param resid: A Dictionary containing the current iterations residuals

:return: An archive object that can be queried for the various quantities in question
"""
class StoredInfo():
    def __init__(self,occupationInfo={},occupationSliceInfo={},denomInfo={},cc_runtype={},tamps={},resid={},integralInfo={}):
        self.occInfo=occupationInfo
        self.occSliceInfo=occupationSliceInfo
        self.denomInfo=denomInfo
        self.cc_runtype=cc_runtype
        self.integralInfo=integralInfo

    def set_occInfo(self,occInfo):
        self.occInfo=occInfo

    def set_occSliceInfo(self,occSliceInfo):
        self.occSliceInfo=occSliceInfo

    def set_denoms(self,denomInfo):
        self.denomInfo=denomInfo

    def set_cc_runtype(self,cc_runtype):
        self.cc_runtype=cc_runtype

    def set_integralInfo(self,integralInfo):
        self.integralInfo=integralInfo

    def get_denoms(self,dataString=None):
        if dataString==None:
            return self.denomInfo
        else:
            return self.denomInfo[str(dataString)]

    def get_occInfo(self,dataString):
        return self.occInfo[str(dataString)]

    def get_occSliceInfo(self,dataString=None):
        if dataString==None:
            return self.occSliceInfo
        else:
            return self.occSliceInfo[str(dataString)]

    def get_cc_runtype(self,dataString=None):
        if dataString==None:
            return self.cc_runtype
        else:
            return self.cc_runtype[str(dataString)]

    def get_integralInfo(self,elec_spec=None,dataString=None):
        if elec_spec==None:
            return self.integralInfo

        dic=self.integralInfo[str(elec_spec)]
        if dataString==None:
            return dic
        else:
            return dic[str(dataString)]


#    fock={"faa":faa,"fbb":fbb}
#    tei={"g_aaaa":g_aaaa,"g_bbbb":g_bbbb,"g_abab":g_abab}
#    integralInfo={"oei":fock,"tei":tei}

""" ccd_main() is a general function that performs the background tasks necessary for a general CC calculation to take place, based on either RHF or UHF orbitals. 

:param mf: A PySCF SCF object, containing pertinent info regarding SCF calculation
:param mol: PySCF object containing info regarding the info regarding the molecule/system in question
:param orb: Set of - presumably converged - SCF coefficients
:param cc_runtype: Dictionary specifying information regarding the correlation calculation that is requested. Has two callable options; "ccdType" which refers to any of the T2-based methods that are implemented, and "fullCCtype" which calls one of CCSDT, CCSDTQ, or CCSDTQf calculations. A third option, "SOccdtype" will lead to spin-orbital formulations of the T2-based methods. Also contains such as convergence criteria, number of diis vectors & length, type of CC calculation, and max. number of iterations allowed in a given CC calculation


:return: Returns the correlation and full (combined correlation + mean field) energy

"""
def ccd_main(mf, mol, orb, cc_runtype):
    storedInfo=StoredInfo()
    cc_runtype.update({"hf_energy":mf.e_tot,"nuclear_energy":mf.energy_nuc()})
    storedInfo = convertSCFinfo(mf, mol, orb, cc_runtype, storedInfo)

    if "ccdType" in cc_runtype: # run all T2 spin-integrt methods
        na=storedInfo.occInfo["nocc_aa"]
        nb=storedInfo.occInfo["nocc_bb"]
        nvirta=storedInfo.occInfo["nvirt_aa"]
        nvirtb=storedInfo.occInfo["nvirt_bb"]

        occ_aa=storedInfo.occSliceInfo["occ_aa"]
        virt_aa=storedInfo.occSliceInfo["virt_aa"]
        occ_bb=storedInfo.occSliceInfo["occ_bb"]
        virt_bb=storedInfo.occSliceInfo["virt_bb"]


        #ints=storedInfo.get_integralInfo
        oei=storedInfo.get_integralInfo("oei")
        tei=storedInfo.get_integralInfo("tei")
        faa=oei["faa"]
        fbb=oei["fbb"]
        gaaaa=tei["g_aaaa"]
        gbbbb=tei["g_bbbb"]
        gabab=tei["g_abab"]

        eabij_aa=storedInfo.get_denoms("D2aa")
        eabij_bb=storedInfo.get_denoms("D2bb")
        eabij_ab=storedInfo.get_denoms("D2ab")

        cc_runtype.update({"max_iter":75,"stopping_eps":10**-10, "diis_size":10, "diis_start_cycle":4})

        CCDobj=kernel.UltT2CC(storedInfo)
        t2, currentE, corrE = CCDobj.kernel()


        t2aaaa, t2bbbb, t2abab, currentE,corrE = ccd_kernel(
            na,
            nb,
            nvirta,
            nvirtb,
            occ_aa,
            virt_aa,
            occ_bb,
            virt_bb,
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
            15,
            4,
            cc_runtype,
        )
#    elif "fullCCtype" in cc_runtype: # running >T2 spin-integrated code


#    elif "ccdTypeSlow" in cc_runtype: # run all T2 spin-orb methods


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



""" convertSCFinfo() is a general function that performs the background tasks necessary for a general CC calculation to take place, based on either RHF, UHF, or spin-orbital formalisms. Harvests relevant information, such as number of alpha/beta electrons, from PySCF mf and mol objects. 

:param mf: A PySCF SCF object, containing pertinent info regarding SCF calculation
:param mol: PySCF object containing info regarding the info regarding the molecule/system in question
:param orb: Set of - presumably converged - SCF coefficients
:param cc_runtype: Dictionary specifying information regarding the correlation calculation that is requested. Has two callable options; "ccdType" which refers to any of the T2-based methods that are implemented, and "fullCCtype" which calls one of CCSDT, CCSDTQ, or CCSDTQf calculations. A third option, "SOccdtype" will lead to spin-orbital formulations of the T2-based methods

:param occupationInfo: Dict of number of alpha/beta occupied/virtual orbitals. Callable with keys "nocc_aa", "nocc_bb", "nvirt_aa", and "nirt_bb".
:param integralInfo: Dict of (transformed) integral information, indexed by "faa", "fbb", "g_aaaa", "g_bbbb", and "g_abab" 
:param occupationSliceInfo: Dict containing slices for the occupied/virtual alpha/beta orbitals, indexed by "occ_aa", "virt_aa", "occ_bb", and "virt_bb"
:param eps: Dict containing alpha/beta molecular orbital energies, indexed by "eps_aa" or "eps_bb"
:param denomInfo: Dict containing denominators for all CC methods, indexed by "D2aa", "D2ab", "D2bb", and so on for arbitrary CC method. 

:return: Returns dictionaries of occupationInfo,integralInfo, eps, denomInfo, occupationSliceInfo

"""
def convertSCFinfo(mf, mol, orb,cc_runtype,storedInfo):
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




""" get_denoms() is a general function that performs the background tasks necessary for a general CC calculation to take place, based on either RHF, UHF, or spin-orbital formalisms. Constructs the denominators based on the CC theory being used.  

:param cc_runtype: Dictionary specifying information regarding the correlation calculation that is requested. Has two callable options; "ccdType" which refers to any of the T2-based methods that are implemented, and "fullCCtype" which calls one of CCSDT, CCSDTQ, or CCSDTQf calculations. A third option, "SOccdtype" will lead to spin-orbital formulations of the T2-based methods

:param occupationSliceInfo: Dict containing slices for the occupied/virtual alpha/beta orbitals
:param eps: Dict containing alpha/beta molecular orbital energies


:return: Dictionary of denominator info 

"""
def get_denoms(cc_runtype,occupationSliceInfo,eps):
    set_spinIntegrt=set(["ccdType", "fullCCtype"])
    denomInfo={}
    n = np.newaxis
    if "ccdTypeSlow" in cc_runtype: # spin-orb formalism
        virt_aa=occupationSliceInfo["virt_aa"]
        occ_aa=occupationSliceInfo["occ_aa"]
        epsaa=eps['occ_aa']
        print('inside two')

    elif "ccdType" in cc_runtype: # spin-integrated formalisms
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

    if "fullCCtype" in cc_runtype: 
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
    
        eabcijk_ab =1.0/ (
            -epsaa[virt_aa, n, n, n, n, n]
            - epsbb[n, virt_bb, n, n, n, n]
            - epsaa[n, n,virt_aa, n, n, n]
            + epsaa[n, n, n, occ_aa, n, n]
            + epsbb[n, n, n, n, occ_bb, n]
            + epsaa[n, n, n, n, n, occ_aa]
        )
        denomInfo.update({"D1aa":eai_aa,"D1bb":eai_bb,"D1ab":eai_ab,
                          "D3aa":eabcijk_aa,"D3bb":eabcijk_bb,"D3ab":eabcijk_ab})
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
