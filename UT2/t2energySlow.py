"""
Drives the spin-orbital based CCD/T2 energy calculations
"""
import numpy as np
from numpy import einsum

import UT2.modify_T2energy_pertQfSlow as pertQf
import UT2.modify_T2resid_T4Qf1Slow as t4resids
import UT2.antisym_t4resids as antisym

def ccd_energyMain(ccd_kernel,get_perturbCorr=False):
    """
    Drives the determination of spin-orbital, CCD energy. This includes unmodified energy, as well as calling subsequent modules to extract perturbative corrections.

    :param ccd_kernel: Object of the UltT2CC class.
    :param get_perturbCorr: Boolean flag to determine if perturbative corrections to the energy are called for

    :return: Returns either the baseline CCD energy, or factorization based perturbative energy corrections
    """
    sliceInfo=ccd_kernel.sliceInfo
    oa=sliceInfo["occ_aa"]
    va=sliceInfo["virt_aa"]

    t2_aaaa=ccd_kernel.tamps["t2aa"]

    fock=ccd_kernel.ints["oei"]
    tei=ccd_kernel.ints["tei"]

    print(np.shape(tei),np.shape(t2_aaaa),oa,va)
    print(np.shape(tei[oa,oa,oa,oa]))

    if get_perturbCorr==True:
        import UT2.modify_T2resid_T4Qf1Slow as t4resids
        l2dic=ccd_kernel.get_l2amps()
        l2=t2_aaaa.transpose(2,3,0,1)
        g2=tei[va,va,oa,oa]*ccd_kernel.denom["D2aa"]
        g2=g2.transpose(2,3,0,1)
        t2_FO_dag=tei[va,va,oa,oa]*ccd_kernel.denom["D2aa"]
        t2_FO_dagger=t2_FO_dag.transpose(2,3,0,1)

        nocc=ccd_kernel.nocca
        nvirt=ccd_kernel.nvrta

        t4_resid=np.zeros((nocc,nocc,nocc,nocc,nvirt,nvirt,nvirt,nvirt))
        t4_resid=antisym.unsym_residQf1(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt,t2_FO_dag) #t4resids.unsym_residQf1(tei,t2_aaaa,oa,va,nocc,nvirt)

#        if ccd_kernel.cc_type == "CCD(Qf*)":
#            print('doing Qf*')
#            t4_resid+=t4resids.unsym_residQf2(tei,t2_aaaa,oa,va,nocc,nvirt)

        antisym_t4_resid = t4_resid.transpose(4,5,6,7,0,1,2,3)
        t2_FO_dag=tei[va,va,oa,oa]*ccd_kernel.denom["D2aa"]
        t2_FO_dagger=t2_FO_dag.transpose(2,3,0,1)

        qf_corr = einsum('klcd,ijab,abcdijkl', t2_FO_dagger, t2_aaaa.transpose(2,3,0,1), antisym_t4_resid[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
        return qf_corr*(1.0/32.0)
    elif ccd_kernel.cc_type == "pCCD":
        return ccdEnergy(t2_aaaa,fock,tei,oa,va)
    else:    
        return ccdEnergy(t2_aaaa,fock,tei,oa,va) 

def pccdEnergy(t2,f,g,o,v):
    energy = einsum('ii',f[o,o])
    energy += -0.50000*einsum('jiji', g[o, o, o, o])
    energy += 0.250000*einsum('iiaa,aaii',g[o,o,v,v],t2)
    return energy

def ccdEnergy(t2,f,g,o,v):
    """
    Exclusively handles the calculation of the CCD-like energy, given set of integrals and T amplitudes.

    :param t2: T2 tensor
    :param f: Fock matrix
    :param g: 2 electron integral tensor
    :param o: occupied orbital slice
    :param v: virtual orbital slice

    :return: CCD energy
    """
    #         1.0000 f(i,i)
    energy =  1.000000000000000 * einsum('ii', f[o, o])
    #        -0.5000 <j,i||j,i>
    energy += -0.500000000000000 * einsum('jiji', g[o, o, o, o])
    #         0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy +=  0.250000000000000 * einsum('jiab,abji', g[o, o, v, v], t2)

    return energy

