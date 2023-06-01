"""
Drives the spin-orbital based CCD/T2 energy calculations
"""
import numpy as np
from numpy import einsum

import UT2.modify_T2energy_pertQfSlow as pertQf
import UT2.modify_T2resid_T4Qf1Slow as t4resids
import UT2.antisym_t4resids as antisym
import UT2.xccd_resid as xccd_resid
import re
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

    nocc=ccd_kernel.nocca
    nvirt=ccd_kernel.nvrta

    XCCD_methods=["XCCD(5)","XCCD(6)","XCCD(7)","XCCD(8)","XCCD(9)"]
    XCCD_Flag=any(element in ccd_kernel.cc_type for element in XCCD_methods)

    if get_perturbCorr==True:
        import UT2.modify_T2resid_T4Qf1Slow as t4resids
        l2dic=ccd_kernel.get_l2amps()
        l2=t2_aaaa.transpose(2,3,0,1)
        g2=tei[va,va,oa,oa]*ccd_kernel.denom["D2aa"]
        g2=g2.transpose(2,3,0,1)
        t2_FO_dag=tei[va,va,oa,oa]*ccd_kernel.denom["D2aa"]
        t2_FO_dagger=t2_FO_dag.transpose(2,3,0,1)


        t4_resid=np.zeros((nocc,nocc,nocc,nocc,nvirt,nvirt,nvirt,nvirt))
        t4_resid=antisym.unsym_residQf1(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt,t2_FO_dag) #t4resids.unsym_residQf1(tei,t2_aaaa,oa,va,nocc,nvirt)

#        if ccd_kernel.cc_type == "CCD(Qf*)":
#            print('doing Qf*')
#            t4_resid+=t4resids.unsym_residQf2(tei,t2_aaaa,oa,va,nocc,nvirt)

        antisym_t4_resid = t4_resid.transpose(4,5,6,7,0,1,2,3)
        t2_FO_dag=tei[va,va,oa,oa]*ccd_kernel.denom["D2aa"]

        t2_FO_dagger=t2_FO_dag.transpose(2,3,0,1)

        qf_corr = einsum('klcd,ijab,abcdijkl', t2_FO_dagger, t2_aaaa.transpose(2,3,0,1), antisym_t4_resid[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])

        qf_corr=(1.0/32.0)*qf_corr
        print('Order 5-6 energy correction:', qf_corr, qf_corr*32.0)

        import UT2.test_qf as test_qf
        test_t2_qf=test_qf.evaluate_residual_2_2(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt,t2_FO_dag)
        test_t2_qf=test_t2_qf.transpose(2,3,0,1)
        test_qf_corr= einsum('ijab,abij',t2_FO_dagger,test_t2_qf[:,:,:,:])
        print('tested Qf corr by contract T4 to T2:', test_qf_corr/8.0, 4.0*test_qf_corr)


       
        order_7E=order_8E=order_9E=0.0

        hgherO=int(ccd_kernel.pert_E_corr)
        if hgherO >= 7:
            t4_7=antisym.unsym_resid7(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt,t2_FO_dag)
            t4_7=t4_7.transpose(4,5,6,7,0,1,2,3)
            order_7E=einsum('klcd,ijab,abcdijkl', t2_FO_dagger, t2_aaaa.transpose(2,3,0,1),t4_7[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
            order_7E=(1.0/128.0)*order_7E
            print('Order 7 energy correction:', order_7E)

            if hgherO >= 8:
                t6_5O=antisym.unsym_resid8(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt,t2_FO_dag)
                t6_5O=t6_5O.transpose(6,7,8,9,10,11,0,1,2,3,4,5)
                order_8E=einsum('mnef,klcd,ijab,abcdefijklmn', t2_FO_dagger, t2_aaaa.transpose(2,3,0,1), t2_aaaa.transpose(2,3,0,1),t6_5O[:, :, :, :, :, :, :, :, : ,: ,: ,:], optimize=['einsum_path', (0, 2), (0, 1)])
                order_8E=(1.0/384.0)*order_8E
                print('Order 8 energy correction:', order_8E)

                if hgherO == 9:
                    t6_6O=antisym.unsym_resid9(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt,t2_FO_dag)
                    t6_6O=t6_6O.transpose(6,7,8,9,10,11,0,1,2,3,4,5)
                    order_9E=einsum('mnef,klcd,ijab,abcdefijklmn', t2_FO_dagger, t2_aaaa.transpose(2,3,0,1), t2_aaaa.transpose(2,3,0,1),t6_6O[:, :, :, :, :, :, :, :, : ,: ,: ,:], optimize=['einsum_path', (0, 2), (0, 1)])
                    order_9E=(1.0/384.0)*order_9E
                    print('Order 8 energy correction:', order_8E)
                                     
        totalEcorr=qf_corr+order_7E+order_8E+order_9E

        return totalEcorr #qf_corr*(1.0/32.0)


    elif XCCD_Flag:
        XCCD_energy=baseCCDE=order_5_6_E=order_7E=order_8E=order_9E=0.0

        baseCCDE=ccdEnergy(t2_aaaa,fock,tei,oa,va)
        energy_mod=extract_integer(ccd_kernel.cc_type)
        xcc_t2Dag=t2_aaaa.transpose(2,3,0,1)
        if energy_mod <= 6:
            t4_resid=antisym.unsym_residQf1(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt)

            antisym_t4_resid=t4_resid.transpose(4,5,6,7,0,1,2,3)
            order_5_6_E=(1.0/32.0)*einsum('klcd,ijab,abcdijkl',xcc_t2Dag,xcc_t2Dag,antisym_t4_resid)

        if energy_mod <= 7:
            t4_t2DagWT23=xccd_resid.xccd_7(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt)
            t4_t2DagWT23=t4_t2DagWT23.transpose(4,5,6,7,0,1,2,3)
            order_7E=(1.0/128.0)*einsum('klcd,ijab,abcdijkl',xcc_t2Dag,xcc_t2Dag,t4_t2DagWT23)

        XCCD_energy=baseCCDE+order_5_6_E+order_7E+order_8E+order_9E
        return XCCD_energy
    else:    
        return ccdEnergy(t2_aaaa,fock,tei,oa,va) 



def extract_UT2(string):
    " Returns the -- UT2 -- from the method strings name, as in UT2-CCD(5), if it is present; otherwise returns None"
    match = re.search(r'UT2', string)
    return match.group() if match else None

def extract_integer(string):
    ''' Returns the integer found within a string. Used to parse method labels and determine which perturbative orders of correction the user wants'''
    match = re.search(r'\d+', string)
    return int(match.group()) if match else None

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

