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
import UT2.kernel as kernel

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
    # only returns perturbative energy corrections
        print('inside pert corr')
        contract_amp_obj=kernel.ContractAdjointAmps(ccd_kernel)
        factorization=True
        Pert_energy,energy_list=add_XCCenergy(factorization,ccd_kernel)
        print('new function; XCC energy by orders:',energy_list)
        print('perturbative energy correction: ', Pert_energy,sum(energy_list))
        return sum(energy_list) 

        # Add debug logic present at the bottom of this file here
        # to check the validity of perturbative energy corrections

    elif XCCD_Flag:
    # returns pert. energy + baseline CCD energy
        baseCCDE=ccdEnergy(t2_aaaa,fock,tei,oa,va)
        print('Inside t2energySlow XCCD_Flag if-statement. ccd pertorder:',ccd_kernel.pertOrder)

        # Extract the energy corrections order-by-order
        factorization=False # Need to specify contracting with final T2^dag instead of Wn-2
        testXCCenergy,energy_list=add_XCCenergy(factorization,ccd_kernel)
        print('new function; XCC energy by orders:',energy_list)
        print('CCD e:',baseCCDE,testXCCenergy)
        return baseCCDE+sum(energy_list)
 
    else:
    # returns standard CCD energy
        return ccdEnergy(t2_aaaa,fock,tei,oa,va) 


def add_XCCenergy(factorization,ccd_kernel):
    '''
    Determines the XCCD-like energy correction. If (Qf)-like correction is requested by factorization=True, then the final cap to build energy uses first-order T2; else, infinite-order (truly XCCD-like) T2^\dag is used to cap to build energy. 

    :param factorization: Boolean that - when True - means that the energy returned will be contracted with a final, first-order T2^\dag. If False, then the energy contraction will use an infinite-order T2^\dag to contract. 

    :param ccd_kernel: an instance of the UltT2CC class
    :return: Returns the total calculated XCCD-like energy, and a list containing the energy associated with orders thru which the calculation has been specified. So, if XCCD(6) calcalation has been requested, then 'energy_list' returns the energy associated with the 5th and 6th order energy corrections. 
 
    '''
    contract_amp_obj=kernel.ContractAdjointAmps(ccd_kernel)    
    XCCenergy=0.0
    energy_list=[]
    for order in ccd_kernel.pertOrder:
        energy=contract_amp_obj.buildXCCD_T2energy(order,factorization)
        energy_list.append(energy)
        XCCenergy+=energy
    return XCCenergy,energy_list

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


## OLD WAY TO CALCULTE 5/6 ORDER PERTURBATIVE CORRECTIONS; USE ONLY IF A CHECK IS NEEDED
#
#        t4_resid=np.zeros((nocc,nocc,nocc,nocc,nvirt,nvirt,nvirt,nvirt))
#        t4_resid=antisym.unsym_residQf1(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt,t2_FO_dag) #t4resids.unsym_residQf1(tei,t2_aaaa,oa,va,nocc,nvirt)
#        t22_dag=t2_aaaa.transpose(2,3,0,1)
#        qf_corr = einsum('klcd,ijab,abcdijkl', t2_FO_dagger, t2_aaaa.transpose(2,3,0,1), antisym_t4_resid[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])

 #       qf_corr=(1.0/32.0)*qf_corr
 #       print('Order 5-6 energy correction:', qf_corr, qf_corr*32.0)


#### NEW WAY TO CALCULATE ONLY 5TH ORDER CORRECTIONS; USE ONLY IF CHECK IS NEEDED
#        import UT2.test_qf as test_qf
#        test_t2_qf=test_qf.evaluate_residual_2_2(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt,t2_FO_dag)
#        test_t2_qf=test_t2_qf.transpose(2,3,0,1)
#        test_qf_corr= einsum('ijab,abij',t2_FO_dagger,test_t2_qf[:,:,:,:])
#        print('tested Qf corr by contract T4 to T2:', test_qf_corr/8.0, 4.0*test_qf_corr)


