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
        import UT2.outfileA as qf_partA
        import UT2.outfileB as qf_partB
        import UT2.outfileC as qf_partC
        import UT2.outfileBresid as qf_residB

        import UT2.modify_T2resid_T4Qf1Slow as t4resids
        l2dic=ccd_kernel.get_l2amps()
        l2=t2_aaaa.transpose(2,3,0,1)
        g2=tei[va,va,oa,oa]*ccd_kernel.denom["D2aa"]
        g2=g2.transpose(2,3,0,1)
        t2_FO_dag=tei[va,va,oa,oa]*ccd_kernel.denom["D2aa"]
        t2_FO_dagger=t2_FO_dag.transpose(2,3,0,1)

        nocc=ccd_kernel.nocca
        nvirt=ccd_kernel.nvrta

        
        def is_antisymmetric(tensor):
            # Get the shape of the tensor
            shape = tensor.shape
            
            # Check if the tensor is rectangular
            if len(shape) != 8:
                raise ValueError('Tensor must have 8 indices')
            for i in range(4):
                if shape[i] != shape[i+4]:
                    raise ValueError('Tensor must be rectangular')
            
            # Check if the tensor is antisymmetric
            for i in range(shape[0]):
                for j in range(i+1, shape[1]):
                    for k in range(shape[2]):
                        for l in range(k+1, shape[3]):
                            for m in range(shape[4]):
                                for n in range(m+1, shape[5]):
                                    for p in range(shape[6]):
                                        for q in range(p+1, shape[7]):
                                            if not np.allclose(tensor[i,j,k,l,m,n,p,q],tensor[j,i,k,l,n,m,p,q]):#tensor[i,j,k,l,m,n,p,q] != -tensor[j,i,k,l,n,m,p,q]:
                                                return False,[i,j,k,l,m,n,p,q],tensor[i,j,k,l,m,n,p,q],tensor[j,i,k,l,n,m,p,q]
            
            return True


        Roovv=np.einsum('abim,cdnl,mnjk->abcdijkl',t2_aaaa,t2_aaaa,tei[oa,oa,oa,oa])
        print('test antisymmetry:', Roovv[0,1,2,3,0,1,2,3],Roovv[1,0,2,3,0,1,2,3],Roovv[1,0,2,3,0,1,3,2])
        antiRes=antisym.permute_t4oo_resid(Roovv)
        print('testing antisymmetry:',is_antisymmetric(antiRes),antiRes.shape,t2_FO_dagger.shape)
        print('5/4 ZWW oo energy:', np.einsum('ijab,klcd,abcdijkl->',t2_FO_dagger,t2_aaaa.transpose(2,3,0,1),antiRes))
        print('test antisymmetry:', antiRes[0,1,2,3,0,1,2,3],antiRes[0,1,2,3,0,1,3,2],Roovv[0,1,2,3,1,0,3,2])
        tvv=np.einsum('aeij,fdkl,bcef->abcdijkl',t2_aaaa,t2_aaaa,tei[va,va,va,va])
        anti_tvv=antisym.permute_t4vv_resid(tvv)
        print('testing antisymmetry of tvv:',is_antisymmetric(anti_tvv))
        e_vv=np.einsum('ijab,klcd,abcdijkl',t2_FO_dagger,t2_aaaa.transpose(2,3,0,1),anti_tvv)
        print('e_vv:',e_vv)

        t4_resid=np.zeros((nocc,nocc,nocc,nocc,nvirt,nvirt,nvirt,nvirt))
        t4_resid=antisym.unsym_residQf1(tei,t2_aaaa,oa,va,nocc,nvirt,t2_FO_dag) #t4resids.unsym_residQf1(tei,t2_aaaa,oa,va,nocc,nvirt)

        if ccd_kernel.cc_type == "CCD(Qf*)":
            print('doing Qf*')
            t4_resid+=t4resids.unsym_residQf2(tei,t2_aaaa,oa,va,nocc,nvirt)

#        # antisymmeterize the T4 residual:
        #antisym_t4_resid = antisym.antisym_t4_residual(t4_resid,nocc,nvirt)
        antisym_t4_resid = t4_resid.transpose(4,5,6,7,0,1,2,3)
        t2_FO_dag=tei[va,va,oa,oa]*ccd_kernel.denom["D2aa"]
        t2_FO_dagger=t2_FO_dag.transpose(2,3,0,1)

        qf_corr = einsum('klcd,ijab,abcdijkl', t2_FO_dagger, t2_aaaa.transpose(2,3,0,1), antisym_t4_resid[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
        print('full qf: COMPLETE ZWW 5/4',qf_corr,qf_corr/32.0)
        import UT2.pdagq_t4resid as pdag_t4
        t4_resid_oo,t4_resid=pdag_t4.t4_test_residual(t2_aaaa,tei,oa,va)
        qf_corr=0.062500000000000 * einsum('lkcd,ijba,cdbaijlk', t2_FO_dagger, t2_aaaa.transpose(2,3,0,1), t4_resid[:, :, :, :, :, :, :, :])
        qf_corr_oo=einsum('lkcd,ijba,cdbaijlk', t2_FO_dagger, t2_aaaa.transpose(2,3,0,1), t4_resid_oo[:, :, :, :, :, :, :, :])
        print('pdagger q qf :',qf_corr,qf_corr*(1.0/2.0))
        print('pdagger q qf only oo:', qf_corr_oo,qf_corr_oo*(1./32.))
        my_testcorr=einsum('lkcd,ijba,cdbaijlk',t2_FO_dagger, t2_aaaa.transpose(2,3,0,1),new_t4resid[:,:,:,:,:,:,:,:])
        print('5/2 test of qf corr e: ',my_testcorr,my_testcorr*(1./32.))
        return qf_corr*(1.0/2.0)
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

