import numpy as np
import UT2.tamps as tamps
import UT2.modify_T2resid_T4Qf1Slow as pdag_xcc5
import UT2.set_denoms as set_denoms
import UT2.build_qf_correction as qf
import sys

def ccenergy_driver(driveCCobj,cc_info):
    o=driveCCobj.occSliceInfo["occ_aa"]
    v=driveCCobj.occSliceInfo["virt_aa"]
    
    W=driveCCobj.integralInfo["tei"]
    Fock=driveCCobj.integralInfo["oei"]
    if "slowSOcalc" in cc_info: #spin-orb energy
        t2=driveCCobj.tamps["t2aa"]
        if "t1aa" in driveCCobj.tamps:
            t1=driveCCobj.tamps["t1aa"]
        else:
            t1=None

        ccsd_energy=spinorbitalCCSDE(t2,W,Fock,o,v,t1)
        print('ccsd e',ccsd_energy) 
        return ccsd_energy




def spinorbitalCCSDE(T2,W,F,o,v,T1=None):
    #         1.0000 f(i,i)
    energy =  1.000000000000000 * np.einsum('ii', F[o, o])
    #        -0.5000 <j,i||j,i>
    energy += -0.500000000000000 * np.einsum('jiji', W[o, o, o, o])
    energy += 0.250000000 * np.einsum("ijab,abij->",T2,W[v,v,o,o],optimize="optimal")

    if T1 is not None:
        energy += 1.000000000 * np.einsum("ai,ia->",F[v,o],T1,optimize="optimal")
        energy += 0.500000000 * np.einsum("ia,jb,abij->",T1,T1,W[v,v,o,o],optimize="optimal")


    return energy





def perturbE_driver(CCobj,cc_type):
    o=CCobj.occSliceInfo["occ_aa"]
    v=CCobj.occSliceInfo["virt_aa"]

    T1=CCobj.tamps.get("t1aa",None)
    T2=CCobj.tamps.get("t2aa",None)
    T3=CCobj.tamps.get("t3aa",None)

    D1=CCobj.denomInfo.get("D1aa",None)
    D2=CCobj.denomInfo.get("D2aa",None)
    D3=CCobj.denomInfo.get("D3aa",set_denoms.D3denomSlow(CCobj.eps,CCobj.occSliceInfo["occ_aa"],CCobj.occSliceInfo["virt_aa"],np.newaxis))

    W=CCobj.integralInfo["tei"]
    if "(qf)" in cc_type: # calculate both fifth and sixth-order contributions
        if T3 is None: # If calculation is CCSD(Qf), **have not done** (T) work
            T3=build_approxT3(W,T2,o,v)
            T3=tamps.antisym_T3(T3,None,None)

        t2_resid = 0.5*pdag_xcc5.residQf1_aaaa(W,T2,T2.transpose(2,3,0,1),o,v)
        fo_t2_b= W[o,o,v,v]*D2 #t2.transpose(2,3,0,1)
        #fo_t2_b=fo_t2_b.transpose(2,3,0,1)
        fifthO_improvement=0.25*np.einsum('jiab,abji',fo_t2_b,t2_resid)
        print('5th order energy:',fifthO_improvement,4.0*fifthO_improvement)


        D3T3=T3*D3
        D3T3=D3T3.transpose(3,4,5,0,1,2)    
    
        sqrBrakT= 0.25*0.111111111 * np.einsum("ijkabc,abcijk->",T3,D3T3,optimize="optimal")
        print('[T] correction to CCSD:', sqrBrakT)
        t2resid_t3 = (0.25)*T3portion_Qf(W,T2.transpose(2,3,0,1),T3,o,v)
        t2resid_t3=tamps.antisym_T2(t2resid_t3,None,None)
        t3contrib=0.25*np.einsum('jiab,abji',fo_t2_b,t2resid_t3.transpose(2,3,0,1))
        print('t3 contribution to Qf:',2.0*t3contrib)
    
        print('Combined Qf correction:', 2.0*t3contrib+fifthO_improvement)
        
        long_testQf5Order(W,T2,T3,o,v,D2)
        print(flush=True)
        sixthOrderQf_wnT2T3(W,T2,T3,o,v,D2)

        sixthOrderQf_wnT2cubed(W,T2,o,v,D2)
        return {"help":2}


def sixthOrderQf_wnT2cubed(W,T2,o,v,D2):
    D4T4 = qf.wnT2cubed_toT4(W,T2,o,v)
    D4T4 = D4T4.transpose(4,5,6,7,0,1,2,3)
    resid_aaaa = (1.0/8.0)*np.einsum('klcd,abcdijkl->abij',T2,D4T4)
    sixthO_wnT2cubed=(1.0/4.0)*np.einsum("ijab,abij",W[o,o,v,v]*D2,resid_aaaa)
    print('WnT2^3 contribution to (Qf):',sixthO_wnT2cubed)
    print(flush=True)

    # Test _toT2 part:
    D2T2=0.5*qf.wnT2cubed_toT2(W,T2,o,v)
    D2T2=D2T2.transpose(2,3,0,1)
    teste=(1.0/4.0)*np.einsum("ijab,abij",W[o,o,v,v]*D2,D2T2)
    print('WnT2^3 contribution to (Qf) faster T2 route:',teste)

def sixthOrderQf_wnT2T3(W,T2,T3,o,v,D2):
    D4T4 = qf.wnT2T3_toT4(W,T2,T3,o,v)
    D4T4 = D4T4.transpose(4,5,6,7,0,1,2,3)
    resid_aaaa = (1.0/8.0)*np.einsum('klcd,abcdijkl->abij',T2,D4T4)
    sixthO_wnT2T3=(1.0/4.0)*np.einsum("ijab,abij",W[o,o,v,v]*D2,resid_aaaa)
    print('WnT2T3 contribution to (Qf):',sixthO_wnT2T3)
    
    # Test _toT2 part:
    D2T2=0.5*qf.wnT2T3_toT2(W,T2,T3,o,v)
    D2T2=D2T2.transpose(2,3,0,1)
    teste=(1.0/4.0)*np.einsum("ijab,abij",W[o,o,v,v]*D2,D2T2)
    print('WnT2T3 contribution to (Qf) faster T2 route:',teste)

def long_testQf5Order(W,T2,T3,o,v,D2):
    v_oo=W[o,o,o,o]
    v_vo=W[o,v,o,v]
    v_vv=W[v,v,v,v]
    no=np.shape(T2)[0]
    nv=np.shape(T2)[2]
    Roooovvvv= np.zeros((no,no,no,no,nv,nv,nv,nv))
    Roooovvvv += -0.062500000 * np.einsum("imab,jncd,klmn->ijklabcd",T2,T2,v_oo,optimize="optimal")
    Roooovvvv += -0.250000000 * np.einsum("imab,jkce,lemd->ijklabcd",T2,T2,v_vo,optimize="optimal")
    Roooovvvv += -0.062500000 * np.einsum("ijae,klbf,efcd->ijklabcd",T2,T2,v_vv,optimize="optimal")

    Roooovvvv = tamps.antisym_T4(Roooovvvv,None,None)
    t4T=Roooovvvv.transpose(4,5,6,7,0,1,2,3)

    resid_aaaa = (1.0/8.0)*np.einsum('klcd,abcdijkl->abij',T2,t4T)
    fifthorder_wnt2=(1.0/4.0)*np.einsum("ijab,abij",W[o,o,v,v]*D2,resid_aaaa)
    print('WnT2^2 contribution to (Qf):',fifthorder_wnt2)

    #Test T2 part
    t2test=0.5*optWnT2sqr(W,T2,o,v)
    t2test=tamps.antisym_T2(t2test,None,None)
    t2test=t2test.transpose(2,3,0,1)
    teste=(1.0/4.0)*np.einsum("ijab,abij",W[o,o,v,v]*D2,t2test)
    print('test WnT2^2 scaling faster:',teste)
    

    # Test fast T3 part
    t2test=0.5*optWnT3(W,T3,T2,o,v)
    t2test=tamps.antisym_T2(t2test,None,None)
    t2test=t2test.transpose(2,3,0,1)
    teste=(1.0/4.0)*np.einsum("ijab,abij",W[o,o,v,v]*D2,t2test)
    print('test WnT3 scaling faster:',teste)

    ### T3 part
    Roooovvvv= np.zeros((no,no,no,no,nv,nv,nv,nv))
    Roooovvvv += -0.041666667 * np.einsum("ijmabc,klmd->ijklabcd",T3,W[o,o,o,v],optimize="optimal")
    Roooovvvv += -0.041666667 * np.einsum("ijkabe,lecd->ijklabcd",T3,W[o,v,v,v],optimize="optimal")

    Roooovvvv = tamps.antisym_T4(Roooovvvv,None,None)
    t4T=Roooovvvv.transpose(4,5,6,7,0,1,2,3)

    resid_aaaa = (1.0/8.0)*np.einsum('klcd,abcdijkl->abij',T2,t4T)
    fifthorder_wnt3=(1.0/4.0)*np.einsum("ijab,abij",W[o,o,v,v]*D2,resid_aaaa)
    print('WnT3 contribution to (Qf):',fifthorder_wnt3)
    print('total Qf:',fifthorder_wnt3+fifthorder_wnt2)

    # test antisym intermediate T4->T2 first:
    Roooovvvv= np.zeros((no,no,no,no,nv,nv,nv,nv))
    Roooovvvv += -0.041666667 * np.einsum("ijmabc,klmd->ijklabcd",T3,W[o,o,o,v],optimize="optimal")
    Roooovvvv += -0.041666667 * np.einsum("ijkabe,lecd->ijklabcd",T3,W[o,v,v,v],optimize="optimal")

    Roooovvvv = tamps.antisym_T4(Roooovvvv,None,None)
    t4T=Roooovvvv.transpose(4,5,6,7,0,1,2,3)

    resid_aaaa = (1.0/16.0)*np.einsum('klcd,abcdijkl->abij',T2,t4T)
    resid_aaaa = tamps.antisym_T2(resid_aaaa,None,None)
    fifthorder_wnt3=(1.0/4.0)*np.einsum("ijab,abij",W[o,o,v,v]*D2,resid_aaaa)
    print('WnT3 contribution to (Qf):',fifthorder_wnt3,fifthorder_wnt3*0.5)

def optWnT3(W,T3,T2,o,v):
    T2dag=T2.transpose(2,3,0,1)

    roovv = 0.125000000 * np.einsum("cdkl,ijmabd,klmc->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("cdkl,ilmabd,jkmc->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("cdkl,klmabd,ijmc->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("cdkl,ijlabe,kecd->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,iklabe,jecd->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,ijmacd,klmb->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("cdkl,ilmacd,jkmb->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,klmacd,ijmb->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("cdkl,ijlade,kebc->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("cdkl,iklade,jebc->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("cdkl,ijlcde,keab->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,iklcde,jeab->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")

    return roovv




def optWnT2sqr(W,T2,o,v):
    T2dag=T2.transpose(2,3,0,1)

    roovv = -0.500000000 * np.einsum("ikab,jlcd,celm,mdke->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikab,jlcd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikab,lmcd,celm,jdke->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikab,lmcd,cdln,jnkm->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klab,ijcd,cekm,mdle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klab,imcd,cekm,jdle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klab,imcd,cdkn,jnlm->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klab,mncd,cdkm,ijln->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijac,klbd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ijac,klbd,dekm,mcle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijac,klde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,klde,dekm,mclb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,jlbd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ikac,jlbd,dekm,mcle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,jlbd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,lmbd,dekl,jcme->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,lmbd,cdln,jnkm->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jlde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlde,dekm,mclb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jlde,cdlm,mekb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikac,lmde,dekl,jcmb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikac,lmde,cdlm,jekb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klac,imbd,dekl,jcme->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,mnbd,cdkm,ijln->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,ijde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klac,ijde,cdkm,melb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klac,imde,dekl,jcmb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("klac,imde,cdkm,jelb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijcd,klef,cekl,dfab->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,jlef,cekl,dfab->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    return roovv



def T3portion_Qf(W,T2dag,T3,o,v):
    # contributions to the residual
    roovv = 0.125000000 * np.einsum("cdkl,ijmabd,klmc->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("cdkl,ilmabd,jkmc->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("cdkl,klmabd,ijmc->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("cdkl,ijlabe,kecd->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,iklabe,jecd->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,ijmacd,klmb->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("cdkl,ilmacd,jkmb->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,klmacd,ijmb->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("cdkl,ijlade,kebc->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("cdkl,iklade,jebc->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("cdkl,ijlcde,keab->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,iklcde,jeab->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    return roovv

def build_approxT3(W,T2,o,v):
    # contributions to the residual
    rooovvv = -0.250000000 * np.einsum("ilab,jklc->ijkabc",T2,W[o,o,o,v],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ijad,kdbc->ijkabc",T2,W[o,v,v,v],optimize="optimal")
    return rooovvv
