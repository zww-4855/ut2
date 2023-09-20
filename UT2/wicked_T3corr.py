import numpy as np


def wicked_main(g,o,v,t1,t1_dag,t2,t2_dag,D3denom,D2denom,nocc,nvirt):
    t3resids_base=build_T3_resids(g,o,v,t2,nocc,nvirt)

    ### test capping with two e integral
    t2_firstO=g[v,v,o,o]*D2denom
    t2_firstO=t2_firstO.transpose(2,3,0,1)
    t3caps=build_T3_resids(g,o,v,t2_firstO,nocc,nvirt)
    t3cap=t3caps[2]
    t3cap=t3cap.transpose(3,4,5,0,1,2)*D3denom
    newwayE=0.111111111 * np.einsum("ijkabc,abcijk->",t3resids_base[2],t3cap,optimize="optimal")
    print('2e cap is: ', newwayE,newwayE*0.25)
    ## end testing

    energy_total=0.0
    energy_list={}

    # Get fourth order correction
    fourthOenergy=get_fourthOrder(t3resids_base[2],D3denom)
    print('fourthOenergy:',fourthOenergy,0.25*fourthOenergy)
    energy_total+=0.25*fourthOenergy
    energy_list.update({4:0.25*fourthOenergy})

    # Get fifth order correction
    fifthOenergy=get_fifthOrder(g,o,v,t2_dag,nocc,nvirt,t3resids_base,D3denom)
    return 0.25*newwayE

def get_fifthOrder(g,o,v,t2_dag,nocc,nvirt,t3resids,D3denom):
    # construct diagram A
    capA=D3denom*fifthOrder_capA(g,o,v,t2_dag,nocc,nvirt)
    capA_energy=contractE(t3resids[2],capA)
    print('capA_energy:',capA_energy,(-1.0/8.0)*capA_energy)

    # construct diagram B
    capB=t3resids[2].transpose(3,4,5,0,1,2)
    capB=capB*D3denom
    capB_energy=contractE(t3resids[3],capB)
    print('capB_energy',capB_energy,0.25*capB_energy)

    # construct diagram C
    capC=fifthOrder_capC(g,o,v,t2_dag,nocc,nvirt)
    capC=capC*D3denom
    capC_energy=contractE(t3resids[2],capC)
    print('capC_energy',capC_energy,0.25*capC_energy)

def fifthOrder_capC(g,o,v,t2_dag,nocc,nvirt):
    rvvvooo = -0.125000000 * np.einsum("abij,dekl,lcde->abcijk",t2_dag,t2_dag,g[o,v,v,v],optimize="optimal")
    rvvvooo += -0.125000000 * np.einsum("abij,cdlm,lmkd->abcijk",t2_dag,t2_dag,g[o,o,o,v],optimize="optimal")
    rvvvooo=rvvvooo.transpose(3,4,5,0,1,2)
    rvvvooo=antisym_T3(rvvvooo,nocc,nvirt)
    rvvvooo=rvvvooo.transpose(3,4,5,0,1,2)
    return rvvvooo

def fifthOrder_capA(g,o,v,t2_dag,nocc,nvirt):
    rvvvooo = 1.000000000 * np.einsum("adij,bekl,lcde->abcijk",t2_dag,t2_dag,g[o,v,v,v],optimize="optimal")
    rvvvooo += -0.250000000 * np.einsum("adij,bclm,lmkd->abcijk",t2_dag,t2_dag,g[o,o,o,v],optimize="optimal")
    rvvvooo += -0.250000000 * np.einsum("deij,abkl,lcde->abcijk",t2_dag,t2_dag,g[o,v,v,v],optimize="optimal")
    rvvvooo += 1.000000000 * np.einsum("abil,cdjm,lmkd->abcijk",t2_dag,t2_dag,g[o,o,o,v],optimize="optimal")
    rvvvooo=rvvvooo.transpose(3,4,5,0,1,2)
    rvvvooo=antisym_T3(rvvvooo,nocc,nvirt)
    rvvvooo=rvvvooo.transpose(3,4,5,0,1,2)
    return rvvvooo


def get_fourthOrder(t3resid,D3denom):
    t3_dag=t3resid.transpose(3,4,5,0,1,2)
    t3_dag=t3_dag*D3denom
    return contractE(t3resid,t3_dag)

def contractE(t3resid,t3_dag):
    return 0.111111111 * np.einsum("ijkabc,abcijk->",t3resid,t3_dag,optimize="optimal")

def build_T3_resids(g,o,v,t2,nocc,nvirt):
    t3resids={}
    # Build second-order T3 resid eqn
    secondOrder=build_T3_secondO(g,o,v,t2)
    secondOrder=antisym_T3(secondOrder,nocc,nvirt)
    t3resids.update({2:secondOrder})

    # Build third-order T3 resid eqn
    thirdOrder=build_T3_thirdO(g,o,v,t2)
    thirdOrder=antisym_T3(thirdOrder,nocc,nvirt)
    t3resids.update({3:thirdOrder})
    return t3resids


def build_T3_thirdO(g,o,v,t2):
    rooovvv = 0.500000000 * np.einsum("ilab,jmcd,kdlm->ijkabc",t2,t2,g[o,v,o,o],optimize="optimal")
    rooovvv += -0.125000000 * np.einsum("ilab,jkde,delc->ijkabc",t2,t2,g[v,v,o,v],optimize="optimal")
    rooovvv += -0.125000000 * np.einsum("lmab,ijcd,kdlm->ijkabc",t2,t2,g[o,v,o,o],optimize="optimal")
    rooovvv += 0.500000000 * np.einsum("ijad,klbe,delc->ijkabc",t2,t2,g[v,v,o,v],optimize="optimal")
    return rooovvv

def build_netT2(g,o,v,t3):
    roovv = -0.250000000 * np.einsum("iklabc,jckl->ijab",t3,g[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijkacd,cdkb->ijab",t3,g[v,v,o,v],optimize="optimal")

    return roovv 

def build_T3_secondO(g,o,v,t2):
    #rooovvv = np.zeros((nocc,nocc,nocc,nvir,nvir,nvir))
    rooovvv = -0.250000000 * np.einsum("ilab,jklc->ijkabc",t2,g[o,o,o,v],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ijad,kdbc->ijkabc",t2,g[o,v,v,v],optimize="optimal")
    return rooovvv

def getE_squareBrackT(g,o,v,t3,t2_dag):
    energy = 0.0
    energy += -0.250000000 * np.einsum("ijkabc,adij,bckd->",t3,t2_dag,g[v,v,o,v],optimize="optimal")
    energy += -0.250000000 * np.einsum("ijkabc,abil,lcjk->",t3,t2_dag,g[o,v,o,o],optimize="optimal")
    return energy

def getE_parenT_t1(g,o,v,t3,t1_dag):
    energy = 0.0
    energy += 0.250000000 * np.einsum("ijkabc,ai,bcjk->",t3,t1_dag,g[v,v,o,o],optimize="optimal")
    return energy









def antisym_T3(Rooovvv, nocc, nvir):
    # antisymmetrize the residual
    Rooovvv_anti = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir))
    Rooovvv_anti += +1 * np.einsum("ijkabc->ijkabc", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ijkacb", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ijkbac", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ijkbca", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ijkcab", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ijkcba", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ikjabc", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ikjacb", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ikjbac", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ikjbca", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->ikjcab", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->ikjcba", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jikabc", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jikacb", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jikbac", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jikbca", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jikcab", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jikcba", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jkiabc", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jkiacb", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jkibac", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jkibca", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->jkicab", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->jkicba", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kijabc", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kijacb", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kijbac", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kijbca", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kijcab", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kijcba", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kjiabc", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kjiacb", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kjibac", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kjibca", Rooovvv)
    Rooovvv_anti += -1 * np.einsum("ijkabc->kjicab", Rooovvv)
    Rooovvv_anti += +1 * np.einsum("ijkabc->kjicba", Rooovvv)
    return Rooovvv_anti


def antisym_T2(Roovv, nocc, nvir):
    # antisymmetrize the residual
    Roovv_anti = np.zeros((nocc, nocc, nvir, nvir))
    Roovv_anti += np.einsum("ijab->ijab", Roovv)
    Roovv_anti -= np.einsum("ijab->jiab", Roovv)
    Roovv_anti -= np.einsum("ijab->ijba", Roovv)
    Roovv_anti += np.einsum("ijab->jiba", Roovv)
    return Roovv_anti
