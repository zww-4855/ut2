import numpy as np
import UT2.t2residEqnsSlow as t2residEqnsSlow
import UT2.t2energySlow as t2energySlow
import UT2.wicked_T3corr as wicked_T3corr

from numpy import einsum

def ucc4_energy_simplified(ccd_kernel):
    ccd_energy=t2energySlow.ccd_energyMain(ccd_kernel)

    t2=ccd_kernel.tamps["t2aa"]
    t2_dag=t2.transpose(2,3,0,1)
    o=ccd_kernel.sliceInfo["occ_aa"]
    v=ccd_kernel.sliceInfo["virt_aa"]
    g=ccd_kernel.ints["tei"]

    t2_resid=2.0*offset_t2resid_t2dag_w_t2(t2,t2_dag,g,o,v)
    t2_resid=t2_resid*ccd_kernel.denom["D2aa"]
    print('CCD portion of E:',ccd_energy)
    print('UCC portion of E:',np.einsum('jiab,abji',t2_dag,t2_resid))
    totalE=ccd_energy+np.einsum('jiab,abji',t2_dag,t2_resid)
    print('UCC(4) energy is: ', totalE)
    return totalE



def resid_main(ccd_kernel):
    sliceInfo=ccd_kernel.sliceInfo
    o=sliceInfo["occ_aa"]
    v=sliceInfo["virt_aa"]
    occaa=o
    virtaa=v
    nocc=ccd_kernel.nocca
    nvirt=ccd_kernel.nvrta
    print('nvirt,nocc',nvirt,nocc)
    print('slice info',v,o)
    print(ccd_kernel.tamps.keys())
    if 't3aa' not in ccd_kernel.tamps.keys():
        t1=np.zeros((nvirt,nocc))
        t3=np.zeros((nvirt,nvirt,nvirt,nocc,nocc,nocc))
        ccd_kernel.tamps.update({'t1aa':t1,'t3aa':t3})

    # all amplitudes are initially stored with nvirt index coming first
    # will need to reshape t1 and t3 to fit into wicked ordering, then reshape 
    # back into denom ordering
    t2=ccd_kernel.tamps["t2aa"]
    t2_dag=t2.transpose(2,3,0,1)

    t1=ccd_kernel.tamps["t1aa"]#.transpose(1,0)
    t3=ccd_kernel.tamps["t3aa"]#.transpose(3,4,5,0,1,2)


    fock=ccd_kernel.ints["oei"]
    g=ccd_kernel.ints["tei"]

    resid_aaaaBKUP = np.zeros((nvirt,nvirt,nocc,nocc))
    eabij_aa=ccd_kernel.denom["D2aa"]
    eai_aa=ccd_kernel.denom["D1aa"]
    eabcijk_aa=ccd_kernel.denom["D3aa"]

    # solve for T1 residuals
    resid_t1=t1_residEqns(t1,t2,fock,g,o,v)


    # solve for T2 residuals
    resid_t2=t2residEqnsSlow.ccd_t2residual(t2,fock,g,o,v)
    resid_t2-=offset_t2resid_t2squared(t2,g,o,v)
    resid_t2+=offset_t2resid_t2dag_w_t2(t2,t2_dag,g,o,v)
    # Add WT1 and WT3 components to T2 resid eqns
    resid_t2+=offset_t2resid_T1_T3(t1,t3,g,o,v)


    # solve for T3 residuals
    resid_t3=t3_resid(t3,t2,fock,g,o,v)
    #resid_t3=wicked_T3corr.antisym_T3(resid_t3, nocc, nvirt)

    # re-order all amplitudes and residuals
    #resid_t1=resid_t1.transpose(1,0)
    #resid_t3=resid_t3.transpose(3,4,5,0,1,2)

    resid_t1=resid_t1+np.reciprocal(eai_aa)*t1#.transpose(1,0)
    resid_t2=resid_t2+np.reciprocal(eabij_aa)*t2
    resid_t3=resid_t3+np.reciprocal(eabcijk_aa)*t3#.transpose(3,4,5,0,1,2)

    t2amp={'t2aa':resid_t2*eabij_aa}
    t1amp={'t1aa':resid_t1*eai_aa}
    t3amp={'t3aa':resid_t3*eabcijk_aa*0.0}

    fin_tamp={'t2aa':resid_t2*eabij_aa,'t1aa':resid_t1*eai_aa,'t3aa':resid_t3*eabcijk_aa}
    ccd_kernel.tamps.update(fin_tamp)
    print(ccd_kernel.tamps.keys())
    print(ccd_kernel.tamps['t1aa'])
    #sys.exit()



def t1_residEqns(t1,t2,f,g,o,v):
    #        -0.5000 <k,j||b,i>*t2(b,a,k,j)
    singles_res = -0.500000000000000 * einsum('kjbi,bakj->ai', g[o, o, v, o], t2)
    
    #        -0.5000 <j,a||b,c>*t2(b,c,i,j)
    singles_res += -0.500000000000000 * einsum('jabc,bcij->ai', g[o, v, v, v], t2)
    
    #         1.0000 f(a,i)
    singles_res +=  1.000000000000000 * einsum('ai->ai', f[v, o])
    
    #        -1.0000 f(j,i)*t1(a,j)
    singles_res += -1.000000000000000 * einsum('ji,aj->ai', f[o, o], t1)
    
    #         1.0000 f(a,b)*t1(b,i)
    singles_res +=  1.000000000000000 * einsum('ab,bi->ai', f[v, v], t1)
    return singles_res

#    t2=t2.transpose(2,3,0,1)
#    print(np.shape(t2),np.shape(t2[o,o,v,v]),np.shape(t2[v,v,o,o]),o,v,np.shape(g[o,v,o,o]))
#    Rov = -1.000000000 * np.einsum("ij,ja->ia",fock[o,o],t1,optimize="optimal")
#    print(np.shape(fock[o,o]),np.shape(t1[o,v]),np.shape(t1))
#    Rov += -0.500000000 * np.einsum("ibjk,jkab->ia",g[o,v,o,o],t2,optimize="optimal")
#    Rov += -0.500000000 * np.einsum("bcja,ijbc->ia",g[v,v,o,v],t2,optimize="optimal")
#    Rov += 1.000000000 * np.einsum("ba,ib->ia",fock[v,v],t1,optimize="optimal")
#return singles_res 

def t3_resid(t3,t2,f,g,o,v):
    #        -1.0000 P(j,k)f(l,k)*t3(a,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lk,abcijl->abcijk', f[o, o], t3)
    triples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)
    
    #        -1.0000 f(l,i)*t3(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('li,abcjkl->abcijk', f[o, o], t3)
    
    #         1.0000 P(a,b)f(a,d)*t3(d,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('ad,dbcijk->abcijk', f[v, v], t3)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)
    
    #         1.0000 f(c,d)*t3(d,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('cd,dabijk->abcijk', f[v, v], t3)
    
    #        -1.0000 P(i,j)*P(a,b)<l,a||j,k>*t2(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lajk,bcil->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)
    
    #        -1.0000 P(a,b)<l,a||i,j>*t2(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('laij,bckl->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)
    
    #        -1.0000 P(i,j)<l,c||j,k>*t2(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)
    
    #        -1.0000 <l,c||i,j>*t2(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g[o, v, o, o], t2)
    
    #        -1.0000 P(j,k)*P(b,c)<a,b||d,k>*t2(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)
    
    #        -1.0000 P(b,c)<a,b||d,i>*t2(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abdi,dcjk->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)
    
    #        -1.0000 P(j,k)<b,c||d,k>*t2(d,a,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcdk,daij->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)
    
    #        -1.0000 <b,c||d,i>*t2(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g[v, v, v, o], t2)
    return triples_res
#    t2=t2.transpose(2,3,0,1)
#    Rooovvv = -0.083333333 * np.einsum("il,jklabc->ijkabc",fock[o,o],t3,optimize="optimal")
#    Rooovvv += -0.250000000 * np.einsum("ijla,klbc->ijkabc",g[o,o,o,v],t2,optimize="optimal")
#    Rooovvv += 0.083333333 * np.einsum("da,ijkbcd->ijkabc",fock[v,v],t3,optimize="optimal")
#    Rooovvv += -0.250000000 * np.einsum("idab,jkcd->ijkabc",g[o,v,v,v],t2,optimize="optimal")
#    return Rooovvv

def offset_t2resid_t2squared(t2,g,o,v):
    #        -0.5000 P(i,j)<l,k||c,d>*t2(c,d,j,k)*t2(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #         0.2500 <l,k||c,d>*t2(c,d,i,j)*t2(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #        -0.5000 <l,k||c,d>*t2(c,a,l,k)*t2(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #         1.0000 P(i,j)<l,k||c,d>*t2(c,a,j,k)*t2(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #        -0.5000 <l,k||c,d>*t2(c,a,i,j)*t2(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])

    return 0.5*doubles_res

def offset_t2resid_t2dag_w_t2(t2,t2_dag,g,o,v):
    #        -0.5000 P(i,j)<l,k||c,d>*t2(c,d,j,k)*t2(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', t2_dag,g[v,v ,o,o],t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #         0.2500 <l,k||c,d>*t2(c,d,i,j)*t2(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk->abij', t2_dag, g[v,v,o,o], t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #        -0.5000 <l,k||c,d>*t2(c,a,l,k)*t2(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', t2_dag, g[v,v,o,o], t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #         1.0000 P(i,j)<l,k||c,d>*t2(c,a,j,k)*t2(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', t2_dag, g[v,v,o,o], t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #        -0.5000 <l,k||c,d>*t2(c,a,i,j)*t2(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', t2_dag, g[v,v,o,o], t2, optimize=['einsum_path', (0, 2), (0, 1)])

    return doubles_res

def offset_t2resid_T1_T3(t1,t3,g,o,v):
    #t1=t1.transpose(1,0)
    #t3=t3.transpose(3,4,5,0,1,2)
    # Building WnT1 into T2 residual
#         1.0000 P(a,b)<k,a||i,j>*t1(b,k)
    contracted_intermediate =  1.000000000000000 * einsum('kaij,bk->abij', g[o, v, o, o], t1)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         1.0000 P(i,j)<a,b||c,j>*t1(c,i)
    contracted_intermediate =  1.000000000000000 * einsum('abcj,ci->abij', g[v, v, v, o], t1)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    ### Add WnT3 into T2 residual eqn:

    #         0.5000 P(i,j)<l,k||c,j>*t3(c,a,b,i,l,k)
    contracted_intermediate =  0.500000000000000 * einsum('lkcj,cabilk->abij', g[o, o, v, o], t3)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.5000 P(a,b)<k,a||c,d>*t3(c,d,b,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('kacd,cdbijk->abij', g[o, v, v, v], t3)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    return doubles_res
