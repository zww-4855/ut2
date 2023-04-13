from numpy import einsum
import UT2.modify_T2resid_T4Qf1Slow as qf1
import UT2.modify_T2resid_T4Qf2Slow as qf2
import numpy as np

def residMain(ccd_kernel):
    sliceInfo=ccd_kernel.sliceInfo
    oa=sliceInfo["occ_aa"]
    va=sliceInfo["virt_aa"]
    occaa=oa
    virtaa=va

    t2_aaaa=ccd_kernel.tamps["t2aa"]

    fock=ccd_kernel.ints["oei"]
    tei=ccd_kernel.ints["tei"]

 

#    g_aaaa=tei["g_aaaa"]

    eabij_aa=ccd_kernel.denom["D2aa"]

 #   g={"aaaa":g_aaaa,"bbbb":g_bbbb,"abab":g_abab}
 #   t2={"aaaa":t2_aaaa,"bbbb":t2_bbbb,"abab":t2_abab}
 #   l2dic=ccd_kernel.get_l2amps()
 #   l2={"aaaa":l2dic["l2aa"],"bbbb":l2dic["l2bb"],"abab":l2dic["l2ab"]}
    
    resid_aaaa=ccd_t2residual(t2_aaaa, fock, tei, oa, va)


    if ccd_kernel.cc_type == "CCDQf-1":
        l2dic=ccd_kernel.get_l2amps()
        l2=l2dic["l2aa"]

        qf1_aaaa = qf1.residQf1_aaaa(tei, l2, t2_aaaa, occaa, virtaa)
        resid_aaaa += 0.5 * qf1_aaaa

    elif ccd_kernel.cc_type == "CCDQf-2":
        l2dic=ccd_kernel.get_l2amps()
        l2=l2dic["l2aa"]

        qf1_aaaa = qf1.residQf1_aaaa(tei, l2, t2_aaaa, occaa, virtaa)

        qf2_aaaa = qf2.residQf2_aaaa(tei, l2, t2_aaaa, occaa, virtaa)

        resid_aaaa += 0.5 * qf1_aaaa + (1.0 / 6.0) * qf2_aaaa


    resid_aaaa=resid_aaaa+np.reciprocal(eabij_aa)*t2_aaaa

    final_resid={"resT2aa":resid_aaaa}
    ccd_kernel.set_resid(final_resid)

    t2amp={"t2aa":resid_aaaa*eabij_aa,"t2bb":resid_aaaa*0.0,"t2ab":resid_aaaa*0.0}#
    #t2amp={"t2aa":resid_aaaa*eabij_aa}
    ccd_kernel.set_tamps(t2amp)

    return ccd_kernel


def ccd_t2residual(t2,f,g,o,v):
    #        -1.0000 P(i,j)f(k,j)*t2(a,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('kj,abik->abij', f[o, o], t2)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         1.0000 P(a,b)f(a,c)*t2(c,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('ac,cbij->abij', f[v, v], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         1.0000 <a,b||i,j>
    doubles_res +=  1.000000000000000 * einsum('abij->abij', g[v, v, o, o])
    
    #         0.5000 <l,k||i,j>*t2(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk->abij', g[o, o, o, o], t2)
    
    #         1.0000 P(i,j)*P(a,b)<k,a||c,j>*t2(c,b,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,cbik->abij', g[o, v, v, o], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.5000 <a,b||c,d>*t2(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij->abij', g[v, v, v, v], t2)
    
    #        -0.5000 P(i,j)<l,k||c,d>*t2(c,d,j,k)*t2(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.2500 <l,k||c,d>*t2(c,d,i,j)*t2(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #        -0.5000 <l,k||c,d>*t2(c,a,l,k)*t2(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #         1.0000 P(i,j)<l,k||c,d>*t2(c,a,j,k)*t2(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.5000 <l,k||c,d>*t2(c,a,i,j)*t2(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    return doubles_res
