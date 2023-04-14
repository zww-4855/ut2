from numpy import einsum
import UT2.modify_T2resid_T4Qf1 as qf1
import UT2.modify_T2resid_T4Qf2 as qf2
import numpy as np

def residMain(ccd_kernel):
    """
    Drives the determination of the spin-integrated, CCD-based residual equations. This includes calls to subroutines that serve to augment the baseline CCDresidual equations using higher order clusters (ie CCDQf-1, CCDQf-2, etc), if requested by the user. 
    
    :param ccd_kernel: Object of the UltT2CC class.
    
    :return: Updated Object of the UltT2CC class, equipped with new T amps and residuals
    """
    sliceInfo=ccd_kernel.sliceInfo
    oa=sliceInfo["occ_aa"]
    ob=sliceInfo["occ_bb"]
    va=sliceInfo["virt_aa"]
    vb=sliceInfo["virt_bb"]
    occaa=oa
    virtaa=va

    t2_aaaa=ccd_kernel.tamps["t2aa"]
    t2_bbbb=ccd_kernel.tamps["t2bb"]
    t2_abab=ccd_kernel.tamps["t2ab"]

    fock=ccd_kernel.ints["oei"]
    tei=ccd_kernel.ints["tei"]

    f_aa=fock["faa"]
    f_bb=fock["fbb"]
 

    g_aaaa=tei["g_aaaa"]
    g_bbbb=tei["g_bbbb"]
    g_abab=tei["g_abab"]

    eabij_aa=ccd_kernel.denom["D2aa"]
    eabij_bb=ccd_kernel.denom["D2bb"]
    eabij_ab=ccd_kernel.denom["D2ab"]



    g={"aaaa":g_aaaa,"bbbb":g_bbbb,"abab":g_abab}
    t2={"aaaa":t2_aaaa,"bbbb":t2_bbbb,"abab":t2_abab}
    l2dic=ccd_kernel.get_l2amps()
    l2={"aaaa":l2dic["l2aa"],"bbbb":l2dic["l2bb"],"abab":l2dic["l2ab"]}
    
    resid_aaaa=ccd_t2_aaaa_residual(t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb,None)
    resid_bbbb=ccd_t2_bbbb_residual(t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb,None)
    resid_abab=ccd_t2_abab_residual(t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb, None)


    if ccd_kernel.cc_type == "CCDQf-1":

        qf1_aaaa = qf1.residQf1_aaaa(g, l2, t2, occaa, virtaa)
        qf1_bbbb = qf1.residQf1_bbbb(g, l2, t2, occaa, virtaa)
        qf1_abab = qf1.residQf1_abab(g, l2, t2, occaa, virtaa)
        resid_aaaa += 0.5 * qf1_aaaa
        resid_bbbb += 0.5 * qf1_bbbb
        resid_abab += 0.5 * qf1_abab

    elif ccd_kernel.cc_type == "CCDQf-2":

        qf1_aaaa = qf1.residQf1_aaaa(g, l2, t2, occaa, virtaa)
        qf1_bbbb = qf1.residQf1_bbbb(g, l2, t2, occaa, virtaa)
        qf1_abab = qf1.residQf1_abab(g, l2, t2, occaa, virtaa)

        qf2_aaaa = qf2.residQf2_aaaa(g, l2, t2, occaa, virtaa)
        qf2_bbbb = qf2.residQf2_bbbb(g, l2, t2, occaa, virtaa)
        qf2_abab = qf2.residQf2_abab(g, l2, t2, occaa, virtaa)

        resid_aaaa += 0.5 * qf1_aaaa + (1.0 / 6.0) * qf2_aaaa
        resid_bbbb += 0.5 * qf1_bbbb + (1.0 / 6.0) * qf2_bbbb
        resid_abab += 0.5 * qf1_abab + (1.0 / 6.0) * qf2_abab


    resid_aaaa=resid_aaaa+np.reciprocal(eabij_aa)*t2_aaaa
    resid_bbbb=resid_bbbb+np.reciprocal(eabij_bb)*t2_bbbb
    resid_abab=resid_abab+np.reciprocal(eabij_ab)*t2_abab

    final_resid={"resT2aa":resid_aaaa,"resT2bb":resid_bbbb,"resT2ab":resid_abab}
    ccd_kernel.set_resid(final_resid)

    t2amp={"t2aa":resid_aaaa*eabij_aa,"t2bb":resid_bbbb*eabij_bb,"t2ab":resid_abab*eabij_ab}
    ccd_kernel.set_tamps(t2amp)
    #ccd_kernel.set_tamps("t2aa")=resid_aaaa*eabij_aa
    #ccd_kernel.set_tamps("t2bb")=resid_bbbb*eabij_bb 
    #ccd_kernel.set_tamps("t2ab")=resid_abab*eabij_ab

    return ccd_kernel

def ccd_t2_aaaa_residual(t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb,cc_runtype):
    
    #    < 0 | m* n* f e e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(m,n)f_aa(i,n)*t2_aaaa(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('in,efmi->efmn', f_aa[o, o], t2_aaaa)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)f_aa(e,a)*t2_aaaa(a,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ea,afmn->efmn', f_aa[v, v], t2_aaaa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 <e,f||m,n>_aaaa
    doubles_res +=  1.000000000000000 * einsum('efmn->efmn', g_aaaa[v, v, o, o])
    
    #	  0.5000 <j,i||m,n>_aaaa*t2_aaaa(e,f,j,i)
    doubles_res +=  0.500000000000000 * einsum('jimn,efji->efmn', g_aaaa[o, o, o, o], t2_aaaa)
    
    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>_aaaa*t2_aaaa(a,f,m,i)
    contracted_intermediate =  1.000000000000000 * einsum('iean,afmi->efmn', g_aaaa[o, v, v, o], t2_aaaa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)*P(e,f)<e,i||n,a>_abab*t2_abab(f,a,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('eina,fami->efmn', g_abab[v, o, o, v], t2_abab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.5000 <e,f||a,b>_aaaa*t2_aaaa(a,b,m,n)
    doubles_res +=  0.500000000000000 * einsum('efab,abmn->efmn', g_aaaa[v, v, v, v], t2_aaaa)
    
    #	 -0.5000 P(m,n)<j,i||a,b>_aaaa*t2_aaaa(a,b,n,i)*t2_aaaa(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('jiab,abni,efmj->efmn', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 P(m,n)<j,i||a,b>_abab*t2_abab(a,b,n,i)*t2_aaaa(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('jiab,abni,efmj->efmn', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 P(m,n)<j,i||b,a>_abab*t2_abab(b,a,n,i)*t2_aaaa(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('jiba,bani,efmj->efmn', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.2500 <j,i||a,b>_aaaa*t2_aaaa(a,b,m,n)*t2_aaaa(e,f,j,i)
    doubles_res +=  0.250000000000000 * einsum('jiab,abmn,efji->efmn', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>_aaaa*t2_aaaa(a,e,j,i)*t2_aaaa(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('jiab,aeji,bfmn->efmn', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||b,a>_abab*t2_abab(e,a,j,i)*t2_aaaa(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('jiba,eaji,bfmn->efmn', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <i,j||b,a>_abab*t2_abab(e,a,i,j)*t2_aaaa(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('ijba,eaij,bfmn->efmn', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)<j,i||a,b>_aaaa*t2_aaaa(a,e,n,i)*t2_aaaa(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,aeni,bfmj->efmn', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(m,n)<i,j||a,b>_abab*t2_aaaa(a,e,n,i)*t2_abab(f,b,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('ijab,aeni,fbmj->efmn', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(m,n)<j,i||b,a>_abab*t2_abab(e,a,n,i)*t2_aaaa(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiba,eani,bfmj->efmn', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(m,n)<j,i||a,b>_bbbb*t2_abab(e,a,n,i)*t2_abab(f,b,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,eani,fbmj->efmn', g_bbbb[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>_aaaa*t2_aaaa(a,e,m,n)*t2_aaaa(b,f,j,i)
    doubles_res += -0.500000000000000 * einsum('jiab,aemn,bfji->efmn', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <j,i||a,b>_abab*t2_aaaa(a,e,m,n)*t2_abab(f,b,j,i)
    doubles_res +=  0.500000000000000 * einsum('jiab,aemn,fbji->efmn', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <i,j||a,b>_abab*t2_aaaa(a,e,m,n)*t2_abab(f,b,i,j)
    doubles_res +=  0.500000000000000 * einsum('ijab,aemn,fbij->efmn', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
   

    return doubles_res


def ccd_t2_bbbb_residual(t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb, cc_runtype):
    
    #    < 0 | m* n* f e e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(m,n)f_bb(i,n)*t2_bbbb(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('in,efmi->efmn', f_bb[o, o], t2_bbbb)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)f_bb(e,a)*t2_bbbb(a,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ea,afmn->efmn', f_bb[v, v], t2_bbbb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 <e,f||m,n>_bbbb
    doubles_res +=  1.000000000000000 * einsum('efmn->efmn', g_bbbb[v, v, o, o])
    
    #	  0.5000 <j,i||m,n>_bbbb*t2_bbbb(e,f,j,i)
    doubles_res +=  0.500000000000000 * einsum('jimn,efji->efmn', g_bbbb[o, o, o, o], t2_bbbb)
    
    #	 -1.0000 P(m,n)*P(e,f)<i,e||a,n>_abab*t2_abab(a,f,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('iean,afim->efmn', g_abab[o, v, v, o], t2_abab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>_bbbb*t2_bbbb(a,f,m,i)
    contracted_intermediate =  1.000000000000000 * einsum('iean,afmi->efmn', g_bbbb[o, v, v, o], t2_bbbb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.5000 <e,f||a,b>_bbbb*t2_bbbb(a,b,m,n)
    doubles_res +=  0.500000000000000 * einsum('efab,abmn->efmn', g_bbbb[v, v, v, v], t2_bbbb)
    
    #	 -0.5000 P(m,n)<i,j||a,b>_abab*t2_abab(a,b,i,n)*t2_bbbb(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('ijab,abin,efmj->efmn', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 P(m,n)<i,j||b,a>_abab*t2_abab(b,a,i,n)*t2_bbbb(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('ijba,bain,efmj->efmn', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 P(m,n)<j,i||a,b>_bbbb*t2_bbbb(a,b,n,i)*t2_bbbb(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('jiab,abni,efmj->efmn', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.2500 <j,i||a,b>_bbbb*t2_bbbb(a,b,m,n)*t2_bbbb(e,f,j,i)
    doubles_res +=  0.250000000000000 * einsum('jiab,abmn,efji->efmn', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>_abab*t2_abab(a,e,j,i)*t2_bbbb(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('jiab,aeji,bfmn->efmn', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <i,j||a,b>_abab*t2_abab(a,e,i,j)*t2_bbbb(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('ijab,aeij,bfmn->efmn', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>_bbbb*t2_bbbb(a,e,j,i)*t2_bbbb(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('jiab,aeji,bfmn->efmn', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)<j,i||a,b>_aaaa*t2_abab(a,e,i,n)*t2_abab(b,f,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,aein,bfjm->efmn', g_aaaa[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(m,n)<i,j||a,b>_abab*t2_abab(a,e,i,n)*t2_bbbb(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('ijab,aein,bfmj->efmn', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(m,n)<j,i||b,a>_abab*t2_bbbb(a,e,n,i)*t2_abab(b,f,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('jiba,aeni,bfjm->efmn', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(m,n)<j,i||a,b>_bbbb*t2_bbbb(a,e,n,i)*t2_bbbb(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,aeni,bfmj->efmn', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.5000 <j,i||b,a>_abab*t2_bbbb(a,e,m,n)*t2_abab(b,f,j,i)
    doubles_res +=  0.500000000000000 * einsum('jiba,aemn,bfji->efmn', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <i,j||b,a>_abab*t2_bbbb(a,e,m,n)*t2_abab(b,f,i,j)
    doubles_res +=  0.500000000000000 * einsum('ijba,aemn,bfij->efmn', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>_bbbb*t2_bbbb(a,e,m,n)*t2_bbbb(b,f,j,i)
    doubles_res += -0.500000000000000 * einsum('jiab,aemn,bfji->efmn', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    return doubles_res


def ccd_t2_abab_residual(t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb, cc_runtype):
    
    #    < 0 | m* n* f e e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_bb(i,n)*t2_abab(e,f,m,i)
    doubles_res = -1.000000000000000 * einsum('in,efmi->efmn', f_bb[o, o], t2_abab)
    
    #	 -1.0000 f_aa(i,m)*t2_abab(e,f,i,n)
    doubles_res += -1.000000000000000 * einsum('im,efin->efmn', f_aa[o, o], t2_abab)
    
    #	  1.0000 f_aa(e,a)*t2_abab(a,f,m,n)
    doubles_res +=  1.000000000000000 * einsum('ea,afmn->efmn', f_aa[v, v], t2_abab)
    
    #	  1.0000 f_bb(f,a)*t2_abab(e,a,m,n)
    doubles_res +=  1.000000000000000 * einsum('fa,eamn->efmn', f_bb[v, v], t2_abab)
    
    #	  1.0000 <e,f||m,n>_abab
    doubles_res +=  1.000000000000000 * einsum('efmn->efmn', g_abab[v, v, o, o])
    
    #	  0.5000 <j,i||m,n>_abab*t2_abab(e,f,j,i)
    doubles_res +=  0.500000000000000 * einsum('jimn,efji->efmn', g_abab[o, o, o, o], t2_abab)
    
    #	  0.5000 <i,j||m,n>_abab*t2_abab(e,f,i,j)
    doubles_res +=  0.500000000000000 * einsum('ijmn,efij->efmn', g_abab[o, o, o, o], t2_abab)
    
    #	 -1.0000 <e,i||a,n>_abab*t2_abab(a,f,m,i)
    doubles_res += -1.000000000000000 * einsum('eian,afmi->efmn', g_abab[v, o, v, o], t2_abab)
    
    #	 -1.0000 <i,f||a,n>_abab*t2_aaaa(a,e,m,i)
    doubles_res += -1.000000000000000 * einsum('ifan,aemi->efmn', g_abab[o, v, v, o], t2_aaaa)
    
    #	  1.0000 <i,f||a,n>_bbbb*t2_abab(e,a,m,i)
    doubles_res +=  1.000000000000000 * einsum('ifan,eami->efmn', g_bbbb[o, v, v, o], t2_abab)
    
    #	  1.0000 <i,e||a,m>_aaaa*t2_abab(a,f,i,n)
    doubles_res +=  1.000000000000000 * einsum('ieam,afin->efmn', g_aaaa[o, v, v, o], t2_abab)
    
    #	 -1.0000 <e,i||m,a>_abab*t2_bbbb(a,f,n,i)
    doubles_res += -1.000000000000000 * einsum('eima,afni->efmn', g_abab[v, o, o, v], t2_bbbb)
    
    #	 -1.0000 <i,f||m,a>_abab*t2_abab(e,a,i,n)
    doubles_res += -1.000000000000000 * einsum('ifma,eain->efmn', g_abab[o, v, o, v], t2_abab)
    
    #	  0.5000 <e,f||a,b>_abab*t2_abab(a,b,m,n)
    doubles_res +=  0.500000000000000 * einsum('efab,abmn->efmn', g_abab[v, v, v, v], t2_abab)
    
    #	  0.5000 <e,f||b,a>_abab*t2_abab(b,a,m,n)
    doubles_res +=  0.500000000000000 * einsum('efba,bamn->efmn', g_abab[v, v, v, v], t2_abab)
    
    #	 -0.5000 <i,j||a,b>_abab*t2_abab(a,b,i,n)*t2_abab(e,f,m,j)
    doubles_res += -0.500000000000000 * einsum('ijab,abin,efmj->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <i,j||b,a>_abab*t2_abab(b,a,i,n)*t2_abab(e,f,m,j)
    doubles_res += -0.500000000000000 * einsum('ijba,bain,efmj->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>_bbbb*t2_bbbb(a,b,n,i)*t2_abab(e,f,m,j)
    doubles_res += -0.500000000000000 * einsum('jiab,abni,efmj->efmn', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>_aaaa*t2_aaaa(a,b,m,i)*t2_abab(e,f,j,n)
    doubles_res += -0.500000000000000 * einsum('jiab,abmi,efjn->efmn', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>_abab*t2_abab(a,b,m,i)*t2_abab(e,f,j,n)
    doubles_res += -0.500000000000000 * einsum('jiab,abmi,efjn->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||b,a>_abab*t2_abab(b,a,m,i)*t2_abab(e,f,j,n)
    doubles_res += -0.500000000000000 * einsum('jiba,bami,efjn->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <j,i||a,b>_abab*t2_abab(a,b,m,n)*t2_abab(e,f,j,i)
    doubles_res +=  0.250000000000000 * einsum('jiab,abmn,efji->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <i,j||a,b>_abab*t2_abab(a,b,m,n)*t2_abab(e,f,i,j)
    doubles_res +=  0.250000000000000 * einsum('ijab,abmn,efij->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <j,i||b,a>_abab*t2_abab(b,a,m,n)*t2_abab(e,f,j,i)
    doubles_res +=  0.250000000000000 * einsum('jiba,bamn,efji->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <i,j||b,a>_abab*t2_abab(b,a,m,n)*t2_abab(e,f,i,j)
    doubles_res +=  0.250000000000000 * einsum('ijba,bamn,efij->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>_aaaa*t2_aaaa(a,e,j,i)*t2_abab(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('jiab,aeji,bfmn->efmn', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||b,a>_abab*t2_abab(e,a,j,i)*t2_abab(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('jiba,eaji,bfmn->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <i,j||b,a>_abab*t2_abab(e,a,i,j)*t2_abab(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('ijba,eaij,bfmn->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <i,j||b,a>_abab*t2_abab(e,a,i,n)*t2_abab(b,f,m,j)
    doubles_res +=  1.000000000000000 * einsum('ijba,eain,bfmj->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,i||a,b>_aaaa*t2_aaaa(a,e,m,i)*t2_abab(b,f,j,n)
    doubles_res +=  1.000000000000000 * einsum('jiab,aemi,bfjn->efmn', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <i,j||a,b>_abab*t2_aaaa(a,e,m,i)*t2_bbbb(b,f,n,j)
    doubles_res +=  1.000000000000000 * einsum('ijab,aemi,bfnj->efmn', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,i||b,a>_abab*t2_abab(e,a,m,i)*t2_abab(b,f,j,n)
    doubles_res +=  1.000000000000000 * einsum('jiba,eami,bfjn->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,i||a,b>_bbbb*t2_abab(e,a,m,i)*t2_bbbb(b,f,n,j)
    doubles_res +=  1.000000000000000 * einsum('jiab,eami,bfnj->efmn', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||b,a>_abab*t2_abab(e,a,m,n)*t2_abab(b,f,j,i)
    doubles_res += -0.500000000000000 * einsum('jiba,eamn,bfji->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <i,j||b,a>_abab*t2_abab(e,a,m,n)*t2_abab(b,f,i,j)
    doubles_res += -0.500000000000000 * einsum('ijba,eamn,bfij->efmn', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <j,i||a,b>_bbbb*t2_abab(e,a,m,n)*t2_bbbb(b,f,j,i)
    doubles_res +=  0.500000000000000 * einsum('jiab,eamn,bfji->efmn', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    return doubles_res

