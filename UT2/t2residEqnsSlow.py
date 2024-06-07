from numpy import einsum
import UT2.modify_T2resid_T4Qf1Slow as qf1
import UT2.modify_T2resid_T4Qf2Slow as qf2
import numpy as np


import UT2.antisym_t4resids as antisym
import UT2.kernel as kernel

def residMain(ccd_kernel):
    """
    Drives the determination of the spin-orbital, CCD-based residual equations. This includes calls to subroutines that serve to augment the baseline CCDresidual equations using higher order clusters (ie CCDQf-1, CCDQf-2, etc), if requested by the user.

    :param ccd_kernel: Object of the UltT2CC class.

    :return: Updated Object of the UltT2CC class, equipped with new T amps and residuals
    """
    sliceInfo=ccd_kernel.sliceInfo
    oa=sliceInfo["occ_aa"]
    va=sliceInfo["virt_aa"]
    occaa=oa
    virtaa=va
    nocc=ccd_kernel.nocca
    nvirt=ccd_kernel.nvrta

    t2_aaaa=ccd_kernel.tamps["t2aa"]
    print('t2shape:', t2_aaaa.shape)
    fock=ccd_kernel.ints["oei"]
    tei=ccd_kernel.ints["tei"]

    resid_aaaaBKUP = np.zeros((nvirt,nvirt,nocc,nocc))
    eabij_aa=ccd_kernel.denom["D2aa"]


    # construct the baseline pCCD or CCD residual equations
    if ccd_kernel.cc_type=="pCCD":
        resid_aaaa=ccd_t2residual(t2_aaaa, fock, tei, oa, va)
#        if np.linalg.norm(ccd_kernel.tamps['t2aa']) > 10E-8:
        tmp=np.zeros((nvirt,nvirt,nocc,nocc ))
        for a in range(nvirt):
            for i in range(nocc):
                tmp[a,a,i,i]=resid_aaaa[a,a,i,i]
#                  print('resid shape',resid_aaaa.shape)
#            resid_aaaa=tmp
    else: 
        resid_aaaa=ccd_t2residual(t2_aaaa, fock, tei, oa, va)
        if ccd_kernel.cc_label=="UCC(4)":
            return resid_aaaa
        if ccd_kernel.cc_type=='LCCD':
            lccd_offset=doubles_resLCCDoffset(t2_aaaa,fock,tei,oa,va)
            resid_aaaa-=lccd_offset

    # determine if we need to augment the CCD T2 equations with terms from XCCD(n)
    if ccd_kernel.xccd_resids:
        resid_aaaaBKUP=resid_aaaa
        amp_obj=kernel.BuildBaseAmps(ccd_kernel.contractInfo)
        for order in ccd_kernel.pertOrder:
            amp_obj.buildXCCDbase(t2_aaaa,order)
            resid_aaaa+=amp_obj.t_base[order]
#            checksum(nvirt,nocc,resid_aaaaBKUP)
#            checksum(nvirt,nocc,resid_aaaa)
#            check_norms(nvirt,nocc,resid_aaaa,resid_aaaaBKUP)



# insert debug code at bottom of file here, if added checks are necessary


    resid_aaaa=resid_aaaa+np.reciprocal(eabij_aa)*t2_aaaa
    final_resid={"resT2aa":resid_aaaa}
    ccd_kernel.set_resid(final_resid)
    if ccd_kernel.cc_type=="pCCD":
        resid_aaaa=resid_aaaa*eabij_aa
        t2amp=np.zeros((nvirt,nvirt,nocc,nocc ))
        for a in range(nvirt):
            for i in range(nocc):
                t2amp[a,a,i,i]=resid_aaaa[a,a,i,i]
                #print('resid_aaaa',resid_aaaa[a,a,i,i])
        t2amp={"t2aa":t2amp,"t2bb":t2amp*0.0,"t2ab":t2amp*0.0}
    else:
        t2amp={"t2aa":resid_aaaa*eabij_aa,"t2bb":resid_aaaa*0.0,"t2ab":resid_aaaa*0.0}#

        #print('t2amp',t2amp["t2aa"])
        zww=t2amp["t2aa"].transpose(0,2,1,3)
        #print('reshaped t2:',zww,np.reshape(zww,(4,4)))
        #print(zww[0,:,:,0],'\n',zww[0,:,0,0])
        #print('a,b,i,j, t2amp')
    #print('\n\n Slice one:',t2amp["t2aa"][3,2,0,:])
    #print('\n\n Slice two: ', t2amp["t2aa"][3,2,1,:])
    #print('\n\n Slice 3: ', t2amp["t2aa"][2,3,1,:])
    #print('\n\n Slice 4: ', t2amp["t2aa"][3,2,:,1])
    #print('\n\n Slice 5: ', t2amp["t2aa"][3,2,:,0])
    zww=t2amp["t2aa"]
    #print('max transition:',zww[0,1,0,1],zww[0,1,2,3])
    #checksum(nvirt,nocc,zww)
    #print(zww[0,2,0,2],zww[1,3,1,3])
    ccd_kernel.set_tamps(t2amp)
    return ccd_kernel


def checksum(nvirt,nocc,val):
    total=0.0
    maxval=0.0
    for a in range(nvirt):
        for b in range(nvirt):
            for i in range(nocc):
                for j in range(nocc):
                    total+=(val[a,b,i,j])**2
                    if abs(val[a,b,i,j])>maxval:
                        maxval=abs(val[a,b,i,j])
                        index=(a,b,i,j)
    #print(total,maxval,index)

def check_norms(nvirt,nocc,resid_aaaa,resid_aaaaBKUP):
    norm=0.0
    for a in range(nvirt):
        for b in range(nvirt):
            for i in range(nocc):
                for j in range(nocc):
                    norm+=abs(abs(resid_aaaa[a,b,i,j])-abs(resid_aaaaBKUP[a,b,i,j]))**2
    print('Norm of the differences between XCCD orig and new', norm,np.sqrt(norm))

def pccd_t2resid(t2, f, g, o, v):
    doubles_res=einsum('aaii',g[v,v,o,o])
    doubles_res+=einsum('aa,aaii',f[v,v],t2)
    doubles_res-=einsum('ii,aaii',f[o,o],t2)
 
    doubles_res-=einsum('jjaa,aajj,aaii',g[o,o,v,v],t2,t2)
    doubles_res-=einsum('iibb,bbii,aaii',g[o,o,v,v],t2,t2)

    reduced=2.0*einsum('iaia',g[o,v,o,v]) - einsum('iaai',g[o,v,o,v]) - einsum('iiaa,aaii',g[o,v,o,v],t2)

    doubles_res-=reduced*t2 #einsum('aaii,aaii',reduced,t2)
    

    doubles_res+=einsum('bbaa,bbii',g[v,v,v,v],t2)
    doubles_res+=einsum('iijj,aajj',g[o,o,o,o],t2)
    doubles_res+=einsum("bbjj,aajj,bbii",g[v,o,v,o],t2,t2)
    return doubles_res

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

def doubles_resLCCDoffset(t2,f,g,o,v):
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

    return doubles_res



#### IN CASE DEBUGGING OF RESIDUAL EQNS IS NEEDED, THESE ALGOS SHOULD BE CORRECT

#
#    resid_mod=[ccd_kernel.pert_wvfxn_corr]
#    XCCD_run=5
#    try:
#        XCCD_run=int(ccd_kernel.pert_wvfxn_corr)
#    except:
#        XCCD_run=0
#    base_calc=ccd_kernel.cc_type
#    print('resid modification is: ', resid_mod,"Qf" in [resid_mod],'XCCD_run',XCCD_run)
#    if "Qf" in resid_mod or "Qf*" in resid_mod or XCCD_run>=5 or base_calc == "CCDQf" or base_calc == "CCDQf*":
#        import UT2.pdagq_t4resid as pdagq
#        print('modifying T2 ansate w/ T4 ish')
#        t4_resid=antisym.unsym_residQf1(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt)
#
#        antisym_t4_resid=t4_resid.transpose(4,5,6,7,0,1,2,3)
#        t2_dag=t2_aaaa*ccd_kernel.denom["D2aa"]
#        t2_dag=t2_aaaa.transpose(2,3,0,1)
#        xcc_t2Dag=t2_aaaa.transpose(2,3,0,1)
#        if XCCD_run>=5:
#            import UT2.xccd_resid as xccd_resid
#            print('inside XCCD run')
#            modify_orders=[i for i in range(5,XCCD_run+1)]
#            resid_aaaa += (1.0/8.0)*einsum('klcd,abcdijkl->abij',xcc_t2Dag,antisym_t4_resid)
#            if 6 in modify_orders:
#                t4_t23=antisym.xccd_6(ccd_kernel,tei,t2_aaaa,oa,va,nocc,nvirt)
#                t4_t23=t4_t23.transpose(4,5,6,7,0,1,2,3)
#                resid_aaaa+=(1.0/8.0)*einsum('klcd,abcdijkl->abij',xcc_t2Dag,t4_t23
#)
#
#        else:
#            print('inside modify T2')
#            resid_aaaa += (1.0/8.0)*einsum('klcd,abcdijkl->abij',t2_dag,antisym_t4_resid)
#
#    print('ccd_kernel.cc_type',ccd_kernel.cc_type)
#    if ccd_kernel.cc_type == "CCDQf-1":
#        l2dic=ccd_kernel.get_l2amps()
#        l2=l2dic["l2aa"]
#
#        qf1_aaaa = qf1.residQf1_aaaa(tei, l2, t2_aaaa, occaa, virtaa)
#        resid_aaaa += 0.5 * qf1_aaaa
