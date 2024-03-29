import numpy as np
import pickle

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


def unsym_residQf1(ccd_kernel,g,t2_aa,o,v,nocc,nvir,g2=None):

    Roooovvvv = np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
    t=t2_aa.transpose(2,3,0,1)
    v_oo=g[o,o,o,o]
    v_vo=g[o,v,o,v]
    v_vv=g[v,v,v,v]
    v_m2=g[v,v,o,o]
    # Do base-line (Qf) style of constructing T4
    Roooovvvv += -0.062500000 * np.einsum("imab,jncd,klmn->ijklabcd",t,t,v_oo,optimize="optimal")
    Roooovvvv += -0.250000000 * np.einsum("imab,jkce,lemd->ijklabcd",t,t,v_vo,optimize="optimal")
    Roooovvvv += -0.062500000 * np.einsum("ijae,klbf,efcd->ijklabcd",t,t,v_vv,optimize="optimal")

    Roooovvvv_t2_dump=antisym_t4_residual(Roooovvvv,nocc,nvir)# dump third-order T4
    with open('roooovvvv_t4_third.pickle', 'wb') as handle:
        pickle.dump(Roooovvvv_t2_dump, handle) 

    if ccd_kernel.cc_type == "CCD(Qf*)" or ccd_kernel.cc_type == "UT2-CCD(6)" or ccd_kernel.cc_type == "CCDQf*":# tack on the W-2 T2^3 contrib to T4
        Roooovvvv_t23 = -0.031250000 * np.einsum("imab,jncd,klef,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")
        Roooovvvv_t23 += 0.250000000 * np.einsum("imab,jkce,lndf,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")
        Roooovvvv_t23 += -0.031250000 * np.einsum("mnab,ijce,kldf,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")
        Roooovvvv_t23_dump = antisym_t4_residual(Roooovvvv_t23,nocc,nvir)  # dump fourth-order T4
        with open('roooovvvv_t4_fourth.pickle', 'wb') as handle:
            pickle.dump(Roooovvvv_t23_dump, handle)   
        Roooovvvv +=Roooovvvv_t23

#    Roooovvvv=antisym_t4_residual(Roooovvvv,nocc,nvir)
#    Roovv=np.einsum('klcd,abcdijkl->abij',t2_aa.transpose(2,3,0,1),Roooovvvv.transpose(4,5,6,7,0,1,2,3))
#    Roovv_anti = np.einsum("ijab->ijab",Roovv)
#    Roovv_anti -= np.einsum("ijab->jiab",Roovv)
#    Roovv_anti -= np.einsum("ijab->ijba",Roovv)
#    Roovv_anti += np.einsum("ijab->jiba",Roovv) 
#    t2_FO_dag=v_vo*ccd_kernel.denom["D2aa"]
#    t2_FO_dagger=t2_FO_dag.transpose(2,3,0,1)
#
#    finalTest=np.einsum('ijab,abij',t2_FO_dagger,Roovv_anti)
#    print('tested Qf energy:',finalTest,finalTest/8.0)
    Roooovvvv=antisym_t4_residual(Roooovvvv,nocc,nvir)

    calculate_t4intermediateE(t,v_vv,g[o,o,v,v],g,nocc,nvir)
    fifth_orderINTERMEDmain(t,v_vv,v_m2,nocc,nvir)
    write_fifthO_loop(t,g,nocc,nvir)
    return Roooovvvv


def write_fifthO_loop(t2,g,nocc,nvirt):
    AA=np.zeros((nocc,nocc,nvirt,nvirt,nvirt,nvirt))
    for i in range(nocc):
        for j in range(nocc):
            for f in range(nvirt):
                for a in range(nvirt): 
                    for b in range(nvirt):
                        for c in range(nvirt):
                            AA[i,j,f,a,b,c]=0.0
                            for e in range(nvirt):
                                AA[i,j,f,a,b,c]+=(t2[i,j,a,e]*g[e,f,b,c]-t2[i,j,b,e]*g[e,f,a,c]-t2[i,j,c,e]*g[e,f,b,a])


    SS=np.zeros((nvirt,nvirt))
    
    for f in range(nvirt):
        for c in range(nvirt):
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvirt):
                        for b in range(nvirt):
                            SS[f,c]+=0.25*(AA[i,j,a,b,c,f]*g[i,j,a,b])


    BB=np.zeros((nocc,nocc,nvirt,nvirt))
    for k in range(nocc):
        for l in range(nocc):
            for c in range(nvirt):
                for d in range(nvirt):
                    for f in range(nvirt):
                        BB+=(SS[f,c]*t2[k,l,f,d]-SS[f,d]*t2[k,l,f,c])
#                    BB[k,l,c,d]-=BB[k,l,d,c]

#    BB=BB-np.einsum('klcd->kldc',BB)

    energy=0.0
    for k in range(nocc):
        for l in range(nocc):
            for c in range(nvirt):
                for d in range(nvirt):
                    energy+=(1.0/4.0)*(BB[k,l,c,d]*g[c,d,k,l])


    print('energy is: ',energy)
  

    
                 


def antisymmetrize_residual_2_2(Roovv, nocc, nvir):
    # antisymmetrize the residual
    Roovv_anti = np.zeros((nocc, nocc, nvir, nvir))
    Roovv_anti += np.einsum("ijab->ijab", Roovv)
    Roovv_anti -= np.einsum("ijab->jiab", Roovv)
    Roovv_anti -= np.einsum("ijab->ijba", Roovv)
    Roovv_anti += np.einsum("ijab->jiba", Roovv)
    return Roovv_anti

def antisym_A(A,nocc,nvirt):
    A_vvv_voo =  1 * np.einsum("ijabcf->ijabcf", A)
    A_vvv_voo +=  -1 * np.einsum("ijabcf->ijacbf", A)
    A_vvv_voo +=  -1 * np.einsum("ijabcf->ijbacf", A)
    A_vvv_voo +=  1 * np.einsum("ijabcf->ijbcaf", A)
    A_vvv_voo +=  -1 * np.einsum("ijabcf->ijcbaf", A)
    A_vvv_voo +=  1 * np.einsum("ijabcf->ijcabf", A)
    A_vvv_voo +=  -1 * np.einsum("ijabcf->jiabcf", A)
    A_vvv_voo +=  1 * np.einsum("ijabcf->jiacbf", A)
    A_vvv_voo +=  1 * np.einsum("ijabcf->jibacf", A)
    A_vvv_voo +=  -1 * np.einsum("ijabcf->jibcaf", A)
    A_vvv_voo +=  1 * np.einsum("ijabcf->jicbaf", A)
    A_vvv_voo +=  -1 * np.einsum("ijabcf->jicabf", A)
    return A_vvv_voo

def antisym_T3(Roooovvvv):
    Roooovvvv_anti =  1 * np.einsum("ijabcd->ijabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijdbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijdbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijdcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijdcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->ijdacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->ijdabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jiabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jiabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jiacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jiacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jiadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jiadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jibacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jibadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jibcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jibcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jibdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jibdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jicbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jicbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jicabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jicadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jicdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jicdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jidbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jidbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jidcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jidcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijabcd->jidacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijabcd->jidabc", Roooovvvv)
    return Roooovvvv_anti

def fifth_orderINTERMEDmain(t2,v_vv,v_vo,nocc,nvirt):
    A_abc_ijf=-0.250000000 * np.einsum("ijae,debc->ijdabc",t2,v_vv,optimize="optimal")
    antiAA=A_abc_ijf #antisym_A(A_abc_ijf,nocc,nvirt)

    S_c_f = -0.250000000 * np.einsum("ijbacd,cdij->ba",antiAA,v_vo,optimize="optimal")
    
    B_cd_kl= -0.500000000 * np.einsum("ca,ijbc->ijab",S_c_f,t2,optimize="optimal")

    netT2=antisymmetrize_residual_2_2(B_cd_kl,nocc,nvirt)

    energy=-0.250000000 * np.einsum("ijab,abij->",netT2,v_vo,optimize="optimal")
 
    print("constructed INTERMEDIATE THRU WICKED: ",energy)

def calculate_t4intermediateE(t,v_vv,v_m2,g,nocc,nvirt):
    Roooovvvv = -0.062500000 * np.einsum("ijae,klbf,efcd->ijklabcd",t,t,v_vv,optimize="optimal")


    Roooovvvv=antisym_t4_residual(Roooovvvv,nocc,nvirt)

    energy=0.0
    for i in range(nocc):
        for j in range(nocc):
            for k in range(nocc):  
                for l in range(nocc):
                    for d in range(nvirt):
                        for c in range(nvirt):
                            for b in range(nvirt):
                                for a in range(nvirt):
                                    energy+=Roooovvvv[i,j,k,l,a,b,c,d]*g[i,j,a,b]*g[k,l,c,d]

    print('newest energyT4 loop:', energy,energy/32.0)


    t4=Roooovvvv.transpose(4,5,6,7,0,1,2,3)
    energy=np.einsum('klcd,ijab,abcdijkl->',v_m2,v_m2,t4[:,:,:,:,:,:,:,:])
    print('t4INTERMEDIATE full E:',energy)
    print('t4INTERMEDIATE rev E:',(1/32.0)*energy)

    e=0.0
    for a in range(nvirt):
        for b in range(nvirt):
            for c in range(nvirt):
                for d in range(nvirt):
                    for i in range(nocc):
                        for j in range(nocc):
                            for k in range(nocc):
                                for l in range(nocc):
                                    e+=t4[a,b,c,d,i,j,k,l]*g[i,j,a,b]*g[k,l,c,d]

    print('e is: ',e,e/32.0)
def xccd_6(ccd_kernel,g,t2_aa,o,v,nocc,nvir,g2=None):
    Roooovvvv = np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
    t=t2_aa.transpose(2,3,0,1)
    v_oo=g[o,o,o,o]
    v_vo=g[o,v,o,v]
    v_vv=g[v,v,v,v]
    v_m2=g[v,v,o,o]

    Roooovvvv_t23 = -0.031250000 * np.einsum("imab,jncd,klef,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")
    Roooovvvv_t23 += 0.250000000 * np.einsum("imab,jkce,lndf,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")
    Roooovvvv_t23 += -0.031250000 * np.einsum("mnab,ijce,kldf,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")

    Roooovvvv=antisym_t4_residual(Roooovvvv,nocc,nvir)
    return Roooovvvv

def xccd_8(ccd_kernel,g,gD_dag,t2,t2_dag,o,v,nocc,nvir,doUT2=False):
    if doUT2:
        cap=gD_dag
    else:
        cap=t2_dag

    midA=midB=np.zeros((nvir,nvir,nvir,nvir,nocc,nocc,nocc,nocc))
    midSum=lambda b,c,e,g,j,k,n,o,a,h,i,l: t[a,h,i,j]*t[b,c,l,k]*cap[i,n,a,e]*cap[l,o,h,g]
    for b in range(nvir):
        for c in range(nvir):
            for e in range(nvir):
                for g in range(nvir):
                    for j in range(nocc):
                        for k in range(nocc):
                            for n in range(nocc):
                                for o in range(nocc):
                                    
                                    for a in range(nvir):
                                        for h in range(nvir):
                                            for i in range(nocc):
                                                for l in range(nocc):
                                                    midA[b,c,e,g,j,k,n,o]=midSum(b,c,e,g,j,k,n,o,a,h,i,l)
                                                


    aftSum=lambda g,e,b,c,n,o,j,k,d,f,m,p: t[d,g,m,n]*t[e,f,o,p]*cap[m,p,d,f]*cap[j,k,b,c]
    for g in range(nvir):
        for e in range(nvir):
            for b in range(nvir):
                for c in range(nvir):
                    for n in range(nocc):
                        for o in range(nocc):
                            for j in range(nocc):
                                for k in range(nocc):

                                    for d in range(nvir):
                                        for f in range(nvir):
                                            for m in range(nocc):
                                                for p in range(nocc):
                                                    midB[g,e,b,c,n,o,j,k]=aftSum(d,f,m,p)

    energy=0.0
    for b in range(nvir):
        for c in range(nvir):
            for e in range(nvir):
                for g in range(nvir):
                    for j in range(nocc):
                        for k in range(nocc):
                            for n in range(nocc):
                                for o in range(nocc):
                                    energy+=midA[b,c,e,g,j,k,n,o]*midB[g,e,b,c,n,o,j,k]

    energy=(1.0/384.0)*energy
    return energy


def xccd8_resid(ccd_kernel,g,gD_dag,t2,t2_dag,o,v,nocc,nvir,doUT2=False):
    if doUT2:
        cap=gD_dag
    else:
        cap=t2_dag

    midA=np.zeros((nvir,nvir,nvir,nvir,nocc,nocc,nocc,nocc))
    midB=np.zeros((nvir,nvir,nocc,nocc))
    midSum=lambda b,c,e,g,j,k,n,o,a,h,i,l: t[a,h,i,j]*t[b,c,l,k]*cap[i,n,a,e]*cap[l,o,h,g]
    for b in range(nvir):
        for c in range(nvir):
            for e in range(nvir):
                for g in range(nvir):
                    for j in range(nocc):
                        for k in range(nocc):
                            for n in range(nocc):
                                for o in range(nocc):

                                    for a in range(nvir):
                                        for h in range(nvir):
                                            for i in range(nocc):
                                                for l in range(nocc):
                                                    midA[b,c,e,g,j,k,n,o]=midSum(b,c,e,g,j,k,n,o,a,h,i,l)


    aftSum=lambda g,e,n,o,d,f,m,p: t[d,g,m,n]*t[e,f,o,p]*cap[m,p,d,f]
    for g in range(nvir):
        for e in range(nvir):
            for n in range(nocc):
                for o in range(nocc):
                    for d in range(nvir):
                        for f in range(nvir):
                            for m in range(nocc):
                                for p in range(nocc):
                                    midB[g,e,n,o]=aftSum(d,f,m,p)

    t2_resid=np.zeros((nvir,nvir,nocc,nocc))
    for b in range(nvir):
        for c in range(nvir):
            for e in range(nvir):
                for g in range(nvir):
                    for j in range(nocc):
                        for k in range(nocc):
                            for n in range(nocc):
                                for o in range(nocc):
                                    t2_resid[b,c,j,k]=midA[b,c,e,g,j,k,n,o]*midB[g,e,n,o]

    t2_resid=(1.0/96.0)*t2_resid
    return t2_resid


def unsym_resid7(ccd_kernel,g,t2_aa,o,v,nocc,nvir,g2=None):
    #load up D4T4^(3) residual and convert to third-order T4
    t4=np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
    with open('roooovvvv_t4_third.pickle', 'rb') as handle:
        t4=pickle.load(handle)


    t4=antisym_t4_residual(t4,nocc,nvir) # antisymmeterize T4 prior to contraction
    t4=t4*ccd_kernel.denom["D4aa"]

    t2=t2_aa.transpose(2,3,0,1)
    v_oo=g[o,o,o,o]
    v_vo=g[o,v,o,v]
    v_vv=g[v,v,v,v]
    v_m2=g[v,v,o,o]

    Roooovvvv = np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
    Roooovvvv += -0.020833333 * np.einsum("imab,jklncdef,efmn->ijklabcd",t2,t4,v_m2,optimize="optimal")
    Roooovvvv += 0.002604167 * np.einsum("mnab,ijklcdef,efmn->ijklabcd",t2,t4,v_m2,optimize="optimal")
    Roooovvvv += -0.020833333 * np.einsum("ijae,klmnbcdf,efmn->ijklabcd",t2,t4,v_m2,optimize="optimal")
    Roooovvvv += 0.027777778 * np.einsum("imae,jklnbcdf,efmn->ijklabcd",t2,t4,v_m2,optimize="optimal")
    Roooovvvv += -0.003472222 * np.einsum("mnae,ijklbcdf,efmn->ijklabcd",t2,t4,v_m2,optimize="optimal")
    Roooovvvv += 0.002604167 * np.einsum("ijef,klmnabcd,efmn->ijklabcd",t2,t4,v_m2,optimize="optimal")
    Roooovvvv += -0.003472222 * np.einsum("imef,jklnabcd,efmn->ijklabcd",t2,t4,v_m2,optimize="optimal")


    Roooovvvv=antisym_t4_residual(Roooovvvv,nocc,nvir) # antisymmeterize resultant T4 prior to E contractin

    return Roooovvvv


def unsym_resid8(ccd_kernel,g,t2_aa,o,v,nocc,nvir,g2=None):
    with open('roooovvvv_t4_third.pickle', 'rb') as handle:
        t4=pickle.load(handle)


    t4=antisym_t4_residual(t4,nocc,nvir)
    t4=t4*ccd_kernel.denom["D4aa"]

    t2=t2_aa.transpose(2,3,0,1)
    v_oo=g[o,o,o,o]
    v_vo=g[o,v,o,v]
    v_vv=g[v,v,v,v]
    v_m2=g[v,v,o,o]

    Roooooovvvvvv = np.zeros((nocc,nocc,nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir,nvir,nvir))
    # R6|(WnT2^4/4!)c|0>
    Roooooovvvvvv += 0.015625000 * np.einsum("iabc,jdef,klgh,mnop,hpad->ijklmnbcefgo",t2,t2,t2,t2,v_m2,optimize="optimal")

    # R6|Wn0 T4 T2)c|0>
    Roooooovvvvvv += -0.001736111 * np.einsum("iabc,jkldefgh,mnad->ijklmnbcefgh",t2,t4,v_oo,optimize="optimal")
    Roooooovvvvvv += -0.003472222 * np.einsum("iabc,jklmdefg,ngah->ijklmnbcdefh",t2,t4,v_vo,optimize="optimal")
    Roooooovvvvvv += -0.003472222 * np.einsum("ijab,klmcdefg,nbch->ijklmnadefgh",t2,t4,v_vo,optimize="optimal")
    Roooooovvvvvv += -0.001736111 * np.einsum("ijab,klmncdef,bfgh->ijklmnacdegh",t2,t4,v_vv,optimize="optimal")


    import UT2.t6antisym as unsym_t6
    t6=unsym_t6.antisym_t6_residual(Roooooovvvvvv,nocc,nvirt)
    return t6

def unsym_resid9(ccd_kernel,g,t2_aa,o,v,nocc,nvir,g2=None):
    with open('roooovvvv_t4_third.pickle', 'rb') as handle:
        t4_3O=pickle.load(handle)

    with open('roooovvvv_t4_fourth.pickle', 'rb') as handle:
        t4_4O=pickle.load(handle)


    t4_3O=antisym_t4_residual(t4,nocc,nvir)
    t4_4O=antisym_t4_residual(t4,nocc,nvir)
    t4_3O=t4_3O*ccd_kernel.denom["D4aa"]
    t4_4O=t4_4O*ccd_kernel.denom["D4aa"]

    t2=t2_aa.transpose(2,3,0,1)
    v_oo=g[o,o,o,o]
    v_vo=g[o,v,o,v]
    v_vv=g[v,v,v,v]
    v_m2=g[v,v,o,o]

    # R6|Wn0 T4 T2)c|0>
    Roooooovvvvvv += -0.001736111 * np.einsum("iabc,jkldefgh,mnad->ijklmnbcefgh",t2,t4_4O,v_oo,optimize="optimal")
    Roooooovvvvvv += -0.003472222 * np.einsum("iabc,jklmdefg,ngah->ijklmnbcdefh",t2,t4_4O,v_vo,optimize="optimal")
    Roooooovvvvvv += -0.003472222 * np.einsum("ijab,klmcdefg,nbch->ijklmnadefgh",t2,t4_4O,v_vo,optimize="optimal")
    Roooooovvvvvv += -0.001736111 * np.einsum("ijab,klmncdef,bfgh->ijklmnacdegh",t2,t4_4O,v_vv,optimize="optimal")

    # R6|Wn-2 T4 T2^2/2)c|0>
    
    Roooooovvvvvv += -0.001302083 * np.einsum("iabc,jdef,klmnghop,opad->ijklmnbcefgh",t2,t2,t4_3O,v_m2,optimize="optimal")
    Roooooovvvvvv += 0.006944444 * np.einsum("iabc,jkde,lmnfghop,epaf->ijklmnbcdgho",t2,t2,t4_3O,v_m2,optimize="optimal")
    Roooooovvvvvv += 0.003472222 * np.einsum("iabc,jdef,klmnghop,fpad->ijklmnbcegho",t2,t2,t4_3O,v_m2,optimize="optimal")
    Roooooovvvvvv += -0.000868056 * np.einsum("iabc,jkde,lmnfghop,deaf->ijklmnbcghop",t2,t2,t4_3O,v_m2,optimize="optimal")
    Roooooovvvvvv += -0.000868056 * np.einsum("abcd,ijef,klmnghop,fpab->ijklmncdegho",t2,t2,t4_3O,v_m2,optimize="optimal")

    Roooooovvvvvv += -0.001302083 * np.einsum("ijab,klcd,mnefghop,bdef->ijklmnacghop",t2,t2,t4_3O,v_m2,optimize="optimal")
    Roooooovvvvvv += 0.003472222 * np.einsum("ijab,kcde,lmnfghop,becf->ijklmnadghop",t2,t2,t4_3O,v_m2,optimize="optimal")



    import UT2.t6antisym as unsym_t6
    t6=unsym_t6.antisym_t6_residual(Roooooovvvvvv,nocc,nvirt)
    return t6

def antisym_t4_residual(Roooovvvv,nocc,nvirt):
    Roooovvvv_anti = np.zeros((nocc,nocc,nocc,nocc,nvirt,nvirt,nvirt,nvirt))

    Roooovvvv_anti =  1 * np.einsum("ijklabcd->ijklabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijklabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijklacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijklacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijkladcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijkladbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijklbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijklbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijklbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijklbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijklbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijklbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijklcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijklcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijklcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijklcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijklcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijklcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijkldbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijkldbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijkldcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijkldcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijkldacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijkldabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkbacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkbadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkbcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkbcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkbdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkbdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkcbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkcbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkcabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkcadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkcdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkcdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkdbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkdbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkdcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkdcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ijlkdacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ijlkdabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjlabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjlabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjlacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjlacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjladcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjladbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjlbacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjlbadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjlbcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjlbcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjlbdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjlbdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjlcbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjlcbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjlcabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjlcadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjlcdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjlcdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjldbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjldbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjldcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjldcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikjldacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikjldabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljdbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljdbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljdcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljdcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ikljdacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ikljdabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjbacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjbadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjbcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjbcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjbdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjbdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjcbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjcbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjcabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjcadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjcdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjcdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjdbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjdbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjdcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjdcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ilkjdacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ilkjdabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkdbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkdbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkdcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkdcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->iljkdacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->iljkdabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jiklabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jiklabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jiklacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jiklacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jikladcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jikladbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jiklbacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jiklbadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jiklbcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jiklbcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jiklbdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jiklbdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jiklcbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jiklcbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jiklcabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jiklcadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jiklcdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jiklcdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jikldbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jikldbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jikldcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jikldcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jikldacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jikldabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkdbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkdbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkdcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkdcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jilkdacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jilkdabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkilabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkilabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkilacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkilacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkiladcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkiladbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkilbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkilbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkilbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkilbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkilbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkilbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkilcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkilcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkilcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkilcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkilcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkilcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkildbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkildbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkildcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkildcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkildacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkildabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkliabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkliabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkliacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkliacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jkliadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jkliadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jklibacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jklibadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jklibcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jklibcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jklibdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jklibdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jklicbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jklicbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jklicabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jklicadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jklicdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jklicdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jklidbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jklidbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jklidcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jklidcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jklidacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jklidabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkiabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkiabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkiacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkiacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkiadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkiadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkibacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkibadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkibcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkibcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkibdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkibdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkicbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkicbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkicabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkicadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkicdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkicdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkidbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkidbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkidcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkidcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlkidacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlkidabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikbacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikbadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikbcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikbcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikbdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikbdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikcbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikcbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikcabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikcadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikcdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikcdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikdbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikdbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikdcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikdcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->jlikdacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->jlikdabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjilabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjilabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjilacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjilacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjiladcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjiladbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjilbacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjilbadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjilbcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjilbcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjilbdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjilbdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjilcbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjilcbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjilcabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjilcadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjilcdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjilcdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjildbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjildbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjildcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjildcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjildacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjildabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjliabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjliabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjliacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjliacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjliadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjliadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjlibacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjlibadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjlibcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjlibcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjlibdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjlibdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjlicbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjlicbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjlicabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjlicadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjlicdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjlicdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjlidbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjlidbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjlidcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjlidcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kjlidacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kjlidabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijlabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijlabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijlacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijlacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijladcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijladbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijlbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijlbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijlbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijlbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijlbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijlbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijlcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijlcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijlcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijlcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijlcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijlcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijldbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijldbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijldcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijldcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kijldacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kijldabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljbacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljbadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljbcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljbcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljbdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljbdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljcbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljcbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljcabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljcadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljcdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljcdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljdbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljdbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljdcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljdcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kiljdacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kiljdabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijdbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijdbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijdcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijdcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->klijdacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->klijdabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljiabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljiabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljiacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljiacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljiadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljiadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljibacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljibadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljibcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljibcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljibdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljibdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljicbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljicbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljicabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljicadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljicdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljicdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljidbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljidbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljidcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljidcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->kljidacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->kljidabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkiabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkiabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkiacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkiacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkiadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkiadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkibacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkibadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkibcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkibcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkibdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkibdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkicbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkicbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkicabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkicadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkicdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkicdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkidbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkidbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkidcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkidcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljkidacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljkidabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikdbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikdbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikdcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikdcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->ljikdacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->ljikdabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjiabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjiabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjiacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjiacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjiadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjiadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjibacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjibadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjibcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjibcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjibdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjibdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjicbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjicbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjicabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjicadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjicdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjicdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjidbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjidbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjidcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjidcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkjidacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkjidabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijbacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijbadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijbcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijbcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijbdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijbdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijcbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijcbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijcabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijcadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijcdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijcdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijdbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijdbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijdcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijdcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lkijdacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lkijdabc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjabcd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjabdc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjacbd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjacdb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjadcb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjadbc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjbacd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjbadc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjbcad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjbcda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjbdca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjbdac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjcbad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjcbda", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjcabd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjcadb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjcdab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjcdba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjdbca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjdbac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjdcba", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjdcab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->likjdacb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->likjdabc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkabcd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkabdc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkacbd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkacdb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkadcb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkadbc", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkbacd", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkbadc", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkbcad", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkbcda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkbdca", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkbdac", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkcbad", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkcbda", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkcabd", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkcadb", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkcdab", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkcdba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkdbca", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkdbac", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkdcba", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkdcab", Roooovvvv)
    Roooovvvv_anti +=  -1 * np.einsum("ijklabcd->lijkdacb", Roooovvvv)
    Roooovvvv_anti +=  1 * np.einsum("ijklabcd->lijkdabc", Roooovvvv)
    return Roooovvvv_anti
