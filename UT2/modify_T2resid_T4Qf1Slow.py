import numpy as np
from numpy import einsum


def unsym_residQf1(g,t2_aa,o,v,nocc,nvir):
    # contributions to the residual
    t=t2_aa.transpose(2,3,0,1)

    g_occ=g[o,o,o,o]
    g_ov=g[o,v,o,v]
    g_virt=g[v,v,v,v]
    Roooovvvv = -0.062500000 * np.einsum("imab,jncd,klmn",t,t,g_occ)

    Roooovvvv += -0.250000000 * np.einsum("imab,jkce,lemd",t,t,g_ov)
    Roooovvvv += -0.062500000 * np.einsum("ijae,klbf,efcd",t,t,g_virt)

    
    return Roooovvvv


def unsym_residQf2(g,t2_aa,o,v,nocc,nvir):
    t={}
    t.update({"oovv":t2_aa.transpose(2,3,0,1)})
    v={}
    v.update({"oooo":g[o,o,o,o],"vvvv":g[v,v,v,v],"ovov":g[ovov]})
    # contributions to the residual
    Roooovvvv = np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
    roooovvvv += -0.031250000 * np.einsum("imab,jncd,klef,efmn->ijklabcd",t["oovv"],t["oovv"],t["oovv"],v["vvoo"],optimize="optimal")
    roooovvvv += 0.250000000 * np.einsum("imab,jkce,lndf,efmn->ijklabcd",t["oovv"],t["oovv"],t["oovv"],v["vvoo"],optimize="optimal")
    roooovvvv += -0.031250000 * np.einsum("mnab,ijce,kldf,efmn->ijklabcd",t["oovv"],t["oovv"],t["oovv"],v["vvoo"],optimize="optimal")
    return Roooovvvv


def residQf1_aaaa(g,l2,t2,o,v):
    
    
    #        -0.2500 P(i,j)<n,m||k,l>*l2(k,l,d,c)*t2(d,c,j,m)*t2(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmkl,kldc,dcjm,abin->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.5000 P(i,j)<n,m||k,l>*l2(k,l,d,c)*t2(d,a,j,m)*t2(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmkl,kldc,dajm,cbin->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.5000 P(i,j)<n,m||j,l>*l2(k,l,d,c)*t2(d,c,k,m)*t2(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,kldc,dckm,abin->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.5000 P(i,j)<n,m||j,l>*l2(k,l,d,c)*t2(d,c,i,m)*t2(a,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmjl,kldc,dcim,abkn->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -1.0000 P(i,j)<n,m||j,l>*l2(k,l,d,c)*t2(d,a,k,m)*t2(c,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,kldc,dakm,cbin->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         1.0000 P(i,j)<n,m||j,l>*l2(k,l,d,c)*t2(d,a,i,m)*t2(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,kldc,daim,cbkn->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.5000 <n,m||i,j>*l2(k,l,d,c)*t2(d,c,l,m)*t2(a,b,k,n)
    double_res += -0.500000000000000 * einsum('nmij,kldc,dclm,abkn->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #         0.5000 P(a,b)<n,m||i,j>*l2(k,l,d,c)*t2(d,a,l,m)*t2(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmij,kldc,dalm,cbkn->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         1.0000 P(i,j)<m,d||e,l>*l2(k,l,d,c)*t2(e,c,j,k)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,kldc,ecjk,abim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         1.0000 <m,d||e,l>*l2(k,l,d,c)*t2(e,c,i,j)*t2(a,b,k,m)
    double_res +=  1.000000000000000 * einsum('mdel,kldc,ecij,abkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #        -1.0000 P(i,j)*P(a,b)<m,d||e,l>*l2(k,l,d,c)*t2(e,a,j,k)*t2(c,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eajk,cbim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -1.0000 P(a,b)<m,d||e,l>*l2(k,l,d,c)*t2(e,a,i,j)*t2(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eaij,cbkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 P(i,j)<m,d||e,j>*l2(k,l,d,c)*t2(e,c,k,l)*t2(a,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdej,kldc,eckl,abim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -1.0000 P(i,j)<m,d||e,j>*l2(k,l,d,c)*t2(e,c,i,l)*t2(a,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,kldc,ecil,abkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.5000 P(i,j)*P(a,b)<m,d||e,j>*l2(k,l,d,c)*t2(e,a,k,l)*t2(c,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('mdej,kldc,eakl,cbim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         1.0000 P(i,j)*P(a,b)<m,d||e,j>*l2(k,l,d,c)*t2(e,a,i,l)*t2(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdej,kldc,eail,cbkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         1.0000 P(i,j)*P(a,b)<m,a||e,l>*l2(k,l,d,c)*t2(e,d,j,k)*t2(c,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,kldc,edjk,cbim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         1.0000 P(a,b)<m,a||e,l>*l2(k,l,d,c)*t2(e,d,i,j)*t2(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,kldc,edij,cbkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 P(i,j)*P(a,b)<m,a||e,l>*l2(k,l,d,c)*t2(e,b,j,k)*t2(d,c,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebjk,dcim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.5000 P(a,b)<m,a||e,l>*l2(k,l,d,c)*t2(e,b,i,j)*t2(d,c,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebij,dckm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 P(i,j)*P(a,b)<m,a||e,j>*l2(k,l,d,c)*t2(e,d,k,l)*t2(c,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('maej,kldc,edkl,cbim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -1.0000 P(i,j)*P(a,b)<m,a||e,j>*l2(k,l,d,c)*t2(e,d,i,l)*t2(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('maej,kldc,edil,cbkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.2500 P(i,j)*P(a,b)<m,a||e,j>*l2(k,l,d,c)*t2(e,b,k,l)*t2(d,c,i,m)
    contracted_intermediate =  0.250000000000000 * einsum('maej,kldc,ebkl,dcim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -0.5000 P(i,j)*P(a,b)<m,a||e,j>*l2(k,l,d,c)*t2(e,b,i,l)*t2(d,c,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,kldc,ebil,dckm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -0.2500 <d,c||e,f>*l2(k,l,d,c)*t2(e,a,k,l)*t2(f,b,i,j)
    double_res += -0.250000000000000 * einsum('dcef,kldc,eakl,fbij->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #         0.5000 P(i,j)<d,c||e,f>*l2(k,l,d,c)*t2(e,a,j,l)*t2(f,b,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('dcef,kldc,eajl,fbik->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.2500 <d,c||e,f>*l2(k,l,d,c)*t2(e,a,i,j)*t2(f,b,k,l)
    double_res += -0.250000000000000 * einsum('dcef,kldc,eaij,fbkl->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #         0.5000 P(a,b)<d,a||e,f>*l2(k,l,d,c)*t2(e,c,k,l)*t2(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,eckl,fbij->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #        -1.0000 P(i,j)*P(a,b)<d,a||e,f>*l2(k,l,d,c)*t2(e,c,j,l)*t2(f,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('daef,kldc,ecjl,fbik->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.5000 P(a,b)<d,a||e,f>*l2(k,l,d,c)*t2(e,c,i,j)*t2(f,b,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,ecij,fbkl->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #        -0.5000 <a,b||e,f>*l2(k,l,d,c)*t2(e,d,k,l)*t2(f,c,i,j)
    double_res += -0.500000000000000 * einsum('abef,kldc,edkl,fcij->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #         0.5000 P(i,j)<a,b||e,f>*l2(k,l,d,c)*t2(e,d,j,l)*t2(f,c,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('abef,kldc,edjl,fcik->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    return double_res
