import numpy
from numpy import einsum



def residQf2_aaaa(g,l2,t2,o,v):
    #        -0.1250 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,f,k,l)*t2(d,c,j,m)*t2(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,kldc,efkl,dcjm,abin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.2500 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,f,k,l)*t2(d,a,j,m)*t2(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efkl,dajm,cbin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.2500 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,f,j,l)*t2(d,c,k,m)*t2(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efjl,dckm,abin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.2500 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,f,j,l)*t2(d,c,i,m)*t2(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,kldc,efjl,dcim,abkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.5000 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,f,j,l)*t2(d,a,k,m)*t2(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,efjl,dakm,cbin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.5000 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,f,j,l)*t2(d,a,i,m)*t2(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,efjl,daim,cbkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.2500 <n,m||e,f>*l2(k,l,d,c)*t2(e,f,i,j)*t2(d,c,l,m)*t2(a,b,k,n)
    double_res += -0.250000000000000 * einsum('nmef,kldc,efij,dclm,abkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #         0.2500 P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,f,i,j)*t2(d,a,l,m)*t2(c,b,k,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efij,dalm,cbkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,l,m)*t2(f,c,j,k)*t2(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edlm,fcjk,abin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         1.0000 <n,m||e,f>*l2(k,l,d,c)*t2(e,d,l,m)*t2(f,c,i,j)*t2(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edlm,fcij,abkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #        -0.7500 P(i,j)*P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,l,m)*t2(f,a,j,k)*t2(c,b,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('nmef,kldc,edlm,fajk,cbin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -1.0000 P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,l,m)*t2(f,a,i,j)*t2(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edlm,faij,cbkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,j,m)*t2(f,c,k,l)*t2(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edjm,fckl,abin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -1.0000 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,j,m)*t2(f,c,i,l)*t2(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edjm,fcil,abkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.5000 P(i,j)*P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,j,m)*t2(f,a,k,l)*t2(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjm,fakl,cbin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         1.0000 P(i,j)*P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,j,m)*t2(f,a,i,l)*t2(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,kldc,edjm,fail,cbkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -0.2500 <n,m||e,f>*l2(k,l,d,c)*t2(e,d,k,l)*t2(f,c,i,j)*t2(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmef,kldc,edkl,fcij,abnm->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #        -0.5000 P(i,j)*P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,k,l)*t2(f,a,j,m)*t2(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edkl,fajm,cbin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.2500 P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,k,l)*t2(f,a,i,j)*t2(c,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edkl,faij,cbnm->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #        -0.5000 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,j,l)*t2(f,c,k,m)*t2(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjl,fckm,abin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.2500 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,j,l)*t2(f,c,i,k)*t2(a,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edjl,fcik,abnm->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.9167 P(i,j)*P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,j,l)*t2(f,a,k,m)*t2(c,b,i,n)
    contracted_intermediate =  0.916666666666670 * einsum('nmef,kldc,edjl,fakm,cbin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -1.0000 P(i,j)*P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,j,l)*t2(f,a,i,m)*t2(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edjl,faim,cbkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -0.5000 P(i,j)*P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,j,l)*t2(f,a,i,k)*t2(c,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjl,faik,cbnm->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -1.0000 P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,i,j)*t2(f,a,l,m)*t2(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edij,falm,cbkn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.2500 P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,d,i,j)*t2(f,a,k,l)*t2(c,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edij,fakl,cbnm->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,a,l,m)*t2(f,b,j,k)*t2(d,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,ealm,fbjk,dcin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.3750 <n,m||e,f>*l2(k,l,d,c)*t2(e,a,l,m)*t2(f,b,i,j)*t2(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('nmef,kldc,ealm,fbij,dckn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #         0.2500 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,a,j,m)*t2(f,b,k,l)*t2(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eajm,fbkl,dcin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.3750 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,a,j,m)*t2(f,b,i,l)*t2(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,kldc,eajm,fbil,dckn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.2500 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,a,k,l)*t2(f,b,j,m)*t2(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eakl,fbjm,dcin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.1250 <n,m||e,f>*l2(k,l,d,c)*t2(e,a,k,l)*t2(f,b,i,j)*t2(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmef,kldc,eakl,fbij,dcnm->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #        -0.5000 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,a,j,l)*t2(f,b,k,m)*t2(d,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,eajl,fbkm,dcin->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.5000 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,a,j,l)*t2(f,b,i,m)*t2(d,c,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,eajl,fbim,dckn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.2500 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,a,j,l)*t2(f,b,i,k)*t2(d,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eajl,fbik,dcnm->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.2500 P(i,j)*P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(e,a,j,k)*t2(d,b,i,m)*t2(f,c,l,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eajk,dbim,fcln->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.5000 <n,m||e,f>*l2(k,l,d,c)*t2(e,a,i,j)*t2(f,b,l,m)*t2(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,eaij,fblm,dckn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #        -0.1250 <n,m||e,f>*l2(k,l,d,c)*t2(e,a,i,j)*t2(f,b,k,l)*t2(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmef,kldc,eaij,fbkl,dcnm->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #        -0.1250 P(i,j)<n,m||e,f>*l2(k,l,d,c)*t2(e,b,j,k)*t2(d,c,l,m)*t2(f,a,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,kldc,ebjk,dclm,fain->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.1250 <n,m||e,f>*l2(k,l,d,c)*t2(e,b,i,j)*t2(d,c,l,m)*t2(f,a,k,n)
    double_res += -0.125000000000000 * einsum('nmef,kldc,ebij,dclm,fakn->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #        -0.0833 P(i,j)*P(a,b)<n,m||e,f>*l2(k,l,d,c)*t2(d,a,j,m)*t2(e,b,k,n)*t2(f,c,i,l)
    contracted_intermediate = -0.083333333333330 * einsum('nmef,kldc,dajm,ebkn,fcil->abij', g[o, o, v, v], l2, t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    
    return double_res


