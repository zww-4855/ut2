import numpy
from numpy import einsum

def energy_pertQf(g,l2,t2,o,v):


    #        -0.1250 <n,m||k,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(d,c,j,m)*t2(b,a,i,n)
    energy = -0.125000000000000 * einsum('nmkl,ijba,kldc,dcjm,bain', g[o, o, o, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #         0.2500 <n,m||k,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(d,b,j,m)*t2(c,a,i,n)
    energy +=  0.250000000000000 * einsum('nmkl,ijba,kldc,dbjm,cain', g[o, o, o, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #         0.2500 <n,m||j,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(d,c,k,m)*t2(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,kldc,dckm,bain', g[o, o, o, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #        -0.2500 <n,m||j,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(d,c,i,m)*t2(b,a,k,n)
    energy += -0.250000000000000 * einsum('nmjl,ijba,kldc,dcim,bakn', g[o, o, o, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #        -0.5000 <n,m||j,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(d,b,k,m)*t2(c,a,i,n)
    energy += -0.500000000000000 * einsum('nmjl,ijba,kldc,dbkm,cain', g[o, o, o, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #         0.5000 <n,m||j,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(d,b,i,m)*t2(c,a,k,n)
    energy +=  0.500000000000000 * einsum('nmjl,ijba,kldc,dbim,cakn', g[o, o, o, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #        -0.1250 <n,m||i,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(d,c,l,m)*t2(b,a,k,n)
    energy += -0.125000000000000 * einsum('nmij,ijba,kldc,dclm,bakn', g[o, o, o, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #         0.2500 <n,m||i,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(d,b,l,m)*t2(c,a,k,n)
    energy +=  0.250000000000000 * einsum('nmij,ijba,kldc,dblm,cakn', g[o, o, o, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #         0.5000 <m,d||e,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,c,j,k)*t2(b,a,i,m)
    energy +=  0.500000000000000 * einsum('mdel,ijba,kldc,ecjk,baim', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #         0.2500 <m,d||e,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,c,i,j)*t2(b,a,k,m)
    energy +=  0.250000000000000 * einsum('mdel,ijba,kldc,ecij,bakm', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #        -1.0000 <m,d||e,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,b,j,k)*t2(c,a,i,m)
    energy += -1.000000000000000 * einsum('mdel,ijba,kldc,ebjk,caim', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #        -0.5000 <m,d||e,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,b,i,j)*t2(c,a,k,m)
    energy += -0.500000000000000 * einsum('mdel,ijba,kldc,ebij,cakm', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #         0.2500 <m,d||e,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,c,k,l)*t2(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,kldc,eckl,baim', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #        -0.5000 <m,d||e,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,c,i,l)*t2(b,a,k,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,kldc,ecil,bakm', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #        -0.5000 <m,d||e,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,b,k,l)*t2(c,a,i,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,kldc,ebkl,caim', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #         1.0000 <m,d||e,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,b,i,l)*t2(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mdej,ijba,kldc,ebil,cakm', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #         1.0000 <m,b||e,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,d,j,k)*t2(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mbel,ijba,kldc,edjk,caim', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #         0.5000 <m,b||e,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,d,i,j)*t2(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mbel,ijba,kldc,edij,cakm', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #         0.5000 <m,b||e,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,a,j,k)*t2(d,c,i,m)
    energy +=  0.500000000000000 * einsum('mbel,ijba,kldc,eajk,dcim', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #         0.2500 <m,b||e,l>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,a,i,j)*t2(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,ijba,kldc,eaij,dckm', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #         0.5000 <m,b||e,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,d,k,l)*t2(c,a,i,m)
    energy +=  0.500000000000000 * einsum('mbej,ijba,kldc,edkl,caim', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #        -1.0000 <m,b||e,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,d,i,l)*t2(c,a,k,m)
    energy += -1.000000000000000 * einsum('mbej,ijba,kldc,edil,cakm', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #         0.2500 <m,b||e,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,a,k,l)*t2(d,c,i,m)
    energy +=  0.250000000000000 * einsum('mbej,ijba,kldc,eakl,dcim', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #        -0.5000 <m,b||e,j>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,a,i,l)*t2(d,c,k,m)
    energy += -0.500000000000000 * einsum('mbej,ijba,kldc,eail,dckm', g[o, v, v, o], l2, l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #        -0.1250 <d,c||e,f>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,b,k,l)*t2(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcef,ijba,kldc,ebkl,faij', g[v, v, v, v], l2, l2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #         0.2500 <d,c||e,f>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,b,j,l)*t2(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcef,ijba,kldc,ebjl,faik', g[v, v, v, v], l2, l2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #         0.2500 <d,b||e,f>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,c,k,l)*t2(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,eckl,faij', g[v, v, v, v], l2, l2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #        -0.6250 <d,b||e,f>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,c,j,l)*t2(f,a,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijba,kldc,ecjl,faik', g[v, v, v, v], l2, l2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #         0.2500 <d,b||e,f>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,c,i,j)*t2(f,a,k,l)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,ecij,fakl', g[v, v, v, v], l2, l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #        -0.3750 <d,a||e,f>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,b,j,l)*t2(f,c,i,k)
    energy += -0.375000000000000 * einsum('daef,ijba,kldc,ebjl,fcik', g[v, v, v, v], l2, l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #        -0.1250 <b,a||e,f>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,d,k,l)*t2(f,c,i,j)
    energy += -0.125000000000000 * einsum('baef,ijba,kldc,edkl,fcij', g[v, v, v, v], l2, l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #         0.2500 <b,a||e,f>*l2(i,j,b,a)*l2(k,l,d,c)*t2(e,d,j,l)*t2(f,c,i,k)
    energy +=  0.250000000000000 * einsum('baef,ijba,kldc,edjl,fcik', g[v, v, v, v], l2, l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    return energy
