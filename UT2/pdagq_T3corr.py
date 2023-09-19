from numpy import einsum
import numpy as np

def build_T3(g,o,v,t2):
    #        -1.0000 P(i,j)*P(a,b)<l,a||j,k>*t2(b,c,i,l)
    print('pdagq shapes:',np.shape(t2),np.shape(g[o,v,o,o]))
    contracted_intermediate = -1.000000000000000 * einsum('lajk,bcil->abcijk', g[o, v, o, o], t2)
    triples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)
    
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

def pdagq_T3energy(g,o,v,l1,l2,t3):
    #         0.2500 <k,j||b,c>*l1(i,a)*t3(b,c,a,i,k,j)
    energy =  0.250000000000000 * einsum('kjbc,ia,bcaikj', g[o, o, v, v], l1, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #         0.2500 <l,k||c,j>*l2(i,j,b,a)*t3(c,b,a,i,l,k)
    energy =  0.250000000000000 * einsum('lkcj,ijba,cbailk', g[o, o, v, o], l2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #         0.2500 <k,b||c,d>*l2(i,j,b,a)*t3(c,d,a,i,j,k)
    energy +=  0.250000000000000 * einsum('kbcd,ijba,cdaijk', g[o, v, v, v], l2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    
    return energy

def ccsd_energy(g,f,o,v,t1,t2):
    #     1.0000 f(i,i)
    energy = 1.0 * einsum('ii', f[o, o])

    print('shapes of fock op and t1:',np.shape(f),np.shape(f[o,v]),np.shape(t1))
    #     1.0000 f(i,a)*t1(a,i)
    energy += 1.0 * einsum('ia,ai', f[o, v], t1)

    #    -0.5000 <j,i||j,i>
    energy += -0.5 * einsum('jiji', g[o, o, o, o])

    #     0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy += 0.25 * einsum('jiab,abji', g[o, o, v, v], t2)

    #    -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += -0.5 * einsum('jiab,ai,bj', g[o, o, v, v], t1, t1,
                            optimize=['einsum_path', (0, 1), (0, 1)])
    print('CCSD energy:',energy)

