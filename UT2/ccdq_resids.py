from numpy import einsum
import numpy as np
def ccdq_t2_aaaa_residual(t2_aaaa, t2_bbbb, t2_abab, 
                            t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                            f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(i,j)f_aa(k,j)*t2_aaaa(a,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('kj,abik->abij', f_aa[o, o], t2_aaaa)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f_aa(a,c)*t2_aaaa(c,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('ac,cbij->abij', f_aa[v, v], t2_aaaa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 <a,b||i,j>_aaaa
    doubles_res +=  1.000000000000000 * einsum('abij->abij', g_aaaa[v, v, o, o])
    
    #	  0.5000 <l,k||i,j>_aaaa*t2_aaaa(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk->abij', g_aaaa[o, o, o, o], t2_aaaa)
    
    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>_aaaa*t2_aaaa(c,b,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,cbik->abij', g_aaaa[o, v, v, o], t2_aaaa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,k||j,c>_abab*t2_abab(b,c,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('akjc,bcik->abij', g_abab[v, o, o, v], t2_abab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 <a,b||c,d>_aaaa*t2_aaaa(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij->abij', g_aaaa[v, v, v, v], t2_aaaa)
    
    #	  0.2500 <l,k||c,d>_aaaa*t4_aaaaaaaa(c,d,a,b,i,j,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdabijlk->abij', g_aaaa[o, o, v, v], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    
    #	 -0.2500 <l,k||c,d>_abab*t4_aaabaaab(c,b,a,d,i,j,l,k)
    doubles_res += -0.250000000000000 * einsum('lkcd,cbadijlk->abij', g_abab[o, o, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	 -0.2500 <k,l||c,d>_abab*t4_aaabaaab(c,b,a,d,i,j,k,l)
    doubles_res += -0.250000000000000 * einsum('klcd,cbadijkl->abij', g_abab[o, o, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  0.2500 <l,k||d,c>_abab*t4_aaabaaab(b,d,a,c,i,j,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkdc,bdacijlk->abij', g_abab[o, o, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  0.2500 <k,l||d,c>_abab*t4_aaabaaab(b,d,a,c,i,j,k,l)
    doubles_res +=  0.250000000000000 * einsum('kldc,bdacijkl->abij', g_abab[o, o, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  0.2500 <l,k||c,d>_bbbb*t4_aabbaabb(a,b,c,d,i,j,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,abcdijlk->abij', g_bbbb[o, o, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(c,d,j,k)*t2_aaaa(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<l,k||c,d>_abab*t2_abab(c,d,j,k)*t2_aaaa(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<l,k||d,c>_abab*t2_abab(d,c,j,k)*t2_aaaa(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkdc,dcjk,abil->abij', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 <l,k||c,d>_aaaa*t2_aaaa(c,d,i,j)*t2_aaaa(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*t2_aaaa(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,c,l,k)*t2_aaaa(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkdc,aclk,dbij->abij', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,c,k,l)*t2_aaaa(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('kldc,ackl,dbij->abij', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(c,a,j,k)*t2_aaaa(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<k,l||c,d>_abab*t2_aaaa(c,a,j,k)*t2_abab(b,d,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('klcd,cajk,bdil->abij', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<l,k||d,c>_abab*t2_abab(a,c,j,k)*t2_aaaa(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkdc,acjk,dbil->abij', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<l,k||c,d>_bbbb*t2_abab(a,c,j,k)*t2_abab(b,d,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,acjk,bdil->abij', g_bbbb[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,a,i,j)*t2_aaaa(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_abab*t2_aaaa(c,a,i,j)*t2_abab(b,d,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,caij,bdlk->abij', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,l||c,d>_abab*t2_aaaa(c,a,i,j)*t2_abab(b,d,k,l)
    doubles_res +=  0.500000000000000 * einsum('klcd,caij,bdkl->abij', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    return doubles_res


def ccdq_t2_abab_residual(t2_aaaa, t2_bbbb, t2_abab, 
                            t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                            f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_bb(k,j)*t2_abab(a,b,i,k)
    doubles_res = -1.000000000000000 * einsum('kj,abik->abij', f_bb[o, o], t2_abab)
    
    #	 -1.0000 f_aa(k,i)*t2_abab(a,b,k,j)
    doubles_res += -1.000000000000000 * einsum('ki,abkj->abij', f_aa[o, o], t2_abab)
    
    #	  1.0000 f_aa(a,c)*t2_abab(c,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('ac,cbij->abij', f_aa[v, v], t2_abab)
    
    #	  1.0000 f_bb(b,c)*t2_abab(a,c,i,j)
    doubles_res +=  1.000000000000000 * einsum('bc,acij->abij', f_bb[v, v], t2_abab)
    
    #	  1.0000 <a,b||i,j>_abab
    doubles_res +=  1.000000000000000 * einsum('abij->abij', g_abab[v, v, o, o])
    
    #	  0.5000 <l,k||i,j>_abab*t2_abab(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk->abij', g_abab[o, o, o, o], t2_abab)
    
    #	  0.5000 <k,l||i,j>_abab*t2_abab(a,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('klij,abkl->abij', g_abab[o, o, o, o], t2_abab)
    
    #	 -1.0000 <a,k||c,j>_abab*t2_abab(c,b,i,k)
    doubles_res += -1.000000000000000 * einsum('akcj,cbik->abij', g_abab[v, o, v, o], t2_abab)
    
    #	 -1.0000 <k,b||c,j>_abab*t2_aaaa(c,a,i,k)
    doubles_res += -1.000000000000000 * einsum('kbcj,caik->abij', g_abab[o, v, v, o], t2_aaaa)
    
    #	  1.0000 <k,b||c,j>_bbbb*t2_abab(a,c,i,k)
    doubles_res +=  1.000000000000000 * einsum('kbcj,acik->abij', g_bbbb[o, v, v, o], t2_abab)
    
    #	  1.0000 <k,a||c,i>_aaaa*t2_abab(c,b,k,j)
    doubles_res +=  1.000000000000000 * einsum('kaci,cbkj->abij', g_aaaa[o, v, v, o], t2_abab)
    
    #	 -1.0000 <a,k||i,c>_abab*t2_bbbb(c,b,j,k)
    doubles_res += -1.000000000000000 * einsum('akic,cbjk->abij', g_abab[v, o, o, v], t2_bbbb)
    
    #	 -1.0000 <k,b||i,c>_abab*t2_abab(a,c,k,j)
    doubles_res += -1.000000000000000 * einsum('kbic,ackj->abij', g_abab[o, v, o, v], t2_abab)
    
    #	  0.5000 <a,b||c,d>_abab*t2_abab(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij->abij', g_abab[v, v, v, v], t2_abab)
    
    #	  0.5000 <a,b||d,c>_abab*t2_abab(d,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('abdc,dcij->abij', g_abab[v, v, v, v], t2_abab)
    
    #	 -0.2500 <l,k||c,d>_aaaa*t4_aaabaaab(c,d,a,b,i,k,l,j)
    doubles_res += -0.250000000000000 * einsum('lkcd,cdabiklj->abij', g_aaaa[o, o, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  0.2500 <l,k||c,d>_abab*t4_aabbaabb(c,a,d,b,i,l,j,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cadbiljk->abij', g_abab[o, o, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.2500 <k,l||c,d>_abab*t4_aabbaabb(c,a,d,b,i,k,l,j)
    doubles_res += -0.250000000000000 * einsum('klcd,cadbiklj->abij', g_abab[o, o, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.2500 <l,k||d,c>_abab*t4_aabbaabb(a,d,c,b,i,l,j,k)
    doubles_res += -0.250000000000000 * einsum('lkdc,adcbiljk->abij', g_abab[o, o, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	  0.2500 <k,l||d,c>_abab*t4_aabbaabb(a,d,c,b,i,k,l,j)
    doubles_res +=  0.250000000000000 * einsum('kldc,adcbiklj->abij', g_abab[o, o, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.2500 <l,k||c,d>_bbbb*t4_abbbabbb(a,d,c,b,i,j,l,k)
    doubles_res += -0.250000000000000 * einsum('lkcd,adcbijlk->abij', g_bbbb[o, o, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 <k,l||c,d>_abab*t2_abab(c,d,k,j)*t2_abab(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('klcd,cdkj,abil->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t2_abab(d,c,k,j)*t2_abab(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('kldc,dckj,abil->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,d,j,k)*t2_abab(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,d,i,k)*t2_abab(a,b,l,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,cdik,ablj->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_abab*t2_abab(c,d,i,k)*t2_abab(a,b,l,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,cdik,ablj->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t2_abab(d,c,i,k)*t2_abab(a,b,l,j)
    doubles_res += -0.500000000000000 * einsum('lkdc,dcik,ablj->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,k||c,d>_abab*t2_abab(c,d,i,j)*t2_abab(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <k,l||c,d>_abab*t2_abab(c,d,i,j)*t2_abab(a,b,k,l)
    doubles_res +=  0.250000000000000 * einsum('klcd,cdij,abkl->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,k||d,c>_abab*t2_abab(d,c,i,j)*t2_abab(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkdc,dcij,ablk->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <k,l||d,c>_abab*t2_abab(d,c,i,j)*t2_abab(a,b,k,l)
    doubles_res +=  0.250000000000000 * einsum('kldc,dcij,abkl->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*t2_abab(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,c,l,k)*t2_abab(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkdc,aclk,dbij->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,c,k,l)*t2_abab(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('kldc,ackl,dbij->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||d,c>_abab*t2_abab(a,c,k,j)*t2_abab(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('kldc,ackj,dbil->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t2_aaaa(c,a,i,k)*t2_abab(d,b,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,caik,dblj->abij', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t2_aaaa(c,a,i,k)*t2_bbbb(d,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,caik,dbjl->abij', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t2_abab(a,c,i,k)*t2_abab(d,b,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkdc,acik,dblj->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t2_abab(a,c,i,k)*t2_bbbb(d,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,acik,dbjl->abij', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,c,i,j)*t2_abab(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkdc,acij,dblk->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,c,i,j)*t2_abab(d,b,k,l)
    doubles_res += -0.500000000000000 * einsum('kldc,acij,dbkl->abij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_bbbb*t2_abab(a,c,i,j)*t2_bbbb(d,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,acij,dblk->abij', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    return doubles_res


def ccdq_t2_bbbb_residual(t2_aaaa, t2_bbbb, t2_abab, 
                            t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                            f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(i,j)f_bb(k,j)*t2_bbbb(a,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('kj,abik->abij', f_bb[o, o], t2_bbbb)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f_bb(a,c)*t2_bbbb(c,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('ac,cbij->abij', f_bb[v, v], t2_bbbb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 <a,b||i,j>_bbbb
    doubles_res +=  1.000000000000000 * einsum('abij->abij', g_bbbb[v, v, o, o])
    
    #	  0.5000 <l,k||i,j>_bbbb*t2_bbbb(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk->abij', g_bbbb[o, o, o, o], t2_bbbb)
    
    #	 -1.0000 P(i,j)*P(a,b)<k,a||c,j>_abab*t2_abab(c,b,k,i)
    contracted_intermediate = -1.000000000000000 * einsum('kacj,cbki->abij', g_abab[o, v, v, o], t2_abab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>_bbbb*t2_bbbb(c,b,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,cbik->abij', g_bbbb[o, v, v, o], t2_bbbb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 <a,b||c,d>_bbbb*t2_bbbb(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij->abij', g_bbbb[v, v, v, v], t2_bbbb)
    
    #	  0.2500 <l,k||c,d>_aaaa*t4_aabbaabb(c,d,a,b,l,k,i,j)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdablkij->abij', g_aaaa[o, o, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.2500 <l,k||c,d>_abab*t4_abbbabbb(c,d,a,b,l,j,i,k)
    doubles_res += -0.250000000000000 * einsum('lkcd,cdabljik->abij', g_abab[o, o, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  0.2500 <k,l||c,d>_abab*t4_abbbabbb(c,d,a,b,k,j,l,i)
    doubles_res +=  0.250000000000000 * einsum('klcd,cdabkjli->abij', g_abab[o, o, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	 -0.2500 <l,k||d,c>_abab*t4_abbbabbb(d,c,a,b,l,j,i,k)
    doubles_res += -0.250000000000000 * einsum('lkdc,dcabljik->abij', g_abab[o, o, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  0.2500 <k,l||d,c>_abab*t4_abbbabbb(d,c,a,b,k,j,l,i)
    doubles_res +=  0.250000000000000 * einsum('kldc,dcabkjli->abij', g_abab[o, o, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  0.2500 <l,k||c,d>_bbbb*t4_bbbbbbbb(c,d,a,b,i,j,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdabijlk->abij', g_bbbb[o, o, v, v], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 P(i,j)<k,l||c,d>_abab*t2_abab(c,d,k,j)*t2_bbbb(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('klcd,cdkj,abil->abij', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<k,l||d,c>_abab*t2_abab(d,c,k,j)*t2_bbbb(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('kldc,dckj,abil->abij', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(c,d,j,k)*t2_bbbb(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 <l,k||c,d>_bbbb*t2_bbbb(c,d,i,j)*t2_bbbb(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk->abij', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_abab*t2_abab(c,a,l,k)*t2_bbbb(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||c,d>_abab*t2_abab(c,a,k,l)*t2_bbbb(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('klcd,cakl,dbij->abij', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,a,l,k)*t2_bbbb(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(i,j)<l,k||c,d>_aaaa*t2_abab(c,a,k,j)*t2_abab(d,b,l,i)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cakj,dbli->abij', g_aaaa[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<k,l||c,d>_abab*t2_abab(c,a,k,j)*t2_bbbb(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('klcd,cakj,dbil->abij', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<l,k||d,c>_abab*t2_bbbb(c,a,j,k)*t2_abab(d,b,l,i)
    contracted_intermediate =  1.000000000000000 * einsum('lkdc,cajk,dbli->abij', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(c,a,j,k)*t2_bbbb(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 <l,k||d,c>_abab*t2_bbbb(c,a,i,j)*t2_abab(d,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkdc,caij,dblk->abij', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,l||d,c>_abab*t2_bbbb(c,a,i,j)*t2_abab(d,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('kldc,caij,dbkl->abij', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,a,i,j)*t2_bbbb(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    return doubles_res


def ccdq_t4_aaaaaaaa_residual(t2_aaaa, t2_bbbb, t2_abab, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(k,l)f_aa(m,l)*t4_aaaaaaaa(a,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000010 * einsum('ml,abcdijkm->abcdijkl', f_aa[o, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)f_aa(m,j)*t4_aaaaaaaa(a,b,c,d,i,k,l,m)
    contracted_intermediate = -1.000000000000010 * einsum('mj,abcdiklm->abcdijkl', f_aa[o, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f_aa(a,e)*t4_aaaaaaaa(e,b,c,d,i,j,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('ae,ebcdijkl->abcdijkl', f_aa[v, v], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)f_aa(c,e)*t4_aaaaaaaa(e,a,b,d,i,j,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('ce,eabdijkl->abcdijkl', f_aa[v, v], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<n,m||k,l>_aaaa*t4_aaaaaaaa(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmkl,abcdijnm->abcdijkl', g_aaaa[o, o, o, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||i,l>_aaaa*t4_aaaaaaaa(a,b,c,d,j,k,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmil,abcdjknm->abcdijkl', g_aaaa[o, o, o, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<n,m||j,k>_aaaa*t4_aaaaaaaa(a,b,c,d,i,l,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmjk,abcdilnm->abcdijkl', g_aaaa[o, o, o, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,l>_aaaa*t4_aaaaaaaa(e,b,c,d,i,j,k,m)
    contracted_intermediate =  1.000000000000020 * einsum('mael,ebcdijkm->abcdijkl', g_aaaa[o, v, v, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<a,m||l,e>_abab*t4_aaabaaab(d,b,c,e,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('amle,dbceijkm->abcdijkl', g_abab[v, o, o, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*t4_aaaaaaaa(e,b,c,d,i,k,l,m)
    contracted_intermediate =  1.000000000000020 * einsum('maej,ebcdiklm->abcdijkl', g_aaaa[o, v, v, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,m||j,e>_abab*t4_aaabaaab(d,b,c,e,i,k,l,m)
    contracted_intermediate = -1.000000000000020 * einsum('amje,dbceiklm->abcdijkl', g_abab[v, o, o, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,l>_aaaa*t4_aaaaaaaa(e,a,b,d,i,j,k,m)
    contracted_intermediate =  1.000000000000020 * einsum('mcel,eabdijkm->abcdijkl', g_aaaa[o, v, v, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<c,m||l,e>_abab*t4_aaabaaab(d,a,b,e,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('cmle,dabeijkm->abcdijkl', g_abab[v, o, o, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,j>_aaaa*t4_aaaaaaaa(e,a,b,d,i,k,l,m)
    contracted_intermediate =  1.000000000000020 * einsum('mcej,eabdiklm->abcdijkl', g_aaaa[o, v, v, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<c,m||j,e>_abab*t4_aaabaaab(d,a,b,e,i,k,l,m)
    contracted_intermediate = -1.000000000000020 * einsum('cmje,dabeiklm->abcdijkl', g_abab[v, o, o, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<a,b||e,f>_aaaa*t4_aaaaaaaa(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('abef,efcdijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<a,d||e,f>_aaaa*t4_aaaaaaaa(e,f,b,c,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('adef,efbcijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,d)<b,c||e,f>_aaaa*t4_aaaaaaaa(e,f,a,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('bcef,efadijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,f,l,m)*t4_aaaaaaaa(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eflm,abcdijkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_abab*t2_abab(e,f,l,m)*t4_aaaaaaaa(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eflm,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||f,e>_abab*t2_abab(f,e,l,m)*t4_aaaaaaaa(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,felm,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,m)*t4_aaaaaaaa(a,b,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,efjm,abcdikln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*t2_abab(e,f,j,m)*t4_aaaaaaaa(a,b,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,efjm,abcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*t2_abab(f,e,j,m)*t4_aaaaaaaa(a,b,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,fejm,abcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.2500 P(j,k)<n,m||e,f>_aaaa*t2_aaaa(e,f,k,l)*t4_aaaaaaaa(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,efkl,abcdijnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.2500 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,l)*t4_aaaaaaaa(a,b,c,d,j,k,n,m)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,efil,abcdjknm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.2500 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,k)*t4_aaaaaaaa(a,b,c,d,i,l,n,m)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,efjk,abcdilnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,n,m)*t4_aaaaaaaa(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eanm,fbcdijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,n,m)*t4_aaaaaaaa(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,aenm,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,m,n)*t4_aaaaaaaa(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,aemn,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,l,m)*t4_aaaaaaaa(f,b,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ealm,fbcdijkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,l,m)*t4_aaabaaab(d,b,c,f,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,ealm,dbcfijkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,l,m)*t4_aaaaaaaa(f,b,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,aelm,fbcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,l,m)*t4_aaabaaab(d,b,c,f,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,aelm,dbcfijkn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t4_aaaaaaaa(f,b,c,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eajm,fbcdikln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,j,m)*t4_aaabaaab(d,b,c,f,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,eajm,dbcfikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t4_aaaaaaaa(f,b,c,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,aejm,fbcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,j,m)*t4_aaabaaab(d,b,c,f,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,aejm,dbcfikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,l)*t4_aaaaaaaa(f,b,c,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eakl,fbcdijnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,k,l)*t4_aaabaaab(d,b,c,f,i,j,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eakl,dbcfijnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,k,l)*t4_aaabaaab(d,b,c,f,i,j,m,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,eakl,dbcfijmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,l)*t4_aaaaaaaa(f,b,c,d,j,k,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eail,fbcdjknm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,i,l)*t4_aaabaaab(d,b,c,f,j,k,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eail,dbcfjknm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,i,l)*t4_aaabaaab(d,b,c,f,j,k,m,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,eail,dbcfjkmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,k)*t4_aaaaaaaa(f,b,c,d,i,l,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eajk,fbcdilnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,j,k)*t4_aaabaaab(d,b,c,f,i,l,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eajk,dbcfilnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,j,k)*t4_aaabaaab(d,b,c,f,i,l,m,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,eajk,dbcfilmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||e,f>_aaaa*t2_aaaa(e,c,n,m)*t4_aaaaaaaa(f,a,b,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecnm,fabdijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||f,e>_abab*t2_abab(c,e,n,m)*t4_aaaaaaaa(f,a,b,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,cenm,fabdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<m,n||f,e>_abab*t2_abab(c,e,m,n)*t4_aaaaaaaa(f,a,b,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,cemn,fabdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>_aaaa*t2_aaaa(e,c,l,m)*t4_aaaaaaaa(f,a,b,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eclm,fabdijkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,n||e,f>_abab*t2_aaaa(e,c,l,m)*t4_aaabaaab(d,a,b,f,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,eclm,dabfijkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||f,e>_abab*t2_abab(c,e,l,m)*t4_aaaaaaaa(f,a,b,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,celm,fabdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>_bbbb*t2_abab(c,e,l,m)*t4_aaabaaab(d,a,b,f,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,celm,dabfijkn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,m)*t4_aaaaaaaa(f,a,b,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ecjm,fabdikln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,n||e,f>_abab*t2_aaaa(e,c,j,m)*t4_aaabaaab(d,a,b,f,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,ecjm,dabfikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||f,e>_abab*t2_abab(c,e,j,m)*t4_aaaaaaaa(f,a,b,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,cejm,fabdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>_bbbb*t2_abab(c,e,j,m)*t4_aaabaaab(d,a,b,f,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,cejm,dabfikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(c,d)<n,m||e,f>_aaaa*t2_aaaa(e,c,k,l)*t4_aaaaaaaa(f,a,b,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eckl,fabdijnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(c,d)<n,m||e,f>_abab*t2_aaaa(e,c,k,l)*t4_aaabaaab(d,a,b,f,i,j,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eckl,dabfijnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(c,d)<m,n||e,f>_abab*t2_aaaa(e,c,k,l)*t4_aaabaaab(d,a,b,f,i,j,m,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,eckl,dabfijmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>_aaaa*t2_aaaa(e,c,i,l)*t4_aaaaaaaa(f,a,b,d,j,k,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecil,fabdjknm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<n,m||e,f>_abab*t2_aaaa(e,c,i,l)*t4_aaabaaab(d,a,b,f,j,k,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecil,dabfjknm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<m,n||e,f>_abab*t2_aaaa(e,c,i,l)*t4_aaabaaab(d,a,b,f,j,k,m,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,ecil,dabfjkmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(c,d)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,k)*t4_aaaaaaaa(f,a,b,d,i,l,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecjk,fabdilnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(c,d)<n,m||e,f>_abab*t2_aaaa(e,c,j,k)*t4_aaabaaab(d,a,b,f,i,l,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecjk,dabfilnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(c,d)<m,n||e,f>_abab*t2_aaaa(e,c,j,k)*t4_aaabaaab(d,a,b,f,i,l,m,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,ecjk,dabfilmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  0.2500 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(a,b,n,m)*t4_aaaaaaaa(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,abnm,efcdijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(a,b,l,m)*t4_aaaaaaaa(e,f,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ablm,efcdijkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<m,n||e,f>_abab*t2_aaaa(a,b,l,m)*t4_aaabaaab(e,d,c,f,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,ablm,edcfijkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,c)<m,n||f,e>_abab*t2_aaaa(a,b,l,m)*t4_aaabaaab(d,f,c,e,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,ablm,dfceijkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(a,b,j,m)*t4_aaaaaaaa(e,f,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,abjm,efcdikln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<m,n||e,f>_abab*t2_aaaa(a,b,j,m)*t4_aaabaaab(e,d,c,f,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,abjm,edcfikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,c)<m,n||f,e>_abab*t2_aaaa(a,b,j,m)*t4_aaabaaab(d,f,c,e,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,abjm,dfceikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(a,d,n,m)*t4_aaaaaaaa(e,f,b,c,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,adnm,efbcijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(a,d,l,m)*t4_aaaaaaaa(e,f,b,c,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adlm,efbcijkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<m,n||e,f>_abab*t2_aaaa(a,d,l,m)*t4_aaabaaab(e,c,b,f,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,adlm,ecbfijkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<m,n||f,e>_abab*t2_aaaa(a,d,l,m)*t4_aaabaaab(c,f,b,e,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,adlm,cfbeijkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(a,d,j,m)*t4_aaaaaaaa(e,f,b,c,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adjm,efbcikln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||e,f>_abab*t2_aaaa(a,d,j,m)*t4_aaabaaab(e,c,b,f,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,adjm,ecbfikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*t2_aaaa(a,d,j,m)*t4_aaabaaab(c,f,b,e,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,adjm,cfbeikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.2500 P(b,d)<n,m||e,f>_aaaa*t2_aaaa(b,c,n,m)*t4_aaaaaaaa(e,f,a,d,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,bcnm,efadijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,d)<n,m||e,f>_aaaa*t2_aaaa(b,c,l,m)*t4_aaaaaaaa(e,f,a,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bclm,efadijkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,d)<m,n||e,f>_abab*t2_aaaa(b,c,l,m)*t4_aaabaaab(e,d,a,f,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,bclm,edafijkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,d)<m,n||f,e>_abab*t2_aaaa(b,c,l,m)*t4_aaabaaab(d,f,a,e,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,bclm,dfaeijkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,d)<n,m||e,f>_aaaa*t2_aaaa(b,c,j,m)*t4_aaaaaaaa(e,f,a,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bcjm,efadikln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,d)<m,n||e,f>_abab*t2_aaaa(b,c,j,m)*t4_aaabaaab(e,d,a,f,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,bcjm,edafikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,d)<m,n||f,e>_abab*t2_aaaa(b,c,j,m)*t4_aaabaaab(d,f,a,e,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,bcjm,dfaeikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||k,l>_aaaa*t2_aaaa(a,b,j,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,abjm,cdin->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||k,l>_aaaa*t2_aaaa(a,d,j,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,adjm,bcin->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||j,l>_aaaa*t2_aaaa(a,b,k,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,abkm,cdin->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||j,l>_aaaa*t2_aaaa(a,d,k,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,adkm,bcin->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||i,l>_aaaa*t2_aaaa(a,b,k,m)*t2_aaaa(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,abkm,cdjn->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||i,l>_aaaa*t2_aaaa(a,d,k,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,adkm,bcjn->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)*P(b,c)<n,m||j,k>_aaaa*t2_aaaa(a,b,l,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,ablm,cdin->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)<n,m||j,k>_aaaa*t2_aaaa(a,d,l,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,adlm,bcin->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,c)<n,m||i,k>_aaaa*t2_aaaa(a,b,l,m)*t2_aaaa(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmik,ablm,cdjn->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||i,k>_aaaa*t2_aaaa(a,d,l,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmik,adlm,bcjn->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||i,j>_aaaa*t2_aaaa(a,b,l,m)*t2_aaaa(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,ablm,cdkn->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||i,j>_aaaa*t2_aaaa(a,d,l,m)*t2_aaaa(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,adlm,bckn->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,l>_aaaa*t2_aaaa(e,b,j,k)*t2_aaaa(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ebjk,cdim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<m,a||e,l>_aaaa*t2_aaaa(e,b,i,j)*t2_aaaa(c,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ebij,cdkm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,l>_aaaa*t2_aaaa(e,d,j,k)*t2_aaaa(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,edjk,bcim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,l>_aaaa*t2_aaaa(e,d,i,j)*t2_aaaa(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,edij,bckm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,a||e,k>_aaaa*t2_aaaa(e,b,j,l)*t2_aaaa(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,ebjl,cdim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,k>_aaaa*t2_aaaa(e,d,j,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,edjl,bcim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<m,a||e,j>_aaaa*t2_aaaa(e,b,k,l)*t2_aaaa(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ebkl,cdim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,j>_aaaa*t2_aaaa(e,b,i,k)*t2_aaaa(c,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ebik,cdlm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,a||e,j>_aaaa*t2_aaaa(e,d,k,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edkl,bcim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*t2_aaaa(e,d,i,k)*t2_aaaa(b,c,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edik,bclm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,a||e,i>_aaaa*t2_aaaa(e,b,k,l)*t2_aaaa(c,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,ebkl,cdjm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,a||e,i>_aaaa*t2_aaaa(e,d,k,l)*t2_aaaa(b,c,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,edkl,bcjm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,l>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,eajk,cdim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,c)<m,b||e,l>_aaaa*t2_aaaa(e,a,i,j)*t2_aaaa(c,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,eaij,cdkm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<m,b||e,k>_aaaa*t2_aaaa(e,a,j,l)*t2_aaaa(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbek,eajl,cdim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,c)<m,b||e,j>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,eakl,cdim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,j>_aaaa*t2_aaaa(e,a,i,k)*t2_aaaa(c,d,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,eaik,cdlm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,c)<m,b||e,i>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(c,d,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbei,eakl,cdjm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,l>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,eajk,bdim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,c||e,l>_aaaa*t2_aaaa(e,a,i,j)*t2_aaaa(b,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,eaij,bdkm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,l>_aaaa*t2_aaaa(e,d,j,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edjk,abim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,l>_aaaa*t2_aaaa(e,d,i,j)*t2_aaaa(a,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edij,abkm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,k>_aaaa*t2_aaaa(e,a,j,l)*t2_aaaa(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,eajl,bdim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,k>_aaaa*t2_aaaa(e,d,j,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,edjl,abim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,c||e,j>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,eakl,bdim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,j>_aaaa*t2_aaaa(e,a,i,k)*t2_aaaa(b,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,eaik,bdlm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)<m,c||e,j>_aaaa*t2_aaaa(e,d,k,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edkl,abim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,j>_aaaa*t2_aaaa(e,d,i,k)*t2_aaaa(a,b,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edik,ablm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,c||e,i>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(b,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,eakl,bdjm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<m,c||e,i>_aaaa*t2_aaaa(e,d,k,l)*t2_aaaa(a,b,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,edkl,abjm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,l>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eajk,bcim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,d||e,l>_aaaa*t2_aaaa(e,a,i,j)*t2_aaaa(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eaij,bckm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,k>_aaaa*t2_aaaa(e,a,j,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdek,eajl,bcim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,d||e,j>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,eakl,bcim->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,j>_aaaa*t2_aaaa(e,a,i,k)*t2_aaaa(b,c,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,eaik,bclm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,d||e,i>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdei,eakl,bcjm->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<a,b||e,f>_aaaa*t2_aaaa(e,c,k,l)*t2_aaaa(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abef,eckl,fdij->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<a,b||e,f>_aaaa*t2_aaaa(e,c,i,l)*t2_aaaa(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecil,fdjk->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<a,b||e,f>_aaaa*t2_aaaa(e,c,j,k)*t2_aaaa(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecjk,fdil->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<a,d||e,f>_aaaa*t2_aaaa(e,b,k,l)*t2_aaaa(f,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebkl,fcij->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<a,d||e,f>_aaaa*t2_aaaa(e,b,i,l)*t2_aaaa(f,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebil,fcjk->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<a,d||e,f>_aaaa*t2_aaaa(e,b,j,k)*t2_aaaa(f,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebjk,fcil->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,d)<b,c||e,f>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eakl,fdij->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,d)<b,c||e,f>_aaaa*t2_aaaa(e,a,i,l)*t2_aaaa(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eail,fdjk->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,d)<b,c||e,f>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eajk,fdil->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbkjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,f,k,l)*t2_aaaa(a,b,j,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efkl,abjm,cdin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,f,k,l)*t2_aaaa(a,d,j,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efkl,adjm,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,l)*t2_aaaa(a,b,k,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efjl,abkm,cdin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,l)*t2_aaaa(a,d,k,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efjl,adkm,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,l)*t2_aaaa(a,b,k,m)*t2_aaaa(c,d,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efil,abkm,cdjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,l)*t2_aaaa(a,d,k,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efil,adkm,bcjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(i,l)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,k)*t2_aaaa(a,b,l,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efjk,ablm,cdin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,l)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,k)*t2_aaaa(a,d,l,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efjk,adlm,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  0.5000 P(j,l)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,k)*t2_aaaa(a,b,l,m)*t2_aaaa(c,d,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efik,ablm,cdjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  0.5000 P(j,l)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,k)*t2_aaaa(a,d,l,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efik,adlm,bcjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,j)*t2_aaaa(a,b,l,m)*t2_aaaa(c,d,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efij,ablm,cdkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,j)*t2_aaaa(a,d,l,m)*t2_aaaa(b,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efij,adlm,bckn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,l,m)*t2_aaaa(f,b,j,k)*t2_aaaa(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fbjk,cdin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||f,e>_abab*t2_abab(a,e,l,m)*t2_aaaa(f,b,j,k)*t2_aaaa(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aelm,fbjk,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,l,m)*t2_aaaa(f,b,i,j)*t2_aaaa(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fbij,cdkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||f,e>_abab*t2_abab(a,e,l,m)*t2_aaaa(f,b,i,j)*t2_aaaa(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aelm,fbij,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,l,m)*t2_aaaa(f,d,j,k)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fdjk,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,l,m)*t2_aaaa(f,d,j,k)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aelm,fdjk,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,l,m)*t2_aaaa(f,d,i,j)*t2_aaaa(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fdij,bckn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,l,m)*t2_aaaa(f,d,i,j)*t2_aaaa(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aelm,fdij,bckn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,m)*t2_aaaa(f,b,j,l)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eakm,fbjl,cdin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||f,e>_abab*t2_abab(a,e,k,m)*t2_aaaa(f,b,j,l)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aekm,fbjl,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,m)*t2_aaaa(f,d,j,l)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eakm,fdjl,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,k,m)*t2_aaaa(f,d,j,l)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aekm,fdjl,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_aaaa(f,b,k,l)*t2_aaaa(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fbkl,cdin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_aaaa(f,b,k,l)*t2_aaaa(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aejm,fbkl,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_aaaa(f,b,i,k)*t2_aaaa(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fbik,cdln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_aaaa(f,b,i,k)*t2_aaaa(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aejm,fbik,cdln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_aaaa(f,d,k,l)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fdkl,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_aaaa(f,d,k,l)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aejm,fdkl,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_aaaa(f,d,i,k)*t2_aaaa(b,c,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fdik,bcln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_aaaa(f,d,i,k)*t2_aaaa(b,c,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aejm,fdik,bcln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,m)*t2_aaaa(f,b,k,l)*t2_aaaa(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaim,fbkl,cdjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||f,e>_abab*t2_abab(a,e,i,m)*t2_aaaa(f,b,k,l)*t2_aaaa(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aeim,fbkl,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,m)*t2_aaaa(f,d,k,l)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaim,fdkl,bcjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,i,m)*t2_aaaa(f,d,k,l)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aeim,fdkl,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(f,b,j,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eakl,fbjm,cdin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,k,l)*t2_abab(b,f,j,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eakl,bfjm,cdin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(f,b,i,j)*t2_aaaa(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eakl,fbij,cdnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(f,d,j,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eakl,fdjm,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,k,l)*t2_abab(d,f,j,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eakl,dfjm,bcin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(f,d,i,j)*t2_aaaa(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eakl,fdij,bcnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,l)*t2_aaaa(f,b,k,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajl,fbkm,cdin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,j,l)*t2_abab(b,f,k,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajl,bfkm,cdin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,l)*t2_aaaa(f,d,k,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajl,fdkm,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,j,l)*t2_abab(d,f,k,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajl,dfkm,bcin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,l)*t2_aaaa(f,b,k,m)*t2_aaaa(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eail,fbkm,cdjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,i,l)*t2_abab(b,f,k,m)*t2_aaaa(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eail,bfkm,cdjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,l)*t2_aaaa(f,b,j,k)*t2_aaaa(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eail,fbjk,cdnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,l)*t2_aaaa(f,d,k,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eail,fdkm,bcjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,i,l)*t2_abab(d,f,k,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eail,dfkm,bcjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,l)*t2_aaaa(f,d,j,k)*t2_aaaa(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eail,fdjk,bcnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,l)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(f,b,l,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajk,fblm,cdin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)*P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,j,k)*t2_abab(b,f,l,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajk,bflm,cdin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(f,b,i,l)*t2_aaaa(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eajk,fbil,cdnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,l)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(f,d,l,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajk,fdlm,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdljki', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,j,k)*t2_abab(d,f,l,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajk,dflm,bcin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(f,d,i,l)*t2_aaaa(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eajk,fdil,bcnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,k)*t2_aaaa(f,b,l,m)*t2_aaaa(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaik,fblm,cdjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,i,k)*t2_abab(b,f,l,m)*t2_aaaa(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaik,bflm,cdjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,k)*t2_aaaa(f,d,l,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaik,fdlm,bcjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,i,k)*t2_abab(d,f,l,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaik,dflm,bcjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,j)*t2_aaaa(f,b,l,m)*t2_aaaa(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fblm,cdkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,m)*t2_aaaa(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaij,bflm,cdkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,j)*t2_aaaa(f,d,l,m)*t2_aaaa(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fdlm,bckn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,i,j)*t2_abab(d,f,l,m)*t2_aaaa(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaij,dflm,bckn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,b,l,m)*t2_aaaa(f,c,j,k)*t2_aaaa(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eblm,fcjk,adin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(b,e,l,m)*t2_aaaa(f,c,j,k)*t2_aaaa(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,belm,fcjk,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,b,l,m)*t2_aaaa(f,c,i,j)*t2_aaaa(a,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eblm,fcij,adkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||f,e>_abab*t2_abab(b,e,l,m)*t2_aaaa(f,c,i,j)*t2_aaaa(a,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,belm,fcij,adkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,b,k,m)*t2_aaaa(f,c,j,l)*t2_aaaa(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebkm,fcjl,adin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(b,e,k,m)*t2_aaaa(f,c,j,l)*t2_aaaa(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,bekm,fcjl,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,b,j,m)*t2_aaaa(f,c,k,l)*t2_aaaa(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjm,fckl,adin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||f,e>_abab*t2_abab(b,e,j,m)*t2_aaaa(f,c,k,l)*t2_aaaa(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,bejm,fckl,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,b,j,m)*t2_aaaa(f,c,i,k)*t2_aaaa(a,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjm,fcik,adln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(b,e,j,m)*t2_aaaa(f,c,i,k)*t2_aaaa(a,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,bejm,fcik,adln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_aaaa*t2_aaaa(e,b,i,m)*t2_aaaa(f,c,k,l)*t2_aaaa(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebim,fckl,adjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||f,e>_abab*t2_abab(b,e,i,m)*t2_aaaa(f,c,k,l)*t2_aaaa(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,beim,fckl,adjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,b,k,l)*t2_aaaa(f,c,j,m)*t2_aaaa(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebkl,fcjm,adin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_abab*t2_aaaa(e,b,k,l)*t2_abab(c,f,j,m)*t2_aaaa(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebkl,cfjm,adin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_aaaa*t2_aaaa(e,b,k,l)*t2_aaaa(f,c,i,j)*t2_aaaa(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebkl,fcij,adnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,b,j,l)*t2_aaaa(f,c,k,m)*t2_aaaa(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebjl,fckm,adin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>_abab*t2_aaaa(e,b,j,l)*t2_abab(c,f,k,m)*t2_aaaa(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjl,cfkm,adin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>_aaaa*t2_aaaa(e,b,i,l)*t2_aaaa(f,c,k,m)*t2_aaaa(a,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebil,fckm,adjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_abab*t2_aaaa(e,b,i,l)*t2_abab(c,f,k,m)*t2_aaaa(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebil,cfkm,adjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,b,i,l)*t2_aaaa(f,c,j,k)*t2_aaaa(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebil,fcjk,adnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,l)<n,m||e,f>_aaaa*t2_aaaa(e,b,j,k)*t2_aaaa(f,c,l,m)*t2_aaaa(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjk,fclm,adin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)<n,m||e,f>_abab*t2_aaaa(e,b,j,k)*t2_abab(c,f,l,m)*t2_aaaa(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebjk,cflm,adin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,b,j,k)*t2_aaaa(f,c,i,l)*t2_aaaa(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebjk,fcil,adnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<n,m||e,f>_aaaa*t2_aaaa(e,b,i,k)*t2_aaaa(f,c,l,m)*t2_aaaa(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebik,fclm,adjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||e,f>_abab*t2_aaaa(e,b,i,k)*t2_abab(c,f,l,m)*t2_aaaa(a,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebik,cflm,adjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,b,i,j)*t2_aaaa(f,c,l,m)*t2_aaaa(a,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebij,fclm,adkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_abab*t2_aaaa(e,b,i,j)*t2_abab(c,f,l,m)*t2_aaaa(a,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebij,cflm,adkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,c,l,m)*t2_aaaa(f,d,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eclm,fdjk,abin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(c,e,l,m)*t2_aaaa(f,d,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,celm,fdjk,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,c,l,m)*t2_aaaa(f,d,i,j)*t2_aaaa(a,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eclm,fdij,abkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||f,e>_abab*t2_abab(c,e,l,m)*t2_aaaa(f,d,i,j)*t2_aaaa(a,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,celm,fdij,abkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,c,k,m)*t2_aaaa(f,d,j,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eckm,fdjl,abin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(c,e,k,m)*t2_aaaa(f,d,j,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,cekm,fdjl,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,m)*t2_aaaa(f,d,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjm,fdkl,abin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||f,e>_abab*t2_abab(c,e,j,m)*t2_aaaa(f,d,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,cejm,fdkl,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,m)*t2_aaaa(f,d,i,k)*t2_aaaa(a,b,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjm,fdik,abln->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(c,e,j,m)*t2_aaaa(f,d,i,k)*t2_aaaa(a,b,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,cejm,fdik,abln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,i,m)*t2_aaaa(f,d,k,l)*t2_aaaa(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecim,fdkl,abjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||f,e>_abab*t2_abab(c,e,i,m)*t2_aaaa(f,d,k,l)*t2_aaaa(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,ceim,fdkl,abjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,c,k,l)*t2_aaaa(f,d,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eckl,fdjm,abin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_abab*t2_aaaa(e,c,k,l)*t2_abab(d,f,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eckl,dfjm,abin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,k,l)*t2_aaaa(f,d,i,j)*t2_aaaa(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eckl,fdij,abnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,l)*t2_aaaa(f,d,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecjl,fdkm,abin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>_abab*t2_aaaa(e,c,j,l)*t2_abab(d,f,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjl,dfkm,abin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,i,l)*t2_aaaa(f,d,k,m)*t2_aaaa(a,b,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecil,fdkm,abjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_abab*t2_aaaa(e,c,i,l)*t2_abab(d,f,k,m)*t2_aaaa(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecil,dfkm,abjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,c,i,l)*t2_aaaa(f,d,j,k)*t2_aaaa(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ecil,fdjk,abnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,l)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,k)*t2_aaaa(f,d,l,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjk,fdlm,abin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)<n,m||e,f>_abab*t2_aaaa(e,c,j,k)*t2_abab(d,f,l,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecjk,dflm,abin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,k)*t2_aaaa(f,d,i,l)*t2_aaaa(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ecjk,fdil,abnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<n,m||e,f>_aaaa*t2_aaaa(e,c,i,k)*t2_aaaa(f,d,l,m)*t2_aaaa(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecik,fdlm,abjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||e,f>_abab*t2_aaaa(e,c,i,k)*t2_abab(d,f,l,m)*t2_aaaa(a,b,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecik,dflm,abjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,c,i,j)*t2_aaaa(f,d,l,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecij,fdlm,abkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_abab*t2_aaaa(e,c,i,j)*t2_abab(d,f,l,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecij,dflm,abkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    return quadruples_res


def ccdq_t4_aaabaaab_residual(t2_aaaa, t2_bbbb, t2_abab, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_bb(m,l)*t4_aaabaaab(a,b,c,d,i,j,k,m)
    quadruples_res = -1.000000000000010 * einsum('ml,abcdijkm->abcdijkl', f_bb[o, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 f_aa(m,k)*t4_aaabaaab(a,b,c,d,i,j,m,l)
    quadruples_res += -1.000000000000010 * einsum('mk,abcdijml->abcdijkl', f_aa[o, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  1.0000 P(i,j)f_aa(m,j)*t4_aaabaaab(a,b,c,d,i,k,m,l)
    contracted_intermediate =  1.000000000000010 * einsum('mj,abcdikml->abcdijkl', f_aa[o, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f_aa(a,e)*t4_aaabaaab(e,b,c,d,i,j,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('ae,ebcdijkl->abcdijkl', f_aa[v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 f_aa(c,e)*t4_aaabaaab(e,a,b,d,i,j,k,l)
    quadruples_res +=  1.000000000000010 * einsum('ce,eabdijkl->abcdijkl', f_aa[v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  1.0000 f_bb(d,e)*t4_aaabaaab(c,a,b,e,i,j,k,l)
    quadruples_res +=  1.000000000000010 * einsum('de,cabeijkl->abcdijkl', f_bb[v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  0.5000 P(j,k)<n,m||k,l>_abab*t4_aaabaaab(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmkl,abcdijnm->abcdijkl', g_abab[o, o, o, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<m,n||k,l>_abab*t4_aaabaaab(a,b,c,d,i,j,m,n)
    contracted_intermediate =  0.500000000000010 * einsum('mnkl,abcdijmn->abcdijkl', g_abab[o, o, o, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 <n,m||i,l>_abab*t4_aaabaaab(a,b,c,d,j,k,n,m)
    quadruples_res +=  0.500000000000010 * einsum('nmil,abcdjknm->abcdijkl', g_abab[o, o, o, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <m,n||i,l>_abab*t4_aaabaaab(a,b,c,d,j,k,m,n)
    quadruples_res +=  0.500000000000010 * einsum('mnil,abcdjkmn->abcdijkl', g_abab[o, o, o, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <n,m||i,k>_aaaa*t4_aaabaaab(a,b,c,d,j,m,n,l)
    quadruples_res +=  0.500000000000010 * einsum('nmik,abcdjmnl->abcdijkl', g_aaaa[o, o, o, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 P(i,k)<n,m||j,k>_aaaa*t4_aaabaaab(a,b,c,d,i,m,n,l)
    contracted_intermediate = -0.500000000000010 * einsum('nmjk,abcdimnl->abcdijkl', g_aaaa[o, o, o, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<a,m||e,l>_abab*t4_aaabaaab(e,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('amel,ebcdijkm->abcdijkl', g_abab[v, o, v, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,a||e,k>_aaaa*t4_aaabaaab(e,b,c,d,i,j,m,l)
    contracted_intermediate =  1.000000000000020 * einsum('maek,ebcdijml->abcdijkl', g_aaaa[o, v, v, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<a,m||k,e>_abab*t4_aabbaabb(c,b,e,d,i,j,l,m)
    contracted_intermediate =  1.000000000000020 * einsum('amke,cbedijlm->abcdijkl', g_abab[v, o, o, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*t4_aaabaaab(e,b,c,d,i,k,m,l)
    contracted_intermediate = -1.000000000000020 * einsum('maej,ebcdikml->abcdijkl', g_aaaa[o, v, v, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,m||j,e>_abab*t4_aabbaabb(c,b,e,d,i,k,l,m)
    contracted_intermediate = -1.000000000000020 * einsum('amje,cbediklm->abcdijkl', g_abab[v, o, o, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 <c,m||e,l>_abab*t4_aaabaaab(e,a,b,d,i,j,k,m)
    quadruples_res += -1.000000000000020 * einsum('cmel,eabdijkm->abcdijkl', g_abab[v, o, v, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 <m,d||e,l>_abab*t4_aaaaaaaa(e,a,b,c,i,j,k,m)
    quadruples_res += -1.000000000000020 * einsum('mdel,eabcijkm->abcdijkl', g_abab[o, v, v, o], t4_aaaaaaaa[:, :, :, :, :, :, :, :])
    
    #	  1.0000 <m,d||e,l>_bbbb*t4_aaabaaab(c,a,b,e,i,j,k,m)
    quadruples_res +=  1.000000000000020 * einsum('mdel,cabeijkm->abcdijkl', g_bbbb[o, v, v, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  1.0000 <m,c||e,k>_aaaa*t4_aaabaaab(e,a,b,d,i,j,m,l)
    quadruples_res +=  1.000000000000020 * einsum('mcek,eabdijml->abcdijkl', g_aaaa[o, v, v, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  1.0000 <c,m||k,e>_abab*t4_aabbaabb(b,a,e,d,i,j,l,m)
    quadruples_res +=  1.000000000000020 * einsum('cmke,baedijlm->abcdijkl', g_abab[v, o, o, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 <m,d||k,e>_abab*t4_aaabaaab(c,a,b,e,i,j,m,l)
    quadruples_res += -1.000000000000020 * einsum('mdke,cabeijml->abcdijkl', g_abab[o, v, o, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 P(i,j)<m,c||e,j>_aaaa*t4_aaabaaab(e,a,b,d,i,k,m,l)
    contracted_intermediate = -1.000000000000020 * einsum('mcej,eabdikml->abcdijkl', g_aaaa[o, v, v, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<c,m||j,e>_abab*t4_aabbaabb(b,a,e,d,i,k,l,m)
    contracted_intermediate = -1.000000000000020 * einsum('cmje,baediklm->abcdijkl', g_abab[v, o, o, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,d||j,e>_abab*t4_aaabaaab(c,a,b,e,i,k,m,l)
    contracted_intermediate =  1.000000000000020 * einsum('mdje,cabeikml->abcdijkl', g_abab[o, v, o, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<a,b||e,f>_aaaa*t4_aaabaaab(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('abef,efcdijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<a,d||e,f>_abab*t4_aaabaaab(e,c,b,f,i,j,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('adef,ecbfijkl->abcdijkl', g_abab[v, v, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<a,d||f,e>_abab*t4_aaabaaab(c,f,b,e,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('adfe,cfbeijkl->abcdijkl', g_abab[v, v, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 <b,c||e,f>_aaaa*t4_aaabaaab(e,f,a,d,i,j,k,l)
    quadruples_res +=  0.500000000000010 * einsum('bcef,efadijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 <c,d||e,f>_abab*t4_aaabaaab(e,b,a,f,i,j,k,l)
    quadruples_res += -0.500000000000010 * einsum('cdef,ebafijkl->abcdijkl', g_abab[v, v, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <c,d||f,e>_abab*t4_aaabaaab(b,f,a,e,i,j,k,l)
    quadruples_res +=  0.500000000000010 * einsum('cdfe,bfaeijkl->abcdijkl', g_abab[v, v, v, v], t4_aaabaaab[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,f,m,l)*t4_aaabaaab(a,b,c,d,i,j,k,n)
    quadruples_res += -0.499999999999950 * einsum('mnef,efml,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(f,e,m,l)*t4_aaabaaab(a,b,c,d,i,j,k,n)
    quadruples_res += -0.499999999999950 * einsum('mnfe,feml,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*t2_bbbb(e,f,l,m)*t4_aaabaaab(a,b,c,d,i,j,k,n)
    quadruples_res += -0.499999999999950 * einsum('nmef,eflm,abcdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*t2_aaaa(e,f,k,m)*t4_aaabaaab(a,b,c,d,i,j,n,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,efkm,abcdijnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,f,k,m)*t4_aaabaaab(a,b,c,d,i,j,n,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,efkm,abcdijnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(f,e,k,m)*t4_aaabaaab(a,b,c,d,i,j,n,l)
    quadruples_res += -0.499999999999950 * einsum('nmfe,fekm,abcdijnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,m)*t4_aaabaaab(a,b,c,d,i,k,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,efjm,abcdiknl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*t2_abab(e,f,j,m)*t4_aaabaaab(a,b,c,d,i,k,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,efjm,abcdiknl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*t2_abab(f,e,j,m)*t4_aaabaaab(a,b,c,d,i,k,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,fejm,abcdiknl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.2500 P(j,k)<n,m||e,f>_abab*t2_abab(e,f,k,l)*t4_aaabaaab(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,efkl,abcdijnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.2500 P(j,k)<m,n||e,f>_abab*t2_abab(e,f,k,l)*t4_aaabaaab(a,b,c,d,i,j,m,n)
    contracted_intermediate =  0.250000000000010 * einsum('mnef,efkl,abcdijmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.2500 P(j,k)<n,m||f,e>_abab*t2_abab(f,e,k,l)*t4_aaabaaab(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.250000000000010 * einsum('nmfe,fekl,abcdijnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.2500 P(j,k)<m,n||f,e>_abab*t2_abab(f,e,k,l)*t4_aaabaaab(a,b,c,d,i,j,m,n)
    contracted_intermediate =  0.250000000000010 * einsum('mnfe,fekl,abcdijmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,f>_abab*t2_abab(e,f,i,l)*t4_aaabaaab(a,b,c,d,j,k,n,m)
    quadruples_res +=  0.250000000000010 * einsum('nmef,efil,abcdjknm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*t2_abab(e,f,i,l)*t4_aaabaaab(a,b,c,d,j,k,m,n)
    quadruples_res +=  0.250000000000010 * einsum('mnef,efil,abcdjkmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*t2_abab(f,e,i,l)*t4_aaabaaab(a,b,c,d,j,k,n,m)
    quadruples_res +=  0.250000000000010 * einsum('nmfe,feil,abcdjknm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*t2_abab(f,e,i,l)*t4_aaabaaab(a,b,c,d,j,k,m,n)
    quadruples_res +=  0.250000000000010 * einsum('mnfe,feil,abcdjkmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*t2_aaaa(e,f,i,k)*t4_aaabaaab(a,b,c,d,j,m,n,l)
    quadruples_res +=  0.250000000000010 * einsum('nmef,efik,abcdjmnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,k)*t4_aaabaaab(a,b,c,d,i,m,n,l)
    contracted_intermediate = -0.250000000000010 * einsum('nmef,efjk,abcdimnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,n,m)*t4_aaabaaab(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eanm,fbcdijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,n,m)*t4_aaabaaab(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,aenm,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,m,n)*t4_aaabaaab(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,aemn,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t4_aaabaaab(f,b,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnfe,aeml,fbcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,m)*t4_aaabaaab(f,b,c,d,i,j,n,l)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eakm,fbcdijnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,k,m)*t4_aabbaabb(c,b,f,d,i,j,l,n)
    contracted_intermediate = -0.999999999999840 * einsum('mnef,eakm,cbfdijln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,k,m)*t4_aaabaaab(f,b,c,d,i,j,n,l)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,aekm,fbcdijnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,k,m)*t4_aabbaabb(c,b,f,d,i,j,l,n)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,aekm,cbfdijln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t4_aaabaaab(f,b,c,d,i,k,n,l)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,eajm,fbcdiknl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,j,m)*t4_aabbaabb(c,b,f,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,eajm,cbfdikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t4_aaabaaab(f,b,c,d,i,k,n,l)
    contracted_intermediate = -0.999999999999840 * einsum('nmfe,aejm,fbcdiknl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,j,m)*t4_aabbaabb(c,b,f,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,aejm,cbfdikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,k,l)*t4_aaabaaab(f,b,c,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,aekl,fbcdijnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<m,n||f,e>_abab*t2_abab(a,e,k,l)*t4_aaabaaab(f,b,c,d,i,j,m,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,aekl,fbcdijmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,k,l)*t4_aabbaabb(c,b,f,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,aekl,cbfdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,i,l)*t4_aaabaaab(f,b,c,d,j,k,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,aeil,fbcdjknm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t4_aaabaaab(f,b,c,d,j,k,m,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,aeil,fbcdjkmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t4_aabbaabb(c,b,f,d,j,k,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,aeil,cbfdjknm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,k)*t4_aaabaaab(f,b,c,d,j,m,n,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eaik,fbcdjmnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,i,k)*t4_aabbaabb(c,b,f,d,j,n,l,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eaik,cbfdjnlm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,i,k)*t4_aabbaabb(c,b,f,d,j,m,n,l)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,eaik,cbfdjmnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,k)*t4_aaabaaab(f,b,c,d,i,m,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eajk,fbcdimnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,j,k)*t4_aabbaabb(c,b,f,d,i,n,l,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eajk,cbfdinlm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,j,k)*t4_aabbaabb(c,b,f,d,i,m,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,eajk,cbfdimnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.5000 <n,m||e,f>_aaaa*t2_aaaa(e,c,n,m)*t4_aaabaaab(f,a,b,d,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,ecnm,fabdijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(c,e,n,m)*t4_aaabaaab(f,a,b,d,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('nmfe,cenm,fabdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(c,e,m,n)*t4_aaabaaab(f,a,b,d,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('mnfe,cemn,fabdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,d,n,m)*t4_aaabaaab(c,a,b,f,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,ednm,cabfijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,d,m,n)*t4_aaabaaab(c,a,b,f,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('mnef,edmn,cabfijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*t2_bbbb(e,d,n,m)*t4_aaabaaab(c,a,b,f,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,ednm,cabfijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(c,e,m,l)*t4_aaabaaab(f,a,b,d,i,j,k,n)
    quadruples_res +=  0.999999999999840 * einsum('mnfe,ceml,fabdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_abab(e,d,m,l)*t4_aaaaaaaa(f,a,b,c,i,j,k,n)
    quadruples_res +=  0.999999999999840 * einsum('nmef,edml,fabcijkn->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*t2_abab(e,d,m,l)*t4_aaabaaab(c,a,b,f,i,j,k,n)
    quadruples_res +=  0.999999999999840 * einsum('mnef,edml,cabfijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,d,l,m)*t4_aaaaaaaa(f,a,b,c,i,j,k,n)
    quadruples_res +=  0.999999999999840 * einsum('nmfe,edlm,fabcijkn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,d,l,m)*t4_aaabaaab(c,a,b,f,i,j,k,n)
    quadruples_res +=  0.999999999999840 * einsum('nmef,edlm,cabfijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,c,k,m)*t4_aaabaaab(f,a,b,d,i,j,n,l)
    quadruples_res +=  0.999999999999840 * einsum('nmef,eckm,fabdijnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*t2_aaaa(e,c,k,m)*t4_aabbaabb(b,a,f,d,i,j,l,n)
    quadruples_res += -0.999999999999840 * einsum('mnef,eckm,bafdijln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_abab(c,e,k,m)*t4_aaabaaab(f,a,b,d,i,j,n,l)
    quadruples_res +=  0.999999999999840 * einsum('nmfe,cekm,fabdijnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(c,e,k,m)*t4_aabbaabb(b,a,f,d,i,j,l,n)
    quadruples_res += -0.999999999999840 * einsum('nmef,cekm,bafdijln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*t2_abab(e,d,k,m)*t4_aaabaaab(c,a,b,f,i,j,n,l)
    quadruples_res +=  0.999999999999840 * einsum('nmef,edkm,cabfijnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,m)*t4_aaabaaab(f,a,b,d,i,k,n,l)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,ecjm,fabdiknl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||e,f>_abab*t2_aaaa(e,c,j,m)*t4_aabbaabb(b,a,f,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,ecjm,bafdikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(c,e,j,m)*t4_aaabaaab(f,a,b,d,i,k,n,l)
    contracted_intermediate = -0.999999999999840 * einsum('nmfe,cejm,fabdiknl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_bbbb*t2_abab(c,e,j,m)*t4_aabbaabb(b,a,f,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,cejm,bafdikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_abab*t2_abab(e,d,j,m)*t4_aaabaaab(c,a,b,f,i,k,n,l)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,edjm,cabfiknl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||f,e>_abab*t2_abab(c,e,k,l)*t4_aaabaaab(f,a,b,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,cekl,fabdijnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<m,n||f,e>_abab*t2_abab(c,e,k,l)*t4_aaabaaab(f,a,b,d,i,j,m,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,cekl,fabdijmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_bbbb*t2_abab(c,e,k,l)*t4_aabbaabb(b,a,f,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,cekl,bafdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<n,m||e,f>_aaaa*t2_abab(e,d,k,l)*t4_aaaaaaaa(f,a,b,c,i,j,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,edkl,fabcijnm->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_abab*t2_abab(e,d,k,l)*t4_aaabaaab(c,a,b,f,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,edkl,cabfijnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<m,n||e,f>_abab*t2_abab(e,d,k,l)*t4_aaabaaab(c,a,b,f,i,j,m,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,edkl,cabfijmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(c,e,i,l)*t4_aaabaaab(f,a,b,d,j,k,n,m)
    quadruples_res += -0.499999999999950 * einsum('nmfe,ceil,fabdjknm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(c,e,i,l)*t4_aaabaaab(f,a,b,d,j,k,m,n)
    quadruples_res += -0.499999999999950 * einsum('mnfe,ceil,fabdjkmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*t2_abab(c,e,i,l)*t4_aabbaabb(b,a,f,d,j,k,n,m)
    quadruples_res += -0.499999999999950 * einsum('nmef,ceil,bafdjknm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*t2_abab(e,d,i,l)*t4_aaaaaaaa(f,a,b,c,j,k,n,m)
    quadruples_res +=  0.499999999999950 * einsum('nmef,edil,fabcjknm->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,d,i,l)*t4_aaabaaab(c,a,b,f,j,k,n,m)
    quadruples_res += -0.499999999999950 * einsum('nmef,edil,cabfjknm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,d,i,l)*t4_aaabaaab(c,a,b,f,j,k,m,n)
    quadruples_res += -0.499999999999950 * einsum('mnef,edil,cabfjkmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*t2_aaaa(e,c,i,k)*t4_aaabaaab(f,a,b,d,j,m,n,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,ecik,fabdjmnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_aaaa(e,c,i,k)*t4_aabbaabb(b,a,f,d,j,n,l,m)
    quadruples_res +=  0.499999999999950 * einsum('nmef,ecik,bafdjnlm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_aaaa(e,c,i,k)*t4_aabbaabb(b,a,f,d,j,m,n,l)
    quadruples_res += -0.499999999999950 * einsum('mnef,ecik,bafdjmnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,k)*t4_aaabaaab(f,a,b,d,i,m,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecjk,fabdimnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)<n,m||e,f>_abab*t2_aaaa(e,c,j,k)*t4_aabbaabb(b,a,f,d,i,n,l,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecjk,bafdinlm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<m,n||e,f>_abab*t2_aaaa(e,c,j,k)*t4_aabbaabb(b,a,f,d,i,m,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,ecjk,bafdimnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  0.2500 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(a,b,n,m)*t4_aaabaaab(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,abnm,efcdijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(a,b,k,m)*t4_aaabaaab(e,f,c,d,i,j,n,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,abkm,efcdijnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<m,n||e,f>_abab*t2_aaaa(a,b,k,m)*t4_aabbaabb(e,c,f,d,i,j,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,abkm,ecfdijln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<m,n||f,e>_abab*t2_aaaa(a,b,k,m)*t4_aabbaabb(c,f,e,d,i,j,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,abkm,cfedijln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(a,b,j,m)*t4_aaabaaab(e,f,c,d,i,k,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,abjm,efcdiknl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<m,n||e,f>_abab*t2_aaaa(a,b,j,m)*t4_aabbaabb(e,c,f,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,abjm,ecfdikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,c)<m,n||f,e>_abab*t2_aaaa(a,b,j,m)*t4_aabbaabb(c,f,e,d,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,abjm,cfedikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_abab*t2_abab(a,d,n,m)*t4_aaabaaab(e,c,b,f,i,j,k,l)
    contracted_intermediate = -0.250000000000010 * einsum('nmef,adnm,ecbfijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||f,e>_abab*t2_abab(a,d,n,m)*t4_aaabaaab(c,f,b,e,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmfe,adnm,cfbeijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,n||e,f>_abab*t2_abab(a,d,m,n)*t4_aaabaaab(e,c,b,f,i,j,k,l)
    contracted_intermediate = -0.250000000000010 * einsum('mnef,admn,ecbfijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,n||f,e>_abab*t2_abab(a,d,m,n)*t4_aaabaaab(c,f,b,e,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('mnfe,admn,cfbeijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||e,f>_aaaa*t2_abab(a,d,m,l)*t4_aaaaaaaa(e,f,b,c,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,adml,efbcijkn->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,n||e,f>_abab*t2_abab(a,d,m,l)*t4_aaabaaab(e,c,b,f,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,adml,ecbfijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||f,e>_abab*t2_abab(a,d,m,l)*t4_aaabaaab(c,f,b,e,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,adml,cfbeijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||e,f>_abab*t2_abab(a,d,k,m)*t4_aaabaaab(e,c,b,f,i,j,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,adkm,ecbfijnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||f,e>_abab*t2_abab(a,d,k,m)*t4_aaabaaab(c,f,b,e,i,j,n,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,adkm,cfbeijnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,d,k,m)*t4_aabbaabb(b,c,e,f,i,j,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,adkm,bcefijln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*t2_abab(a,d,j,m)*t4_aaabaaab(e,c,b,f,i,k,n,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adjm,ecbfiknl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,d,j,m)*t4_aaabaaab(c,f,b,e,i,k,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,adjm,cfbeiknl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_abab(a,d,j,m)*t4_aabbaabb(b,c,e,f,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adjm,bcefikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,f>_aaaa*t2_aaaa(b,c,n,m)*t4_aaabaaab(e,f,a,d,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('nmef,bcnm,efadijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*t2_abab(c,d,n,m)*t4_aaabaaab(e,b,a,f,i,j,k,l)
    quadruples_res += -0.250000000000010 * einsum('nmef,cdnm,ebafijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*t2_abab(c,d,n,m)*t4_aaabaaab(b,f,a,e,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('nmfe,cdnm,bfaeijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*t2_abab(c,d,m,n)*t4_aaabaaab(e,b,a,f,i,j,k,l)
    quadruples_res += -0.250000000000010 * einsum('mnef,cdmn,ebafijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*t2_abab(c,d,m,n)*t4_aaabaaab(b,f,a,e,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('mnfe,cdmn,bfaeijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*t2_abab(c,d,m,l)*t4_aaaaaaaa(e,f,a,b,i,j,k,n)
    quadruples_res +=  0.499999999999950 * einsum('nmef,cdml,efabijkn->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_abab(c,d,m,l)*t4_aaabaaab(e,b,a,f,i,j,k,n)
    quadruples_res +=  0.499999999999950 * einsum('mnef,cdml,ebafijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(c,d,m,l)*t4_aaabaaab(b,f,a,e,i,j,k,n)
    quadruples_res += -0.499999999999950 * einsum('mnfe,cdml,bfaeijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*t2_aaaa(b,c,k,m)*t4_aaabaaab(e,f,a,d,i,j,n,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,bckm,efadijnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_aaaa(b,c,k,m)*t4_aabbaabb(e,a,f,d,i,j,l,n)
    quadruples_res +=  0.499999999999950 * einsum('mnef,bckm,eafdijln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_aaaa(b,c,k,m)*t4_aabbaabb(a,f,e,d,i,j,l,n)
    quadruples_res += -0.499999999999950 * einsum('mnfe,bckm,afedijln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_abab(c,d,k,m)*t4_aaabaaab(e,b,a,f,i,j,n,l)
    quadruples_res +=  0.499999999999950 * einsum('nmef,cdkm,ebafijnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(c,d,k,m)*t4_aaabaaab(b,f,a,e,i,j,n,l)
    quadruples_res += -0.499999999999950 * einsum('nmfe,cdkm,bfaeijnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*t2_abab(c,d,k,m)*t4_aabbaabb(a,b,e,f,i,j,l,n)
    quadruples_res +=  0.499999999999950 * einsum('nmef,cdkm,abefijln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(b,c,j,m)*t4_aaabaaab(e,f,a,d,i,k,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,bcjm,efadiknl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*t2_aaaa(b,c,j,m)*t4_aabbaabb(e,a,f,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,bcjm,eafdikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*t2_aaaa(b,c,j,m)*t4_aabbaabb(a,f,e,d,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,bcjm,afedikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*t2_abab(c,d,j,m)*t4_aaabaaab(e,b,a,f,i,k,n,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,cdjm,ebafiknl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*t2_abab(c,d,j,m)*t4_aaabaaab(b,f,a,e,i,k,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,cdjm,bfaeiknl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*t2_abab(c,d,j,m)*t4_aabbaabb(a,b,e,f,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,cdjm,abefikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,n||k,l>_abab*t2_aaaa(a,b,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnkl,abjm,cdin->abcdijkl', g_abab[o, o, o, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||k,l>_abab*t2_abab(a,d,j,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,adjm,bcin->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<m,n||j,l>_abab*t2_aaaa(a,b,k,m)*t2_abab(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnjl,abkm,cdin->abcdijkl', g_abab[o, o, o, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||j,l>_abab*t2_abab(a,d,k,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,adkm,bcin->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<m,n||i,l>_abab*t2_aaaa(a,b,k,m)*t2_abab(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnil,abkm,cdjn->abcdijkl', g_abab[o, o, o, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||i,l>_abab*t2_abab(a,d,k,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,adkm,bcjn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<n,m||j,k>_aaaa*t2_aaaa(a,b,i,m)*t2_abab(c,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,abim,cdnl->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 <n,m||j,k>_aaaa*t2_abab(a,d,m,l)*t2_aaaa(b,c,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmjk,adml,bcin->abcdijkl', g_aaaa[o, o, o, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(b,c)<n,m||i,k>_aaaa*t2_aaaa(a,b,j,m)*t2_abab(c,d,n,l)
    contracted_intermediate =  1.000000000000000 * einsum('nmik,abjm,cdnl->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 <n,m||i,k>_aaaa*t2_abab(a,d,m,l)*t2_aaaa(b,c,j,n)
    quadruples_res += -1.000000000000000 * einsum('nmik,adml,bcjn->abcdijkl', g_aaaa[o, o, o, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(b,c)<n,m||i,j>_aaaa*t2_aaaa(a,b,k,m)*t2_abab(c,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,abkm,cdnl->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 <n,m||i,j>_aaaa*t2_abab(a,d,m,l)*t2_aaaa(b,c,k,n)
    quadruples_res +=  1.000000000000000 * einsum('nmij,adml,bckn->abcdijkl', g_aaaa[o, o, o, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(b,c)<a,m||e,l>_abab*t2_aaaa(e,b,j,k)*t2_abab(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('amel,ebjk,cdim->abcdijkl', g_abab[v, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<a,m||e,l>_abab*t2_aaaa(e,b,i,j)*t2_abab(c,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('amel,ebij,cdkm->abcdijkl', g_abab[v, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<m,a||e,k>_aaaa*t2_aaaa(e,b,i,j)*t2_abab(c,d,m,l)
    contracted_intermediate =  1.000000000000000 * einsum('maek,ebij,cdml->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<a,m||k,e>_abab*t2_abab(b,e,j,l)*t2_abab(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('amke,bejl,cdim->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,k>_aaaa*t2_abab(e,d,j,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,edjl,bcim->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<a,m||j,e>_abab*t2_abab(b,e,k,l)*t2_abab(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('amje,bekl,cdim->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,a||e,j>_aaaa*t2_aaaa(e,b,i,k)*t2_abab(c,d,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('maej,ebik,cdml->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,a||e,j>_aaaa*t2_abab(e,d,k,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edkl,bcim->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<a,m||i,e>_abab*t2_abab(b,e,k,l)*t2_abab(c,d,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('amie,bekl,cdjm->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,a||e,i>_aaaa*t2_abab(e,d,k,l)*t2_aaaa(b,c,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,edkl,bcjm->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<b,m||e,l>_abab*t2_aaaa(e,a,j,k)*t2_abab(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('bmel,eajk,cdim->abcdijkl', g_abab[v, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,c)<b,m||e,l>_abab*t2_aaaa(e,a,i,j)*t2_abab(c,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('bmel,eaij,cdkm->abcdijkl', g_abab[v, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,c)<m,b||e,k>_aaaa*t2_aaaa(e,a,i,j)*t2_abab(c,d,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('mbek,eaij,cdml->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<b,m||k,e>_abab*t2_abab(a,e,j,l)*t2_abab(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('bmke,aejl,cdim->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,c)<b,m||j,e>_abab*t2_abab(a,e,k,l)*t2_abab(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('bmje,aekl,cdim->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<m,b||e,j>_aaaa*t2_aaaa(e,a,i,k)*t2_abab(c,d,m,l)
    contracted_intermediate =  1.000000000000000 * einsum('mbej,eaik,cdml->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,c)<b,m||i,e>_abab*t2_abab(a,e,k,l)*t2_abab(c,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('bmie,aekl,cdjm->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<c,m||e,l>_abab*t2_aaaa(e,a,j,k)*t2_abab(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('cmel,eajk,bdim->abcdijkl', g_abab[v, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<c,m||e,l>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('cmel,eaij,bdkm->abcdijkl', g_abab[v, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,c||e,k>_aaaa*t2_aaaa(e,a,i,j)*t2_abab(b,d,m,l)
    contracted_intermediate =  1.000000000000000 * einsum('mcek,eaij,bdml->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||e,l>_abab*t2_aaaa(e,c,j,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,ecjk,abim->abcdijkl', g_abab[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 <m,d||e,l>_abab*t2_aaaa(e,c,i,j)*t2_aaaa(a,b,k,m)
    quadruples_res += -1.000000000000000 * einsum('mdel,ecij,abkm->abcdijkl', g_abab[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(i,j)*P(a,b)<c,m||k,e>_abab*t2_abab(a,e,j,l)*t2_abab(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('cmke,aejl,bdim->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,c||e,k>_aaaa*t2_abab(e,d,j,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,edjl,abim->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,d||k,e>_abab*t2_abab(c,e,j,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdke,cejl,abim->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<c,m||j,e>_abab*t2_abab(a,e,k,l)*t2_abab(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('cmje,aekl,bdim->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,j>_aaaa*t2_aaaa(e,a,i,k)*t2_abab(b,d,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('mcej,eaik,bdml->abcdijkl', g_aaaa[o, v, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<m,c||e,j>_aaaa*t2_abab(e,d,k,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edkl,abim->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<m,d||j,e>_abab*t2_abab(c,e,k,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdje,cekl,abim->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<c,m||i,e>_abab*t2_abab(a,e,k,l)*t2_abab(b,d,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('cmie,aekl,bdjm->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,c||e,i>_aaaa*t2_abab(e,d,k,l)*t2_aaaa(a,b,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,edkl,abjm->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,d||i,e>_abab*t2_abab(c,e,k,l)*t2_aaaa(a,b,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdie,cekl,abjm->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,l>_abab*t2_aaaa(e,a,j,k)*t2_aaaa(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eajk,bcim->abcdijkl', g_abab[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,d||e,l>_abab*t2_aaaa(e,a,i,j)*t2_aaaa(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eaij,bckm->abcdijkl', g_abab[o, v, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||k,e>_abab*t2_abab(a,e,j,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdke,aejl,bcim->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,d||j,e>_abab*t2_abab(a,e,k,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdje,aekl,bcim->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,d||i,e>_abab*t2_abab(a,e,k,l)*t2_aaaa(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdie,aekl,bcjm->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<a,b||e,f>_aaaa*t2_aaaa(e,c,i,k)*t2_abab(f,d,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('abef,ecik,fdjl->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<a,b||e,f>_aaaa*t2_aaaa(e,c,j,k)*t2_abab(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecjk,fdil->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<a,d||f,e>_abab*t2_abab(b,e,k,l)*t2_aaaa(f,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('adfe,bekl,fcij->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<a,d||f,e>_abab*t2_abab(b,e,i,l)*t2_aaaa(f,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adfe,beil,fcjk->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<a,d||e,f>_abab*t2_aaaa(e,b,i,k)*t2_abab(c,f,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebik,cfjl->abcdijkl', g_abab[v, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<a,d||e,f>_abab*t2_aaaa(e,b,j,k)*t2_abab(c,f,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('adef,ebjk,cfil->abcdijkl', g_abab[v, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<c,d||f,e>_abab*t2_abab(a,e,k,l)*t2_aaaa(f,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('cdfe,aekl,fbij->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 <c,d||f,e>_abab*t2_abab(a,e,i,l)*t2_aaaa(f,b,j,k)
    quadruples_res += -1.000000000000000 * einsum('cdfe,aeil,fbjk->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,c||e,f>_aaaa*t2_aaaa(e,a,i,k)*t2_abab(f,d,j,l)
    quadruples_res +=  1.000000000000000 * einsum('bcef,eaik,fdjl->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <c,d||e,f>_abab*t2_aaaa(e,a,i,k)*t2_abab(b,f,j,l)
    quadruples_res += -1.000000000000000 * einsum('cdef,eaik,bfjl->abcdijkl', g_abab[v, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,k)<b,c||e,f>_aaaa*t2_aaaa(e,a,j,k)*t2_abab(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eajk,fdil->abcdijkl', g_aaaa[v, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<c,d||e,f>_abab*t2_aaaa(e,a,j,k)*t2_abab(b,f,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('cdef,eajk,bfil->abcdijkl', g_abab[v, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,c)<m,n||e,f>_abab*t2_abab(e,f,k,l)*t2_aaaa(a,b,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,efkl,abjm,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,c)<m,n||f,e>_abab*t2_abab(f,e,k,l)*t2_aaaa(a,b,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,fekl,abjm,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*t2_abab(e,f,k,l)*t2_abab(a,d,j,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efkl,adjm,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*t2_abab(f,e,k,l)*t2_abab(a,d,j,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,fekl,adjm,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(b,c)<m,n||e,f>_abab*t2_abab(e,f,j,l)*t2_aaaa(a,b,k,m)*t2_abab(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,efjl,abkm,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(b,c)<m,n||f,e>_abab*t2_abab(f,e,j,l)*t2_aaaa(a,b,k,m)*t2_abab(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,fejl,abkm,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<n,m||e,f>_abab*t2_abab(e,f,j,l)*t2_abab(a,d,k,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efjl,adkm,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<n,m||f,e>_abab*t2_abab(f,e,j,l)*t2_abab(a,d,k,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,fejl,adkm,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(b,c)<m,n||e,f>_abab*t2_abab(e,f,i,l)*t2_aaaa(a,b,k,m)*t2_abab(c,d,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,efil,abkm,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(b,c)<m,n||f,e>_abab*t2_abab(f,e,i,l)*t2_aaaa(a,b,k,m)*t2_abab(c,d,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,feil,abkm,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_abab*t2_abab(e,f,i,l)*t2_abab(a,d,k,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efil,adkm,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||f,e>_abab*t2_abab(f,e,i,l)*t2_abab(a,d,k,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,feil,adkm,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,k)*t2_aaaa(a,b,i,m)*t2_abab(c,d,n,l)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efjk,abim,cdnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 <n,m||e,f>_aaaa*t2_aaaa(e,f,j,k)*t2_abab(a,d,m,l)*t2_aaaa(b,c,i,n)
    quadruples_res +=  0.500000000000000 * einsum('nmef,efjk,adml,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,k)*t2_aaaa(a,b,j,m)*t2_abab(c,d,n,l)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efik,abjm,cdnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 <n,m||e,f>_aaaa*t2_aaaa(e,f,i,k)*t2_abab(a,d,m,l)*t2_aaaa(b,c,j,n)
    quadruples_res += -0.500000000000000 * einsum('nmef,efik,adml,bcjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,j)*t2_aaaa(a,b,k,m)*t2_abab(c,d,n,l)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efij,abkm,cdnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 <n,m||e,f>_aaaa*t2_aaaa(e,f,i,j)*t2_abab(a,d,m,l)*t2_aaaa(b,c,k,n)
    quadruples_res +=  0.500000000000000 * einsum('nmef,efij,adml,bckn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(i,j)*P(b,c)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t2_aaaa(f,b,j,k)*t2_abab(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeml,fbjk,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t2_aaaa(f,b,i,j)*t2_abab(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeml,fbij,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,m)*t2_aaaa(f,b,i,j)*t2_abab(c,d,n,l)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eakm,fbij,cdnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<n,m||f,e>_abab*t2_abab(a,e,k,m)*t2_aaaa(f,b,i,j)*t2_abab(c,d,n,l)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aekm,fbij,cdnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,k,m)*t2_abab(b,f,j,l)*t2_abab(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eakm,bfjl,cdin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,k,m)*t2_abab(b,f,j,l)*t2_abab(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aekm,bfjl,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,k,m)*t2_abab(f,d,j,l)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eakm,fdjl,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,k,m)*t2_abab(f,d,j,l)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aekm,fdjl,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,j,m)*t2_abab(b,f,k,l)*t2_abab(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eajm,bfkl,cdin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,j,m)*t2_abab(b,f,k,l)*t2_abab(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aejm,bfkl,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_aaaa(f,b,i,k)*t2_abab(c,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajm,fbik,cdnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_aaaa(f,b,i,k)*t2_abab(c,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aejm,fbik,cdnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_abab(f,d,k,l)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fdkl,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_abab(f,d,k,l)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aejm,fdkl,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,i,m)*t2_abab(b,f,k,l)*t2_abab(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eaim,bfkl,cdjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,i,m)*t2_abab(b,f,k,l)*t2_abab(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aeim,bfkl,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,m)*t2_abab(f,d,k,l)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaim,fdkl,bcjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,i,m)*t2_abab(f,d,k,l)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aeim,fdkl,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,n||f,e>_abab*t2_abab(a,e,k,l)*t2_aaaa(f,b,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aekl,fbjm,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,k,l)*t2_abab(b,f,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aekl,bfjm,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<n,m||f,e>_abab*t2_abab(a,e,k,l)*t2_aaaa(f,b,i,j)*t2_abab(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,aekl,fbij,cdnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<m,n||f,e>_abab*t2_abab(a,e,k,l)*t2_aaaa(f,b,i,j)*t2_abab(c,d,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,aekl,fbij,cdmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,k,l)*t2_abab(f,d,j,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aekl,fdjm,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<m,n||f,e>_abab*t2_abab(a,e,j,l)*t2_aaaa(f,b,k,m)*t2_abab(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aejl,fbkm,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,j,l)*t2_abab(b,f,k,m)*t2_abab(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aejl,bfkm,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,d,k,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aejl,fdkm,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t2_aaaa(f,b,k,m)*t2_abab(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aeil,fbkm,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t2_abab(b,f,k,m)*t2_abab(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aeil,bfkm,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<n,m||f,e>_abab*t2_abab(a,e,i,l)*t2_aaaa(f,b,j,k)*t2_abab(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,aeil,fbjk,cdnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t2_aaaa(f,b,j,k)*t2_abab(c,d,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,aeil,fbjk,cdmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,i,k)*t2_abab(b,f,j,l)*t2_abab(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eaik,bfjl,cdnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,i,k)*t2_abab(b,f,j,l)*t2_abab(c,d,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,eaik,bfjl,cdmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,d,k,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aeil,fdkm,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,k)*t2_abab(f,d,j,l)*t2_aaaa(b,c,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,eaik,fdjl,bcnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,j,k)*t2_abab(b,f,m,l)*t2_abab(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eajk,bfml,cdin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(f,b,i,m)*t2_abab(c,d,n,l)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajk,fbim,cdnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,j,k)*t2_abab(b,f,i,m)*t2_abab(c,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajk,bfim,cdnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,j,k)*t2_abab(b,f,i,l)*t2_abab(c,d,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,eajk,bfil,cdnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,j,k)*t2_abab(b,f,i,l)*t2_abab(c,d,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,eajk,bfil,cdmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,k)*t2_abab(f,d,m,l)*t2_aaaa(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajk,fdml,bcin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,j,k)*t2_bbbb(f,d,l,m)*t2_aaaa(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajk,fdlm,bcin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,k)*t2_abab(f,d,i,l)*t2_aaaa(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eajk,fdil,bcnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,i,k)*t2_abab(b,f,m,l)*t2_abab(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eaik,bfml,cdjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,k)*t2_aaaa(f,b,j,m)*t2_abab(c,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaik,fbjm,cdnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,i,k)*t2_abab(b,f,j,m)*t2_abab(c,d,n,l)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaik,bfjm,cdnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,k)*t2_abab(f,d,m,l)*t2_aaaa(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaik,fdml,bcjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,i,k)*t2_bbbb(f,d,l,m)*t2_aaaa(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaik,fdlm,bcjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,f,m,l)*t2_abab(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eaij,bfml,cdkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,j)*t2_aaaa(f,b,k,m)*t2_abab(c,d,n,l)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fbkm,cdnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<n,m||e,f>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,f,k,m)*t2_abab(c,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaij,bfkm,cdnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,j)*t2_abab(f,d,m,l)*t2_aaaa(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaij,fdml,bckn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,i,j)*t2_bbbb(f,d,l,m)*t2_aaaa(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fdlm,bckn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||f,e>_abab*t2_abab(b,e,m,l)*t2_aaaa(f,c,j,k)*t2_abab(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,beml,fcjk,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(b,e,m,l)*t2_aaaa(f,c,i,j)*t2_abab(a,d,k,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,beml,fcij,adkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,b,k,m)*t2_aaaa(f,c,i,j)*t2_abab(a,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebkm,fcij,adnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_abab(b,e,k,m)*t2_aaaa(f,c,i,j)*t2_abab(a,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,bekm,fcij,adnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)<m,n||e,f>_abab*t2_aaaa(e,b,k,m)*t2_abab(c,f,j,l)*t2_abab(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,ebkm,cfjl,adin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_bbbb*t2_abab(b,e,k,m)*t2_abab(c,f,j,l)*t2_abab(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,bekm,cfjl,adin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<m,n||e,f>_abab*t2_aaaa(e,b,j,m)*t2_abab(c,f,k,l)*t2_abab(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,ebjm,cfkl,adin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>_bbbb*t2_abab(b,e,j,m)*t2_abab(c,f,k,l)*t2_abab(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,bejm,cfkl,adin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,b,j,m)*t2_aaaa(f,c,i,k)*t2_abab(a,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebjm,fcik,adnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(b,e,j,m)*t2_aaaa(f,c,i,k)*t2_abab(a,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,bejm,fcik,adnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,n||e,f>_abab*t2_aaaa(e,b,i,m)*t2_abab(c,f,k,l)*t2_abab(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,ebim,cfkl,adjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_bbbb*t2_abab(b,e,i,m)*t2_abab(c,f,k,l)*t2_abab(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,beim,cfkl,adjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||f,e>_abab*t2_abab(b,e,k,l)*t2_aaaa(f,c,j,m)*t2_abab(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,bekl,fcjm,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_bbbb*t2_abab(b,e,k,l)*t2_abab(c,f,j,m)*t2_abab(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,bekl,cfjm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||f,e>_abab*t2_abab(b,e,k,l)*t2_aaaa(f,c,i,j)*t2_abab(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,bekl,fcij,adnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<m,n||f,e>_abab*t2_abab(b,e,k,l)*t2_aaaa(f,c,i,j)*t2_abab(a,d,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,bekl,fcij,admn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<m,n||f,e>_abab*t2_abab(b,e,j,l)*t2_aaaa(f,c,k,m)*t2_abab(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,bejl,fckm,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,f>_bbbb*t2_abab(b,e,j,l)*t2_abab(c,f,k,m)*t2_abab(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,bejl,cfkm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,n||f,e>_abab*t2_abab(b,e,i,l)*t2_aaaa(f,c,k,m)*t2_abab(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,beil,fckm,adjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>_bbbb*t2_abab(b,e,i,l)*t2_abab(c,f,k,m)*t2_abab(a,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,beil,cfkm,adjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(b,e,i,l)*t2_aaaa(f,c,j,k)*t2_abab(a,d,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmfe,beil,fcjk,adnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(b,e,i,l)*t2_aaaa(f,c,j,k)*t2_abab(a,d,m,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,beil,fcjk,admn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_aaaa(e,b,i,k)*t2_abab(c,f,j,l)*t2_abab(a,d,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmef,ebik,cfjl,adnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_aaaa(e,b,i,k)*t2_abab(c,f,j,l)*t2_abab(a,d,m,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,ebik,cfjl,admn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*t2_aaaa(e,b,j,k)*t2_abab(c,f,m,l)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnef,ebjk,cfml,adin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,b,j,k)*t2_aaaa(f,c,i,m)*t2_abab(a,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebjk,fcim,adnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*t2_aaaa(e,b,j,k)*t2_abab(c,f,i,m)*t2_abab(a,d,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebjk,cfim,adnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 P(i,k)<n,m||e,f>_abab*t2_aaaa(e,b,j,k)*t2_abab(c,f,i,l)*t2_abab(a,d,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,ebjk,cfil,adnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<m,n||e,f>_abab*t2_aaaa(e,b,j,k)*t2_abab(c,f,i,l)*t2_abab(a,d,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,ebjk,cfil,admn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 <m,n||e,f>_abab*t2_aaaa(e,b,i,k)*t2_abab(c,f,m,l)*t2_abab(a,d,j,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,ebik,cfml,adjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,b,i,k)*t2_aaaa(f,c,j,m)*t2_abab(a,d,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebik,fcjm,adnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*t2_aaaa(e,b,i,k)*t2_abab(c,f,j,m)*t2_abab(a,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebik,cfjm,adnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*t2_aaaa(e,b,i,j)*t2_abab(c,f,m,l)*t2_abab(a,d,k,n)
    quadruples_res += -1.000000000000000 * einsum('mnef,ebij,cfml,adkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,b,i,j)*t2_aaaa(f,c,k,m)*t2_abab(a,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebij,fckm,adnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*t2_aaaa(e,b,i,j)*t2_abab(c,f,k,m)*t2_abab(a,d,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebij,cfkm,adnl->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,c,k,m)*t2_abab(f,d,j,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eckm,fdjl,abin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(c,e,k,m)*t2_abab(f,d,j,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,cekm,fdjl,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,m)*t2_abab(f,d,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjm,fdkl,abin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||f,e>_abab*t2_abab(c,e,j,m)*t2_abab(f,d,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,cejm,fdkl,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,i,m)*t2_abab(f,d,k,l)*t2_aaaa(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecim,fdkl,abjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||f,e>_abab*t2_abab(c,e,i,m)*t2_abab(f,d,k,l)*t2_aaaa(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,ceim,fdkl,abjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(c,e,k,l)*t2_abab(f,d,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,cekl,fdjm,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||f,e>_abab*t2_abab(c,e,j,l)*t2_abab(f,d,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,cejl,fdkm,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||f,e>_abab*t2_abab(c,e,i,l)*t2_abab(f,d,k,m)*t2_aaaa(a,b,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,ceil,fdkm,abjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 <n,m||e,f>_aaaa*t2_aaaa(e,c,i,k)*t2_abab(f,d,j,l)*t2_aaaa(a,b,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmef,ecik,fdjl,abnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,c,j,k)*t2_abab(f,d,m,l)*t2_aaaa(a,b,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecjk,fdml,abin->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*t2_aaaa(e,c,j,k)*t2_bbbb(f,d,l,m)*t2_aaaa(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecjk,fdlm,abin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 P(i,k)<n,m||e,f>_aaaa*t2_aaaa(e,c,j,k)*t2_abab(f,d,i,l)*t2_aaaa(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ecjk,fdil,abnm->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,c,i,k)*t2_abab(f,d,m,l)*t2_aaaa(a,b,j,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecik,fdml,abjn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*t2_aaaa(e,c,i,k)*t2_bbbb(f,d,l,m)*t2_aaaa(a,b,j,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecik,fdlm,abjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,c,i,j)*t2_abab(f,d,m,l)*t2_aaaa(a,b,k,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecij,fdml,abkn->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*t2_aaaa(e,c,i,j)*t2_bbbb(f,d,l,m)*t2_aaaa(a,b,k,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecij,fdlm,abkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    return quadruples_res


def ccdq_t4_aabbaabb_residual(t2_aaaa, t2_bbbb, t2_abab, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(k,l)f_bb(m,l)*t4_aabbaabb(a,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000010 * einsum('ml,abcdijkm->abcdijkl', f_bb[o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)f_aa(m,j)*t4_aabbaabb(a,b,c,d,i,m,l,k)
    contracted_intermediate =  1.000000000000010 * einsum('mj,abcdimlk->abcdijkl', f_aa[o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f_aa(a,e)*t4_aabbaabb(e,b,c,d,i,j,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('ae,ebcdijkl->abcdijkl', f_aa[v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)f_bb(c,e)*t4_aabbaabb(b,a,e,d,i,j,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('ce,baedijkl->abcdijkl', f_bb[v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 <n,m||k,l>_bbbb*t4_aabbaabb(a,b,c,d,i,j,n,m)
    quadruples_res +=  0.500000000000010 * einsum('nmkl,abcdijnm->abcdijkl', g_bbbb[o, o, o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <n,m||j,l>_abab*t4_aabbaabb(a,b,c,d,i,n,k,m)
    quadruples_res +=  0.500000000000010 * einsum('nmjl,abcdinkm->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 <m,n||j,l>_abab*t4_aabbaabb(a,b,c,d,i,m,n,k)
    quadruples_res += -0.500000000000010 * einsum('mnjl,abcdimnk->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 P(k,l)<n,m||i,l>_abab*t4_aabbaabb(a,b,c,d,j,n,k,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmil,abcdjnkm->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<m,n||i,l>_abab*t4_aabbaabb(a,b,c,d,j,m,n,k)
    contracted_intermediate =  0.500000000000010 * einsum('mnil,abcdjmnk->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 <n,m||j,k>_abab*t4_aabbaabb(a,b,c,d,i,n,l,m)
    quadruples_res += -0.500000000000010 * einsum('nmjk,abcdinlm->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <m,n||j,k>_abab*t4_aabbaabb(a,b,c,d,i,m,n,l)
    quadruples_res +=  0.500000000000010 * einsum('mnjk,abcdimnl->abcdijkl', g_abab[o, o, o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 <n,m||j,i>_aaaa*t4_aabbaabb(a,b,c,d,n,m,k,l)
    quadruples_res += -0.500000000000010 * einsum('nmji,abcdnmkl->abcdijkl', g_aaaa[o, o, o, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 P(k,l)*P(a,b)<a,m||e,l>_abab*t4_aabbaabb(e,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('amel,ebcdijkm->abcdijkl', g_abab[v, o, v, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*t4_aabbaabb(e,b,c,d,i,m,l,k)
    contracted_intermediate = -1.000000000000020 * einsum('maej,ebcdimlk->abcdijkl', g_aaaa[o, v, v, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,m||j,e>_abab*t4_abbbabbb(b,e,c,d,i,k,l,m)
    contracted_intermediate = -1.000000000000020 * einsum('amje,becdiklm->abcdijkl', g_abab[v, o, o, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,l>_abab*t4_aaabaaab(e,a,b,d,i,j,m,k)
    contracted_intermediate = -1.000000000000020 * einsum('mcel,eabdijmk->abcdijkl', g_abab[o, v, v, o], t4_aaabaaab[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,l>_bbbb*t4_aabbaabb(b,a,e,d,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('mcel,baedijkm->abcdijkl', g_bbbb[o, v, v, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||j,e>_abab*t4_aabbaabb(b,a,e,d,i,m,l,k)
    contracted_intermediate = -1.000000000000020 * einsum('mcje,baedimlk->abcdijkl', g_abab[o, v, o, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 <a,b||e,f>_aaaa*t4_aabbaabb(e,f,c,d,i,j,k,l)
    quadruples_res +=  0.500000000000010 * einsum('abef,efcdijkl->abcdijkl', g_aaaa[v, v, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <a,c||e,f>_abab*t4_aabbaabb(e,b,f,d,i,j,k,l)
    quadruples_res +=  0.500000000000010 * einsum('acef,ebfdijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 <a,c||f,e>_abab*t4_aabbaabb(b,f,e,d,i,j,k,l)
    quadruples_res += -0.500000000000010 * einsum('acfe,bfedijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 P(a,b)<a,d||e,f>_abab*t4_aabbaabb(e,b,f,c,i,j,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('adef,ebfcijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<a,d||f,e>_abab*t4_aabbaabb(b,f,e,c,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('adfe,bfecijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 <b,c||e,f>_abab*t4_aabbaabb(e,a,f,d,i,j,k,l)
    quadruples_res += -0.500000000000010 * einsum('bcef,eafdijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <b,c||f,e>_abab*t4_aabbaabb(a,f,e,d,i,j,k,l)
    quadruples_res +=  0.500000000000010 * einsum('bcfe,afedijkl->abcdijkl', g_abab[v, v, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 <d,c||e,f>_bbbb*t4_aabbaabb(a,b,e,f,i,j,k,l)
    quadruples_res += -0.500000000000010 * einsum('dcef,abefijkl->abcdijkl', g_bbbb[v, v, v, v], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(e,f,m,l)*t4_aabbaabb(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,efml,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(f,e,m,l)*t4_aabbaabb(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,feml,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,f,l,m)*t4_aabbaabb(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eflm,abcdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,f,j,m)*t4_aabbaabb(a,b,c,d,i,n,l,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,efjm,abcdinlk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*t2_abab(e,f,j,m)*t4_aabbaabb(a,b,c,d,i,n,l,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,efjm,abcdinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*t2_abab(f,e,j,m)*t4_aabbaabb(a,b,c,d,i,n,l,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,fejm,abcdinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,f>_bbbb*t2_bbbb(e,f,k,l)*t4_aabbaabb(a,b,c,d,i,j,n,m)
    quadruples_res +=  0.250000000000010 * einsum('nmef,efkl,abcdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*t2_abab(e,f,j,l)*t4_aabbaabb(a,b,c,d,i,n,k,m)
    quadruples_res +=  0.250000000000010 * einsum('nmef,efjl,abcdinkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*t2_abab(e,f,j,l)*t4_aabbaabb(a,b,c,d,i,m,n,k)
    quadruples_res += -0.250000000000010 * einsum('mnef,efjl,abcdimnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*t2_abab(f,e,j,l)*t4_aabbaabb(a,b,c,d,i,n,k,m)
    quadruples_res +=  0.250000000000010 * einsum('nmfe,fejl,abcdinkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*t2_abab(f,e,j,l)*t4_aabbaabb(a,b,c,d,i,m,n,k)
    quadruples_res += -0.250000000000010 * einsum('mnfe,fejl,abcdimnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 P(k,l)<n,m||e,f>_abab*t2_abab(e,f,i,l)*t4_aabbaabb(a,b,c,d,j,n,k,m)
    contracted_intermediate = -0.250000000000010 * einsum('nmef,efil,abcdjnkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.2500 P(k,l)<m,n||e,f>_abab*t2_abab(e,f,i,l)*t4_aabbaabb(a,b,c,d,j,m,n,k)
    contracted_intermediate =  0.250000000000010 * einsum('mnef,efil,abcdjmnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.2500 P(k,l)<n,m||f,e>_abab*t2_abab(f,e,i,l)*t4_aabbaabb(a,b,c,d,j,n,k,m)
    contracted_intermediate = -0.250000000000010 * einsum('nmfe,feil,abcdjnkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.2500 P(k,l)<m,n||f,e>_abab*t2_abab(f,e,i,l)*t4_aabbaabb(a,b,c,d,j,m,n,k)
    contracted_intermediate =  0.250000000000010 * einsum('mnfe,feil,abcdjmnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.2500 <n,m||e,f>_abab*t2_abab(e,f,j,k)*t4_aabbaabb(a,b,c,d,i,n,l,m)
    quadruples_res += -0.250000000000010 * einsum('nmef,efjk,abcdinlm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*t2_abab(e,f,j,k)*t4_aabbaabb(a,b,c,d,i,m,n,l)
    quadruples_res +=  0.250000000000010 * einsum('mnef,efjk,abcdimnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*t2_abab(f,e,j,k)*t4_aabbaabb(a,b,c,d,i,n,l,m)
    quadruples_res += -0.250000000000010 * einsum('nmfe,fejk,abcdinlm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*t2_abab(f,e,j,k)*t4_aabbaabb(a,b,c,d,i,m,n,l)
    quadruples_res +=  0.250000000000010 * einsum('mnfe,fejk,abcdimnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_aaaa*t2_aaaa(e,f,j,i)*t4_aabbaabb(a,b,c,d,n,m,k,l)
    quadruples_res += -0.250000000000010 * einsum('nmef,efji,abcdnmkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,n,m)*t4_aabbaabb(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eanm,fbcdijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,n,m)*t4_aabbaabb(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,aenm,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,m,n)*t4_aabbaabb(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,aemn,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t4_aabbaabb(f,b,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnfe,aeml,fbcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t4_aabbaabb(f,b,c,d,i,n,l,k)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,eajm,fbcdinlk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,j,m)*t4_abbbabbb(b,f,c,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,eajm,bfcdikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t4_aabbaabb(f,b,c,d,i,n,l,k)
    contracted_intermediate = -0.999999999999840 * einsum('nmfe,aejm,fbcdinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,j,m)*t4_abbbabbb(b,f,c,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,aejm,bfcdikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,l)*t4_aabbaabb(f,b,c,d,i,n,k,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,aejl,fbcdinkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,j,l)*t4_aabbaabb(f,b,c,d,i,m,n,k)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,aejl,fbcdimnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,j,l)*t4_abbbabbb(b,f,c,d,i,k,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,aejl,bfcdiknm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,i,l)*t4_aabbaabb(f,b,c,d,j,n,k,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,aeil,fbcdjnkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t4_aabbaabb(f,b,c,d,j,m,n,k)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,aeil,fbcdjmnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t4_abbbabbb(b,f,c,d,j,k,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,aeil,bfcdjknm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,k)*t4_aabbaabb(f,b,c,d,i,n,l,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,aejk,fbcdinlm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,j,k)*t4_aabbaabb(f,b,c,d,i,m,n,l)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,aejk,fbcdimnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,j,k)*t4_abbbabbb(b,f,c,d,i,l,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,aejk,bfcdilnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,i)*t4_aabbaabb(f,b,c,d,n,m,k,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eaji,fbcdnmkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,j,i)*t4_abbbabbb(b,f,c,d,n,l,k,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eaji,bfcdnlkm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,j,i)*t4_abbbabbb(b,f,c,d,m,l,n,k)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,eaji,bfcdmlnk->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<n,m||e,f>_abab*t2_abab(e,c,n,m)*t4_aabbaabb(b,a,f,d,i,j,k,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecnm,bafdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<m,n||e,f>_abab*t2_abab(e,c,m,n)*t4_aabbaabb(b,a,f,d,i,j,k,l)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,ecmn,bafdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,n,m)*t4_aabbaabb(b,a,f,d,i,j,k,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecnm,bafdijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,m,l)*t4_aaabaaab(f,a,b,d,i,j,n,k)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ecml,fabdijnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,n||e,f>_abab*t2_abab(e,c,m,l)*t4_aabbaabb(b,a,f,d,i,j,k,n)
    contracted_intermediate = -0.999999999999840 * einsum('mnef,ecml,bafdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,l,m)*t4_aaabaaab(f,a,b,d,i,j,n,k)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,eclm,fabdijnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,l,m)*t4_aabbaabb(b,a,f,d,i,j,k,n)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,eclm,bafdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>_abab*t2_abab(e,c,j,m)*t4_aabbaabb(b,a,f,d,i,n,l,k)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ecjm,bafdinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,k,l)*t4_aaabaaab(f,a,b,d,i,j,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,eckl,fabdijnm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<m,n||f,e>_abab*t2_bbbb(e,c,k,l)*t4_aaabaaab(f,a,b,d,i,j,m,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,eckl,fabdijmn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,k,l)*t4_aabbaabb(b,a,f,d,i,j,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eckl,bafdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,j,l)*t4_aaabaaab(f,a,b,d,i,m,n,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecjl,fabdimnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<n,m||e,f>_abab*t2_abab(e,c,j,l)*t4_aabbaabb(b,a,f,d,i,n,k,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecjl,bafdinkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<m,n||e,f>_abab*t2_abab(e,c,j,l)*t4_aabbaabb(b,a,f,d,i,m,n,k)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,ecjl,bafdimnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,i,l)*t4_aaabaaab(f,a,b,d,j,m,n,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecil,fabdjmnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>_abab*t2_abab(e,c,i,l)*t4_aabbaabb(b,a,f,d,j,n,k,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecil,bafdjnkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<m,n||e,f>_abab*t2_abab(e,c,i,l)*t4_aabbaabb(b,a,f,d,j,m,n,k)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,ecil,bafdjmnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,j,k)*t4_aaabaaab(f,a,b,d,i,m,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecjk,fabdimnl->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||e,f>_abab*t2_abab(e,c,j,k)*t4_aabbaabb(b,a,f,d,i,n,l,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecjk,bafdinlm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<m,n||e,f>_abab*t2_abab(e,c,j,k)*t4_aabbaabb(b,a,f,d,i,m,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,ecjk,bafdimnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,f>_aaaa*t2_aaaa(a,b,n,m)*t4_aabbaabb(e,f,c,d,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('nmef,abnm,efcdijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*t2_abab(a,c,n,m)*t4_aabbaabb(e,b,f,d,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('nmef,acnm,ebfdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*t2_abab(a,c,n,m)*t4_aabbaabb(b,f,e,d,i,j,k,l)
    quadruples_res += -0.250000000000010 * einsum('nmfe,acnm,bfedijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*t2_abab(a,c,m,n)*t4_aabbaabb(e,b,f,d,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('mnef,acmn,ebfdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*t2_abab(a,c,m,n)*t4_aabbaabb(b,f,e,d,i,j,k,l)
    quadruples_res += -0.250000000000010 * einsum('mnfe,acmn,bfedijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 P(k,l)<n,m||e,f>_aaaa*t2_abab(a,c,m,l)*t4_aaabaaab(e,f,b,d,i,j,n,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,acml,efbdijnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(a,c,m,l)*t4_aabbaabb(e,b,f,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,acml,ebfdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(a,c,m,l)*t4_aabbaabb(b,f,e,d,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,acml,bfedijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(a,b,j,m)*t4_aabbaabb(e,f,c,d,i,n,l,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,abjm,efcdinlk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||e,f>_abab*t2_aaaa(a,b,j,m)*t4_abbbabbb(e,f,c,d,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,abjm,efcdikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*t2_aaaa(a,b,j,m)*t4_abbbabbb(f,e,c,d,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,abjm,fecdikln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*t2_abab(a,c,j,m)*t4_aabbaabb(e,b,f,d,i,n,l,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,acjm,ebfdinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*t2_abab(a,c,j,m)*t4_aabbaabb(b,f,e,d,i,n,l,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,acjm,bfedinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*t2_abab(a,c,j,m)*t4_abbbabbb(b,f,e,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,acjm,bfedikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_abab*t2_abab(a,d,n,m)*t4_aabbaabb(e,b,f,c,i,j,k,l)
    contracted_intermediate = -0.250000000000010 * einsum('nmef,adnm,ebfcijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||f,e>_abab*t2_abab(a,d,n,m)*t4_aabbaabb(b,f,e,c,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmfe,adnm,bfecijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,n||e,f>_abab*t2_abab(a,d,m,n)*t4_aabbaabb(e,b,f,c,i,j,k,l)
    contracted_intermediate = -0.250000000000010 * einsum('mnef,admn,ebfcijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,n||f,e>_abab*t2_abab(a,d,m,n)*t4_aabbaabb(b,f,e,c,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('mnfe,admn,bfecijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>_aaaa*t2_abab(a,d,m,l)*t4_aaabaaab(e,f,b,c,i,j,n,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adml,efbcijnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<m,n||e,f>_abab*t2_abab(a,d,m,l)*t4_aabbaabb(e,b,f,c,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,adml,ebfcijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<m,n||f,e>_abab*t2_abab(a,d,m,l)*t4_aabbaabb(b,f,e,c,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,adml,bfecijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*t2_abab(a,d,j,m)*t4_aabbaabb(e,b,f,c,i,n,l,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adjm,ebfcinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,d,j,m)*t4_aabbaabb(b,f,e,c,i,n,l,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,adjm,bfecinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_abab(a,d,j,m)*t4_abbbabbb(b,f,e,c,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,adjm,bfecikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.2500 <n,m||e,f>_abab*t2_abab(b,c,n,m)*t4_aabbaabb(e,a,f,d,i,j,k,l)
    quadruples_res += -0.250000000000010 * einsum('nmef,bcnm,eafdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*t2_abab(b,c,n,m)*t4_aabbaabb(a,f,e,d,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('nmfe,bcnm,afedijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*t2_abab(b,c,m,n)*t4_aabbaabb(e,a,f,d,i,j,k,l)
    quadruples_res += -0.250000000000010 * einsum('mnef,bcmn,eafdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*t2_abab(b,c,m,n)*t4_aabbaabb(a,f,e,d,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('mnfe,bcmn,afedijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*t2_bbbb(d,c,n,m)*t4_aabbaabb(a,b,e,f,i,j,k,l)
    quadruples_res += -0.250000000000010 * einsum('nmef,dcnm,abefijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(k,l)<n,m||e,f>_aaaa*t2_abab(b,c,m,l)*t4_aaabaaab(e,f,a,d,i,j,n,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bcml,efadijnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(b,c,m,l)*t4_aabbaabb(e,a,f,d,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,bcml,eafdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(b,c,m,l)*t4_aabbaabb(a,f,e,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,bcml,afedijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,f>_abab*t2_bbbb(d,c,l,m)*t4_aaabaaab(e,b,a,f,i,j,n,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,dclm,ebafijnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||f,e>_abab*t2_bbbb(d,c,l,m)*t4_aaabaaab(b,f,a,e,i,j,n,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,dclm,bfaeijnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(d,c,l,m)*t4_aabbaabb(a,b,e,f,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,dclm,abefijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*t2_abab(b,c,j,m)*t4_aabbaabb(e,a,f,d,i,n,l,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bcjm,eafdinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*t2_abab(b,c,j,m)*t4_aabbaabb(a,f,e,d,i,n,l,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,bcjm,afedinlk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*t2_abab(b,c,j,m)*t4_abbbabbb(a,f,e,d,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,bcjm,afedikln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||k,l>_bbbb*t2_abab(a,c,j,m)*t2_abab(b,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmkl,acjm,bdin->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||k,l>_bbbb*t2_abab(a,d,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,adjm,bcin->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 <m,n||j,l>_abab*t2_abab(a,c,m,k)*t2_abab(b,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnjl,acmk,bdin->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||j,l>_abab*t2_aaaa(a,b,i,m)*t2_bbbb(c,d,k,n)
    quadruples_res +=  1.000000000000000 * einsum('mnjl,abim,cdkn->abcdijkl', g_abab[o, o, o, o], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <n,m||j,l>_abab*t2_abab(a,c,i,m)*t2_abab(b,d,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmjl,acim,bdnk->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||j,l>_abab*t2_abab(a,d,m,k)*t2_abab(b,c,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnjl,admk,bcin->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||j,l>_abab*t2_abab(a,d,i,m)*t2_abab(b,c,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmjl,adim,bcnk->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||i,l>_abab*t2_abab(a,c,m,k)*t2_abab(b,d,j,n)
    quadruples_res +=  1.000000000000000 * einsum('mnil,acmk,bdjn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,n||i,l>_abab*t2_aaaa(a,b,j,m)*t2_bbbb(c,d,k,n)
    quadruples_res += -1.000000000000000 * einsum('mnil,abjm,cdkn->abcdijkl', g_abab[o, o, o, o], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||i,l>_abab*t2_abab(a,c,j,m)*t2_abab(b,d,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmil,acjm,bdnk->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,n||i,l>_abab*t2_abab(a,d,m,k)*t2_abab(b,c,j,n)
    quadruples_res += -1.000000000000000 * einsum('mnil,admk,bcjn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <n,m||i,l>_abab*t2_abab(a,d,j,m)*t2_abab(b,c,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmil,adjm,bcnk->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||j,k>_abab*t2_abab(a,c,m,l)*t2_abab(b,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnjk,acml,bdin->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,n||j,k>_abab*t2_aaaa(a,b,i,m)*t2_bbbb(c,d,l,n)
    quadruples_res += -1.000000000000000 * einsum('mnjk,abim,cdln->abcdijkl', g_abab[o, o, o, o], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||j,k>_abab*t2_abab(a,c,i,m)*t2_abab(b,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmjk,acim,bdnl->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,n||j,k>_abab*t2_abab(a,d,m,l)*t2_abab(b,c,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnjk,adml,bcin->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <n,m||j,k>_abab*t2_abab(a,d,i,m)*t2_abab(b,c,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmjk,adim,bcnl->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,n||i,k>_abab*t2_abab(a,c,m,l)*t2_abab(b,d,j,n)
    quadruples_res += -1.000000000000000 * einsum('mnik,acml,bdjn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||i,k>_abab*t2_aaaa(a,b,j,m)*t2_bbbb(c,d,l,n)
    quadruples_res +=  1.000000000000000 * einsum('mnik,abjm,cdln->abcdijkl', g_abab[o, o, o, o], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <n,m||i,k>_abab*t2_abab(a,c,j,m)*t2_abab(b,d,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmik,acjm,bdnl->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||i,k>_abab*t2_abab(a,d,m,l)*t2_abab(b,c,j,n)
    quadruples_res +=  1.000000000000000 * einsum('mnik,adml,bcjn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||i,k>_abab*t2_abab(a,d,j,m)*t2_abab(b,c,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmik,adjm,bcnl->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(k,l)<n,m||i,j>_aaaa*t2_abab(a,c,m,l)*t2_abab(b,d,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmij,acml,bdnk->abcdijkl', g_aaaa[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||i,j>_aaaa*t2_abab(a,d,m,l)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,adml,bcnk->abcdijkl', g_aaaa[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<a,m||e,l>_abab*t2_abab(e,c,j,k)*t2_abab(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('amel,ecjk,bdim->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<a,m||e,l>_abab*t2_aaaa(e,b,i,j)*t2_bbbb(c,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('amel,ebij,cdkm->abcdijkl', g_abab[v, o, v, o], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,m||e,l>_abab*t2_abab(e,d,j,k)*t2_abab(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('amel,edjk,bcim->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<a,m||e,k>_abab*t2_abab(e,c,j,l)*t2_abab(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('amek,ecjl,bdim->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<a,m||e,k>_abab*t2_abab(e,d,j,l)*t2_abab(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('amek,edjl,bcim->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 <a,m||j,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(b,d,i,m)
    quadruples_res += -1.000000000000000 * einsum('amje,eckl,bdim->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,m||j,e>_abab*t2_abab(b,e,i,l)*t2_bbbb(c,d,k,m)
    quadruples_res +=  1.000000000000000 * einsum('amje,beil,cdkm->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,a||e,j>_aaaa*t2_abab(e,c,i,l)*t2_abab(b,d,m,k)
    quadruples_res += -1.000000000000000 * einsum('maej,ecil,bdmk->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,j)<a,m||j,e>_abab*t2_abab(b,e,i,k)*t2_bbbb(c,d,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('amje,beik,cdlm->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,a||e,j>_aaaa*t2_abab(e,c,i,k)*t2_abab(b,d,m,l)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ecik,bdml->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<a,m||j,e>_abab*t2_bbbb(e,d,k,l)*t2_abab(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('amje,edkl,bcim->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,a||e,j>_aaaa*t2_abab(e,d,i,l)*t2_abab(b,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edil,bcmk->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*t2_abab(e,d,i,k)*t2_abab(b,c,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('maej,edik,bcml->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 <a,m||i,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(b,d,j,m)
    quadruples_res +=  1.000000000000000 * einsum('amie,eckl,bdjm->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,m||i,e>_abab*t2_abab(b,e,j,l)*t2_bbbb(c,d,k,m)
    quadruples_res += -1.000000000000000 * einsum('amie,bejl,cdkm->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,a||e,i>_aaaa*t2_abab(e,c,j,l)*t2_abab(b,d,m,k)
    quadruples_res +=  1.000000000000000 * einsum('maei,ecjl,bdmk->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(a,b)<a,m||i,e>_abab*t2_bbbb(e,d,k,l)*t2_abab(b,c,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('amie,edkl,bcjm->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,a||e,i>_aaaa*t2_abab(e,d,j,l)*t2_abab(b,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('maei,edjl,bcmk->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<b,m||e,l>_abab*t2_abab(e,c,j,k)*t2_abab(a,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('bmel,ecjk,adim->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<b,m||e,l>_abab*t2_aaaa(e,a,i,j)*t2_bbbb(c,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('bmel,eaij,cdkm->abcdijkl', g_abab[v, o, v, o], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<b,m||e,k>_abab*t2_abab(e,c,j,l)*t2_abab(a,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('bmek,ecjl,adim->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 <b,m||j,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(a,d,i,m)
    quadruples_res +=  1.000000000000000 * einsum('bmje,eckl,adim->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,m||j,e>_abab*t2_abab(a,e,i,l)*t2_bbbb(c,d,k,m)
    quadruples_res += -1.000000000000000 * einsum('bmje,aeil,cdkm->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_aaaa*t2_abab(e,c,i,l)*t2_abab(a,d,m,k)
    quadruples_res +=  1.000000000000000 * einsum('mbej,ecil,admk->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(i,j)<b,m||j,e>_abab*t2_abab(a,e,i,k)*t2_bbbb(c,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('bmje,aeik,cdlm->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,b||e,j>_aaaa*t2_abab(e,c,i,k)*t2_abab(a,d,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,ecik,adml->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 <b,m||i,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(a,d,j,m)
    quadruples_res += -1.000000000000000 * einsum('bmie,eckl,adjm->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||i,e>_abab*t2_abab(a,e,j,l)*t2_bbbb(c,d,k,m)
    quadruples_res +=  1.000000000000000 * einsum('bmie,aejl,cdkm->abcdijkl', g_abab[v, o, o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,i>_aaaa*t2_abab(e,c,j,l)*t2_abab(a,d,m,k)
    quadruples_res += -1.000000000000000 * einsum('mbei,ecjl,admk->abcdijkl', g_aaaa[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,l>_bbbb*t2_abab(a,e,j,k)*t2_abab(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcel,aejk,bdim->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,c||e,l>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,d,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mcel,eaij,bdmk->abcdijkl', g_abab[o, v, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,l>_abab*t2_abab(e,d,j,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edjk,abim->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,k>_bbbb*t2_abab(a,e,j,l)*t2_abab(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcek,aejl,bdim->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,k>_abab*t2_abab(e,d,j,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,edjl,abim->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,c||j,e>_abab*t2_abab(a,e,i,l)*t2_abab(b,d,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mcje,aeil,bdmk->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||j,e>_abab*t2_abab(a,e,i,k)*t2_abab(b,d,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('mcje,aeik,bdml->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||j,e>_abab*t2_bbbb(e,d,k,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcje,edkl,abim->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,c||i,e>_abab*t2_abab(a,e,j,l)*t2_abab(b,d,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mcie,aejl,bdmk->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<m,c||i,e>_abab*t2_bbbb(e,d,k,l)*t2_aaaa(a,b,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcie,edkl,abjm->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,l>_bbbb*t2_abab(a,e,j,k)*t2_abab(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,aejk,bcim->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,d||e,l>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,eaij,bcmk->abcdijkl', g_abab[o, v, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,k>_bbbb*t2_abab(a,e,j,l)*t2_abab(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdek,aejl,bcim->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,d||j,e>_abab*t2_abab(a,e,i,l)*t2_abab(b,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mdje,aeil,bcmk->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||j,e>_abab*t2_abab(a,e,i,k)*t2_abab(b,c,m,l)
    contracted_intermediate =  1.000000000000000 * einsum('mdje,aeik,bcml->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,d||i,e>_abab*t2_abab(a,e,j,l)*t2_abab(b,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mdie,aejl,bcmk->abcdijkl', g_abab[o, v, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 <a,b||e,f>_aaaa*t2_abab(e,c,j,l)*t2_abab(f,d,i,k)
    quadruples_res +=  1.000000000000000 * einsum('abef,ecjl,fdik->abcdijkl', g_aaaa[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,c||f,e>_abab*t2_abab(b,e,j,l)*t2_abab(f,d,i,k)
    quadruples_res += -1.000000000000000 * einsum('acfe,bejl,fdik->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(k,l)<a,b||e,f>_aaaa*t2_abab(e,c,i,l)*t2_abab(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecil,fdjk->abcdijkl', g_aaaa[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<a,c||f,e>_abab*t2_abab(b,e,i,l)*t2_abab(f,d,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('acfe,beil,fdjk->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <a,b||e,f>_aaaa*t2_abab(e,c,j,k)*t2_abab(f,d,i,l)
    quadruples_res += -1.000000000000000 * einsum('abef,ecjk,fdil->abcdijkl', g_aaaa[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,c||f,e>_abab*t2_abab(b,e,j,k)*t2_abab(f,d,i,l)
    quadruples_res +=  1.000000000000000 * einsum('acfe,bejk,fdil->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,c||e,f>_abab*t2_aaaa(e,b,j,i)*t2_bbbb(f,d,k,l)
    quadruples_res += -1.000000000000000 * einsum('acef,ebji,fdkl->abcdijkl', g_abab[v, v, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(a,b)<a,d||f,e>_abab*t2_abab(b,e,j,l)*t2_abab(f,c,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('adfe,bejl,fcik->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<a,d||f,e>_abab*t2_abab(b,e,i,l)*t2_abab(f,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adfe,beil,fcjk->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<a,d||f,e>_abab*t2_abab(b,e,j,k)*t2_abab(f,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('adfe,bejk,fcil->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<a,d||e,f>_abab*t2_aaaa(e,b,j,i)*t2_bbbb(f,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('adef,ebji,fckl->abcdijkl', g_abab[v, v, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 <b,c||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,d,i,k)
    quadruples_res +=  1.000000000000000 * einsum('bcfe,aejl,fdik->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <d,c||e,f>_bbbb*t2_abab(a,e,j,l)*t2_abab(b,f,i,k)
    quadruples_res += -1.000000000000000 * einsum('dcef,aejl,bfik->abcdijkl', g_bbbb[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(k,l)<b,c||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('bcfe,aeil,fdjk->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<d,c||e,f>_bbbb*t2_abab(a,e,i,l)*t2_abab(b,f,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('dcef,aeil,bfjk->abcdijkl', g_bbbb[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <b,c||f,e>_abab*t2_abab(a,e,j,k)*t2_abab(f,d,i,l)
    quadruples_res += -1.000000000000000 * einsum('bcfe,aejk,fdil->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <d,c||e,f>_bbbb*t2_abab(a,e,j,k)*t2_abab(b,f,i,l)
    quadruples_res +=  1.000000000000000 * einsum('dcef,aejk,bfil->abcdijkl', g_bbbb[v, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,c||e,f>_abab*t2_aaaa(e,a,j,i)*t2_bbbb(f,d,k,l)
    quadruples_res +=  1.000000000000000 * einsum('bcef,eaji,fdkl->abcdijkl', g_abab[v, v, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,f,k,l)*t2_abab(a,c,j,m)*t2_abab(b,d,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efkl,acjm,bdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,f,k,l)*t2_abab(a,d,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efkl,adjm,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,f,j,l)*t2_abab(a,c,m,k)*t2_abab(b,d,i,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,efjl,acmk,bdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(f,e,j,l)*t2_abab(a,c,m,k)*t2_abab(b,d,i,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,fejl,acmk,bdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_abab(e,f,j,l)*t2_aaaa(a,b,i,m)*t2_bbbb(c,d,k,n)
    quadruples_res +=  0.500000000000000 * einsum('mnef,efjl,abim,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_abab(f,e,j,l)*t2_aaaa(a,b,i,m)*t2_bbbb(c,d,k,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,fejl,abim,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,f,j,l)*t2_abab(a,c,i,m)*t2_abab(b,d,n,k)
    quadruples_res += -0.500000000000000 * einsum('nmef,efjl,acim,bdnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(f,e,j,l)*t2_abab(a,c,i,m)*t2_abab(b,d,n,k)
    quadruples_res += -0.500000000000000 * einsum('nmfe,fejl,acim,bdnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_abab(e,f,j,l)*t2_abab(a,d,m,k)*t2_abab(b,c,i,n)
    quadruples_res +=  0.500000000000000 * einsum('mnef,efjl,admk,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_abab(f,e,j,l)*t2_abab(a,d,m,k)*t2_abab(b,c,i,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,fejl,admk,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_abab(e,f,j,l)*t2_abab(a,d,i,m)*t2_abab(b,c,n,k)
    quadruples_res +=  0.500000000000000 * einsum('nmef,efjl,adim,bcnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*t2_abab(f,e,j,l)*t2_abab(a,d,i,m)*t2_abab(b,c,n,k)
    quadruples_res +=  0.500000000000000 * einsum('nmfe,fejl,adim,bcnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_abab(e,f,i,l)*t2_abab(a,c,m,k)*t2_abab(b,d,j,n)
    quadruples_res +=  0.500000000000000 * einsum('mnef,efil,acmk,bdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_abab(f,e,i,l)*t2_abab(a,c,m,k)*t2_abab(b,d,j,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,feil,acmk,bdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,f,i,l)*t2_aaaa(a,b,j,m)*t2_bbbb(c,d,k,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,efil,abjm,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(f,e,i,l)*t2_aaaa(a,b,j,m)*t2_bbbb(c,d,k,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,feil,abjm,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_abab(e,f,i,l)*t2_abab(a,c,j,m)*t2_abab(b,d,n,k)
    quadruples_res +=  0.500000000000000 * einsum('nmef,efil,acjm,bdnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*t2_abab(f,e,i,l)*t2_abab(a,c,j,m)*t2_abab(b,d,n,k)
    quadruples_res +=  0.500000000000000 * einsum('nmfe,feil,acjm,bdnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,f,i,l)*t2_abab(a,d,m,k)*t2_abab(b,c,j,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,efil,admk,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(f,e,i,l)*t2_abab(a,d,m,k)*t2_abab(b,c,j,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,feil,admk,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,f,i,l)*t2_abab(a,d,j,m)*t2_abab(b,c,n,k)
    quadruples_res += -0.500000000000000 * einsum('nmef,efil,adjm,bcnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(f,e,i,l)*t2_abab(a,d,j,m)*t2_abab(b,c,n,k)
    quadruples_res += -0.500000000000000 * einsum('nmfe,feil,adjm,bcnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_abab(e,f,j,k)*t2_abab(a,c,m,l)*t2_abab(b,d,i,n)
    quadruples_res +=  0.500000000000000 * einsum('mnef,efjk,acml,bdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_abab(f,e,j,k)*t2_abab(a,c,m,l)*t2_abab(b,d,i,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,fejk,acml,bdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,f,j,k)*t2_aaaa(a,b,i,m)*t2_bbbb(c,d,l,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,efjk,abim,cdln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(f,e,j,k)*t2_aaaa(a,b,i,m)*t2_bbbb(c,d,l,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,fejk,abim,cdln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_abab(e,f,j,k)*t2_abab(a,c,i,m)*t2_abab(b,d,n,l)
    quadruples_res +=  0.500000000000000 * einsum('nmef,efjk,acim,bdnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*t2_abab(f,e,j,k)*t2_abab(a,c,i,m)*t2_abab(b,d,n,l)
    quadruples_res +=  0.500000000000000 * einsum('nmfe,fejk,acim,bdnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,f,j,k)*t2_abab(a,d,m,l)*t2_abab(b,c,i,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,efjk,adml,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(f,e,j,k)*t2_abab(a,d,m,l)*t2_abab(b,c,i,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,fejk,adml,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,f,j,k)*t2_abab(a,d,i,m)*t2_abab(b,c,n,l)
    quadruples_res += -0.500000000000000 * einsum('nmef,efjk,adim,bcnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(f,e,j,k)*t2_abab(a,d,i,m)*t2_abab(b,c,n,l)
    quadruples_res += -0.500000000000000 * einsum('nmfe,fejk,adim,bcnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,f,i,k)*t2_abab(a,c,m,l)*t2_abab(b,d,j,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,efik,acml,bdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(f,e,i,k)*t2_abab(a,c,m,l)*t2_abab(b,d,j,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,feik,acml,bdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_abab(e,f,i,k)*t2_aaaa(a,b,j,m)*t2_bbbb(c,d,l,n)
    quadruples_res +=  0.500000000000000 * einsum('mnef,efik,abjm,cdln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_abab(f,e,i,k)*t2_aaaa(a,b,j,m)*t2_bbbb(c,d,l,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,feik,abjm,cdln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,f,i,k)*t2_abab(a,c,j,m)*t2_abab(b,d,n,l)
    quadruples_res += -0.500000000000000 * einsum('nmef,efik,acjm,bdnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(f,e,i,k)*t2_abab(a,c,j,m)*t2_abab(b,d,n,l)
    quadruples_res += -0.500000000000000 * einsum('nmfe,feik,acjm,bdnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_abab(e,f,i,k)*t2_abab(a,d,m,l)*t2_abab(b,c,j,n)
    quadruples_res +=  0.500000000000000 * einsum('mnef,efik,adml,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_abab(f,e,i,k)*t2_abab(a,d,m,l)*t2_abab(b,c,j,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,feik,adml,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_abab(e,f,i,k)*t2_abab(a,d,j,m)*t2_abab(b,c,n,l)
    quadruples_res +=  0.500000000000000 * einsum('nmef,efik,adjm,bcnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*t2_abab(f,e,i,k)*t2_abab(a,d,j,m)*t2_abab(b,c,n,l)
    quadruples_res +=  0.500000000000000 * einsum('nmfe,feik,adjm,bcnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,j)*t2_abab(a,c,m,l)*t2_abab(b,d,n,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efij,acml,bdnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,f,i,j)*t2_abab(a,d,m,l)*t2_abab(b,c,n,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efij,adml,bcnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t2_abab(f,c,j,k)*t2_abab(b,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aeml,fcjk,bdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t2_aaaa(f,b,i,j)*t2_bbbb(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeml,fbij,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t2_abab(f,d,j,k)*t2_abab(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeml,fdjk,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||f,e>_abab*t2_abab(a,e,m,k)*t2_abab(f,c,j,l)*t2_abab(b,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aemk,fcjl,bdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*t2_abab(a,e,m,k)*t2_abab(f,d,j,l)*t2_abab(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aemk,fdjl,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 <m,n||e,f>_abab*t2_aaaa(e,a,j,m)*t2_bbbb(f,c,k,l)*t2_abab(b,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,eajm,fckl,bdin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,j,m)*t2_bbbb(f,c,k,l)*t2_abab(b,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,aejm,fckl,bdin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*t2_aaaa(e,a,j,m)*t2_abab(b,f,i,l)*t2_bbbb(c,d,k,n)
    quadruples_res += -1.000000000000000 * einsum('mnef,eajm,bfil,cdkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,j,m)*t2_abab(b,f,i,l)*t2_bbbb(c,d,k,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,aejm,bfil,cdkn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_abab(f,c,i,l)*t2_abab(b,d,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmef,eajm,fcil,bdnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_abab(f,c,i,l)*t2_abab(b,d,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmfe,aejm,fcil,bdnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(i,j)<m,n||e,f>_abab*t2_aaaa(e,a,j,m)*t2_abab(b,f,i,k)*t2_bbbb(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eajm,bfik,cdln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_bbbb*t2_abab(a,e,j,m)*t2_abab(b,f,i,k)*t2_bbbb(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aejm,bfik,cdln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_abab(f,c,i,k)*t2_abab(b,d,n,l)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fcik,bdnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_abab(f,c,i,k)*t2_abab(b,d,n,l)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aejm,fcik,bdnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,j,m)*t2_bbbb(f,d,k,l)*t2_abab(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eajm,fdkl,bcin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,j,m)*t2_bbbb(f,d,k,l)*t2_abab(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aejm,fdkl,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_abab(f,d,i,l)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fdil,bcnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_abab(f,d,i,l)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aejm,fdil,bcnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,j,m)*t2_abab(f,d,i,k)*t2_abab(b,c,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajm,fdik,bcnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,m)*t2_abab(f,d,i,k)*t2_abab(b,c,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aejm,fdik,bcnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 <m,n||e,f>_abab*t2_aaaa(e,a,i,m)*t2_bbbb(f,c,k,l)*t2_abab(b,d,j,n)
    quadruples_res += -1.000000000000000 * einsum('mnef,eaim,fckl,bdjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,i,m)*t2_bbbb(f,c,k,l)*t2_abab(b,d,j,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,aeim,fckl,bdjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*t2_aaaa(e,a,i,m)*t2_abab(b,f,j,l)*t2_bbbb(c,d,k,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,eaim,bfjl,cdkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,i,m)*t2_abab(b,f,j,l)*t2_bbbb(c,d,k,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,aeim,bfjl,cdkn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,a,i,m)*t2_abab(f,c,j,l)*t2_abab(b,d,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmef,eaim,fcjl,bdnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_abab(a,e,i,m)*t2_abab(f,c,j,l)*t2_abab(b,d,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,aeim,fcjl,bdnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,i,m)*t2_bbbb(f,d,k,l)*t2_abab(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eaim,fdkl,bcjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,i,m)*t2_bbbb(f,d,k,l)*t2_abab(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aeim,fdkl,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,m)*t2_abab(f,d,j,l)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaim,fdjl,bcnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,i,m)*t2_abab(f,d,j,l)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aeim,fdjl,bcnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 <n,m||e,f>_bbbb*t2_abab(a,e,j,l)*t2_abab(b,f,i,k)*t2_bbbb(c,d,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmef,aejl,bfik,cdnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,c,i,k)*t2_abab(b,d,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmfe,aejl,fcik,bdnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,c,i,k)*t2_abab(b,d,m,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,aejl,fcik,bdmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,d,i,k)*t2_abab(b,c,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,aejl,fdik,bcnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,d,i,k)*t2_abab(b,c,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,aejl,fdik,bcmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,c,m,k)*t2_abab(b,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,aejl,fcmk,bdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,j,l)*t2_bbbb(f,c,k,m)*t2_abab(b,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,aejl,fckm,bdin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,m)*t2_bbbb(c,d,k,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,aejl,fbim,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,j,l)*t2_abab(b,f,i,m)*t2_bbbb(c,d,k,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,aejl,bfim,cdkn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,c,i,m)*t2_abab(b,d,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,aejl,fcim,bdnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,d,m,k)*t2_abab(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aejl,fdmk,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,j,l)*t2_bbbb(f,d,k,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aejl,fdkm,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,l)*t2_abab(f,d,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aejl,fdim,bcnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 <m,n||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,c,m,k)*t2_abab(b,d,j,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,aeil,fcmk,bdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(f,c,k,m)*t2_abab(b,d,j,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,aeil,fckm,bdjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(a,e,i,l)*t2_aaaa(f,b,j,m)*t2_bbbb(c,d,k,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,aeil,fbjm,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t2_abab(b,f,j,m)*t2_bbbb(c,d,k,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,aeil,bfjm,cdkn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,c,j,m)*t2_abab(b,d,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmfe,aeil,fcjm,bdnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 P(k,l)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t2_abab(b,f,j,k)*t2_bbbb(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,aeil,bfjk,cdnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,c,j,k)*t2_abab(b,d,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,aeil,fcjk,bdnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,c,j,k)*t2_abab(b,d,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,aeil,fcjk,bdmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,d,m,k)*t2_abab(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeil,fdmk,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(f,d,k,m)*t2_abab(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aeil,fdkm,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,d,j,m)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aeil,fdjm,bcnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,d,j,k)*t2_abab(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,aeil,fdjk,bcnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,d,j,k)*t2_abab(b,c,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,aeil,fdjk,bcmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,n||f,e>_abab*t2_abab(a,e,j,k)*t2_abab(f,c,m,l)*t2_abab(b,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,aejk,fcml,bdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,j,k)*t2_bbbb(f,c,l,m)*t2_abab(b,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,aejk,fclm,bdin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(a,e,j,k)*t2_aaaa(f,b,i,m)*t2_bbbb(c,d,l,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,aejk,fbim,cdln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,j,k)*t2_abab(b,f,i,m)*t2_bbbb(c,d,l,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,aejk,bfim,cdln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_abab(a,e,j,k)*t2_abab(f,c,i,m)*t2_abab(b,d,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmfe,aejk,fcim,bdnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*t2_abab(a,e,j,k)*t2_abab(b,f,i,l)*t2_bbbb(c,d,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmef,aejk,bfil,cdnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*t2_abab(a,e,j,k)*t2_abab(f,c,i,l)*t2_abab(b,d,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmfe,aejk,fcil,bdnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_abab(a,e,j,k)*t2_abab(f,c,i,l)*t2_abab(b,d,m,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,aejk,fcil,bdmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_aaaa(e,a,j,i)*t2_bbbb(f,c,k,l)*t2_abab(b,d,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmef,eaji,fckl,bdnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_aaaa(e,a,j,i)*t2_bbbb(f,c,k,l)*t2_abab(b,d,m,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,eaji,fckl,bdmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,j,k)*t2_abab(f,d,m,l)*t2_abab(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aejk,fdml,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,j,k)*t2_bbbb(f,d,l,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aejk,fdlm,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,k)*t2_abab(f,d,i,m)*t2_abab(b,c,n,l)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,aejk,fdim,bcnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,j,k)*t2_abab(f,d,i,l)*t2_abab(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,aejk,fdil,bcnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,j,k)*t2_abab(f,d,i,l)*t2_abab(b,c,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,aejk,fdil,bcmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,j,i)*t2_bbbb(f,d,k,l)*t2_abab(b,c,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,eaji,fdkl,bcnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,n||e,f>_abab*t2_aaaa(e,a,j,i)*t2_bbbb(f,d,k,l)*t2_abab(b,c,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,eaji,fdkl,bcmn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(a,e,i,k)*t2_abab(f,c,m,l)*t2_abab(b,d,j,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,aeik,fcml,bdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(f,c,l,m)*t2_abab(b,d,j,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,aeik,fclm,bdjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*t2_abab(a,e,i,k)*t2_aaaa(f,b,j,m)*t2_bbbb(c,d,l,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,aeik,fbjm,cdln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,i,k)*t2_abab(b,f,j,m)*t2_bbbb(c,d,l,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,aeik,bfjm,cdln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_abab(a,e,i,k)*t2_abab(f,c,j,m)*t2_abab(b,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,aeik,fcjm,bdnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 P(a,b)<m,n||f,e>_abab*t2_abab(a,e,i,k)*t2_abab(f,d,m,l)*t2_abab(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aeik,fdml,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(f,d,l,m)*t2_abab(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aeik,fdlm,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||f,e>_abab*t2_abab(a,e,i,k)*t2_abab(f,d,j,m)*t2_abab(b,c,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,aeik,fdjm,bcnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<m,n||e,f>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,f,m,l)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eaij,bfml,cdkn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,j)*t2_abab(f,c,m,l)*t2_abab(b,d,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaij,fcml,bdnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_abab*t2_aaaa(e,a,i,j)*t2_bbbb(f,c,l,m)*t2_abab(b,d,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fclm,bdnk->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>_aaaa*t2_aaaa(e,a,i,j)*t2_abab(f,d,m,l)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fdml,bcnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,f>_abab*t2_aaaa(e,a,i,j)*t2_bbbb(f,d,l,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaij,fdlm,bcnk->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||f,e>_abab*t2_abab(b,e,m,l)*t2_abab(f,c,j,k)*t2_abab(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,beml,fcjk,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||f,e>_abab*t2_abab(b,e,m,k)*t2_abab(f,c,j,l)*t2_abab(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,bemk,fcjl,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 <m,n||e,f>_abab*t2_aaaa(e,b,j,m)*t2_bbbb(f,c,k,l)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnef,ebjm,fckl,adin->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(b,e,j,m)*t2_bbbb(f,c,k,l)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,bejm,fckl,adin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,b,j,m)*t2_abab(f,c,i,l)*t2_abab(a,d,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebjm,fcil,adnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_abab(b,e,j,m)*t2_abab(f,c,i,l)*t2_abab(a,d,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,bejm,fcil,adnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)<n,m||e,f>_aaaa*t2_aaaa(e,b,j,m)*t2_abab(f,c,i,k)*t2_abab(a,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebjm,fcik,adnl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*t2_abab(b,e,j,m)*t2_abab(f,c,i,k)*t2_abab(a,d,n,l)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,bejm,fcik,adnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 <m,n||e,f>_abab*t2_aaaa(e,b,i,m)*t2_bbbb(f,c,k,l)*t2_abab(a,d,j,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,ebim,fckl,adjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(b,e,i,m)*t2_bbbb(f,c,k,l)*t2_abab(a,d,j,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,beim,fckl,adjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,b,i,m)*t2_abab(f,c,j,l)*t2_abab(a,d,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebim,fcjl,adnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_abab(b,e,i,m)*t2_abab(f,c,j,l)*t2_abab(a,d,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmfe,beim,fcjl,adnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*t2_abab(b,e,j,l)*t2_abab(f,c,i,k)*t2_abab(a,d,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmfe,bejl,fcik,adnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_abab(b,e,j,l)*t2_abab(f,c,i,k)*t2_abab(a,d,m,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,bejl,fcik,admn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*t2_abab(b,e,j,l)*t2_abab(f,c,m,k)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,bejl,fcmk,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(b,e,j,l)*t2_bbbb(f,c,k,m)*t2_abab(a,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,bejl,fckm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_abab(b,e,j,l)*t2_abab(f,c,i,m)*t2_abab(a,d,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmfe,bejl,fcim,adnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(b,e,i,l)*t2_abab(f,c,m,k)*t2_abab(a,d,j,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,beil,fcmk,adjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(b,e,i,l)*t2_bbbb(f,c,k,m)*t2_abab(a,d,j,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,beil,fckm,adjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_abab(b,e,i,l)*t2_abab(f,c,j,m)*t2_abab(a,d,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,beil,fcjm,adnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 P(k,l)<n,m||f,e>_abab*t2_abab(b,e,i,l)*t2_abab(f,c,j,k)*t2_abab(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,beil,fcjk,adnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(b,e,i,l)*t2_abab(f,c,j,k)*t2_abab(a,d,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,beil,fcjk,admn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(b,e,j,k)*t2_abab(f,c,m,l)*t2_abab(a,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,bejk,fcml,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_abab(b,e,j,k)*t2_bbbb(f,c,l,m)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,bejk,fclm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_abab(b,e,j,k)*t2_abab(f,c,i,m)*t2_abab(a,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,bejk,fcim,adnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(b,e,j,k)*t2_abab(f,c,i,l)*t2_abab(a,d,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmfe,bejk,fcil,adnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(b,e,j,k)*t2_abab(f,c,i,l)*t2_abab(a,d,m,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,bejk,fcil,admn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_aaaa(e,b,j,i)*t2_bbbb(f,c,k,l)*t2_abab(a,d,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmef,ebji,fckl,adnm->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_aaaa(e,b,j,i)*t2_bbbb(f,c,k,l)*t2_abab(a,d,m,n)
    quadruples_res +=  0.500000000000000 * einsum('mnef,ebji,fckl,admn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*t2_abab(b,e,i,k)*t2_abab(f,c,m,l)*t2_abab(a,d,j,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,beik,fcml,adjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(b,e,i,k)*t2_bbbb(f,c,l,m)*t2_abab(a,d,j,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,beik,fclm,adjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_abab(b,e,i,k)*t2_abab(f,c,j,m)*t2_abab(a,d,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmfe,beik,fcjm,adnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 P(k,l)<n,m||e,f>_aaaa*t2_aaaa(e,b,i,j)*t2_abab(f,c,m,l)*t2_abab(a,d,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebij,fcml,adnk->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_abab*t2_aaaa(e,b,i,j)*t2_bbbb(f,c,l,m)*t2_abab(a,d,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebij,fclm,adnk->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_aaaa*t2_abab(e,c,m,l)*t2_abab(f,d,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecml,fdjk,abin->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*t2_bbbb(e,c,l,m)*t2_abab(f,d,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,eclm,fdjk,abin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_aaaa*t2_abab(e,c,m,k)*t2_abab(f,d,j,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecmk,fdjl,abin->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||f,e>_abab*t2_bbbb(e,c,k,m)*t2_abab(f,d,j,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,eckm,fdjl,abin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 <n,m||e,f>_abab*t2_abab(e,c,j,m)*t2_bbbb(f,d,k,l)*t2_aaaa(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecjm,fdkl,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*t2_abab(e,c,i,m)*t2_bbbb(f,d,k,l)*t2_aaaa(a,b,j,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecim,fdkl,abjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(f,d,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,eckl,fdjm,abin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 <n,m||e,f>_aaaa*t2_abab(e,c,j,l)*t2_abab(f,d,i,k)*t2_aaaa(a,b,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmef,ecjl,fdik,abnm->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_abab(e,c,j,l)*t2_abab(f,d,m,k)*t2_aaaa(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecjl,fdmk,abin->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*t2_abab(e,c,j,l)*t2_bbbb(f,d,k,m)*t2_aaaa(a,b,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecjl,fdkm,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_abab(e,c,i,l)*t2_abab(f,d,m,k)*t2_aaaa(a,b,j,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecil,fdmk,abjn->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*t2_abab(e,c,i,l)*t2_bbbb(f,d,k,m)*t2_aaaa(a,b,j,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecil,fdkm,abjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 P(k,l)<n,m||e,f>_aaaa*t2_abab(e,c,i,l)*t2_abab(f,d,j,k)*t2_aaaa(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ecil,fdjk,abnm->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_abab(e,c,j,k)*t2_abab(f,d,m,l)*t2_aaaa(a,b,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecjk,fdml,abin->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*t2_abab(e,c,j,k)*t2_bbbb(f,d,l,m)*t2_aaaa(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecjk,fdlm,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*t2_abab(e,c,j,k)*t2_abab(f,d,i,l)*t2_aaaa(a,b,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmef,ecjk,fdil,abnm->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_abab(e,c,i,k)*t2_abab(f,d,m,l)*t2_aaaa(a,b,j,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecik,fdml,abjn->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*t2_abab(e,c,i,k)*t2_bbbb(f,d,l,m)*t2_aaaa(a,b,j,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecik,fdlm,abjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    return quadruples_res


def ccdq_t4_abbbabbb_residual(t2_aaaa, t2_bbbb, t2_abab, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(k,l)f_bb(m,l)*t4_abbbabbb(a,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000010 * einsum('ml,abcdijkm->abcdijkl', f_bb[o, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 f_bb(m,j)*t4_abbbabbb(a,b,c,d,i,k,l,m)
    quadruples_res += -1.000000000000010 * einsum('mj,abcdiklm->abcdijkl', f_bb[o, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 f_aa(m,i)*t4_abbbabbb(a,b,c,d,m,k,l,j)
    quadruples_res += -1.000000000000010 * einsum('mi,abcdmklj->abcdijkl', f_aa[o, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  1.0000 f_aa(a,e)*t4_abbbabbb(e,b,c,d,i,j,k,l)
    quadruples_res +=  1.000000000000010 * einsum('ae,ebcdijkl->abcdijkl', f_aa[v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  1.0000 f_bb(b,e)*t4_abbbabbb(a,e,c,d,i,j,k,l)
    quadruples_res +=  1.000000000000010 * einsum('be,aecdijkl->abcdijkl', f_bb[v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 P(c,d)f_bb(c,e)*t4_abbbabbb(a,e,b,d,i,j,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('ce,aebdijkl->abcdijkl', f_bb[v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<n,m||k,l>_bbbb*t4_abbbabbb(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmkl,abcdijnm->abcdijkl', g_bbbb[o, o, o, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||i,l>_abab*t4_abbbabbb(a,b,c,d,n,k,j,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmil,abcdnkjm->abcdijkl', g_abab[o, o, o, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<m,n||i,l>_abab*t4_abbbabbb(a,b,c,d,m,k,n,j)
    contracted_intermediate =  0.500000000000010 * einsum('mnil,abcdmknj->abcdijkl', g_abab[o, o, o, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 <n,m||j,k>_bbbb*t4_abbbabbb(a,b,c,d,i,l,n,m)
    quadruples_res +=  0.500000000000010 * einsum('nmjk,abcdilnm->abcdijkl', g_bbbb[o, o, o, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 <n,m||i,j>_abab*t4_abbbabbb(a,b,c,d,n,l,k,m)
    quadruples_res += -0.500000000000010 * einsum('nmij,abcdnlkm->abcdijkl', g_abab[o, o, o, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <m,n||i,j>_abab*t4_abbbabbb(a,b,c,d,m,l,n,k)
    quadruples_res +=  0.500000000000010 * einsum('mnij,abcdmlnk->abcdijkl', g_abab[o, o, o, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 P(k,l)<a,m||e,l>_abab*t4_abbbabbb(e,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('amel,ebcdijkm->abcdijkl', g_abab[v, o, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,b||e,l>_abab*t4_aabbaabb(e,a,c,d,i,m,k,j)
    contracted_intermediate =  1.000000000000020 * einsum('mbel,eacdimkj->abcdijkl', g_abab[o, v, v, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,b||e,l>_bbbb*t4_abbbabbb(a,e,c,d,i,j,k,m)
    contracted_intermediate =  1.000000000000020 * einsum('mbel,aecdijkm->abcdijkl', g_bbbb[o, v, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <a,m||e,j>_abab*t4_abbbabbb(e,b,c,d,i,k,l,m)
    quadruples_res += -1.000000000000020 * einsum('amej,ebcdiklm->abcdijkl', g_abab[v, o, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  1.0000 <m,b||e,j>_abab*t4_aabbaabb(e,a,c,d,i,m,l,k)
    quadruples_res +=  1.000000000000020 * einsum('mbej,eacdimlk->abcdijkl', g_abab[o, v, v, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    
    #	  1.0000 <m,b||e,j>_bbbb*t4_abbbabbb(a,e,c,d,i,k,l,m)
    quadruples_res +=  1.000000000000020 * einsum('mbej,aecdiklm->abcdijkl', g_bbbb[o, v, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  1.0000 <m,a||e,i>_aaaa*t4_abbbabbb(e,b,c,d,m,k,l,j)
    quadruples_res +=  1.000000000000020 * einsum('maei,ebcdmklj->abcdijkl', g_aaaa[o, v, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 <a,m||i,e>_abab*t4_bbbbbbbb(e,b,c,d,j,k,l,m)
    quadruples_res += -1.000000000000020 * einsum('amie,ebcdjklm->abcdijkl', g_abab[v, o, o, v], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 <m,b||i,e>_abab*t4_abbbabbb(a,e,c,d,m,k,l,j)
    quadruples_res += -1.000000000000020 * einsum('mbie,aecdmklj->abcdijkl', g_abab[o, v, o, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,l>_abab*t4_aabbaabb(e,a,b,d,i,m,k,j)
    contracted_intermediate = -1.000000000000020 * einsum('mcel,eabdimkj->abcdijkl', g_abab[o, v, v, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,l>_bbbb*t4_abbbabbb(a,e,b,d,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('mcel,aebdijkm->abcdijkl', g_bbbb[o, v, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||e,j>_abab*t4_aabbaabb(e,a,b,d,i,m,l,k)
    contracted_intermediate = -1.000000000000020 * einsum('mcej,eabdimlk->abcdijkl', g_abab[o, v, v, o], t4_aabbaabb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||e,j>_bbbb*t4_abbbabbb(a,e,b,d,i,k,l,m)
    contracted_intermediate = -1.000000000000020 * einsum('mcej,aebdiklm->abcdijkl', g_bbbb[o, v, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<m,c||i,e>_abab*t4_abbbabbb(a,e,b,d,m,k,l,j)
    contracted_intermediate =  1.000000000000020 * einsum('mcie,aebdmklj->abcdijkl', g_abab[o, v, o, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<a,b||e,f>_abab*t4_abbbabbb(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('abef,efcdijkl->abcdijkl', g_abab[v, v, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<a,b||f,e>_abab*t4_abbbabbb(f,e,c,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('abfe,fecdijkl->abcdijkl', g_abab[v, v, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 <a,d||e,f>_abab*t4_abbbabbb(e,f,b,c,i,j,k,l)
    quadruples_res +=  0.500000000000010 * einsum('adef,efbcijkl->abcdijkl', g_abab[v, v, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <a,d||f,e>_abab*t4_abbbabbb(f,e,b,c,i,j,k,l)
    quadruples_res +=  0.500000000000010 * einsum('adfe,febcijkl->abcdijkl', g_abab[v, v, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	  0.5000 <b,d||e,f>_bbbb*t4_abbbabbb(a,f,e,c,i,j,k,l)
    quadruples_res +=  0.500000000000010 * einsum('bdef,afecijkl->abcdijkl', g_bbbb[v, v, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 P(b,d)<b,c||e,f>_bbbb*t4_abbbabbb(a,f,e,d,i,j,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('bcef,afedijkl->abcdijkl', g_bbbb[v, v, v, v], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(e,f,m,l)*t4_abbbabbb(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,efml,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(f,e,m,l)*t4_abbbabbb(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,feml,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,f,l,m)*t4_abbbabbb(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eflm,abcdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,f,m,j)*t4_abbbabbb(a,b,c,d,i,k,l,n)
    quadruples_res += -0.499999999999950 * einsum('mnef,efmj,abcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(f,e,m,j)*t4_abbbabbb(a,b,c,d,i,k,l,n)
    quadruples_res += -0.499999999999950 * einsum('mnfe,femj,abcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*t2_bbbb(e,f,j,m)*t4_abbbabbb(a,b,c,d,i,k,l,n)
    quadruples_res += -0.499999999999950 * einsum('nmef,efjm,abcdikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*t2_aaaa(e,f,i,m)*t4_abbbabbb(a,b,c,d,n,k,l,j)
    quadruples_res += -0.499999999999950 * einsum('nmef,efim,abcdnklj->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,f,i,m)*t4_abbbabbb(a,b,c,d,n,k,l,j)
    quadruples_res += -0.499999999999950 * einsum('nmef,efim,abcdnklj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(f,e,i,m)*t4_abbbabbb(a,b,c,d,n,k,l,j)
    quadruples_res += -0.499999999999950 * einsum('nmfe,feim,abcdnklj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,f,k,l)*t4_abbbabbb(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,efkl,abcdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.2500 P(k,l)<n,m||e,f>_abab*t2_abab(e,f,i,l)*t4_abbbabbb(a,b,c,d,n,k,j,m)
    contracted_intermediate = -0.250000000000010 * einsum('nmef,efil,abcdnkjm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.2500 P(k,l)<m,n||e,f>_abab*t2_abab(e,f,i,l)*t4_abbbabbb(a,b,c,d,m,k,n,j)
    contracted_intermediate =  0.250000000000010 * einsum('mnef,efil,abcdmknj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.2500 P(k,l)<n,m||f,e>_abab*t2_abab(f,e,i,l)*t4_abbbabbb(a,b,c,d,n,k,j,m)
    contracted_intermediate = -0.250000000000010 * einsum('nmfe,feil,abcdnkjm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.2500 P(k,l)<m,n||f,e>_abab*t2_abab(f,e,i,l)*t4_abbbabbb(a,b,c,d,m,k,n,j)
    contracted_intermediate =  0.250000000000010 * einsum('mnfe,feil,abcdmknj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,f>_bbbb*t2_bbbb(e,f,j,k)*t4_abbbabbb(a,b,c,d,i,l,n,m)
    quadruples_res +=  0.250000000000010 * einsum('nmef,efjk,abcdilnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*t2_abab(e,f,i,j)*t4_abbbabbb(a,b,c,d,n,l,k,m)
    quadruples_res += -0.250000000000010 * einsum('nmef,efij,abcdnlkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*t2_abab(e,f,i,j)*t4_abbbabbb(a,b,c,d,m,l,n,k)
    quadruples_res +=  0.250000000000010 * einsum('mnef,efij,abcdmlnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*t2_abab(f,e,i,j)*t4_abbbabbb(a,b,c,d,n,l,k,m)
    quadruples_res += -0.250000000000010 * einsum('nmfe,feij,abcdnlkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*t2_abab(f,e,i,j)*t4_abbbabbb(a,b,c,d,m,l,n,k)
    quadruples_res +=  0.250000000000010 * einsum('mnfe,feij,abcdmlnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*t2_aaaa(e,a,n,m)*t4_abbbabbb(f,b,c,d,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,eanm,fbcdijkl->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(a,e,n,m)*t4_abbbabbb(f,b,c,d,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('nmfe,aenm,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(a,e,m,n)*t4_abbbabbb(f,b,c,d,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('mnfe,aemn,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,b,n,m)*t4_abbbabbb(a,f,c,d,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,ebnm,afcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,b,m,n)*t4_abbbabbb(a,f,c,d,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('mnef,ebmn,afcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*t2_bbbb(e,b,n,m)*t4_abbbabbb(a,f,c,d,i,j,k,l)
    quadruples_res += -0.499999999999950 * einsum('nmef,ebnm,afcdijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(k,l)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t4_abbbabbb(f,b,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnfe,aeml,fbcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_aaaa*t2_abab(e,b,m,l)*t4_aabbaabb(f,a,c,d,i,n,k,j)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,ebml,facdinkj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,n||e,f>_abab*t2_abab(e,b,m,l)*t4_abbbabbb(a,f,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,ebml,afcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||f,e>_abab*t2_bbbb(e,b,l,m)*t4_aabbaabb(f,a,c,d,i,n,k,j)
    contracted_intermediate = -0.999999999999840 * einsum('nmfe,eblm,facdinkj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,b,l,m)*t4_abbbabbb(a,f,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eblm,afcdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(a,e,m,j)*t4_abbbabbb(f,b,c,d,i,k,l,n)
    quadruples_res +=  0.999999999999840 * einsum('mnfe,aemj,fbcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_abab(e,b,m,j)*t4_aabbaabb(f,a,c,d,i,n,l,k)
    quadruples_res += -0.999999999999840 * einsum('nmef,ebmj,facdinlk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*t2_abab(e,b,m,j)*t4_abbbabbb(a,f,c,d,i,k,l,n)
    quadruples_res +=  0.999999999999840 * einsum('mnef,ebmj,afcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,j,m)*t4_aabbaabb(f,a,c,d,i,n,l,k)
    quadruples_res += -0.999999999999840 * einsum('nmfe,ebjm,facdinlk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,j,m)*t4_abbbabbb(a,f,c,d,i,k,l,n)
    quadruples_res +=  0.999999999999840 * einsum('nmef,ebjm,afcdikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_aaaa(e,a,i,m)*t4_abbbabbb(f,b,c,d,n,k,l,j)
    quadruples_res +=  0.999999999999840 * einsum('nmef,eaim,fbcdnklj->abcdijkl', g_aaaa[o, o, v, v], t2_aaaa, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*t2_aaaa(e,a,i,m)*t4_bbbbbbbb(f,b,c,d,j,k,l,n)
    quadruples_res +=  0.999999999999840 * einsum('mnef,eaim,fbcdjkln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_abab(a,e,i,m)*t4_abbbabbb(f,b,c,d,n,k,l,j)
    quadruples_res +=  0.999999999999840 * einsum('nmfe,aeim,fbcdnklj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,i,m)*t4_bbbbbbbb(f,b,c,d,j,k,l,n)
    quadruples_res +=  0.999999999999840 * einsum('nmef,aeim,fbcdjkln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*t2_abab(e,b,i,m)*t4_abbbabbb(a,f,c,d,n,k,l,j)
    quadruples_res +=  0.999999999999840 * einsum('nmef,ebim,afcdnklj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 P(j,k)<n,m||f,e>_abab*t2_bbbb(e,b,k,l)*t4_aabbaabb(f,a,c,d,i,n,j,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,ebkl,facdinjm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<m,n||f,e>_abab*t2_bbbb(e,b,k,l)*t4_aabbaabb(f,a,c,d,i,m,n,j)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,ebkl,facdimnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,b,k,l)*t4_abbbabbb(a,f,c,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ebkl,afcdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||f,e>_abab*t2_abab(a,e,i,l)*t4_abbbabbb(f,b,c,d,n,k,j,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,aeil,fbcdnkjm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t4_abbbabbb(f,b,c,d,m,k,n,j)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,aeil,fbcdmknj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t4_bbbbbbbb(f,b,c,d,j,k,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,aeil,fbcdjknm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,f>_aaaa*t2_abab(e,b,i,l)*t4_aabbaabb(f,a,c,d,n,m,j,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ebil,facdnmjk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,f>_abab*t2_abab(e,b,i,l)*t4_abbbabbb(a,f,c,d,n,k,j,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ebil,afcdnkjm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(e,b,i,l)*t4_abbbabbb(a,f,c,d,m,k,n,j)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,ebil,afcdmknj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 <n,m||f,e>_abab*t2_bbbb(e,b,j,k)*t4_aabbaabb(f,a,c,d,i,n,l,m)
    quadruples_res +=  0.499999999999950 * einsum('nmfe,ebjk,facdinlm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_bbbb(e,b,j,k)*t4_aabbaabb(f,a,c,d,i,m,n,l)
    quadruples_res += -0.499999999999950 * einsum('mnfe,ebjk,facdimnl->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*t2_bbbb(e,b,j,k)*t4_abbbabbb(a,f,c,d,i,l,n,m)
    quadruples_res += -0.499999999999950 * einsum('nmef,ebjk,afcdilnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*t2_abab(a,e,i,j)*t4_abbbabbb(f,b,c,d,n,l,k,m)
    quadruples_res +=  0.499999999999950 * einsum('nmfe,aeij,fbcdnlkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(a,e,i,j)*t4_abbbabbb(f,b,c,d,m,l,n,k)
    quadruples_res += -0.499999999999950 * einsum('mnfe,aeij,fbcdmlnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*t2_abab(a,e,i,j)*t4_bbbbbbbb(f,b,c,d,k,l,n,m)
    quadruples_res +=  0.499999999999950 * einsum('nmef,aeij,fbcdklnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*t2_abab(e,b,i,j)*t4_aabbaabb(f,a,c,d,n,m,k,l)
    quadruples_res +=  0.499999999999950 * einsum('nmef,ebij,facdnmkl->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_abab(e,b,i,j)*t4_abbbabbb(a,f,c,d,n,l,k,m)
    quadruples_res +=  0.499999999999950 * einsum('nmef,ebij,afcdnlkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,b,i,j)*t4_abbbabbb(a,f,c,d,m,l,n,k)
    quadruples_res += -0.499999999999950 * einsum('mnef,ebij,afcdmlnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 P(c,d)<n,m||e,f>_abab*t2_abab(e,c,n,m)*t4_abbbabbb(a,f,b,d,i,j,k,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecnm,afbdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<m,n||e,f>_abab*t2_abab(e,c,m,n)*t4_abbbabbb(a,f,b,d,i,j,k,l)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,ecmn,afbdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,n,m)*t4_abbbabbb(a,f,b,d,i,j,k,l)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecnm,afbdijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,m,l)*t4_aabbaabb(f,a,b,d,i,n,k,j)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ecml,fabdinkj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,n||e,f>_abab*t2_abab(e,c,m,l)*t4_abbbabbb(a,f,b,d,i,j,k,n)
    contracted_intermediate = -0.999999999999840 * einsum('mnef,ecml,afbdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,l,m)*t4_aabbaabb(f,a,b,d,i,n,k,j)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,eclm,fabdinkj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,l,m)*t4_abbbabbb(a,f,b,d,i,j,k,n)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,eclm,afbdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,m,j)*t4_aabbaabb(f,a,b,d,i,n,l,k)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ecmj,fabdinlk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,n||e,f>_abab*t2_abab(e,c,m,j)*t4_abbbabbb(a,f,b,d,i,k,l,n)
    contracted_intermediate = -0.999999999999840 * einsum('mnef,ecmj,afbdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,j,m)*t4_aabbaabb(f,a,b,d,i,n,l,k)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,ecjm,fabdinlk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,j,m)*t4_abbbabbb(a,f,b,d,i,k,l,n)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,ecjm,afbdikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<n,m||e,f>_abab*t2_abab(e,c,i,m)*t4_abbbabbb(a,f,b,d,n,k,l,j)
    contracted_intermediate = -0.999999999999840 * einsum('nmef,ecim,afbdnklj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,k,l)*t4_aabbaabb(f,a,b,d,i,n,j,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,eckl,fabdinjm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(c,d)<m,n||f,e>_abab*t2_bbbb(e,c,k,l)*t4_aabbaabb(f,a,b,d,i,m,n,j)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,eckl,fabdimnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,k,l)*t4_abbbabbb(a,f,b,d,i,j,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,eckl,afbdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,i,l)*t4_aabbaabb(f,a,b,d,n,m,j,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecil,fabdnmjk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>_abab*t2_abab(e,c,i,l)*t4_abbbabbb(a,f,b,d,n,k,j,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecil,afbdnkjm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<m,n||e,f>_abab*t2_abab(e,c,i,l)*t4_abbbabbb(a,f,b,d,m,k,n,j)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,ecil,afbdmknj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,j,k)*t4_aabbaabb(f,a,b,d,i,n,l,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,ecjk,fabdinlm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<m,n||f,e>_abab*t2_bbbb(e,c,j,k)*t4_aabbaabb(f,a,b,d,i,m,n,l)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,ecjk,fabdimnl->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,j,k)*t4_abbbabbb(a,f,b,d,i,l,n,m)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ecjk,afbdilnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,i,j)*t4_aabbaabb(f,a,b,d,n,m,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecij,fabdnmkl->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||e,f>_abab*t2_abab(e,c,i,j)*t4_abbbabbb(a,f,b,d,n,l,k,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecij,afbdnlkm->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<m,n||e,f>_abab*t2_abab(e,c,i,j)*t4_abbbabbb(a,f,b,d,m,l,n,k)
    contracted_intermediate =  0.499999999999950 * einsum('mnef,ecij,afbdmlnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.2500 P(b,c)<n,m||e,f>_abab*t2_abab(a,b,n,m)*t4_abbbabbb(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,abnm,efcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.2500 P(b,c)<n,m||f,e>_abab*t2_abab(a,b,n,m)*t4_abbbabbb(f,e,c,d,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmfe,abnm,fecdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.2500 P(b,c)<m,n||e,f>_abab*t2_abab(a,b,m,n)*t4_abbbabbb(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('mnef,abmn,efcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.2500 P(b,c)<m,n||f,e>_abab*t2_abab(a,b,m,n)*t4_abbbabbb(f,e,c,d,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('mnfe,abmn,fecdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>_aaaa*t2_abab(a,b,m,l)*t4_aabbaabb(e,f,c,d,i,n,k,j)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,abml,efcdinkj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<m,n||e,f>_abab*t2_abab(a,b,m,l)*t4_abbbabbb(e,f,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,abml,efcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<m,n||f,e>_abab*t2_abab(a,b,m,l)*t4_abbbabbb(f,e,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,abml,fecdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<n,m||e,f>_aaaa*t2_abab(a,b,m,j)*t4_aabbaabb(e,f,c,d,i,n,l,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,abmj,efcdinlk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<m,n||e,f>_abab*t2_abab(a,b,m,j)*t4_abbbabbb(e,f,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,abmj,efcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<m,n||f,e>_abab*t2_abab(a,b,m,j)*t4_abbbabbb(f,e,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,abmj,fecdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<n,m||e,f>_abab*t2_abab(a,b,i,m)*t4_abbbabbb(e,f,c,d,n,k,l,j)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,abim,efcdnklj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<n,m||f,e>_abab*t2_abab(a,b,i,m)*t4_abbbabbb(f,e,c,d,n,k,l,j)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,abim,fecdnklj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<n,m||e,f>_bbbb*t2_abab(a,b,i,m)*t4_bbbbbbbb(e,f,c,d,j,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,abim,efcdjkln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,f>_abab*t2_abab(a,d,n,m)*t4_abbbabbb(e,f,b,c,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('nmef,adnm,efbcijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*t2_abab(a,d,n,m)*t4_abbbabbb(f,e,b,c,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('nmfe,adnm,febcijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*t2_abab(a,d,m,n)*t4_abbbabbb(e,f,b,c,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('mnef,admn,efbcijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*t2_abab(a,d,m,n)*t4_abbbabbb(f,e,b,c,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('mnfe,admn,febcijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*t2_bbbb(b,d,n,m)*t4_abbbabbb(a,f,e,c,i,j,k,l)
    quadruples_res +=  0.250000000000010 * einsum('nmef,bdnm,afecijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(k,l)<n,m||e,f>_aaaa*t2_abab(a,d,m,l)*t4_aabbaabb(e,f,b,c,i,n,k,j)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adml,efbcinkj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(a,d,m,l)*t4_abbbabbb(e,f,b,c,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,adml,efbcijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(a,d,m,l)*t4_abbbabbb(f,e,b,c,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,adml,febcijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,f>_abab*t2_bbbb(b,d,l,m)*t4_aabbaabb(e,a,f,c,i,n,k,j)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,bdlm,eafcinkj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||f,e>_abab*t2_bbbb(b,d,l,m)*t4_aabbaabb(a,f,e,c,i,n,k,j)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,bdlm,afecinkj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(b,d,l,m)*t4_abbbabbb(a,f,e,c,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bdlm,afecijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 <n,m||e,f>_aaaa*t2_abab(a,d,m,j)*t4_aabbaabb(e,f,b,c,i,n,l,k)
    quadruples_res += -0.499999999999950 * einsum('nmef,admj,efbcinlk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(a,d,m,j)*t4_abbbabbb(e,f,b,c,i,k,l,n)
    quadruples_res += -0.499999999999950 * einsum('mnef,admj,efbcikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_abab(a,d,m,j)*t4_abbbabbb(f,e,b,c,i,k,l,n)
    quadruples_res += -0.499999999999950 * einsum('mnfe,admj,febcikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_bbbb(b,d,j,m)*t4_aabbaabb(e,a,f,c,i,n,l,k)
    quadruples_res +=  0.499999999999950 * einsum('nmef,bdjm,eafcinlk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_bbbb(b,d,j,m)*t4_aabbaabb(a,f,e,c,i,n,l,k)
    quadruples_res += -0.499999999999950 * einsum('nmfe,bdjm,afecinlk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*t2_bbbb(b,d,j,m)*t4_abbbabbb(a,f,e,c,i,k,l,n)
    quadruples_res += -0.499999999999950 * einsum('nmef,bdjm,afecikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(a,d,i,m)*t4_abbbabbb(e,f,b,c,n,k,l,j)
    quadruples_res += -0.499999999999950 * einsum('nmef,adim,efbcnklj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_abab(a,d,i,m)*t4_abbbabbb(f,e,b,c,n,k,l,j)
    quadruples_res += -0.499999999999950 * einsum('nmfe,adim,febcnklj->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*t2_abab(a,d,i,m)*t4_bbbbbbbb(e,f,b,c,j,k,l,n)
    quadruples_res +=  0.499999999999950 * einsum('nmef,adim,efbcjkln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 P(b,d)<n,m||e,f>_bbbb*t2_bbbb(b,c,n,m)*t4_abbbabbb(a,f,e,d,i,j,k,l)
    contracted_intermediate = -0.250000000000010 * einsum('nmef,bcnm,afedijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,d)<n,m||e,f>_abab*t2_bbbb(b,c,l,m)*t4_aabbaabb(e,a,f,d,i,n,k,j)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bclm,eafdinkj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,d)<n,m||f,e>_abab*t2_bbbb(b,c,l,m)*t4_aabbaabb(a,f,e,d,i,n,k,j)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,bclm,afedinkj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,d)<n,m||e,f>_bbbb*t2_bbbb(b,c,l,m)*t4_abbbabbb(a,f,e,d,i,j,k,n)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,bclm,afedijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -0.5000 P(b,d)<n,m||e,f>_abab*t2_bbbb(b,c,j,m)*t4_aabbaabb(e,a,f,d,i,n,l,k)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bcjm,eafdinlk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,d)<n,m||f,e>_abab*t2_bbbb(b,c,j,m)*t4_aabbaabb(a,f,e,d,i,n,l,k)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,bcjm,afedinlk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,d)<n,m||e,f>_bbbb*t2_bbbb(b,c,j,m)*t4_abbbabbb(a,f,e,d,i,k,l,n)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,bcjm,afedikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<n,m||k,l>_bbbb*t2_abab(a,b,i,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmkl,abim,cdjn->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 <n,m||k,l>_bbbb*t2_abab(a,d,i,m)*t2_bbbb(b,c,j,n)
    quadruples_res +=  1.000000000000000 * einsum('nmkl,adim,bcjn->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(b,c)<n,m||j,l>_bbbb*t2_abab(a,b,i,m)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,abim,cdkn->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 <n,m||j,l>_bbbb*t2_abab(a,d,i,m)*t2_bbbb(b,c,k,n)
    quadruples_res += -1.000000000000000 * einsum('nmjl,adim,bckn->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(j,k)*P(b,c)<m,n||i,l>_abab*t2_abab(a,b,m,k)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnil,abmk,cdjn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,n||i,l>_abab*t2_abab(a,d,m,k)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnil,admk,bcjn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<n,m||j,k>_bbbb*t2_abab(a,b,i,m)*t2_bbbb(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjk,abim,cdln->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 <n,m||j,k>_bbbb*t2_abab(a,d,i,m)*t2_bbbb(b,c,l,n)
    quadruples_res +=  1.000000000000000 * einsum('nmjk,adim,bcln->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(j,l)*P(b,c)<m,n||i,k>_abab*t2_abab(a,b,m,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnik,abml,cdjn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<m,n||i,k>_abab*t2_abab(a,d,m,l)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnik,adml,bcjn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<m,n||i,j>_abab*t2_abab(a,b,m,l)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnij,abml,cdkn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<m,n||i,j>_abab*t2_abab(a,d,m,l)*t2_bbbb(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnij,adml,bckn->abcdijkl', g_abab[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<a,m||e,l>_abab*t2_abab(e,b,i,k)*t2_bbbb(c,d,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('amel,ebik,cdjm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<a,m||e,l>_abab*t2_abab(e,b,i,j)*t2_bbbb(c,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('amel,ebij,cdkm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,b||e,l>_bbbb*t2_bbbb(e,d,j,k)*t2_abab(a,c,i,m)
    quadruples_res += -1.000000000000000 * einsum('mbel,edjk,acim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,m||e,l>_abab*t2_abab(e,d,i,k)*t2_bbbb(b,c,j,m)
    quadruples_res +=  1.000000000000000 * einsum('amel,edik,bcjm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,l>_abab*t2_abab(e,d,i,k)*t2_abab(a,c,m,j)
    quadruples_res += -1.000000000000000 * einsum('mbel,edik,acmj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(k,l)<a,m||e,l>_abab*t2_abab(e,d,i,j)*t2_bbbb(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('amel,edij,bckm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,b||e,l>_abab*t2_abab(e,d,i,j)*t2_abab(a,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mbel,edij,acmk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<a,m||e,k>_abab*t2_abab(e,b,i,l)*t2_bbbb(c,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('amek,ebil,cdjm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 <m,b||e,k>_bbbb*t2_bbbb(e,d,j,l)*t2_abab(a,c,i,m)
    quadruples_res +=  1.000000000000000 * einsum('mbek,edjl,acim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,m||e,k>_abab*t2_abab(e,d,i,l)*t2_bbbb(b,c,j,m)
    quadruples_res += -1.000000000000000 * einsum('amek,edil,bcjm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,k>_abab*t2_abab(e,d,i,l)*t2_abab(a,c,m,j)
    quadruples_res +=  1.000000000000000 * einsum('mbek,edil,acmj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(b,c)<a,m||e,j>_abab*t2_abab(e,b,i,l)*t2_bbbb(c,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('amej,ebil,cdkm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<a,m||e,j>_abab*t2_abab(e,b,i,k)*t2_bbbb(c,d,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('amej,ebik,cdlm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<a,m||i,e>_abab*t2_bbbb(e,b,j,k)*t2_bbbb(c,d,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('amie,ebjk,cdlm->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 <m,b||e,j>_bbbb*t2_bbbb(e,d,k,l)*t2_abab(a,c,i,m)
    quadruples_res += -1.000000000000000 * einsum('mbej,edkl,acim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,m||e,j>_abab*t2_abab(e,d,i,l)*t2_bbbb(b,c,k,m)
    quadruples_res +=  1.000000000000000 * einsum('amej,edil,bckm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_abab*t2_abab(e,d,i,l)*t2_abab(a,c,m,k)
    quadruples_res += -1.000000000000000 * einsum('mbej,edil,acmk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,m||e,j>_abab*t2_abab(e,d,i,k)*t2_bbbb(b,c,l,m)
    quadruples_res += -1.000000000000000 * einsum('amej,edik,bclm->abcdijkl', g_abab[v, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_abab*t2_abab(e,d,i,k)*t2_abab(a,c,m,l)
    quadruples_res +=  1.000000000000000 * einsum('mbej,edik,acml->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,m||i,e>_abab*t2_bbbb(e,d,j,k)*t2_bbbb(b,c,l,m)
    quadruples_res += -1.000000000000000 * einsum('amie,edjk,bclm->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||i,e>_abab*t2_bbbb(e,d,j,k)*t2_abab(a,c,m,l)
    quadruples_res +=  1.000000000000000 * einsum('mbie,edjk,acml->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(j,k)*P(b,c)<a,m||i,e>_abab*t2_bbbb(e,b,k,l)*t2_bbbb(c,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('amie,ebkl,cdjm->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<a,m||i,e>_abab*t2_bbbb(e,d,k,l)*t2_bbbb(b,c,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('amie,edkl,bcjm->abcdijkl', g_abab[v, o, o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,b||i,e>_abab*t2_bbbb(e,d,k,l)*t2_abab(a,c,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('mbie,edkl,acmj->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 <m,b||e,l>_bbbb*t2_bbbb(e,c,j,k)*t2_abab(a,d,i,m)
    quadruples_res +=  1.000000000000000 * einsum('mbel,ecjk,adim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,l>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(c,d,j,m)
    quadruples_res += -1.000000000000000 * einsum('mbel,aeik,cdjm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_abab*t2_abab(e,c,i,k)*t2_abab(a,d,m,j)
    quadruples_res +=  1.000000000000000 * einsum('mbel,ecik,admj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(k,l)<m,b||e,l>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(c,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbel,aeij,cdkm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<m,b||e,l>_abab*t2_abab(e,c,i,j)*t2_abab(a,d,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,ecij,admk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,b||e,k>_bbbb*t2_bbbb(e,c,j,l)*t2_abab(a,d,i,m)
    quadruples_res += -1.000000000000000 * einsum('mbek,ecjl,adim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,k>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(c,d,j,m)
    quadruples_res +=  1.000000000000000 * einsum('mbek,aeil,cdjm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,k>_abab*t2_abab(e,c,i,l)*t2_abab(a,d,m,j)
    quadruples_res += -1.000000000000000 * einsum('mbek,ecil,admj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_bbbb*t2_bbbb(e,c,k,l)*t2_abab(a,d,i,m)
    quadruples_res +=  1.000000000000000 * einsum('mbej,eckl,adim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(c,d,k,m)
    quadruples_res += -1.000000000000000 * einsum('mbej,aeil,cdkm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_abab*t2_abab(e,c,i,l)*t2_abab(a,d,m,k)
    quadruples_res +=  1.000000000000000 * einsum('mbej,ecil,admk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(c,d,l,m)
    quadruples_res +=  1.000000000000000 * einsum('mbej,aeik,cdlm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_abab*t2_abab(e,c,i,k)*t2_abab(a,d,m,l)
    quadruples_res += -1.000000000000000 * einsum('mbej,ecik,adml->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||i,e>_abab*t2_bbbb(e,c,j,k)*t2_abab(a,d,m,l)
    quadruples_res += -1.000000000000000 * einsum('mbie,ecjk,adml->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(j,k)<m,b||i,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(a,d,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('mbie,eckl,admj->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 <m,c||e,l>_bbbb*t2_bbbb(e,b,j,k)*t2_abab(a,d,i,m)
    quadruples_res += -1.000000000000000 * einsum('mcel,ebjk,adim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,c||e,l>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(b,d,j,m)
    quadruples_res +=  1.000000000000000 * einsum('mcel,aeik,bdjm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,c||e,l>_abab*t2_abab(e,b,i,k)*t2_abab(a,d,m,j)
    quadruples_res += -1.000000000000000 * einsum('mcel,ebik,admj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(k,l)<m,c||e,l>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(b,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcel,aeij,bdkm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,c||e,l>_abab*t2_abab(e,b,i,j)*t2_abab(a,d,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,ebij,admk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<m,c||e,l>_bbbb*t2_bbbb(e,d,j,k)*t2_abab(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edjk,abim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<m,c||e,l>_abab*t2_abab(e,d,i,k)*t2_abab(a,b,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edik,abmj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,l>_abab*t2_abab(e,d,i,j)*t2_abab(a,b,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mcel,edij,abmk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 <m,c||e,k>_bbbb*t2_bbbb(e,b,j,l)*t2_abab(a,d,i,m)
    quadruples_res +=  1.000000000000000 * einsum('mcek,ebjl,adim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,c||e,k>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(b,d,j,m)
    quadruples_res += -1.000000000000000 * einsum('mcek,aeil,bdjm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,c||e,k>_abab*t2_abab(e,b,i,l)*t2_abab(a,d,m,j)
    quadruples_res +=  1.000000000000000 * einsum('mcek,ebil,admj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(c,d)<m,c||e,k>_bbbb*t2_bbbb(e,d,j,l)*t2_abab(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,edjl,abim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||e,k>_abab*t2_abab(e,d,i,l)*t2_abab(a,b,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,edil,abmj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 <m,c||e,j>_bbbb*t2_bbbb(e,b,k,l)*t2_abab(a,d,i,m)
    quadruples_res += -1.000000000000000 * einsum('mcej,ebkl,adim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,c||e,j>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(b,d,k,m)
    quadruples_res +=  1.000000000000000 * einsum('mcej,aeil,bdkm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,c||e,j>_abab*t2_abab(e,b,i,l)*t2_abab(a,d,m,k)
    quadruples_res += -1.000000000000000 * einsum('mcej,ebil,admk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,c||e,j>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(b,d,l,m)
    quadruples_res += -1.000000000000000 * einsum('mcej,aeik,bdlm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,c||e,j>_abab*t2_abab(e,b,i,k)*t2_abab(a,d,m,l)
    quadruples_res +=  1.000000000000000 * einsum('mcej,ebik,adml->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,c||i,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(a,d,m,l)
    quadruples_res +=  1.000000000000000 * einsum('mcie,ebjk,adml->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(c,d)<m,c||e,j>_bbbb*t2_bbbb(e,d,k,l)*t2_abab(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edkl,abim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<m,c||e,j>_abab*t2_abab(e,d,i,l)*t2_abab(a,b,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edil,abmk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||e,j>_abab*t2_abab(e,d,i,k)*t2_abab(a,b,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('mcej,edik,abml->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||i,e>_abab*t2_bbbb(e,d,j,k)*t2_abab(a,b,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('mcie,edjk,abml->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,c||i,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(a,d,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('mcie,ebkl,admj->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<m,c||i,e>_abab*t2_bbbb(e,d,k,l)*t2_abab(a,b,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('mcie,edkl,abmj->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 <m,d||e,l>_bbbb*t2_bbbb(e,b,j,k)*t2_abab(a,c,i,m)
    quadruples_res +=  1.000000000000000 * einsum('mdel,ebjk,acim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(b,c,j,m)
    quadruples_res += -1.000000000000000 * einsum('mdel,aeik,bcjm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_abab*t2_abab(e,b,i,k)*t2_abab(a,c,m,j)
    quadruples_res +=  1.000000000000000 * einsum('mdel,ebik,acmj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(k,l)<m,d||e,l>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,aeij,bckm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<m,d||e,l>_abab*t2_abab(e,b,i,j)*t2_abab(a,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,ebij,acmk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,d||e,k>_bbbb*t2_bbbb(e,b,j,l)*t2_abab(a,c,i,m)
    quadruples_res += -1.000000000000000 * einsum('mdek,ebjl,acim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,k>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(b,c,j,m)
    quadruples_res +=  1.000000000000000 * einsum('mdek,aeil,bcjm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,k>_abab*t2_abab(e,b,i,l)*t2_abab(a,c,m,j)
    quadruples_res += -1.000000000000000 * einsum('mdek,ebil,acmj->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_bbbb*t2_bbbb(e,b,k,l)*t2_abab(a,c,i,m)
    quadruples_res +=  1.000000000000000 * einsum('mdej,ebkl,acim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(b,c,k,m)
    quadruples_res += -1.000000000000000 * einsum('mdej,aeil,bckm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_abab*t2_abab(e,b,i,l)*t2_abab(a,c,m,k)
    quadruples_res +=  1.000000000000000 * einsum('mdej,ebil,acmk->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(b,c,l,m)
    quadruples_res +=  1.000000000000000 * einsum('mdej,aeik,bclm->abcdijkl', g_bbbb[o, v, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_abab*t2_abab(e,b,i,k)*t2_abab(a,c,m,l)
    quadruples_res += -1.000000000000000 * einsum('mdej,ebik,acml->abcdijkl', g_abab[o, v, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||i,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(a,c,m,l)
    quadruples_res += -1.000000000000000 * einsum('mdie,ebjk,acml->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(j,k)<m,d||i,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(a,c,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('mdie,ebkl,acmj->abcdijkl', g_abab[o, v, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<a,b||f,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(f,d,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('abfe,eckl,fdij->abcdijkl', g_abab[v, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<a,b||e,f>_abab*t2_abab(e,c,i,l)*t2_bbbb(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecil,fdjk->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<a,b||f,e>_abab*t2_bbbb(e,c,j,k)*t2_abab(f,d,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('abfe,ecjk,fdil->abcdijkl', g_abab[v, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<a,b||e,f>_abab*t2_abab(e,c,i,j)*t2_bbbb(f,d,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecij,fdkl->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<a,d||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('adfe,ebkl,fcij->abcdijkl', g_abab[v, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<a,d||e,f>_abab*t2_abab(e,b,i,l)*t2_bbbb(f,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebil,fcjk->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<b,d||e,f>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(f,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('bdef,aeil,fcjk->abcdijkl', g_bbbb[v, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 <a,d||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,c,i,l)
    quadruples_res +=  1.000000000000000 * einsum('adfe,ebjk,fcil->abcdijkl', g_abab[v, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,d||e,f>_abab*t2_abab(e,b,i,j)*t2_bbbb(f,c,k,l)
    quadruples_res += -1.000000000000000 * einsum('adef,ebij,fckl->abcdijkl', g_abab[v, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,d||e,f>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(f,c,k,l)
    quadruples_res += -1.000000000000000 * einsum('bdef,aeij,fckl->abcdijkl', g_bbbb[v, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(k,l)*P(b,d)<b,c||e,f>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(f,d,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('bcef,aeil,fdjk->abcdijkl', g_bbbb[v, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  1.0000 P(b,d)<b,c||e,f>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(f,d,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('bcef,aeij,fdkl->abcdijkl', g_bbbb[v, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,f,k,l)*t2_abab(a,b,i,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efkl,abim,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 <n,m||e,f>_bbbb*t2_bbbb(e,f,k,l)*t2_abab(a,d,i,m)*t2_bbbb(b,c,j,n)
    quadruples_res +=  0.500000000000000 * einsum('nmef,efkl,adim,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,f,j,l)*t2_abab(a,b,i,m)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efjl,abim,cdkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 <n,m||e,f>_bbbb*t2_bbbb(e,f,j,l)*t2_abab(a,d,i,m)*t2_bbbb(b,c,k,n)
    quadruples_res += -0.500000000000000 * einsum('nmef,efjl,adim,bckn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 P(j,k)*P(b,c)<m,n||e,f>_abab*t2_abab(e,f,i,l)*t2_abab(a,b,m,k)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,efil,abmk,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<m,n||f,e>_abab*t2_abab(f,e,i,l)*t2_abab(a,b,m,k)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,feil,abmk,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<m,n||e,f>_abab*t2_abab(e,f,i,l)*t2_abab(a,d,m,k)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,efil,admk,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<m,n||f,e>_abab*t2_abab(f,e,i,l)*t2_abab(a,d,m,k)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,feil,admk,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,f,j,k)*t2_abab(a,b,i,m)*t2_bbbb(c,d,l,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efjk,abim,cdln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 <n,m||e,f>_bbbb*t2_bbbb(e,f,j,k)*t2_abab(a,d,i,m)*t2_bbbb(b,c,l,n)
    quadruples_res +=  0.500000000000000 * einsum('nmef,efjk,adim,bcln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 P(j,l)*P(b,c)<m,n||e,f>_abab*t2_abab(e,f,i,k)*t2_abab(a,b,m,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,efik,abml,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  0.5000 P(j,l)*P(b,c)<m,n||f,e>_abab*t2_abab(f,e,i,k)*t2_abab(a,b,m,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,feik,abml,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  0.5000 P(j,l)<m,n||e,f>_abab*t2_abab(e,f,i,k)*t2_abab(a,d,m,l)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,efik,adml,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  0.5000 P(j,l)<m,n||f,e>_abab*t2_abab(f,e,i,k)*t2_abab(a,d,m,l)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,feik,adml,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<m,n||e,f>_abab*t2_abab(e,f,i,j)*t2_abab(a,b,m,l)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,efij,abml,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<m,n||f,e>_abab*t2_abab(f,e,i,j)*t2_abab(a,b,m,l)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,feij,abml,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(e,f,i,j)*t2_abab(a,d,m,l)*t2_bbbb(b,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,efij,adml,bckn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(f,e,i,j)*t2_abab(a,d,m,l)*t2_bbbb(b,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,feij,adml,bckn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t2_abab(f,b,i,k)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aeml,fbik,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t2_abab(f,b,i,j)*t2_bbbb(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeml,fbij,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,n||e,f>_abab*t2_abab(e,b,m,l)*t2_bbbb(f,d,j,k)*t2_abab(a,c,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnef,ebml,fdjk,acin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,l,m)*t2_bbbb(f,d,j,k)*t2_abab(a,c,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,eblm,fdjk,acin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*t2_abab(a,e,m,l)*t2_abab(f,d,i,k)*t2_bbbb(b,c,j,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,aeml,fdik,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_abab(e,b,m,l)*t2_abab(f,d,i,k)*t2_abab(a,c,n,j)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebml,fdik,acnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,l,m)*t2_abab(f,d,i,k)*t2_abab(a,c,n,j)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,eblm,fdik,acnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(k,l)<m,n||f,e>_abab*t2_abab(a,e,m,l)*t2_abab(f,d,i,j)*t2_bbbb(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeml,fdij,bckn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_aaaa*t2_abab(e,b,m,l)*t2_abab(f,d,i,j)*t2_abab(a,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebml,fdij,acnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||f,e>_abab*t2_bbbb(e,b,l,m)*t2_abab(f,d,i,j)*t2_abab(a,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,eblm,fdij,acnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<m,n||f,e>_abab*t2_abab(a,e,m,k)*t2_abab(f,b,i,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aemk,fbil,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 <m,n||e,f>_abab*t2_abab(e,b,m,k)*t2_bbbb(f,d,j,l)*t2_abab(a,c,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,ebmk,fdjl,acin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,k,m)*t2_bbbb(f,d,j,l)*t2_abab(a,c,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebkm,fdjl,acin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(a,e,m,k)*t2_abab(f,d,i,l)*t2_bbbb(b,c,j,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,aemk,fdil,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_abab(e,b,m,k)*t2_abab(f,d,i,l)*t2_abab(a,c,n,j)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebmk,fdil,acnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,k,m)*t2_abab(f,d,i,l)*t2_abab(a,c,n,j)
    quadruples_res += -1.000000000000000 * einsum('nmfe,ebkm,fdil,acnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(b,c)<m,n||f,e>_abab*t2_abab(a,e,m,j)*t2_abab(f,b,i,l)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aemj,fbil,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<m,n||f,e>_abab*t2_abab(a,e,m,j)*t2_abab(f,b,i,k)*t2_bbbb(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aemj,fbik,cdln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,i,m)*t2_bbbb(f,b,j,k)*t2_bbbb(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eaim,fbjk,cdln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,i,m)*t2_bbbb(f,b,j,k)*t2_bbbb(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aeim,fbjk,cdln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -1.0000 <m,n||e,f>_abab*t2_abab(e,b,m,j)*t2_bbbb(f,d,k,l)*t2_abab(a,c,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnef,ebmj,fdkl,acin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,j,m)*t2_bbbb(f,d,k,l)*t2_abab(a,c,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebjm,fdkl,acin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*t2_abab(a,e,m,j)*t2_abab(f,d,i,l)*t2_bbbb(b,c,k,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,aemj,fdil,bckn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_abab(e,b,m,j)*t2_abab(f,d,i,l)*t2_abab(a,c,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebmj,fdil,acnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,j,m)*t2_abab(f,d,i,l)*t2_abab(a,c,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,ebjm,fdil,acnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*t2_abab(a,e,m,j)*t2_abab(f,d,i,k)*t2_bbbb(b,c,l,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,aemj,fdik,bcln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_abab(e,b,m,j)*t2_abab(f,d,i,k)*t2_abab(a,c,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebmj,fdik,acnl->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,j,m)*t2_abab(f,d,i,k)*t2_abab(a,c,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmfe,ebjm,fdik,acnl->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*t2_aaaa(e,a,i,m)*t2_bbbb(f,d,j,k)*t2_bbbb(b,c,l,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,eaim,fdjk,bcln->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_abab(a,e,i,m)*t2_bbbb(f,d,j,k)*t2_bbbb(b,c,l,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,aeim,fdjk,bcln->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*t2_abab(e,b,i,m)*t2_bbbb(f,d,j,k)*t2_abab(a,c,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebim,fdjk,acnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(j,k)*P(b,c)<m,n||e,f>_abab*t2_aaaa(e,a,i,m)*t2_bbbb(f,b,k,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eaim,fbkl,cdjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,i,m)*t2_bbbb(f,b,k,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aeim,fbkl,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,n||e,f>_abab*t2_aaaa(e,a,i,m)*t2_bbbb(f,d,k,l)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eaim,fdkl,bcjn->abcdijkl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>_bbbb*t2_abab(a,e,i,m)*t2_bbbb(f,d,k,l)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aeim,fdkl,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_abab*t2_abab(e,b,i,m)*t2_bbbb(f,d,k,l)*t2_abab(a,c,n,j)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebim,fdkl,acnj->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 <m,n||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,d,m,j)*t2_abab(a,c,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,ebkl,fdmj,acin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,k,l)*t2_bbbb(f,d,j,m)*t2_abab(a,c,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebkl,fdjm,acin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,d,i,m)*t2_abab(a,c,n,j)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,ebkl,fdim,acnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 P(j,k)<n,m||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,d,i,j)*t2_abab(a,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,ebkl,fdij,acnm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<m,n||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,d,i,j)*t2_abab(a,c,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,ebkl,fdij,acmn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 <m,n||f,e>_abab*t2_bbbb(e,b,j,l)*t2_abab(f,d,m,k)*t2_abab(a,c,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,ebjl,fdmk,acin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,j,l)*t2_bbbb(f,d,k,m)*t2_abab(a,c,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebjl,fdkm,acin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,j,l)*t2_abab(f,d,i,m)*t2_abab(a,c,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmfe,ebjl,fdim,acnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 P(j,k)*P(b,c)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,b,m,k)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeil,fbmk,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(f,b,k,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aeil,fbkm,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(f,b,j,k)*t2_bbbb(c,d,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,aeil,fbjk,cdnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,n||f,e>_abab*t2_abab(a,e,i,l)*t2_abab(f,d,m,k)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeil,fdmk,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(f,d,k,m)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aeil,fdkm,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_aaaa*t2_abab(e,b,i,l)*t2_abab(f,d,m,k)*t2_abab(a,c,n,j)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebil,fdmk,acnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>_abab*t2_abab(e,b,i,l)*t2_bbbb(f,d,k,m)*t2_abab(a,c,n,j)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebil,fdkm,acnj->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,f>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(f,d,j,k)*t2_bbbb(b,c,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,aeil,fdjk,bcnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,f>_abab*t2_abab(e,b,i,l)*t2_bbbb(f,d,j,k)*t2_abab(a,c,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,ebil,fdjk,acnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(e,b,i,l)*t2_bbbb(f,d,j,k)*t2_abab(a,c,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,ebil,fdjk,acmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(f,b,k,l)*t2_bbbb(c,d,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,aeij,fbkl,cdnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 <m,n||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,d,m,l)*t2_abab(a,c,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,ebjk,fdml,acin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,j,k)*t2_bbbb(f,d,l,m)*t2_abab(a,c,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebjk,fdlm,acin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,d,i,m)*t2_abab(a,c,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,ebjk,fdim,acnl->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,d,i,l)*t2_abab(a,c,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmfe,ebjk,fdil,acnm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,d,i,l)*t2_abab(a,c,m,n)
    quadruples_res += -0.500000000000000 * einsum('mnfe,ebjk,fdil,acmn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(f,d,k,l)*t2_bbbb(b,c,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmef,aeij,fdkl,bcnm->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*t2_abab(e,b,i,j)*t2_bbbb(f,d,k,l)*t2_abab(a,c,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmef,ebij,fdkl,acnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*t2_abab(e,b,i,j)*t2_bbbb(f,d,k,l)*t2_abab(a,c,m,n)
    quadruples_res +=  0.500000000000000 * einsum('mnef,ebij,fdkl,acmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 P(j,l)*P(b,c)<m,n||f,e>_abab*t2_abab(a,e,i,k)*t2_abab(f,b,m,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aeik,fbml,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(f,b,l,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aeik,fblm,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<m,n||f,e>_abab*t2_abab(a,e,i,k)*t2_abab(f,d,m,l)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,aeik,fdml,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||e,f>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(f,d,l,m)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,aeik,fdlm,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||e,f>_aaaa*t2_abab(e,b,i,k)*t2_abab(f,d,m,l)*t2_abab(a,c,n,j)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebik,fdml,acnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<n,m||e,f>_abab*t2_abab(e,b,i,k)*t2_bbbb(f,d,l,m)*t2_abab(a,c,n,j)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebik,fdlm,acnj->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<m,n||f,e>_abab*t2_abab(a,e,i,j)*t2_abab(f,b,m,l)*t2_bbbb(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeij,fbml,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,f>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(f,b,l,m)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aeij,fblm,cdkn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,n||f,e>_abab*t2_abab(a,e,i,j)*t2_abab(f,d,m,l)*t2_bbbb(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,aeij,fdml,bckn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(f,d,l,m)*t2_bbbb(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,aeij,fdlm,bckn->abcdijkl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_aaaa*t2_abab(e,b,i,j)*t2_abab(f,d,m,l)*t2_abab(a,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebij,fdml,acnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_abab*t2_abab(e,b,i,j)*t2_bbbb(f,d,l,m)*t2_abab(a,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebij,fdlm,acnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 <m,n||e,f>_abab*t2_abab(e,b,m,l)*t2_bbbb(f,c,j,k)*t2_abab(a,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,ebml,fcjk,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,l,m)*t2_bbbb(f,c,j,k)*t2_abab(a,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,eblm,fcjk,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_abab(e,b,m,l)*t2_abab(f,c,i,k)*t2_abab(a,d,n,j)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebml,fcik,adnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,l,m)*t2_abab(f,c,i,k)*t2_abab(a,d,n,j)
    quadruples_res += -1.000000000000000 * einsum('nmfe,eblm,fcik,adnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(k,l)<n,m||e,f>_aaaa*t2_abab(e,b,m,l)*t2_abab(f,c,i,j)*t2_abab(a,d,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebml,fcij,adnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||f,e>_abab*t2_bbbb(e,b,l,m)*t2_abab(f,c,i,j)*t2_abab(a,d,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,eblm,fcij,adnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,n||e,f>_abab*t2_abab(e,b,m,k)*t2_bbbb(f,c,j,l)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnef,ebmk,fcjl,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,k,m)*t2_bbbb(f,c,j,l)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebkm,fcjl,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_abab(e,b,m,k)*t2_abab(f,c,i,l)*t2_abab(a,d,n,j)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebmk,fcil,adnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,k,m)*t2_abab(f,c,i,l)*t2_abab(a,d,n,j)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,ebkm,fcil,adnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*t2_abab(e,b,m,j)*t2_bbbb(f,c,k,l)*t2_abab(a,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,ebmj,fckl,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,j,m)*t2_bbbb(f,c,k,l)*t2_abab(a,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebjm,fckl,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_abab(e,b,m,j)*t2_abab(f,c,i,l)*t2_abab(a,d,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebmj,fcil,adnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,j,m)*t2_abab(f,c,i,l)*t2_abab(a,d,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmfe,ebjm,fcil,adnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_abab(e,b,m,j)*t2_abab(f,c,i,k)*t2_abab(a,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebmj,fcik,adnl->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,j,m)*t2_abab(f,c,i,k)*t2_abab(a,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,ebjm,fcik,adnl->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*t2_abab(e,b,i,m)*t2_bbbb(f,c,j,k)*t2_abab(a,d,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebim,fcjk,adnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(j,k)<n,m||e,f>_abab*t2_abab(e,b,i,m)*t2_bbbb(f,c,k,l)*t2_abab(a,d,n,j)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebim,fckl,adnj->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 <m,n||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,c,m,j)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,ebkl,fcmj,adin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,k,l)*t2_bbbb(f,c,j,m)*t2_abab(a,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebkl,fcjm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,c,i,m)*t2_abab(a,d,n,j)
    quadruples_res += -1.000000000000000 * einsum('nmfe,ebkl,fcim,adnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 P(j,k)<n,m||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,c,i,j)*t2_abab(a,d,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,ebkl,fcij,adnm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<m,n||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,c,i,j)*t2_abab(a,d,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,ebkl,fcij,admn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 <m,n||f,e>_abab*t2_bbbb(e,b,j,l)*t2_abab(f,c,m,k)*t2_abab(a,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,ebjl,fcmk,adin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,j,l)*t2_bbbb(f,c,k,m)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ebjl,fckm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,j,l)*t2_abab(f,c,i,m)*t2_abab(a,d,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,ebjl,fcim,adnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 P(j,k)<n,m||e,f>_aaaa*t2_abab(e,b,i,l)*t2_abab(f,c,m,k)*t2_abab(a,d,n,j)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebil,fcmk,adnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_abab*t2_abab(e,b,i,l)*t2_bbbb(f,c,k,m)*t2_abab(a,d,n,j)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebil,fckm,adnj->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_abab*t2_abab(e,b,i,l)*t2_bbbb(f,c,j,k)*t2_abab(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebil,fcjk,adnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(e,b,i,l)*t2_bbbb(f,c,j,k)*t2_abab(a,d,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,ebil,fcjk,admn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,n||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,c,m,l)*t2_abab(a,d,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,ebjk,fcml,adin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,b,j,k)*t2_bbbb(f,c,l,m)*t2_abab(a,d,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ebjk,fclm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,c,i,m)*t2_abab(a,d,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmfe,ebjk,fcim,adnl->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,c,i,l)*t2_abab(a,d,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmfe,ebjk,fcil,adnm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,c,i,l)*t2_abab(a,d,m,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,ebjk,fcil,admn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,b,i,j)*t2_bbbb(f,c,k,l)*t2_abab(a,d,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmef,ebij,fckl,adnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,b,i,j)*t2_bbbb(f,c,k,l)*t2_abab(a,d,m,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,ebij,fckl,admn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 P(j,l)<n,m||e,f>_aaaa*t2_abab(e,b,i,k)*t2_abab(f,c,m,l)*t2_abab(a,d,n,j)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebik,fcml,adnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||e,f>_abab*t2_abab(e,b,i,k)*t2_bbbb(f,c,l,m)*t2_abab(a,d,n,j)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebik,fclm,adnj->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_aaaa*t2_abab(e,b,i,j)*t2_abab(f,c,m,l)*t2_abab(a,d,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebij,fcml,adnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_abab*t2_abab(e,b,i,j)*t2_bbbb(f,c,l,m)*t2_abab(a,d,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebij,fclm,adnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 <m,n||e,f>_abab*t2_abab(e,c,m,l)*t2_bbbb(f,d,j,k)*t2_abab(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,ecml,fdjk,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,c,l,m)*t2_bbbb(f,d,j,k)*t2_abab(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,eclm,fdjk,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_abab(e,c,m,l)*t2_abab(f,d,i,k)*t2_abab(a,b,n,j)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecml,fdik,abnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,c,l,m)*t2_abab(f,d,i,k)*t2_abab(a,b,n,j)
    quadruples_res += -1.000000000000000 * einsum('nmfe,eclm,fdik,abnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(k,l)<n,m||e,f>_aaaa*t2_abab(e,c,m,l)*t2_abab(f,d,i,j)*t2_abab(a,b,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecml,fdij,abnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||f,e>_abab*t2_bbbb(e,c,l,m)*t2_abab(f,d,i,j)*t2_abab(a,b,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,eclm,fdij,abnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,n||e,f>_abab*t2_abab(e,c,m,k)*t2_bbbb(f,d,j,l)*t2_abab(a,b,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnef,ecmk,fdjl,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,c,k,m)*t2_bbbb(f,d,j,l)*t2_abab(a,b,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,eckm,fdjl,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_abab(e,c,m,k)*t2_abab(f,d,i,l)*t2_abab(a,b,n,j)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecmk,fdil,abnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,c,k,m)*t2_abab(f,d,i,l)*t2_abab(a,b,n,j)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,eckm,fdil,abnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*t2_abab(e,c,m,j)*t2_bbbb(f,d,k,l)*t2_abab(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnef,ecmj,fdkl,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,c,j,m)*t2_bbbb(f,d,k,l)*t2_abab(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecjm,fdkl,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*t2_abab(e,c,m,j)*t2_abab(f,d,i,l)*t2_abab(a,b,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecmj,fdil,abnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,c,j,m)*t2_abab(f,d,i,l)*t2_abab(a,b,n,k)
    quadruples_res += -1.000000000000000 * einsum('nmfe,ecjm,fdil,abnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*t2_abab(e,c,m,j)*t2_abab(f,d,i,k)*t2_abab(a,b,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecmj,fdik,abnl->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,c,j,m)*t2_abab(f,d,i,k)*t2_abab(a,b,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,ecjm,fdik,abnl->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*t2_abab(e,c,i,m)*t2_bbbb(f,d,j,k)*t2_abab(a,b,n,l)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecim,fdjk,abnl->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(j,k)<n,m||e,f>_abab*t2_abab(e,c,i,m)*t2_bbbb(f,d,k,l)*t2_abab(a,b,n,j)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecim,fdkl,abnj->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 <m,n||f,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(f,d,m,j)*t2_abab(a,b,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,eckl,fdmj,abin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,c,k,l)*t2_bbbb(f,d,j,m)*t2_abab(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,eckl,fdjm,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(f,d,i,m)*t2_abab(a,b,n,j)
    quadruples_res += -1.000000000000000 * einsum('nmfe,eckl,fdim,abnj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 P(j,k)<n,m||f,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(f,d,i,j)*t2_abab(a,b,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,eckl,fdij,abnm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<m,n||f,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(f,d,i,j)*t2_abab(a,b,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,eckl,fdij,abmn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 <m,n||f,e>_abab*t2_bbbb(e,c,j,l)*t2_abab(f,d,m,k)*t2_abab(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('mnfe,ecjl,fdmk,abin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,c,j,l)*t2_bbbb(f,d,k,m)*t2_abab(a,b,i,n)
    quadruples_res += -1.000000000000000 * einsum('nmef,ecjl,fdkm,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*t2_bbbb(e,c,j,l)*t2_abab(f,d,i,m)*t2_abab(a,b,n,k)
    quadruples_res +=  1.000000000000000 * einsum('nmfe,ecjl,fdim,abnk->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 P(j,k)<n,m||e,f>_aaaa*t2_abab(e,c,i,l)*t2_abab(f,d,m,k)*t2_abab(a,b,n,j)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecil,fdmk,abnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_abab*t2_abab(e,c,i,l)*t2_bbbb(f,d,k,m)*t2_abab(a,b,n,j)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecil,fdkm,abnj->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_abab*t2_abab(e,c,i,l)*t2_bbbb(f,d,j,k)*t2_abab(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ecil,fdjk,abnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(e,c,i,l)*t2_bbbb(f,d,j,k)*t2_abab(a,b,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,ecil,fdjk,abmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,n||f,e>_abab*t2_bbbb(e,c,j,k)*t2_abab(f,d,m,l)*t2_abab(a,b,i,n)
    quadruples_res += -1.000000000000000 * einsum('mnfe,ecjk,fdml,abin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*t2_bbbb(e,c,j,k)*t2_bbbb(f,d,l,m)*t2_abab(a,b,i,n)
    quadruples_res +=  1.000000000000000 * einsum('nmef,ecjk,fdlm,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*t2_bbbb(e,c,j,k)*t2_abab(f,d,i,m)*t2_abab(a,b,n,l)
    quadruples_res += -1.000000000000000 * einsum('nmfe,ecjk,fdim,abnl->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*t2_bbbb(e,c,j,k)*t2_abab(f,d,i,l)*t2_abab(a,b,n,m)
    quadruples_res +=  0.500000000000000 * einsum('nmfe,ecjk,fdil,abnm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*t2_bbbb(e,c,j,k)*t2_abab(f,d,i,l)*t2_abab(a,b,m,n)
    quadruples_res +=  0.500000000000000 * einsum('mnfe,ecjk,fdil,abmn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*t2_abab(e,c,i,j)*t2_bbbb(f,d,k,l)*t2_abab(a,b,n,m)
    quadruples_res += -0.500000000000000 * einsum('nmef,ecij,fdkl,abnm->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*t2_abab(e,c,i,j)*t2_bbbb(f,d,k,l)*t2_abab(a,b,m,n)
    quadruples_res += -0.500000000000000 * einsum('mnef,ecij,fdkl,abmn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 P(j,l)<n,m||e,f>_aaaa*t2_abab(e,c,i,k)*t2_abab(f,d,m,l)*t2_abab(a,b,n,j)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecik,fdml,abnj->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||e,f>_abab*t2_abab(e,c,i,k)*t2_bbbb(f,d,l,m)*t2_abab(a,b,n,j)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecik,fdlm,abnj->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_aaaa*t2_abab(e,c,i,j)*t2_abab(f,d,m,l)*t2_abab(a,b,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecij,fdml,abnk->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>_abab*t2_abab(e,c,i,j)*t2_bbbb(f,d,l,m)*t2_abab(a,b,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecij,fdlm,abnk->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    return quadruples_res


def ccdq_t4_bbbbbbbb_residual(t2_aaaa, t2_bbbb, t2_abab, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(k,l)f_bb(m,l)*t4_bbbbbbbb(a,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000010 * einsum('ml,abcdijkm->abcdijkl', f_bb[o, o], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)f_bb(m,j)*t4_bbbbbbbb(a,b,c,d,i,k,l,m)
    contracted_intermediate = -1.000000000000010 * einsum('mj,abcdiklm->abcdijkl', f_bb[o, o], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f_bb(a,e)*t4_bbbbbbbb(e,b,c,d,i,j,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('ae,ebcdijkl->abcdijkl', f_bb[v, v], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)f_bb(c,e)*t4_bbbbbbbb(e,a,b,d,i,j,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('ce,eabdijkl->abcdijkl', f_bb[v, v], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<n,m||k,l>_bbbb*t4_bbbbbbbb(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmkl,abcdijnm->abcdijkl', g_bbbb[o, o, o, o], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||i,l>_bbbb*t4_bbbbbbbb(a,b,c,d,j,k,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmil,abcdjknm->abcdijkl', g_bbbb[o, o, o, o], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<n,m||j,k>_bbbb*t4_bbbbbbbb(a,b,c,d,i,l,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmjk,abcdilnm->abcdijkl', g_bbbb[o, o, o, o], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,a||e,l>_abab*t4_abbbabbb(e,b,c,d,m,j,k,i)
    contracted_intermediate = -1.000000000000020 * einsum('mael,ebcdmjki->abcdijkl', g_abab[o, v, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,l>_bbbb*t4_bbbbbbbb(e,b,c,d,i,j,k,m)
    contracted_intermediate =  1.000000000000020 * einsum('mael,ebcdijkm->abcdijkl', g_bbbb[o, v, v, o], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,j>_abab*t4_abbbabbb(e,b,c,d,m,k,l,i)
    contracted_intermediate = -1.000000000000020 * einsum('maej,ebcdmkli->abcdijkl', g_abab[o, v, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*t4_bbbbbbbb(e,b,c,d,i,k,l,m)
    contracted_intermediate =  1.000000000000020 * einsum('maej,ebcdiklm->abcdijkl', g_bbbb[o, v, v, o], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,l>_abab*t4_abbbabbb(e,a,b,d,m,j,k,i)
    contracted_intermediate = -1.000000000000020 * einsum('mcel,eabdmjki->abcdijkl', g_abab[o, v, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,l>_bbbb*t4_bbbbbbbb(e,a,b,d,i,j,k,m)
    contracted_intermediate =  1.000000000000020 * einsum('mcel,eabdijkm->abcdijkl', g_bbbb[o, v, v, o], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,j>_abab*t4_abbbabbb(e,a,b,d,m,k,l,i)
    contracted_intermediate = -1.000000000000020 * einsum('mcej,eabdmkli->abcdijkl', g_abab[o, v, v, o], t4_abbbabbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,j>_bbbb*t4_bbbbbbbb(e,a,b,d,i,k,l,m)
    contracted_intermediate =  1.000000000000020 * einsum('mcej,eabdiklm->abcdijkl', g_bbbb[o, v, v, o], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<a,b||e,f>_bbbb*t4_bbbbbbbb(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('abef,efcdijkl->abcdijkl', g_bbbb[v, v, v, v], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<a,d||e,f>_bbbb*t4_bbbbbbbb(e,f,b,c,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('adef,efbcijkl->abcdijkl', g_bbbb[v, v, v, v], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,d)<b,c||e,f>_bbbb*t4_bbbbbbbb(e,f,a,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('bcef,efadijkl->abcdijkl', g_bbbb[v, v, v, v], t4_bbbbbbbb[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||e,f>_abab*t2_abab(e,f,m,l)*t4_bbbbbbbb(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,efml,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<m,n||f,e>_abab*t2_abab(f,e,m,l)*t4_bbbbbbbb(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,feml,abcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,f,l,m)*t4_bbbbbbbb(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eflm,abcdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*t2_abab(e,f,m,j)*t4_bbbbbbbb(a,b,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,efmj,abcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*t2_abab(f,e,m,j)*t4_bbbbbbbb(a,b,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('mnfe,femj,abcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,f,j,m)*t4_bbbbbbbb(a,b,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,efjm,abcdikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.2500 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,f,k,l)*t4_bbbbbbbb(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,efkl,abcdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.2500 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,f,i,l)*t4_bbbbbbbb(a,b,c,d,j,k,n,m)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,efil,abcdjknm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.2500 P(i,k)<n,m||e,f>_bbbb*t2_bbbb(e,f,j,k)*t4_bbbbbbbb(a,b,c,d,i,l,n,m)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,efjk,abcdilnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,f>_abab*t2_abab(e,a,n,m)*t4_bbbbbbbb(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eanm,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,n||e,f>_abab*t2_abab(e,a,m,n)*t4_bbbbbbbb(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,eamn,fbcdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,n,m)*t4_bbbbbbbb(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eanm,fbcdijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>_aaaa*t2_abab(e,a,m,l)*t4_abbbabbb(f,b,c,d,n,j,k,i)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eaml,fbcdnjki->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,n||e,f>_abab*t2_abab(e,a,m,l)*t4_bbbbbbbb(f,b,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,eaml,fbcdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||f,e>_abab*t2_bbbb(e,a,l,m)*t4_abbbabbb(f,b,c,d,n,j,k,i)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,ealm,fbcdnjki->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,l,m)*t4_bbbbbbbb(f,b,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ealm,fbcdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*t2_abab(e,a,m,j)*t4_abbbabbb(f,b,c,d,n,k,l,i)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eamj,fbcdnkli->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*t2_abab(e,a,m,j)*t4_bbbbbbbb(f,b,c,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,eamj,fbcdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_bbbb(e,a,j,m)*t4_abbbabbb(f,b,c,d,n,k,l,i)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,eajm,fbcdnkli->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,m)*t4_bbbbbbbb(f,b,c,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eajm,fbcdikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||f,e>_abab*t2_bbbb(e,a,k,l)*t4_abbbabbb(f,b,c,d,n,j,i,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,eakl,fbcdnjim->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(a,b)<m,n||f,e>_abab*t2_bbbb(e,a,k,l)*t4_abbbabbb(f,b,c,d,m,j,n,i)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,eakl,fbcdmjni->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,k,l)*t4_bbbbbbbb(f,b,c,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eakl,fbcdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||f,e>_abab*t2_bbbb(e,a,i,l)*t4_abbbabbb(f,b,c,d,n,k,j,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,eail,fbcdnkjm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<m,n||f,e>_abab*t2_bbbb(e,a,i,l)*t4_abbbabbb(f,b,c,d,m,k,n,j)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,eail,fbcdmknj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,l)*t4_bbbbbbbb(f,b,c,d,j,k,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eail,fbcdjknm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||f,e>_abab*t2_bbbb(e,a,j,k)*t4_abbbabbb(f,b,c,d,n,l,i,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,eajk,fbcdnlim->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(a,b)<m,n||f,e>_abab*t2_bbbb(e,a,j,k)*t4_abbbabbb(f,b,c,d,m,l,n,i)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,eajk,fbcdmlni->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,k)*t4_bbbbbbbb(f,b,c,d,i,l,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eajk,fbcdilnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||e,f>_abab*t2_abab(e,c,n,m)*t4_bbbbbbbb(f,a,b,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecnm,fabdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<m,n||e,f>_abab*t2_abab(e,c,m,n)*t4_bbbbbbbb(f,a,b,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('mnef,ecmn,fabdijkl->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,n,m)*t4_bbbbbbbb(f,a,b,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecnm,fabdijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,m,l)*t4_abbbabbb(f,a,b,d,n,j,k,i)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ecml,fabdnjki->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,n||e,f>_abab*t2_abab(e,c,m,l)*t4_bbbbbbbb(f,a,b,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,ecml,fabdijkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,l,m)*t4_abbbabbb(f,a,b,d,n,j,k,i)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,eclm,fabdnjki->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,l,m)*t4_bbbbbbbb(f,a,b,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eclm,fabdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>_aaaa*t2_abab(e,c,m,j)*t4_abbbabbb(f,a,b,d,n,k,l,i)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ecmj,fabdnkli->abcdijkl', g_aaaa[o, o, v, v], t2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,n||e,f>_abab*t2_abab(e,c,m,j)*t4_bbbbbbbb(f,a,b,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('mnef,ecmj,fabdikln->abcdijkl', g_abab[o, o, v, v], t2_abab, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,j,m)*t4_abbbabbb(f,a,b,d,n,k,l,i)
    contracted_intermediate =  0.999999999999840 * einsum('nmfe,ecjm,fabdnkli->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,j,m)*t4_bbbbbbbb(f,a,b,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ecjm,fabdikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,k,l)*t4_abbbabbb(f,a,b,d,n,j,i,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,eckl,fabdnjim->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(c,d)<m,n||f,e>_abab*t2_bbbb(e,c,k,l)*t4_abbbabbb(f,a,b,d,m,j,n,i)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,eckl,fabdmjni->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,k,l)*t4_bbbbbbbb(f,a,b,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eckl,fabdijnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,i,l)*t4_abbbabbb(f,a,b,d,n,k,j,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,ecil,fabdnkjm->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<m,n||f,e>_abab*t2_bbbb(e,c,i,l)*t4_abbbabbb(f,a,b,d,m,k,n,j)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,ecil,fabdmknj->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,i,l)*t4_bbbbbbbb(f,a,b,d,j,k,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecil,fabdjknm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(c,d)<n,m||f,e>_abab*t2_bbbb(e,c,j,k)*t4_abbbabbb(f,a,b,d,n,l,i,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmfe,ecjk,fabdnlim->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(c,d)<m,n||f,e>_abab*t2_bbbb(e,c,j,k)*t4_abbbabbb(f,a,b,d,m,l,n,i)
    contracted_intermediate =  0.499999999999950 * einsum('mnfe,ecjk,fabdmlni->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(c,d)<n,m||e,f>_bbbb*t2_bbbb(e,c,j,k)*t4_bbbbbbbb(f,a,b,d,i,l,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecjk,fabdilnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  0.2500 P(b,c)<n,m||e,f>_bbbb*t2_bbbb(a,b,n,m)*t4_bbbbbbbb(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,abnm,efcdijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,c)<n,m||e,f>_abab*t2_bbbb(a,b,l,m)*t4_abbbabbb(e,f,c,d,n,j,k,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,ablm,efcdnjki->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,c)<n,m||f,e>_abab*t2_bbbb(a,b,l,m)*t4_abbbabbb(f,e,c,d,n,j,k,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,ablm,fecdnjki->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(a,b,l,m)*t4_bbbbbbbb(e,f,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ablm,efcdijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,c)<n,m||e,f>_abab*t2_bbbb(a,b,j,m)*t4_abbbabbb(e,f,c,d,n,k,l,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,abjm,efcdnkli->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,c)<n,m||f,e>_abab*t2_bbbb(a,b,j,m)*t4_abbbabbb(f,e,c,d,n,k,l,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,abjm,fecdnkli->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(a,b,j,m)*t4_bbbbbbbb(e,f,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,abjm,efcdikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_bbbb*t2_bbbb(a,d,n,m)*t4_bbbbbbbb(e,f,b,c,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,adnm,efbcijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<n,m||e,f>_abab*t2_bbbb(a,d,l,m)*t4_abbbabbb(e,f,b,c,n,j,k,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,adlm,efbcnjki->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<n,m||f,e>_abab*t2_bbbb(a,d,l,m)*t4_abbbabbb(f,e,b,c,n,j,k,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,adlm,febcnjki->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(a,d,l,m)*t4_bbbbbbbb(e,f,b,c,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adlm,efbcijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*t2_bbbb(a,d,j,m)*t4_abbbabbb(e,f,b,c,n,k,l,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,adjm,efbcnkli->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||f,e>_abab*t2_bbbb(a,d,j,m)*t4_abbbabbb(f,e,b,c,n,k,l,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,adjm,febcnkli->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(a,d,j,m)*t4_bbbbbbbb(e,f,b,c,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adjm,efbcikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.2500 P(b,d)<n,m||e,f>_bbbb*t2_bbbb(b,c,n,m)*t4_bbbbbbbb(e,f,a,d,i,j,k,l)
    contracted_intermediate =  0.250000000000010 * einsum('nmef,bcnm,efadijkl->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,d)<n,m||e,f>_abab*t2_bbbb(b,c,l,m)*t4_abbbabbb(e,f,a,d,n,j,k,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,bclm,efadnjki->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,d)<n,m||f,e>_abab*t2_bbbb(b,c,l,m)*t4_abbbabbb(f,e,a,d,n,j,k,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,bclm,feadnjki->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,d)<n,m||e,f>_bbbb*t2_bbbb(b,c,l,m)*t4_bbbbbbbb(e,f,a,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bclm,efadijkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,d)<n,m||e,f>_abab*t2_bbbb(b,c,j,m)*t4_abbbabbb(e,f,a,d,n,k,l,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmef,bcjm,efadnkli->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,d)<n,m||f,e>_abab*t2_bbbb(b,c,j,m)*t4_abbbabbb(f,e,a,d,n,k,l,i)
    contracted_intermediate =  0.499999999999950 * einsum('nmfe,bcjm,feadnkli->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,d)<n,m||e,f>_bbbb*t2_bbbb(b,c,j,m)*t4_bbbbbbbb(e,f,a,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bcjm,efadikln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||k,l>_bbbb*t2_bbbb(a,b,j,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,abjm,cdin->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||k,l>_bbbb*t2_bbbb(a,d,j,m)*t2_bbbb(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,adjm,bcin->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||j,l>_bbbb*t2_bbbb(a,b,k,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,abkm,cdin->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||j,l>_bbbb*t2_bbbb(a,d,k,m)*t2_bbbb(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,adkm,bcin->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||i,l>_bbbb*t2_bbbb(a,b,k,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,abkm,cdjn->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||i,l>_bbbb*t2_bbbb(a,d,k,m)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,adkm,bcjn->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)*P(b,c)<n,m||j,k>_bbbb*t2_bbbb(a,b,l,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,ablm,cdin->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)<n,m||j,k>_bbbb*t2_bbbb(a,d,l,m)*t2_bbbb(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,adlm,bcin->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,c)<n,m||i,k>_bbbb*t2_bbbb(a,b,l,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmik,ablm,cdjn->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||i,k>_bbbb*t2_bbbb(a,d,l,m)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmik,adlm,bcjn->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||i,j>_bbbb*t2_bbbb(a,b,l,m)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,ablm,cdkn->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||i,j>_bbbb*t2_bbbb(a,d,l,m)*t2_bbbb(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,adlm,bckn->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,l>_bbbb*t2_bbbb(e,b,j,k)*t2_bbbb(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ebjk,cdim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<m,a||e,l>_bbbb*t2_bbbb(e,b,i,j)*t2_bbbb(c,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ebij,cdkm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,l>_bbbb*t2_bbbb(e,d,j,k)*t2_bbbb(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,edjk,bcim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,l>_bbbb*t2_bbbb(e,d,i,j)*t2_bbbb(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,edij,bckm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,a||e,k>_bbbb*t2_bbbb(e,b,j,l)*t2_bbbb(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,ebjl,cdim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,k>_bbbb*t2_bbbb(e,d,j,l)*t2_bbbb(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,edjl,bcim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<m,a||e,j>_bbbb*t2_bbbb(e,b,k,l)*t2_bbbb(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ebkl,cdim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,j>_bbbb*t2_bbbb(e,b,i,k)*t2_bbbb(c,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ebik,cdlm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,a||e,j>_bbbb*t2_bbbb(e,d,k,l)*t2_bbbb(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edkl,bcim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*t2_bbbb(e,d,i,k)*t2_bbbb(b,c,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edik,bclm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,a||e,i>_bbbb*t2_bbbb(e,b,k,l)*t2_bbbb(c,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,ebkl,cdjm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,a||e,i>_bbbb*t2_bbbb(e,d,k,l)*t2_bbbb(b,c,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,edkl,bcjm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,l>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,eajk,cdim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,c)<m,b||e,l>_bbbb*t2_bbbb(e,a,i,j)*t2_bbbb(c,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,eaij,cdkm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<m,b||e,k>_bbbb*t2_bbbb(e,a,j,l)*t2_bbbb(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbek,eajl,cdim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,c)<m,b||e,j>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,eakl,cdim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,j>_bbbb*t2_bbbb(e,a,i,k)*t2_bbbb(c,d,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,eaik,cdlm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,c)<m,b||e,i>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(c,d,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbei,eakl,cdjm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,l>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,eajk,bdim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,c||e,l>_bbbb*t2_bbbb(e,a,i,j)*t2_bbbb(b,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,eaij,bdkm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,l>_bbbb*t2_bbbb(e,d,j,k)*t2_bbbb(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edjk,abim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,l>_bbbb*t2_bbbb(e,d,i,j)*t2_bbbb(a,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edij,abkm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,k>_bbbb*t2_bbbb(e,a,j,l)*t2_bbbb(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,eajl,bdim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,k>_bbbb*t2_bbbb(e,d,j,l)*t2_bbbb(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,edjl,abim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,c||e,j>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,eakl,bdim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,j>_bbbb*t2_bbbb(e,a,i,k)*t2_bbbb(b,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,eaik,bdlm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)<m,c||e,j>_bbbb*t2_bbbb(e,d,k,l)*t2_bbbb(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edkl,abim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,j>_bbbb*t2_bbbb(e,d,i,k)*t2_bbbb(a,b,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edik,ablm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,c||e,i>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(b,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,eakl,bdjm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<m,c||e,i>_bbbb*t2_bbbb(e,d,k,l)*t2_bbbb(a,b,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,edkl,abjm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,l>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eajk,bcim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,d||e,l>_bbbb*t2_bbbb(e,a,i,j)*t2_bbbb(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eaij,bckm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,k>_bbbb*t2_bbbb(e,a,j,l)*t2_bbbb(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdek,eajl,bcim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,d||e,j>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,eakl,bcim->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,j>_bbbb*t2_bbbb(e,a,i,k)*t2_bbbb(b,c,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,eaik,bclm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,d||e,i>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdei,eakl,bcjm->abcdijkl', g_bbbb[o, v, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<a,b||e,f>_bbbb*t2_bbbb(e,c,k,l)*t2_bbbb(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abef,eckl,fdij->abcdijkl', g_bbbb[v, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<a,b||e,f>_bbbb*t2_bbbb(e,c,i,l)*t2_bbbb(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecil,fdjk->abcdijkl', g_bbbb[v, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<a,b||e,f>_bbbb*t2_bbbb(e,c,j,k)*t2_bbbb(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecjk,fdil->abcdijkl', g_bbbb[v, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<a,d||e,f>_bbbb*t2_bbbb(e,b,k,l)*t2_bbbb(f,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebkl,fcij->abcdijkl', g_bbbb[v, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<a,d||e,f>_bbbb*t2_bbbb(e,b,i,l)*t2_bbbb(f,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebil,fcjk->abcdijkl', g_bbbb[v, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<a,d||e,f>_bbbb*t2_bbbb(e,b,j,k)*t2_bbbb(f,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebjk,fcil->abcdijkl', g_bbbb[v, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,d)<b,c||e,f>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eakl,fdij->abcdijkl', g_bbbb[v, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,d)<b,c||e,f>_bbbb*t2_bbbb(e,a,i,l)*t2_bbbb(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eail,fdjk->abcdijkl', g_bbbb[v, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,d)<b,c||e,f>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eajk,fdil->abcdijkl', g_bbbb[v, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbkjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,f,k,l)*t2_bbbb(a,b,j,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efkl,abjm,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,f,k,l)*t2_bbbb(a,d,j,m)*t2_bbbb(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efkl,adjm,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,f,j,l)*t2_bbbb(a,b,k,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efjl,abkm,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<n,m||e,f>_bbbb*t2_bbbb(e,f,j,l)*t2_bbbb(a,d,k,m)*t2_bbbb(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efjl,adkm,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,f,i,l)*t2_bbbb(a,b,k,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efil,abkm,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,f,i,l)*t2_bbbb(a,d,k,m)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efil,adkm,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(i,l)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,f,j,k)*t2_bbbb(a,b,l,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efjk,ablm,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,l)<n,m||e,f>_bbbb*t2_bbbb(e,f,j,k)*t2_bbbb(a,d,l,m)*t2_bbbb(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efjk,adlm,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  0.5000 P(j,l)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,f,i,k)*t2_bbbb(a,b,l,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efik,ablm,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  0.5000 P(j,l)<n,m||e,f>_bbbb*t2_bbbb(e,f,i,k)*t2_bbbb(a,d,l,m)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efik,adlm,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,f,i,j)*t2_bbbb(a,b,l,m)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efij,ablm,cdkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,f,i,j)*t2_bbbb(a,d,l,m)*t2_bbbb(b,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efij,adlm,bckn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,n||e,f>_abab*t2_abab(e,a,m,l)*t2_bbbb(f,b,j,k)*t2_bbbb(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eaml,fbjk,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,l,m)*t2_bbbb(f,b,j,k)*t2_bbbb(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fbjk,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<m,n||e,f>_abab*t2_abab(e,a,m,l)*t2_bbbb(f,b,i,j)*t2_bbbb(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eaml,fbij,cdkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,l,m)*t2_bbbb(f,b,i,j)*t2_bbbb(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fbij,cdkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*t2_abab(e,a,m,l)*t2_bbbb(f,d,j,k)*t2_bbbb(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eaml,fdjk,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,l,m)*t2_bbbb(f,d,j,k)*t2_bbbb(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fdjk,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,n||e,f>_abab*t2_abab(e,a,m,l)*t2_bbbb(f,d,i,j)*t2_bbbb(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eaml,fdij,bckn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,l,m)*t2_bbbb(f,d,i,j)*t2_bbbb(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fdij,bckn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,n||e,f>_abab*t2_abab(e,a,m,k)*t2_bbbb(f,b,j,l)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eamk,fbjl,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,k,m)*t2_bbbb(f,b,j,l)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eakm,fbjl,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*t2_abab(e,a,m,k)*t2_bbbb(f,d,j,l)*t2_bbbb(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eamk,fdjl,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,k,m)*t2_bbbb(f,d,j,l)*t2_bbbb(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eakm,fdjl,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<m,n||e,f>_abab*t2_abab(e,a,m,j)*t2_bbbb(f,b,k,l)*t2_bbbb(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eamj,fbkl,cdin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,m)*t2_bbbb(f,b,k,l)*t2_bbbb(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fbkl,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,n||e,f>_abab*t2_abab(e,a,m,j)*t2_bbbb(f,b,i,k)*t2_bbbb(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eamj,fbik,cdln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,m)*t2_bbbb(f,b,i,k)*t2_bbbb(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fbik,cdln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,n||e,f>_abab*t2_abab(e,a,m,j)*t2_bbbb(f,d,k,l)*t2_bbbb(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eamj,fdkl,bcin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,m)*t2_bbbb(f,d,k,l)*t2_bbbb(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fdkl,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*t2_abab(e,a,m,j)*t2_bbbb(f,d,i,k)*t2_bbbb(b,c,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,eamj,fdik,bcln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,m)*t2_bbbb(f,d,i,k)*t2_bbbb(b,c,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fdik,bcln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,n||e,f>_abab*t2_abab(e,a,m,i)*t2_bbbb(f,b,k,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eami,fbkl,cdjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,m)*t2_bbbb(f,b,k,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaim,fbkl,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,n||e,f>_abab*t2_abab(e,a,m,i)*t2_bbbb(f,d,k,l)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,eami,fdkl,bcjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,m)*t2_bbbb(f,d,k,l)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaim,fdkl,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,n||f,e>_abab*t2_bbbb(e,a,k,l)*t2_abab(f,b,m,j)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,eakl,fbmj,cdin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(f,b,j,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eakl,fbjm,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(f,b,i,j)*t2_bbbb(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eakl,fbij,cdnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*t2_bbbb(e,a,k,l)*t2_abab(f,d,m,j)*t2_bbbb(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,eakl,fdmj,bcin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(f,d,j,m)*t2_bbbb(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eakl,fdjm,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(f,d,i,j)*t2_bbbb(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eakl,fdij,bcnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<m,n||f,e>_abab*t2_bbbb(e,a,j,l)*t2_abab(f,b,m,k)*t2_bbbb(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,eajl,fbmk,cdin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,l)*t2_bbbb(f,b,k,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajl,fbkm,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,n||f,e>_abab*t2_bbbb(e,a,j,l)*t2_abab(f,d,m,k)*t2_bbbb(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,eajl,fdmk,bcin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,l)*t2_bbbb(f,d,k,m)*t2_bbbb(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajl,fdkm,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,n||f,e>_abab*t2_bbbb(e,a,i,l)*t2_abab(f,b,m,k)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,eail,fbmk,cdjn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,l)*t2_bbbb(f,b,k,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eail,fbkm,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,l)*t2_bbbb(f,b,j,k)*t2_bbbb(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eail,fbjk,cdnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,n||f,e>_abab*t2_bbbb(e,a,i,l)*t2_abab(f,d,m,k)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,eail,fdmk,bcjn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,l)*t2_bbbb(f,d,k,m)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eail,fdkm,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,l)*t2_bbbb(f,d,j,k)*t2_bbbb(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eail,fdjk,bcnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)*P(b,c)<m,n||f,e>_abab*t2_bbbb(e,a,j,k)*t2_abab(f,b,m,l)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,eajk,fbml,cdin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	  1.0000 P(i,l)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(f,b,l,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajk,fblm,cdin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(f,b,i,l)*t2_bbbb(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eajk,fbil,cdnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)*P(a,b)<m,n||f,e>_abab*t2_bbbb(e,a,j,k)*t2_abab(f,d,m,l)*t2_bbbb(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,eajk,fdml,bcin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdljki', contracted_intermediate) 
    
    #	  1.0000 P(i,l)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(f,d,l,m)*t2_bbbb(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajk,fdlm,bcin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(f,d,i,l)*t2_bbbb(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eajk,fdil,bcnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,c)<m,n||f,e>_abab*t2_bbbb(e,a,i,k)*t2_abab(f,b,m,l)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,eaik,fbml,cdjn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,k)*t2_bbbb(f,b,l,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaik,fblm,cdjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(a,b)<m,n||f,e>_abab*t2_bbbb(e,a,i,k)*t2_abab(f,d,m,l)*t2_bbbb(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,eaik,fdml,bcjn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,k)*t2_bbbb(f,d,l,m)*t2_bbbb(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaik,fdlm,bcjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<m,n||f,e>_abab*t2_bbbb(e,a,i,j)*t2_abab(f,b,m,l)*t2_bbbb(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,eaij,fbml,cdkn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,j)*t2_bbbb(f,b,l,m)*t2_bbbb(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fblm,cdkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,n||f,e>_abab*t2_bbbb(e,a,i,j)*t2_abab(f,d,m,l)*t2_bbbb(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,eaij,fdml,bckn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>_bbbb*t2_bbbb(e,a,i,j)*t2_bbbb(f,d,l,m)*t2_bbbb(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fdlm,bckn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||e,f>_abab*t2_abab(e,b,m,l)*t2_bbbb(f,c,j,k)*t2_bbbb(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,ebml,fcjk,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,b,l,m)*t2_bbbb(f,c,j,k)*t2_bbbb(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eblm,fcjk,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,n||e,f>_abab*t2_abab(e,b,m,l)*t2_bbbb(f,c,i,j)*t2_bbbb(a,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,ebml,fcij,adkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,b,l,m)*t2_bbbb(f,c,i,j)*t2_bbbb(a,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eblm,fcij,adkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||e,f>_abab*t2_abab(e,b,m,k)*t2_bbbb(f,c,j,l)*t2_bbbb(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,ebmk,fcjl,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,b,k,m)*t2_bbbb(f,c,j,l)*t2_bbbb(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebkm,fcjl,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<m,n||e,f>_abab*t2_abab(e,b,m,j)*t2_bbbb(f,c,k,l)*t2_bbbb(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,ebmj,fckl,adin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>_bbbb*t2_bbbb(e,b,j,m)*t2_bbbb(f,c,k,l)*t2_bbbb(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjm,fckl,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||e,f>_abab*t2_abab(e,b,m,j)*t2_bbbb(f,c,i,k)*t2_bbbb(a,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,ebmj,fcik,adln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,b,j,m)*t2_bbbb(f,c,i,k)*t2_bbbb(a,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjm,fcik,adln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,n||e,f>_abab*t2_abab(e,b,m,i)*t2_bbbb(f,c,k,l)*t2_bbbb(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,ebmi,fckl,adjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,b,i,m)*t2_bbbb(f,c,k,l)*t2_bbbb(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebim,fckl,adjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||f,e>_abab*t2_bbbb(e,b,k,l)*t2_abab(f,c,m,j)*t2_bbbb(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,ebkl,fcmj,adin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,b,k,l)*t2_bbbb(f,c,j,m)*t2_bbbb(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebkl,fcjm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,b,k,l)*t2_bbbb(f,c,i,j)*t2_bbbb(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebkl,fcij,adnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<m,n||f,e>_abab*t2_bbbb(e,b,j,l)*t2_abab(f,c,m,k)*t2_bbbb(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,ebjl,fcmk,adin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,f>_bbbb*t2_bbbb(e,b,j,l)*t2_bbbb(f,c,k,m)*t2_bbbb(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebjl,fckm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,n||f,e>_abab*t2_bbbb(e,b,i,l)*t2_abab(f,c,m,k)*t2_bbbb(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,ebil,fcmk,adjn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,b,i,l)*t2_bbbb(f,c,k,m)*t2_bbbb(a,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebil,fckm,adjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,b,i,l)*t2_bbbb(f,c,j,k)*t2_bbbb(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebil,fcjk,adnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)<m,n||f,e>_abab*t2_bbbb(e,b,j,k)*t2_abab(f,c,m,l)*t2_bbbb(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,ebjk,fcml,adin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  1.0000 P(i,l)<n,m||e,f>_bbbb*t2_bbbb(e,b,j,k)*t2_bbbb(f,c,l,m)*t2_bbbb(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjk,fclm,adin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)<n,m||e,f>_bbbb*t2_bbbb(e,b,j,k)*t2_bbbb(f,c,i,l)*t2_bbbb(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebjk,fcil,adnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<m,n||f,e>_abab*t2_bbbb(e,b,i,k)*t2_abab(f,c,m,l)*t2_bbbb(a,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,ebik,fcml,adjn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<n,m||e,f>_bbbb*t2_bbbb(e,b,i,k)*t2_bbbb(f,c,l,m)*t2_bbbb(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebik,fclm,adjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<m,n||f,e>_abab*t2_bbbb(e,b,i,j)*t2_abab(f,c,m,l)*t2_bbbb(a,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,ebij,fcml,adkn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,b,i,j)*t2_bbbb(f,c,l,m)*t2_bbbb(a,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebij,fclm,adkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||e,f>_abab*t2_abab(e,c,m,l)*t2_bbbb(f,d,j,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,ecml,fdjk,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,c,l,m)*t2_bbbb(f,d,j,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eclm,fdjk,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,n||e,f>_abab*t2_abab(e,c,m,l)*t2_bbbb(f,d,i,j)*t2_bbbb(a,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,ecml,fdij,abkn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,c,l,m)*t2_bbbb(f,d,i,j)*t2_bbbb(a,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eclm,fdij,abkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||e,f>_abab*t2_abab(e,c,m,k)*t2_bbbb(f,d,j,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,ecmk,fdjl,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,c,k,m)*t2_bbbb(f,d,j,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eckm,fdjl,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<m,n||e,f>_abab*t2_abab(e,c,m,j)*t2_bbbb(f,d,k,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,ecmj,fdkl,abin->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>_bbbb*t2_bbbb(e,c,j,m)*t2_bbbb(f,d,k,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjm,fdkl,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||e,f>_abab*t2_abab(e,c,m,j)*t2_bbbb(f,d,i,k)*t2_bbbb(a,b,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,ecmj,fdik,abln->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,c,j,m)*t2_bbbb(f,d,i,k)*t2_bbbb(a,b,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjm,fdik,abln->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,n||e,f>_abab*t2_abab(e,c,m,i)*t2_bbbb(f,d,k,l)*t2_bbbb(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,ecmi,fdkl,abjn->abcdijkl', g_abab[o, o, v, v], t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,c,i,m)*t2_bbbb(f,d,k,l)*t2_bbbb(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecim,fdkl,abjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||f,e>_abab*t2_bbbb(e,c,k,l)*t2_abab(f,d,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,eckl,fdmj,abin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>_bbbb*t2_bbbb(e,c,k,l)*t2_bbbb(f,d,j,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eckl,fdjm,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,c,k,l)*t2_bbbb(f,d,i,j)*t2_bbbb(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eckl,fdij,abnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<m,n||f,e>_abab*t2_bbbb(e,c,j,l)*t2_abab(f,d,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,ecjl,fdmk,abin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,f>_bbbb*t2_bbbb(e,c,j,l)*t2_bbbb(f,d,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecjl,fdkm,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,n||f,e>_abab*t2_bbbb(e,c,i,l)*t2_abab(f,d,m,k)*t2_bbbb(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,ecil,fdmk,abjn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>_bbbb*t2_bbbb(e,c,i,l)*t2_bbbb(f,d,k,m)*t2_bbbb(a,b,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecil,fdkm,abjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,c,i,l)*t2_bbbb(f,d,j,k)*t2_bbbb(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ecil,fdjk,abnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)<m,n||f,e>_abab*t2_bbbb(e,c,j,k)*t2_abab(f,d,m,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,ecjk,fdml,abin->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  1.0000 P(i,l)<n,m||e,f>_bbbb*t2_bbbb(e,c,j,k)*t2_bbbb(f,d,l,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjk,fdlm,abin->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)<n,m||e,f>_bbbb*t2_bbbb(e,c,j,k)*t2_bbbb(f,d,i,l)*t2_bbbb(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ecjk,fdil,abnm->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<m,n||f,e>_abab*t2_bbbb(e,c,i,k)*t2_abab(f,d,m,l)*t2_bbbb(a,b,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,ecik,fdml,abjn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<n,m||e,f>_bbbb*t2_bbbb(e,c,i,k)*t2_bbbb(f,d,l,m)*t2_bbbb(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecik,fdlm,abjn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<m,n||f,e>_abab*t2_bbbb(e,c,i,j)*t2_abab(f,d,m,l)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,ecij,fdml,abkn->abcdijkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>_bbbb*t2_bbbb(e,c,i,j)*t2_bbbb(f,d,l,m)*t2_bbbb(a,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecij,fdlm,abkn->abcdijkl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    return quadruples_res

