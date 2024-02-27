import numpy 
from numpy import einsum


def ccsdtq_t3_aaaaaa_residual(t2_aaaa, t2_bbbb, t2_abab, 
                              g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(i,j)*P(a,b)<l,a||j,k>_aaaa*t2_aaaa(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lajk,bcil->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    triples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<l,a||i,j>_aaaa*t2_aaaa(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('laij,bckl->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<l,c||j,k>_aaaa*t2_aaaa(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -1.0000 <l,c||i,j>_aaaa*t2_aaaa(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	 -1.0000 P(j,k)*P(b,c)<a,b||d,k>_aaaa*t2_aaaa(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<a,b||d,i>_aaaa*t2_aaaa(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<b,c||d,k>_aaaa*t2_aaaa(d,a,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcdk,daij->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 <b,c||d,i>_aaaa*t2_aaaa(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    return triples_res


def ccsdtq_t3_aabaab_residual(t2_aaaa, t2_bbbb, t2_abab, 
                              g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	  1.0000 P(i,j)*P(a,b)<a,l||j,k>_abab*t2_abab(b,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('aljk,bcil->abcijk', g_abab[v, o, o, o], t2_abab)
    triples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<l,a||i,j>_aaaa*t2_abab(b,c,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('laij,bclk->abcijk', g_aaaa[o, v, o, o], t2_abab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<l,c||j,k>_abab*t2_aaaa(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g_abab[o, v, o, o], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  1.0000 <a,c||d,k>_abab*t2_aaaa(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('acdk,dbij->abcijk', g_abab[v, v, v, o], t2_aaaa)
    
    #	  1.0000 <a,b||d,j>_aaaa*t2_abab(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('abdj,dcik->abcijk', g_aaaa[v, v, v, o], t2_abab)
    
    #	 -1.0000 <a,c||j,d>_abab*t2_abab(b,d,i,k)
    triples_res += -1.000000000000000 * einsum('acjd,bdik->abcijk', g_abab[v, v, o, v], t2_abab)
    
    #	 -1.0000 <a,b||d,i>_aaaa*t2_abab(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_abab)
    
    #	  1.0000 <a,c||i,d>_abab*t2_abab(b,d,j,k)
    triples_res +=  1.000000000000000 * einsum('acid,bdjk->abcijk', g_abab[v, v, o, v], t2_abab)
    
    #	 -1.0000 <b,c||d,k>_abab*t2_aaaa(d,a,i,j)
    triples_res += -1.000000000000000 * einsum('bcdk,daij->abcijk', g_abab[v, v, v, o], t2_aaaa)
    
    #	  1.0000 <b,c||j,d>_abab*t2_abab(a,d,i,k)
    triples_res +=  1.000000000000000 * einsum('bcjd,adik->abcijk', g_abab[v, v, o, v], t2_abab)
    
    #	 -1.0000 <b,c||i,d>_abab*t2_abab(a,d,j,k)
    triples_res += -1.000000000000000 * einsum('bcid,adjk->abcijk', g_abab[v, v, o, v], t2_abab)
    
    
    return triples_res


def ccsdtq_t3_abbabb_residual(t2_aaaa, t2_bbbb, t2_abab, 
                              g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	  1.0000 <l,b||j,k>_bbbb*t2_abab(a,c,i,l)
    triples_res =  1.000000000000000 * einsum('lbjk,acil->abcijk', g_bbbb[o, v, o, o], t2_abab)
    
    #	 -1.0000 <a,l||i,k>_abab*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('alik,bcjl->abcijk', g_abab[v, o, o, o], t2_bbbb)
    
    #	  1.0000 <l,b||i,k>_abab*t2_abab(a,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lbik,aclj->abcijk', g_abab[o, v, o, o], t2_abab)
    
    #	  1.0000 <a,l||i,j>_abab*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('alij,bckl->abcijk', g_abab[v, o, o, o], t2_bbbb)
    
    #	 -1.0000 <l,b||i,j>_abab*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('lbij,aclk->abcijk', g_abab[o, v, o, o], t2_abab)
    
    #	 -1.0000 <l,c||j,k>_bbbb*t2_abab(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcjk,abil->abcijk', g_bbbb[o, v, o, o], t2_abab)
    
    #	 -1.0000 <l,c||i,k>_abab*t2_abab(a,b,l,j)
    triples_res += -1.000000000000000 * einsum('lcik,ablj->abcijk', g_abab[o, v, o, o], t2_abab)
    
    #	  1.0000 <l,c||i,j>_abab*t2_abab(a,b,l,k)
    triples_res +=  1.000000000000000 * einsum('lcij,ablk->abcijk', g_abab[o, v, o, o], t2_abab)
    
    #	 -1.0000 P(j,k)*P(b,c)<a,b||d,k>_abab*t2_abab(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g_abab[v, v, v, o], t2_abab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<a,b||i,d>_abab*t2_bbbb(d,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('abid,dcjk->abcijk', g_abab[v, v, o, v], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<b,c||d,k>_bbbb*t2_abab(a,d,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('bcdk,adij->abcijk', g_bbbb[v, v, v, o], t2_abab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    
    return triples_res


def ccsdtq_t3_bbbbbb_residual(t2_aaaa, t2_bbbb, t2_abab, 
                              g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(i,j)*P(a,b)<l,a||j,k>_bbbb*t2_bbbb(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lajk,bcil->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    triples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<l,a||i,j>_bbbb*t2_bbbb(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('laij,bckl->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<l,c||j,k>_bbbb*t2_bbbb(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -1.0000 <l,c||i,j>_bbbb*t2_bbbb(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	 -1.0000 P(j,k)*P(b,c)<a,b||d,k>_bbbb*t2_bbbb(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<a,b||d,i>_bbbb*t2_bbbb(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abdi,dcjk->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<b,c||d,k>_bbbb*t2_bbbb(d,a,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcdk,daij->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 <b,c||d,i>_bbbb*t2_bbbb(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    
    return triples_res

