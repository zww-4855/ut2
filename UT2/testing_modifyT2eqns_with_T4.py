import numpy as np
from numpy import einsum

def ccsdt_t2_aaaa_residual_Qf(t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa,t3_bbbbbb,t3_aabaab,t3_abbabb,  g_aaaa, g_bbbb, g_abab,l2_aaaa, l2_bbbb, l2_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -0.2500 P(i,j)<n,m||k,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmkl,kldc,dcjm,abin->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||k,l>_abab*l2_abab(k,l,d,c)*t2_abab(d,c,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmkl,kldc,dcjm,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||k,l>_abab*l2_abab(k,l,c,d)*t2_abab(c,d,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmkl,klcd,cdjm,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||l,k>_abab*l2_abab(l,k,d,c)*t2_abab(d,c,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmlk,lkdc,dcjm,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||l,k>_abab*l2_abab(l,k,c,d)*t2_abab(c,d,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmlk,lkcd,cdjm,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||k,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmkl,kldc,dajm,cbin->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||k,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(d,a,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnkl,kldc,dajm,bcin->abij', g_abab[o, o, o, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||k,l>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmkl,klcd,adjm,cbin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||l,k>_abab*l2_abab(l,k,d,c)*t2_aaaa(d,a,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnlk,lkdc,dajm,bcin->abij', g_abab[o, o, o, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||l,k>_abab*l2_abab(l,k,c,d)*t2_abab(a,d,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmlk,lkcd,adjm,cbin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||k,l>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,d,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmkl,kldc,adjm,bcin->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||j,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,kldc,dckm,abin->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||j,l>_abab*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,kldc,dckm,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||j,l>_abab*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,klcd,cdkm,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||j,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,lkdc,dcmk,abin->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||j,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,lkcd,cdmk,abin->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||j,l>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,kldc,dckm,abin->abij', g_abab[o, o, o, o], l2_bbbb, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||j,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,i,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmjl,kldc,dcim,abkn->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||j,l>_abab*l2_abab(k,l,d,c)*t2_abab(d,c,i,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmjl,kldc,dcim,abkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||j,l>_abab*l2_abab(k,l,c,d)*t2_abab(c,d,i,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmjl,klcd,cdim,abkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||j,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,k,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,kldc,dakm,cbin->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||j,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(d,a,k,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnjl,kldc,dakm,bcin->abij', g_abab[o, o, o, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||j,l>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,k,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,klcd,adkm,cbin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||j,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(a,d,m,k)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,lkcd,admk,cbin->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||j,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,d,m,k)*t2_abab(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnjl,kldc,admk,bcin->abij', g_abab[o, o, o, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||j,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,i,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,kldc,daim,cbkn->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||j,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(d,a,i,m)*t2_abab(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnjl,kldc,daim,bckn->abij', g_abab[o, o, o, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||j,l>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,i,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,klcd,adim,cbkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||j,l>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(d,a,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,lkdc,daim,bcnk->abij', g_aaaa[o, o, o, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||j,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,d,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,kldc,adim,bcnk->abij', g_abab[o, o, o, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 <n,m||i,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,l,m)*t2_aaaa(a,b,k,n)
    doubles_res += -0.500000000000000 * einsum('nmij,kldc,dclm,abkn->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,j>_aaaa*l2_abab(k,l,d,c)*t2_abab(d,c,m,l)*t2_aaaa(a,b,k,n)
    doubles_res +=  0.500000000000000 * einsum('nmij,kldc,dcml,abkn->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,j>_aaaa*l2_abab(k,l,c,d)*t2_abab(c,d,m,l)*t2_aaaa(a,b,k,n)
    doubles_res +=  0.500000000000000 * einsum('nmij,klcd,cdml,abkn->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 P(a,b)<n,m||i,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,l,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmij,kldc,dalm,cbkn->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||i,j>_aaaa*l2_abab(k,l,c,d)*t2_abab(a,d,m,l)*t2_aaaa(c,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmij,klcd,adml,cbkn->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||i,j>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(d,a,l,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmij,lkdc,dalm,bcnk->abij', g_aaaa[o, o, o, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||i,j>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(a,d,m,l)*t2_abab(b,c,n,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmij,kldc,adml,bcnk->abij', g_aaaa[o, o, o, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,d||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,j,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,kldc,ecjk,abim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||e,l>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,c,j,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,klcd,ecjk,abim->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||e,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,j,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,lkdc,ecjk,abim->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,d||l,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,j,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdle,lkcd,cejk,abim->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,d||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,kldc,ecjk,abim->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 <m,d||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,j)*t2_aaaa(a,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdel,kldc,ecij,abkm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,c,i,j)*t2_aaaa(a,b,k,m)
    doubles_res += -1.000000000000000 * einsum('mdel,klcd,ecij,abkm->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,k)*t2_aaaa(c,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eajk,cbim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,m||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,j,k)*t2_abab(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('dmel,kldc,eajk,bcim->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,l>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,j,k)*t2_aaaa(c,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,klcd,eajk,cbim->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<d,m||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,j,k)*t2_abab(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmle,lkdc,aejk,bcim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||l,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,j,k)*t2_aaaa(c,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdle,lkcd,aejk,cbim->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,k)*t2_abab(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,aejk,bcim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,d||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_aaaa(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eaij,cbkm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<d,m||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('dmel,kldc,eaij,bckm->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,d||e,l>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,j)*t2_aaaa(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,klcd,eaij,cbkm->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,d||e,l>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,lkdc,eaij,bcmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,d||e,l>_abab*l2_bbbb(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eaij,bcmk->abij', g_abab[o, v, v, o], l2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,d||e,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdej,kldc,eckl,abim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,d||e,j>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdej,kldc,eckl,abim->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,d||j,e>_abab*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('mdje,klcd,cekl,abim->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,d||e,j>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdej,lkdc,eclk,abim->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,d||j,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('mdje,lkcd,celk,abim->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,d||j,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_aaaa(a,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('mdje,kldc,eckl,abim->abij', g_abab[o, v, o, v], l2_bbbb, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||e,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,l)*t2_aaaa(a,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,kldc,ecil,abkm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||e,j>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,c,i,l)*t2_aaaa(a,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,kldc,ecil,abkm->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,d||j,e>_abab*l2_abab(k,l,c,d)*t2_abab(c,e,i,l)*t2_aaaa(a,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdje,klcd,ceil,abkm->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,d||e,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_aaaa(c,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('mdej,kldc,eakl,cbim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<d,m||j,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(b,c,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('dmje,kldc,aekl,bcim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,d||j,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_aaaa(c,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdje,klcd,aekl,cbim->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<d,m||j,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(b,c,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('dmje,lkdc,aelk,bcim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,d||j,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_aaaa(c,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdje,lkcd,aelk,cbim->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,l)*t2_aaaa(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdej,kldc,eail,cbkm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<d,m||j,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmje,kldc,aeil,bckm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||j,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_aaaa(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdje,klcd,aeil,cbkm->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,j>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_abab(b,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,lkdc,eail,bcmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||j,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(b,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mdje,kldc,aeil,bcmk->abij', g_abab[o, v, o, v], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,k)*t2_aaaa(c,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,kldc,edjk,cbim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<a,m||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,j,k)*t2_abab(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('amel,kldc,edjk,bcim->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,m||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,j,k)*t2_abab(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('amle,lkdc,dejk,bcim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,j,k)*t2_aaaa(c,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,lkcd,edjk,cbim->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<a,m||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,k)*t2_abab(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('amel,kldc,edjk,bcim->abij', g_abab[v, o, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,a||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_aaaa(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,kldc,edij,cbkm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<a,m||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_abab(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('amel,kldc,edij,bckm->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,a||e,l>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,j)*t2_abab(b,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mael,lkdc,edij,bcmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_aaaa(d,c,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebjk,dcim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<a,m||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_abab(d,c,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('amel,kldc,ebjk,dcim->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<a,m||e,l>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,b,j,k)*t2_abab(c,d,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('amel,klcd,ebjk,cdim->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<a,m||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(b,e,j,k)*t2_abab(d,c,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('amle,lkdc,bejk,dcim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<a,m||l,e>_abab*l2_abab(l,k,c,d)*t2_abab(b,e,j,k)*t2_abab(c,d,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('amle,lkcd,bejk,cdim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,a||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_aaaa(d,c,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebij,dckm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<a,m||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_abab(d,c,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('amel,kldc,ebij,dckm->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<a,m||e,l>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,b,i,j)*t2_abab(c,d,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('amel,klcd,ebij,cdkm->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,a||e,l>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,b,i,j)*t2_abab(d,c,m,k)
    contracted_intermediate =  0.500000000000000 * einsum('mael,lkdc,ebij,dcmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,a||e,l>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,b,i,j)*t2_abab(c,d,m,k)
    contracted_intermediate =  0.500000000000000 * einsum('mael,lkcd,ebij,cdmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<a,m||e,l>_abab*l2_bbbb(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_bbbb(d,c,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('amel,kldc,ebij,dckm->abij', g_abab[v, o, v, o], l2_bbbb, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(c,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('maej,kldc,edkl,cbim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(b,c,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('amje,kldc,dekl,bcim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(c,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,klcd,edkl,cbim->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(b,c,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('amje,lkdc,delk,bcim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(c,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,lkcd,edlk,cbim->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(b,c,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('amje,kldc,edkl,bcim->abij', g_abab[v, o, o, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_aaaa(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('maej,kldc,edil,cbkm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('amje,kldc,deil,bckm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_aaaa(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,klcd,edil,cbkm->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_abab(b,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('maej,lkdc,edil,bcmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_abab(b,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('maej,kldc,edil,bcmk->abij', g_aaaa[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,k,l)*t2_aaaa(d,c,i,m)
    contracted_intermediate =  0.250000000000000 * einsum('maej,kldc,ebkl,dcim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_abab(k,l,d,c)*t2_abab(b,e,k,l)*t2_abab(d,c,i,m)
    contracted_intermediate = -0.250000000000000 * einsum('amje,kldc,bekl,dcim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_abab(k,l,c,d)*t2_abab(b,e,k,l)*t2_abab(c,d,i,m)
    contracted_intermediate = -0.250000000000000 * einsum('amje,klcd,bekl,cdim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_abab(l,k,d,c)*t2_abab(b,e,l,k)*t2_abab(d,c,i,m)
    contracted_intermediate = -0.250000000000000 * einsum('amje,lkdc,belk,dcim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_abab(l,k,c,d)*t2_abab(b,e,l,k)*t2_abab(c,d,i,m)
    contracted_intermediate = -0.250000000000000 * einsum('amje,lkcd,belk,cdim->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,i,l)*t2_aaaa(d,c,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,kldc,ebil,dckm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_abab(k,l,d,c)*t2_abab(b,e,i,l)*t2_abab(d,c,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('amje,kldc,beil,dckm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_abab(k,l,c,d)*t2_abab(b,e,i,l)*t2_abab(c,d,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('amje,klcd,beil,cdkm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,b,i,l)*t2_abab(d,c,m,k)
    contracted_intermediate = -0.500000000000000 * einsum('maej,lkdc,ebil,dcmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,b,i,l)*t2_abab(c,d,m,k)
    contracted_intermediate = -0.500000000000000 * einsum('maej,lkcd,ebil,cdmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<a,m||j,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(b,e,i,l)*t2_bbbb(d,c,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('amje,kldc,beil,dckm->abij', g_abab[v, o, o, v], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 <d,c||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_aaaa(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('dcef,kldc,eakl,fbij->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <d,c||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_aaaa(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('dcfe,kldc,aekl,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <c,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_aaaa(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('cdfe,klcd,aekl,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <d,c||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_aaaa(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('dcfe,lkdc,aelk,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <c,d||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_aaaa(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('cdfe,lkcd,aelk,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 P(i,j)<d,c||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,l)*t2_aaaa(f,b,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('dcef,kldc,eajl,fbik->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<d,c||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('dcfe,kldc,aejl,fbik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<c,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('cdfe,klcd,aejl,fbik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<d,c||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,j,l)*t2_abab(b,f,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('dcef,lkdc,eajl,bfik->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<c,d||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,j,l)*t2_abab(b,f,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('cdef,lkcd,eajl,bfik->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<d,c||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,l)*t2_abab(b,f,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('dcef,kldc,aejl,bfik->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 <d,c||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_aaaa(f,b,k,l)
    doubles_res += -0.250000000000000 * einsum('dcef,kldc,eaij,fbkl->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,k,l)
    doubles_res +=  0.250000000000000 * einsum('dcef,kldc,eaij,bfkl->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <c,d||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,j)*t2_abab(b,f,k,l)
    doubles_res +=  0.250000000000000 * einsum('cdef,klcd,eaij,bfkl->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,k)
    doubles_res +=  0.250000000000000 * einsum('dcef,lkdc,eaij,bflk->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <c,d||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,k)
    doubles_res +=  0.250000000000000 * einsum('cdef,lkcd,eaij,bflk->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 P(a,b)<d,a||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_aaaa(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,eckl,fbij->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<d,a||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_aaaa(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,eckl,fbij->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<a,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_aaaa(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('adfe,klcd,cekl,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<d,a||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_aaaa(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,lkdc,eclk,fbij->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<a,d||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_aaaa(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('adfe,lkcd,celk,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<a,d||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_aaaa(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('adfe,kldc,eckl,fbij->abij', g_abab[v, v, v, v], l2_bbbb, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,a||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,j,l)*t2_aaaa(f,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('daef,kldc,ecjl,fbik->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,a||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,c,j,l)*t2_aaaa(f,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('daef,kldc,ecjl,fbik->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(c,e,j,l)*t2_aaaa(f,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('adfe,klcd,cejl,fbik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,d||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,c,j,l)*t2_abab(b,f,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('adef,lkcd,ecjl,bfik->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,d||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,l)*t2_abab(b,f,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('adef,kldc,ecjl,bfik->abij', g_abab[v, v, v, v], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<d,a||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,j)*t2_aaaa(f,b,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,ecij,fbkl->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<a,d||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,c,i,j)*t2_abab(b,f,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('adef,klcd,ecij,bfkl->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<a,d||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,c,i,j)*t2_abab(b,f,l,k)
    contracted_intermediate = -0.500000000000000 * einsum('adef,lkcd,ecij,bflk->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 <a,b||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(f,c,i,j)
    doubles_res += -0.500000000000000 * einsum('abef,kldc,edkl,fcij->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,b||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(f,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('abef,klcd,edkl,fcij->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,b||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(f,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('abef,lkcd,edlk,fcij->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 P(i,j)<a,b||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_aaaa(f,c,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('abef,kldc,edjl,fcik->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<a,b||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_aaaa(f,c,i,k)
    contracted_intermediate = -0.500000000000000 * einsum('abef,klcd,edjl,fcik->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<a,b||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_abab(f,c,i,k)
    contracted_intermediate = -0.500000000000000 * einsum('abef,lkdc,edjl,fcik->abij', g_aaaa[v, v, v, v], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<a,b||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_abab(f,c,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('abef,kldc,edjl,fcik->abij', g_aaaa[v, v, v, v], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 <m,d||k,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(c,a,b,i,j,m)
    doubles_res +=  0.500000000000000 * einsum('mdkl,kldc,cabijm->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||k,l>_abab*l2_abab(k,l,d,c)*t3_aabaab(b,a,c,i,j,m)
    doubles_res +=  0.500000000000000 * einsum('dmkl,kldc,bacijm->abij', g_abab[v, o, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||k,l>_abab*l2_abab(k,l,c,d)*t3_aaaaaa(c,a,b,i,j,m)
    doubles_res += -0.500000000000000 * einsum('mdkl,klcd,cabijm->abij', g_abab[o, v, o, o], l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,k>_abab*l2_abab(l,k,d,c)*t3_aabaab(b,a,c,i,j,m)
    doubles_res +=  0.500000000000000 * einsum('dmlk,lkdc,bacijm->abij', g_abab[v, o, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||l,k>_abab*l2_abab(l,k,c,d)*t3_aaaaaa(c,a,b,i,j,m)
    doubles_res += -0.500000000000000 * einsum('mdlk,lkcd,cabijm->abij', g_abab[o, v, o, o], l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||k,l>_bbbb*l2_bbbb(k,l,d,c)*t3_aabaab(b,a,c,i,j,m)
    doubles_res += -0.500000000000000 * einsum('mdkl,kldc,bacijm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,j)<m,d||j,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(c,a,b,i,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdjl,kldc,cabikm->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<d,m||j,l>_abab*l2_abab(k,l,d,c)*t3_aabaab(b,a,c,i,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('dmjl,kldc,bacikm->abij', g_abab[v, o, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,d||j,l>_abab*l2_abab(k,l,c,d)*t3_aaaaaa(c,a,b,i,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdjl,klcd,cabikm->abij', g_abab[o, v, o, o], l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,d||j,l>_aaaa*l2_abab(l,k,d,c)*t3_aabaab(b,a,c,i,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mdjl,lkdc,bacimk->abij', g_aaaa[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||j,l>_abab*l2_bbbb(k,l,d,c)*t3_aabaab(b,a,c,i,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mdjl,kldc,bacimk->abij', g_abab[o, v, o, o], l2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 <m,d||i,j>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(c,a,b,k,l,m)
    doubles_res +=  0.500000000000000 * einsum('mdij,kldc,cabklm->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <m,d||i,j>_aaaa*l2_abab(k,l,d,c)*t3_aabaab(b,a,c,k,m,l)
    doubles_res +=  0.500000000000000 * einsum('mdij,kldc,backml->abij', g_aaaa[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,d||i,j>_aaaa*l2_abab(l,k,d,c)*t3_aabaab(b,a,c,m,l,k)
    doubles_res += -0.500000000000000 * einsum('mdij,lkdc,bacmlk->abij', g_aaaa[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 P(a,b)<m,a||k,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,b,i,j,m)
    contracted_intermediate =  0.250000000000000 * einsum('makl,kldc,dcbijm->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<a,m||k,l>_abab*l2_abab(k,l,d,c)*t3_aabaab(d,b,c,i,j,m)
    contracted_intermediate =  0.250000000000000 * einsum('amkl,kldc,dbcijm->abij', g_abab[v, o, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<a,m||k,l>_abab*l2_abab(k,l,c,d)*t3_aabaab(b,c,d,i,j,m)
    contracted_intermediate = -0.250000000000000 * einsum('amkl,klcd,bcdijm->abij', g_abab[v, o, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<a,m||l,k>_abab*l2_abab(l,k,d,c)*t3_aabaab(d,b,c,i,j,m)
    contracted_intermediate =  0.250000000000000 * einsum('amlk,lkdc,dbcijm->abij', g_abab[v, o, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<a,m||l,k>_abab*l2_abab(l,k,c,d)*t3_aabaab(b,c,d,i,j,m)
    contracted_intermediate = -0.250000000000000 * einsum('amlk,lkcd,bcdijm->abij', g_abab[v, o, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||j,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,b,i,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('majl,kldc,dcbikm->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<a,m||j,l>_abab*l2_abab(k,l,d,c)*t3_aabaab(d,b,c,i,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('amjl,kldc,dbcikm->abij', g_abab[v, o, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<a,m||j,l>_abab*l2_abab(k,l,c,d)*t3_aabaab(b,c,d,i,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('amjl,klcd,bcdikm->abij', g_abab[v, o, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||j,l>_aaaa*l2_abab(l,k,d,c)*t3_aabaab(d,b,c,i,m,k)
    contracted_intermediate =  0.500000000000000 * einsum('majl,lkdc,dbcimk->abij', g_aaaa[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||j,l>_aaaa*l2_abab(l,k,c,d)*t3_aabaab(b,c,d,i,m,k)
    contracted_intermediate = -0.500000000000000 * einsum('majl,lkcd,bcdimk->abij', g_aaaa[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<a,m||j,l>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(b,c,d,i,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('amjl,kldc,bcdikm->abij', g_abab[v, o, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,b,k,l,m)
    contracted_intermediate =  0.250000000000000 * einsum('maij,kldc,dcbklm->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_aaaa*l2_abab(k,l,d,c)*t3_aabaab(d,b,c,k,m,l)
    contracted_intermediate =  0.250000000000000 * einsum('maij,kldc,dbckml->abij', g_aaaa[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,a||i,j>_aaaa*l2_abab(k,l,c,d)*t3_aabaab(b,c,d,k,m,l)
    contracted_intermediate = -0.250000000000000 * einsum('maij,klcd,bcdkml->abij', g_aaaa[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,a||i,j>_aaaa*l2_abab(l,k,d,c)*t3_aabaab(d,b,c,m,l,k)
    contracted_intermediate = -0.250000000000000 * einsum('maij,lkdc,dbcmlk->abij', g_aaaa[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_aaaa*l2_abab(l,k,c,d)*t3_aabaab(b,c,d,m,l,k)
    contracted_intermediate =  0.250000000000000 * einsum('maij,lkcd,bcdmlk->abij', g_aaaa[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_aaaa*l2_bbbb(k,l,d,c)*t3_abbabb(b,c,d,m,l,k)
    contracted_intermediate =  0.250000000000000 * einsum('maij,kldc,bcdmlk->abij', g_aaaa[o, v, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 <d,c||e,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,a,b,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('dcel,kldc,eabijk->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <d,c||e,l>_abab*l2_abab(k,l,d,c)*t3_aaaaaa(e,a,b,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('dcel,kldc,eabijk->abij', g_abab[v, v, v, o], l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <c,d||e,l>_abab*l2_abab(k,l,c,d)*t3_aaaaaa(e,a,b,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('cdel,klcd,eabijk->abij', g_abab[v, v, v, o], l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <d,c||l,e>_abab*l2_abab(l,k,d,c)*t3_aabaab(b,a,e,i,j,k)
    doubles_res += -0.500000000000000 * einsum('dcle,lkdc,baeijk->abij', g_abab[v, v, o, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <c,d||l,e>_abab*l2_abab(l,k,c,d)*t3_aabaab(b,a,e,i,j,k)
    doubles_res += -0.500000000000000 * einsum('cdle,lkcd,baeijk->abij', g_abab[v, v, o, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <d,c||e,l>_bbbb*l2_bbbb(k,l,d,c)*t3_aabaab(b,a,e,i,j,k)
    doubles_res += -0.500000000000000 * einsum('dcel,kldc,baeijk->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 P(i,j)<d,c||e,j>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,a,b,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('dcej,kldc,eabikl->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<d,c||j,e>_abab*l2_abab(k,l,d,c)*t3_aabaab(b,a,e,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('dcje,kldc,baeikl->abij', g_abab[v, v, o, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<c,d||j,e>_abab*l2_abab(k,l,c,d)*t3_aabaab(b,a,e,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('cdje,klcd,baeikl->abij', g_abab[v, v, o, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<d,c||j,e>_abab*l2_abab(l,k,d,c)*t3_aabaab(b,a,e,i,l,k)
    contracted_intermediate =  0.250000000000000 * einsum('dcje,lkdc,baeilk->abij', g_abab[v, v, o, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<c,d||j,e>_abab*l2_abab(l,k,c,d)*t3_aabaab(b,a,e,i,l,k)
    contracted_intermediate =  0.250000000000000 * einsum('cdje,lkcd,baeilk->abij', g_abab[v, v, o, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<d,a||e,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,c,b,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('dael,kldc,ecbijk->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<a,d||e,l>_abab*l2_abab(k,l,c,d)*t3_aaaaaa(e,c,b,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adel,klcd,ecbijk->abij', g_abab[v, v, v, o], l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<d,a||e,l>_aaaa*l2_abab(l,k,d,c)*t3_aabaab(e,b,c,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('dael,lkdc,ebcijk->abij', g_aaaa[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<a,d||l,e>_abab*l2_abab(l,k,c,d)*t3_aabaab(b,c,e,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('adle,lkcd,bceijk->abij', g_abab[v, v, o, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<a,d||e,l>_abab*l2_bbbb(k,l,d,c)*t3_aabaab(e,b,c,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adel,kldc,ebcijk->abij', g_abab[v, v, v, o], l2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<d,a||e,j>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,c,b,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('daej,kldc,ecbikl->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<d,a||e,j>_aaaa*l2_abab(k,l,d,c)*t3_aabaab(e,b,c,i,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('daej,kldc,ebcikl->abij', g_aaaa[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<a,d||j,e>_abab*l2_abab(k,l,c,d)*t3_aabaab(b,c,e,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('adje,klcd,bceikl->abij', g_abab[v, v, o, v], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<d,a||e,j>_aaaa*l2_abab(l,k,d,c)*t3_aabaab(e,b,c,i,l,k)
    contracted_intermediate =  0.500000000000000 * einsum('daej,lkdc,ebcilk->abij', g_aaaa[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<a,d||j,e>_abab*l2_abab(l,k,c,d)*t3_aabaab(b,c,e,i,l,k)
    contracted_intermediate = -0.500000000000000 * einsum('adje,lkcd,bceilk->abij', g_abab[v, v, o, v], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<a,d||j,e>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(b,c,e,i,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('adje,kldc,bceikl->abij', g_abab[v, v, o, v], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 <a,b||e,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,d,c,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('abel,kldc,edcijk->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,l>_aaaa*l2_abab(l,k,d,c)*t3_aabaab(e,d,c,i,j,k)
    doubles_res += -0.500000000000000 * einsum('abel,lkdc,edcijk->abij', g_aaaa[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,l>_aaaa*l2_abab(l,k,c,d)*t3_aabaab(e,c,d,i,j,k)
    doubles_res += -0.500000000000000 * einsum('abel,lkcd,ecdijk->abij', g_aaaa[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 P(i,j)<a,b||e,j>_aaaa*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,d,c,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('abej,kldc,edcikl->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<a,b||e,j>_aaaa*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('abej,kldc,edcikl->abij', g_aaaa[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<a,b||e,j>_aaaa*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('abej,klcd,ecdikl->abij', g_aaaa[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<a,b||e,j>_aaaa*l2_abab(l,k,d,c)*t3_aabaab(e,d,c,i,l,k)
    contracted_intermediate =  0.250000000000000 * einsum('abej,lkdc,edcilk->abij', g_aaaa[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<a,b||e,j>_aaaa*l2_abab(l,k,c,d)*t3_aabaab(e,c,d,i,l,k)
    contracted_intermediate =  0.250000000000000 * einsum('abej,lkcd,ecdilk->abij', g_aaaa[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<a,b||e,j>_aaaa*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('abej,kldc,edcikl->abij', g_aaaa[v, v, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    return doubles_res


def ccsdt_t2_bbbb_residual_Qf(t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa,t3_bbbbbb,t3_aabaab,t3_abbabb,  g_aaaa, g_bbbb, g_abab,l2_aaaa, l2_bbbb, l2_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -0.2500 P(i,j)<m,n||k,l>_abab*l2_abab(k,l,d,c)*t2_abab(d,c,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnkl,kldc,dcmj,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||k,l>_abab*l2_abab(k,l,c,d)*t2_abab(c,d,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnkl,klcd,cdmj,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||l,k>_abab*l2_abab(l,k,d,c)*t2_abab(d,c,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnlk,lkdc,dcmj,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||l,k>_abab*l2_abab(l,k,c,d)*t2_abab(c,d,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnlk,lkcd,cdmj,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||k,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,j,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmkl,kldc,dcjm,abin->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||k,l>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(d,a,m,j)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmkl,kldc,damj,cbni->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||k,l>_abab*l2_abab(k,l,d,c)*t2_abab(d,a,m,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnkl,kldc,damj,cbin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||k,l>_abab*l2_abab(k,l,c,d)*t2_bbbb(d,a,j,m)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmkl,klcd,dajm,cbni->abij', g_abab[o, o, o, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||l,k>_abab*l2_abab(l,k,d,c)*t2_abab(d,a,m,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnlk,lkdc,damj,cbin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||l,k>_abab*l2_abab(l,k,c,d)*t2_bbbb(d,a,j,m)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmlk,lkcd,dajm,cbni->abij', g_abab[o, o, o, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||k,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,a,j,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmkl,kldc,dajm,cbin->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||l,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnlj,kldc,dckm,abin->abij', g_abab[o, o, o, o], l2_aaaa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||j,l>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,kldc,dckm,abin->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||j,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,klcd,cdkm,abin->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||l,j>_abab*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnlj,lkdc,dcmk,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||l,j>_abab*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnlj,lkcd,cdmk,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||j,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,kldc,dckm,abin->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||l,j>_abab*l2_abab(l,k,d,c)*t2_abab(d,c,m,i)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnlj,lkdc,dcmi,abkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||l,j>_abab*l2_abab(l,k,c,d)*t2_abab(c,d,m,i)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnlj,lkcd,cdmi,abkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||j,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,i,m)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmjl,kldc,dcim,abkn->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||l,j>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,a,k,m)*t2_abab(c,b,n,i)
    contracted_intermediate = -1.000000000000000 * einsum('nmlj,kldc,dakm,cbni->abij', g_abab[o, o, o, o], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||j,l>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,a,k,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,kldc,dakm,cbin->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||l,j>_abab*l2_abab(l,k,d,c)*t2_abab(d,a,m,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnlj,lkdc,damk,cbin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||l,j>_abab*l2_abab(l,k,c,d)*t2_bbbb(d,a,k,m)*t2_abab(c,b,n,i)
    contracted_intermediate = -1.000000000000000 * einsum('nmlj,lkcd,dakm,cbni->abij', g_abab[o, o, o, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||j,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,a,k,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,kldc,dakm,cbin->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||l,j>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,a,m,i)*t2_abab(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnlj,kldc,dami,cbkn->abij', g_abab[o, o, o, o], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||j,l>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(d,a,i,m)*t2_abab(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,klcd,daim,cbkn->abij', g_bbbb[o, o, o, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,n||l,j>_abab*l2_abab(l,k,d,c)*t2_abab(d,a,m,i)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnlj,lkdc,dami,cbkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||l,j>_abab*l2_abab(l,k,c,d)*t2_bbbb(d,a,i,m)*t2_abab(c,b,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmlj,lkcd,daim,cbnk->abij', g_abab[o, o, o, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||j,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,a,i,m)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,kldc,daim,cbkn->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 <n,m||i,j>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,c,l,m)*t2_bbbb(a,b,k,n)
    doubles_res +=  0.500000000000000 * einsum('nmij,lkdc,dclm,abkn->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,j>_bbbb*l2_abab(l,k,c,d)*t2_abab(c,d,l,m)*t2_bbbb(a,b,k,n)
    doubles_res +=  0.500000000000000 * einsum('nmij,lkcd,cdlm,abkn->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||i,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,l,m)*t2_bbbb(a,b,k,n)
    doubles_res += -0.500000000000000 * einsum('nmij,kldc,dclm,abkn->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 P(a,b)<n,m||i,j>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,a,l,m)*t2_abab(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmij,kldc,dalm,cbkn->abij', g_bbbb[o, o, o, o], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||i,j>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(d,a,l,m)*t2_abab(c,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmij,klcd,dalm,cbkn->abij', g_bbbb[o, o, o, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||i,j>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,a,l,m)*t2_bbbb(c,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmij,lkdc,dalm,cbkn->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<n,m||i,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,a,l,m)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmij,kldc,dalm,cbkn->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<d,m||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(c,e,k,j)*t2_bbbb(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmle,kldc,cekj,abim->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<d,m||e,l>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,k,j)*t2_bbbb(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmel,kldc,eckj,abim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||e,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,k,j)*t2_bbbb(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,klcd,cekj,abim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<d,m||l,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,c,j,k)*t2_bbbb(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('dmle,lkdc,ecjk,abim->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,d||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,k)*t2_bbbb(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,kldc,ecjk,abim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 <d,m||l,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,c,i,j)*t2_bbbb(a,b,k,m)
    doubles_res += -1.000000000000000 * einsum('dmle,lkdc,ecij,abkm->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,i,j)*t2_bbbb(a,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdel,kldc,ecij,abkm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,a,k,j)*t2_abab(c,b,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eakj,cbmi->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,m||e,l>_abab*l2_abab(k,l,d,c)*t2_abab(e,a,k,j)*t2_bbbb(c,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('dmel,kldc,eakj,cbim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,a,k,j)*t2_abab(c,b,m,i)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,klcd,eakj,cbmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<d,m||l,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,a,j,k)*t2_bbbb(c,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmle,lkdc,eajk,cbim->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,a,j,k)*t2_abab(c,b,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('mdle,lkcd,eajk,cbmi->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,k)*t2_bbbb(c,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eajk,cbim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<d,m||l,e>_abab*l2_aaaa(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_abab(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('dmle,kldc,eaij,cbkm->abij', g_abab[v, o, o, v], l2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,d||e,l>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,j)*t2_abab(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,klcd,eaij,cbkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<d,m||l,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,a,i,j)*t2_bbbb(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmle,lkdc,eaij,cbkm->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,d||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,a,i,j)*t2_abab(c,b,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mdle,lkcd,eaij,cbmk->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,d||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_bbbb(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eaij,cbkm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<d,m||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_bbbb(a,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('dmej,kldc,eckl,abim->abij', g_abab[v, o, v, o], l2_aaaa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<d,m||e,j>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_bbbb(a,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('dmej,kldc,eckl,abim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,d||e,j>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_bbbb(a,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdej,klcd,cekl,abim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<d,m||e,j>_abab*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_bbbb(a,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('dmej,lkdc,eclk,abim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,d||e,j>_bbbb*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_bbbb(a,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdej,lkcd,celk,abim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,d||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_bbbb(a,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdej,kldc,eckl,abim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<d,m||e,j>_abab*l2_abab(l,k,d,c)*t2_abab(e,c,l,i)*t2_bbbb(a,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmej,lkdc,ecli,abkm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||e,j>_bbbb*l2_abab(l,k,c,d)*t2_abab(c,e,l,i)*t2_bbbb(a,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,lkcd,celi,abkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,i,l)*t2_bbbb(a,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,kldc,ecil,abkm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<d,m||e,j>_abab*l2_abab(k,l,d,c)*t2_abab(e,a,k,l)*t2_bbbb(c,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('dmej,kldc,eakl,cbim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,d||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,a,k,l)*t2_abab(c,b,m,i)
    contracted_intermediate = -0.500000000000000 * einsum('mdej,klcd,eakl,cbmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<d,m||e,j>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,k)*t2_bbbb(c,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('dmej,lkdc,ealk,cbim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,d||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,k)*t2_abab(c,b,m,i)
    contracted_intermediate = -0.500000000000000 * einsum('mdej,lkcd,ealk,cbmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,d||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,k,l)*t2_bbbb(c,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('mdej,kldc,eakl,cbim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<d,m||e,j>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,a,l,i)*t2_abab(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmej,kldc,eali,cbkm->abij', g_abab[v, o, v, o], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,j>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,l)*t2_abab(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,klcd,eail,cbkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,m||e,j>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,i)*t2_bbbb(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('dmej,lkdc,eali,cbkm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,i)*t2_abab(c,b,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mdej,lkcd,eali,cbmk->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,i,l)*t2_bbbb(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdej,kldc,eail,cbkm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,k,j)*t2_abab(c,b,m,i)
    contracted_intermediate =  1.000000000000000 * einsum('male,kldc,dekj,cbmi->abij', g_abab[o, v, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,l>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,j)*t2_bbbb(c,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,kldc,dekj,cbim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,j)*t2_abab(c,b,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('mael,klcd,edkj,cbmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,k)*t2_abab(c,b,m,i)
    contracted_intermediate =  1.000000000000000 * einsum('male,lkcd,edjk,cbmi->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,k)*t2_bbbb(c,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,kldc,edjk,cbim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,a||e,l>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,j)*t2_abab(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mael,klcd,edij,cbkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,a||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,i,j)*t2_abab(c,b,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('male,lkcd,edij,cbmk->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,a||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,i,j)*t2_bbbb(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,kldc,edij,cbkm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,l>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_abab(d,c,m,i)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebkj,dcmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_abab(c,d,m,i)
    contracted_intermediate =  0.500000000000000 * einsum('mael,klcd,ebkj,cdmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||l,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_abab(d,c,m,i)
    contracted_intermediate = -0.500000000000000 * einsum('male,lkdc,ebjk,dcmi->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_abab(c,d,m,i)
    contracted_intermediate = -0.500000000000000 * einsum('male,lkcd,ebjk,cdmi->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_bbbb(d,c,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebjk,dcim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,a||l,e>_abab*l2_aaaa(k,l,d,c)*t2_bbbb(e,b,i,j)*t2_aaaa(d,c,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('male,kldc,ebij,dckm->abij', g_abab[o, v, o, v], l2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,a||e,l>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,b,i,j)*t2_abab(d,c,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebij,dckm->abij', g_bbbb[o, v, v, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,a||e,l>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,b,i,j)*t2_abab(c,d,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,klcd,ebij,cdkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,a||l,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,b,i,j)*t2_abab(d,c,m,k)
    contracted_intermediate = -0.500000000000000 * einsum('male,lkdc,ebij,dcmk->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,a||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,b,i,j)*t2_abab(c,d,m,k)
    contracted_intermediate = -0.500000000000000 * einsum('male,lkcd,ebij,cdmk->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,a||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,i,j)*t2_bbbb(d,c,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebij,dckm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,b,m,i)
    contracted_intermediate = -0.500000000000000 * einsum('maej,kldc,edkl,cbmi->abij', g_abab[o, v, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(c,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,kldc,dekl,cbim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,b,m,i)
    contracted_intermediate =  0.500000000000000 * einsum('maej,klcd,edkl,cbmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(c,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,lkdc,delk,cbim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,b,m,i)
    contracted_intermediate =  0.500000000000000 * einsum('maej,lkcd,edlk,cbmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(c,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('maej,kldc,edkl,cbim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,i)*t2_abab(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('maej,kldc,deli,cbkm->abij', g_bbbb[o, v, v, o], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,l)*t2_abab(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,klcd,edil,cbkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,i)*t2_bbbb(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,lkdc,deli,cbkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,i)*t2_abab(c,b,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('maej,lkcd,edli,cbmk->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,i,l)*t2_bbbb(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('maej,kldc,edil,cbkm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_abab(d,c,m,i)
    contracted_intermediate = -0.250000000000000 * einsum('maej,kldc,ebkl,dcmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,k,l)*t2_abab(c,d,m,i)
    contracted_intermediate = -0.250000000000000 * einsum('maej,klcd,ebkl,cdmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_abab(d,c,m,i)
    contracted_intermediate = -0.250000000000000 * einsum('maej,lkdc,eblk,dcmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,b,l,k)*t2_abab(c,d,m,i)
    contracted_intermediate = -0.250000000000000 * einsum('maej,lkcd,eblk,cdmi->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_bbbb(d,c,i,m)
    contracted_intermediate =  0.250000000000000 * einsum('maej,kldc,ebkl,dcim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,b,l,i)*t2_aaaa(d,c,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('maej,kldc,ebli,dckm->abij', g_abab[o, v, v, o], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,b,i,l)*t2_abab(d,c,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,kldc,ebil,dckm->abij', g_bbbb[o, v, v, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,b,i,l)*t2_abab(c,d,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,klcd,ebil,cdkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_abab(l,k,d,c)*t2_abab(e,b,l,i)*t2_abab(d,c,m,k)
    contracted_intermediate =  0.500000000000000 * einsum('maej,lkdc,ebli,dcmk->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,b,l,i)*t2_abab(c,d,m,k)
    contracted_intermediate =  0.500000000000000 * einsum('maej,lkcd,ebli,cdmk->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,i,l)*t2_bbbb(d,c,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,kldc,ebil,dckm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 <d,c||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,a,k,l)*t2_bbbb(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('dcef,kldc,eakl,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <c,d||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,a,k,l)*t2_bbbb(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('cdef,klcd,eakl,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <d,c||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,k)*t2_bbbb(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('dcef,lkdc,ealk,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <c,d||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,k)*t2_bbbb(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('cdef,lkcd,ealk,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <d,c||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,k,l)*t2_bbbb(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('dcef,kldc,eakl,fbij->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 P(i,j)<d,c||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,a,l,j)*t2_abab(f,b,k,i)
    contracted_intermediate =  0.500000000000000 * einsum('dcef,kldc,ealj,fbki->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<d,c||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_abab(f,b,k,i)
    contracted_intermediate =  0.500000000000000 * einsum('dcfe,kldc,eajl,fbki->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<c,d||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,j,l)*t2_abab(f,b,k,i)
    contracted_intermediate =  0.500000000000000 * einsum('cdfe,klcd,eajl,fbki->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<d,c||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,j)*t2_bbbb(f,b,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('dcef,lkdc,ealj,fbik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<c,d||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,j)*t2_bbbb(f,b,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('cdef,lkcd,ealj,fbik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<d,c||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_bbbb(f,b,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('dcef,kldc,eajl,fbik->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 <d,c||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,k,l)
    doubles_res +=  0.250000000000000 * einsum('dcfe,kldc,eaij,fbkl->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <c,d||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,j)*t2_abab(f,b,k,l)
    doubles_res +=  0.250000000000000 * einsum('cdfe,klcd,eaij,fbkl->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <d,c||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('dcfe,lkdc,eaij,fblk->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <c,d||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,a,i,j)*t2_abab(f,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('cdfe,lkcd,eaij,fblk->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <d,c||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_bbbb(f,b,k,l)
    doubles_res += -0.250000000000000 * einsum('dcef,kldc,eaij,fbkl->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 P(a,b)<d,a||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_bbbb(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,eckl,fbij->abij', g_abab[v, v, v, v], l2_aaaa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<d,a||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_bbbb(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,eckl,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<d,a||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_bbbb(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,klcd,cekl,fbij->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<d,a||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_bbbb(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,lkdc,eclk,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<d,a||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_bbbb(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,lkcd,celk,fbij->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<d,a||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_bbbb(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,eckl,fbij->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,a||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(c,e,l,j)*t2_abab(f,b,k,i)
    contracted_intermediate = -1.000000000000000 * einsum('dafe,kldc,celj,fbki->abij', g_abab[v, v, v, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,a||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_abab(f,b,k,i)
    contracted_intermediate = -1.000000000000000 * einsum('dafe,kldc,ecjl,fbki->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,a||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,c,l,j)*t2_bbbb(f,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('daef,lkdc,eclj,fbik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,a||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(c,e,l,j)*t2_bbbb(f,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('daef,lkcd,celj,fbik->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<d,a||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_bbbb(f,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('daef,kldc,ecjl,fbik->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<d,a||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,c,i,j)*t2_abab(f,b,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('dafe,kldc,ecij,fbkl->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<d,a||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,c,i,j)*t2_abab(f,b,l,k)
    contracted_intermediate = -0.500000000000000 * einsum('dafe,lkdc,ecij,fblk->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<d,a||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,i,j)*t2_bbbb(f,b,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,ecij,fbkl->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 <a,b||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(f,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('abef,kldc,dekl,fcij->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,b||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(f,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('abef,lkdc,delk,fcij->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(f,c,i,j)
    doubles_res += -0.500000000000000 * einsum('abef,kldc,edkl,fcij->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 P(i,j)<a,b||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(c,f,k,i)
    contracted_intermediate =  0.500000000000000 * einsum('abef,kldc,delj,cfki->abij', g_bbbb[v, v, v, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<a,b||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(c,f,k,i)
    contracted_intermediate = -0.500000000000000 * einsum('abef,klcd,edjl,cfki->abij', g_bbbb[v, v, v, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<a,b||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,c,i,k)
    contracted_intermediate = -0.500000000000000 * einsum('abef,lkdc,delj,fcik->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<a,b||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(f,c,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('abef,kldc,edjl,fcik->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 <m,d||k,l>_aaaa*l2_aaaa(k,l,d,c)*t3_abbabb(c,a,b,m,j,i)
    doubles_res += -0.500000000000000 * einsum('mdkl,kldc,cabmji->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||k,l>_abab*l2_abab(k,l,d,c)*t3_bbbbbb(c,a,b,i,j,m)
    doubles_res += -0.500000000000000 * einsum('dmkl,kldc,cabijm->abij', g_abab[v, o, o, o], l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||k,l>_abab*l2_abab(k,l,c,d)*t3_abbabb(c,a,b,m,j,i)
    doubles_res +=  0.500000000000000 * einsum('mdkl,klcd,cabmji->abij', g_abab[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,k>_abab*l2_abab(l,k,d,c)*t3_bbbbbb(c,a,b,i,j,m)
    doubles_res += -0.500000000000000 * einsum('dmlk,lkdc,cabijm->abij', g_abab[v, o, o, o], l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||l,k>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,a,b,m,j,i)
    doubles_res +=  0.500000000000000 * einsum('mdlk,lkcd,cabmji->abij', g_abab[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||k,l>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(c,a,b,i,j,m)
    doubles_res +=  0.500000000000000 * einsum('mdkl,kldc,cabijm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(i,j)<d,m||l,j>_abab*l2_aaaa(k,l,d,c)*t3_abbabb(c,a,b,k,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmlj,kldc,cabkim->abij', g_abab[v, o, o, o], l2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||j,l>_bbbb*l2_abab(k,l,c,d)*t3_abbabb(c,a,b,k,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdjl,klcd,cabkim->abij', g_bbbb[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<d,m||l,j>_abab*l2_abab(l,k,d,c)*t3_bbbbbb(c,a,b,i,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('dmlj,lkdc,cabikm->abij', g_abab[v, o, o, o], l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||l,j>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,a,b,m,k,i)
    contracted_intermediate = -1.000000000000000 * einsum('mdlj,lkcd,cabmki->abij', g_abab[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,d||j,l>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(c,a,b,i,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdjl,kldc,cabikm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 <m,d||i,j>_bbbb*l2_abab(k,l,c,d)*t3_abbabb(c,a,b,k,l,m)
    doubles_res += -0.500000000000000 * einsum('mdij,klcd,cabklm->abij', g_bbbb[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,d||i,j>_bbbb*l2_abab(l,k,c,d)*t3_abbabb(c,a,b,l,k,m)
    doubles_res += -0.500000000000000 * einsum('mdij,lkcd,cablkm->abij', g_bbbb[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <m,d||i,j>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(c,a,b,k,l,m)
    doubles_res +=  0.500000000000000 * einsum('mdij,kldc,cabklm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 P(a,b)<m,a||k,l>_abab*l2_abab(k,l,d,c)*t3_abbabb(d,c,b,m,j,i)
    contracted_intermediate = -0.250000000000000 * einsum('makl,kldc,dcbmji->abij', g_abab[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,a||k,l>_abab*l2_abab(k,l,c,d)*t3_abbabb(c,d,b,m,j,i)
    contracted_intermediate = -0.250000000000000 * einsum('makl,klcd,cdbmji->abij', g_abab[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,a||l,k>_abab*l2_abab(l,k,d,c)*t3_abbabb(d,c,b,m,j,i)
    contracted_intermediate = -0.250000000000000 * einsum('malk,lkdc,dcbmji->abij', g_abab[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,a||l,k>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,d,b,m,j,i)
    contracted_intermediate = -0.250000000000000 * einsum('malk,lkcd,cdbmji->abij', g_abab[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||k,l>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,b,i,j,m)
    contracted_intermediate =  0.250000000000000 * einsum('makl,kldc,dcbijm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||l,j>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,b,m,k,i)
    contracted_intermediate = -0.500000000000000 * einsum('malj,kldc,dcbmki->abij', g_abab[o, v, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||j,l>_bbbb*l2_abab(k,l,d,c)*t3_abbabb(d,c,b,k,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('majl,kldc,dcbkim->abij', g_bbbb[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||j,l>_bbbb*l2_abab(k,l,c,d)*t3_abbabb(c,d,b,k,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('majl,klcd,cdbkim->abij', g_bbbb[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||l,j>_abab*l2_abab(l,k,d,c)*t3_abbabb(d,c,b,m,k,i)
    contracted_intermediate =  0.500000000000000 * einsum('malj,lkdc,dcbmki->abij', g_abab[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||l,j>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,d,b,m,k,i)
    contracted_intermediate =  0.500000000000000 * einsum('malj,lkcd,cdbmki->abij', g_abab[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||j,l>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,b,i,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('majl,kldc,dcbikm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_bbbb*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,b,k,l,m)
    contracted_intermediate =  0.250000000000000 * einsum('maij,kldc,dcbklm->abij', g_bbbb[o, v, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_bbbb*l2_abab(k,l,d,c)*t3_abbabb(d,c,b,k,l,m)
    contracted_intermediate =  0.250000000000000 * einsum('maij,kldc,dcbklm->abij', g_bbbb[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_bbbb*l2_abab(k,l,c,d)*t3_abbabb(c,d,b,k,l,m)
    contracted_intermediate =  0.250000000000000 * einsum('maij,klcd,cdbklm->abij', g_bbbb[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_bbbb*l2_abab(l,k,d,c)*t3_abbabb(d,c,b,l,k,m)
    contracted_intermediate =  0.250000000000000 * einsum('maij,lkdc,dcblkm->abij', g_bbbb[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_bbbb*l2_abab(l,k,c,d)*t3_abbabb(c,d,b,l,k,m)
    contracted_intermediate =  0.250000000000000 * einsum('maij,lkcd,cdblkm->abij', g_bbbb[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,b,k,l,m)
    contracted_intermediate =  0.250000000000000 * einsum('maij,kldc,dcbklm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 <d,c||e,l>_aaaa*l2_aaaa(k,l,d,c)*t3_abbabb(e,a,b,k,j,i)
    doubles_res += -0.500000000000000 * einsum('dcel,kldc,eabkji->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <d,c||e,l>_abab*l2_abab(k,l,d,c)*t3_abbabb(e,a,b,k,j,i)
    doubles_res += -0.500000000000000 * einsum('dcel,kldc,eabkji->abij', g_abab[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <c,d||e,l>_abab*l2_abab(k,l,c,d)*t3_abbabb(e,a,b,k,j,i)
    doubles_res += -0.500000000000000 * einsum('cdel,klcd,eabkji->abij', g_abab[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <d,c||l,e>_abab*l2_abab(l,k,d,c)*t3_bbbbbb(e,a,b,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('dcle,lkdc,eabijk->abij', g_abab[v, v, o, v], l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <c,d||l,e>_abab*l2_abab(l,k,c,d)*t3_bbbbbb(e,a,b,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('cdle,lkcd,eabijk->abij', g_abab[v, v, o, v], l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <d,c||e,l>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,a,b,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('dcel,kldc,eabijk->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 P(i,j)<d,c||e,j>_abab*l2_abab(k,l,d,c)*t3_abbabb(e,a,b,k,i,l)
    contracted_intermediate = -0.250000000000000 * einsum('dcej,kldc,eabkil->abij', g_abab[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<c,d||e,j>_abab*l2_abab(k,l,c,d)*t3_abbabb(e,a,b,k,i,l)
    contracted_intermediate = -0.250000000000000 * einsum('cdej,klcd,eabkil->abij', g_abab[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<d,c||e,j>_abab*l2_abab(l,k,d,c)*t3_abbabb(e,a,b,l,k,i)
    contracted_intermediate =  0.250000000000000 * einsum('dcej,lkdc,eablki->abij', g_abab[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<c,d||e,j>_abab*l2_abab(l,k,c,d)*t3_abbabb(e,a,b,l,k,i)
    contracted_intermediate =  0.250000000000000 * einsum('cdej,lkcd,eablki->abij', g_abab[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<d,c||e,j>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,a,b,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('dcej,kldc,eabikl->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<d,a||l,e>_abab*l2_aaaa(k,l,d,c)*t3_abbabb(c,e,b,k,j,i)
    contracted_intermediate =  1.000000000000000 * einsum('dale,kldc,cebkji->abij', g_abab[v, v, o, v], l2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<d,a||e,l>_abab*l2_abab(k,l,d,c)*t3_abbabb(e,c,b,k,j,i)
    contracted_intermediate =  1.000000000000000 * einsum('dael,kldc,ecbkji->abij', g_abab[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<d,a||e,l>_bbbb*l2_abab(k,l,c,d)*t3_abbabb(c,e,b,k,j,i)
    contracted_intermediate =  1.000000000000000 * einsum('dael,klcd,cebkji->abij', g_bbbb[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<d,a||l,e>_abab*l2_abab(l,k,d,c)*t3_bbbbbb(e,c,b,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('dale,lkdc,ecbijk->abij', g_abab[v, v, o, v], l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<d,a||e,l>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,c,b,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('dael,kldc,ecbijk->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<d,a||e,j>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(e,c,b,l,k,i)
    contracted_intermediate =  0.500000000000000 * einsum('daej,kldc,ecblki->abij', g_abab[v, v, v, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<d,a||e,j>_abab*l2_abab(k,l,d,c)*t3_abbabb(e,c,b,k,i,l)
    contracted_intermediate =  0.500000000000000 * einsum('daej,kldc,ecbkil->abij', g_abab[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<d,a||e,j>_bbbb*l2_abab(k,l,c,d)*t3_abbabb(c,e,b,k,i,l)
    contracted_intermediate =  0.500000000000000 * einsum('daej,klcd,cebkil->abij', g_bbbb[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<d,a||e,j>_abab*l2_abab(l,k,d,c)*t3_abbabb(e,c,b,l,k,i)
    contracted_intermediate = -0.500000000000000 * einsum('daej,lkdc,ecblki->abij', g_abab[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<d,a||e,j>_bbbb*l2_abab(l,k,c,d)*t3_abbabb(c,e,b,l,k,i)
    contracted_intermediate = -0.500000000000000 * einsum('daej,lkcd,ceblki->abij', g_bbbb[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<d,a||e,j>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,c,b,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('daej,kldc,ecbikl->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 <a,b||e,l>_bbbb*l2_abab(k,l,d,c)*t3_abbabb(d,e,c,k,j,i)
    doubles_res +=  0.500000000000000 * einsum('abel,kldc,deckji->abij', g_bbbb[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,l>_bbbb*l2_abab(k,l,c,d)*t3_abbabb(c,d,e,k,j,i)
    doubles_res += -0.500000000000000 * einsum('abel,klcd,cdekji->abij', g_bbbb[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,b||e,l>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,d,c,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('abel,kldc,edcijk->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 P(i,j)<a,b||e,j>_bbbb*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,l,k,i)
    contracted_intermediate =  0.250000000000000 * einsum('abej,kldc,cdelki->abij', g_bbbb[v, v, v, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<a,b||e,j>_bbbb*l2_abab(k,l,d,c)*t3_abbabb(d,e,c,k,i,l)
    contracted_intermediate =  0.250000000000000 * einsum('abej,kldc,deckil->abij', g_bbbb[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<a,b||e,j>_bbbb*l2_abab(k,l,c,d)*t3_abbabb(c,d,e,k,i,l)
    contracted_intermediate = -0.250000000000000 * einsum('abej,klcd,cdekil->abij', g_bbbb[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<a,b||e,j>_bbbb*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,l,k,i)
    contracted_intermediate = -0.250000000000000 * einsum('abej,lkdc,declki->abij', g_bbbb[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<a,b||e,j>_bbbb*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,l,k,i)
    contracted_intermediate =  0.250000000000000 * einsum('abej,lkcd,cdelki->abij', g_bbbb[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<a,b||e,j>_bbbb*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,d,c,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('abej,kldc,edcikl->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    return doubles_res



def ccsdt_t2_abab_residual_Qf(t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa,t3_bbbbbb,t3_aabaab,t3_abbabb,  g_aaaa, g_bbbb, g_abab, l2_aaaa, l2_bbbb, l2_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    print('inside modify T2 thru T4')
    o = oa
    v = va
    
    #	 -0.2500 <m,n||k,l>_abab*l2_abab(k,l,d,c)*t2_abab(d,c,m,j)*t2_abab(a,b,i,n)
    doubles_res = -0.250000000000000 * einsum('mnkl,kldc,dcmj,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||k,l>_abab*l2_abab(k,l,c,d)*t2_abab(c,d,m,j)*t2_abab(a,b,i,n)
    doubles_res += -0.250000000000000 * einsum('mnkl,klcd,cdmj,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||l,k>_abab*l2_abab(l,k,d,c)*t2_abab(d,c,m,j)*t2_abab(a,b,i,n)
    doubles_res += -0.250000000000000 * einsum('mnlk,lkdc,dcmj,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||l,k>_abab*l2_abab(l,k,c,d)*t2_abab(c,d,m,j)*t2_abab(a,b,i,n)
    doubles_res += -0.250000000000000 * einsum('mnlk,lkcd,cdmj,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||k,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,j,m)*t2_abab(a,b,i,n)
    doubles_res += -0.250000000000000 * einsum('nmkl,kldc,dcjm,abin->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||k,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,i,m)*t2_abab(a,b,n,j)
    doubles_res += -0.250000000000000 * einsum('nmkl,kldc,dcim,abnj->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||k,l>_abab*l2_abab(k,l,d,c)*t2_abab(d,c,i,m)*t2_abab(a,b,n,j)
    doubles_res += -0.250000000000000 * einsum('nmkl,kldc,dcim,abnj->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||k,l>_abab*l2_abab(k,l,c,d)*t2_abab(c,d,i,m)*t2_abab(a,b,n,j)
    doubles_res += -0.250000000000000 * einsum('nmkl,klcd,cdim,abnj->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||l,k>_abab*l2_abab(l,k,d,c)*t2_abab(d,c,i,m)*t2_abab(a,b,n,j)
    doubles_res += -0.250000000000000 * einsum('nmlk,lkdc,dcim,abnj->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||l,k>_abab*l2_abab(l,k,c,d)*t2_abab(c,d,i,m)*t2_abab(a,b,n,j)
    doubles_res += -0.250000000000000 * einsum('nmlk,lkcd,cdim,abnj->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||k,l>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,m,j)*t2_abab(c,b,i,n)
    doubles_res +=  0.500000000000000 * einsum('mnkl,klcd,admj,cbin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||l,k>_abab*l2_abab(l,k,c,d)*t2_abab(a,d,m,j)*t2_abab(c,b,i,n)
    doubles_res +=  0.500000000000000 * einsum('mnlk,lkcd,admj,cbin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||k,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,i,m)*t2_abab(c,b,n,j)
    doubles_res +=  0.500000000000000 * einsum('nmkl,kldc,daim,cbnj->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||k,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(d,a,i,m)*t2_bbbb(c,b,j,n)
    doubles_res +=  0.500000000000000 * einsum('mnkl,kldc,daim,cbjn->abij', g_abab[o, o, o, o], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||k,l>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,i,m)*t2_abab(c,b,n,j)
    doubles_res +=  0.500000000000000 * einsum('nmkl,klcd,adim,cbnj->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||l,k>_abab*l2_abab(l,k,d,c)*t2_aaaa(d,a,i,m)*t2_bbbb(c,b,j,n)
    doubles_res +=  0.500000000000000 * einsum('mnlk,lkdc,daim,cbjn->abij', g_abab[o, o, o, o], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||l,k>_abab*l2_abab(l,k,c,d)*t2_abab(a,d,i,m)*t2_abab(c,b,n,j)
    doubles_res +=  0.500000000000000 * einsum('nmlk,lkcd,adim,cbnj->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||k,l>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,d,i,m)*t2_bbbb(c,b,j,n)
    doubles_res +=  0.500000000000000 * einsum('nmkl,kldc,adim,cbjn->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||l,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_abab(a,b,i,n)
    doubles_res +=  0.500000000000000 * einsum('mnlj,kldc,dckm,abin->abij', g_abab[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_abab(a,b,i,n)
    doubles_res +=  0.500000000000000 * einsum('nmjl,kldc,dckm,abin->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_abab(a,b,i,n)
    doubles_res +=  0.500000000000000 * einsum('nmjl,klcd,cdkm,abin->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||l,j>_abab*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_abab(a,b,i,n)
    doubles_res +=  0.500000000000000 * einsum('mnlj,lkdc,dcmk,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||l,j>_abab*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_abab(a,b,i,n)
    doubles_res +=  0.500000000000000 * einsum('mnlj,lkcd,cdmk,abin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_abab(a,b,i,n)
    doubles_res +=  0.500000000000000 * einsum('nmjl,kldc,dckm,abin->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_abab(a,b,n,j)
    doubles_res +=  0.500000000000000 * einsum('nmil,kldc,dckm,abnj->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,l>_abab*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_abab(a,b,n,j)
    doubles_res +=  0.500000000000000 * einsum('nmil,kldc,dckm,abnj->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,l>_abab*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_abab(a,b,n,j)
    doubles_res +=  0.500000000000000 * einsum('nmil,klcd,cdkm,abnj->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_abab(a,b,n,j)
    doubles_res +=  0.500000000000000 * einsum('nmil,lkdc,dcmk,abnj->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_abab(a,b,n,j)
    doubles_res +=  0.500000000000000 * einsum('nmil,lkcd,cdmk,abnj->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,l>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_abab(a,b,n,j)
    doubles_res +=  0.500000000000000 * einsum('nmil,kldc,dckm,abnj->abij', g_abab[o, o, o, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||l,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,i,m)*t2_abab(a,b,k,n)
    doubles_res += -0.500000000000000 * einsum('mnlj,kldc,dcim,abkn->abij', g_abab[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,c,i,m)*t2_abab(a,b,k,n)
    doubles_res += -0.500000000000000 * einsum('nmjl,kldc,dcim,abkn->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,d,i,m)*t2_abab(a,b,k,n)
    doubles_res += -0.500000000000000 * einsum('nmjl,klcd,cdim,abkn->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||l,j>_abab*l2_abab(l,k,d,c)*t2_abab(d,c,i,m)*t2_abab(a,b,n,k)
    doubles_res +=  0.500000000000000 * einsum('nmlj,lkdc,dcim,abnk->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||l,j>_abab*l2_abab(l,k,c,d)*t2_abab(c,d,i,m)*t2_abab(a,b,n,k)
    doubles_res +=  0.500000000000000 * einsum('nmlj,lkcd,cdim,abnk->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||i,l>_abab*l2_abab(k,l,d,c)*t2_abab(d,c,m,j)*t2_abab(a,b,k,n)
    doubles_res +=  0.500000000000000 * einsum('mnil,kldc,dcmj,abkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||i,l>_abab*l2_abab(k,l,c,d)*t2_abab(c,d,m,j)*t2_abab(a,b,k,n)
    doubles_res +=  0.500000000000000 * einsum('mnil,klcd,cdmj,abkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||i,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(d,c,m,j)*t2_abab(a,b,n,k)
    doubles_res += -0.500000000000000 * einsum('nmil,lkdc,dcmj,abnk->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||i,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(c,d,m,j)*t2_abab(a,b,n,k)
    doubles_res += -0.500000000000000 * einsum('nmil,lkcd,cdmj,abnk->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||i,l>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,j,m)*t2_abab(a,b,n,k)
    doubles_res += -0.500000000000000 * einsum('nmil,kldc,dcjm,abnk->abij', g_abab[o, o, o, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||l,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,k,m)*t2_abab(c,b,i,n)
    doubles_res += -1.000000000000000 * einsum('mnlj,kldc,dakm,cbin->abij', g_abab[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||j,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,d,k,m)*t2_abab(c,b,i,n)
    doubles_res += -1.000000000000000 * einsum('nmjl,klcd,adkm,cbin->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||l,j>_abab*l2_abab(l,k,c,d)*t2_abab(a,d,m,k)*t2_abab(c,b,i,n)
    doubles_res += -1.000000000000000 * einsum('mnlj,lkcd,admk,cbin->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||i,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,k,m)*t2_abab(c,b,n,j)
    doubles_res += -1.000000000000000 * einsum('nmil,kldc,dakm,cbnj->abij', g_aaaa[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||i,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(d,a,k,m)*t2_bbbb(c,b,j,n)
    doubles_res += -1.000000000000000 * einsum('mnil,kldc,dakm,cbjn->abij', g_abab[o, o, o, o], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||i,l>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,k,m)*t2_abab(c,b,n,j)
    doubles_res += -1.000000000000000 * einsum('nmil,klcd,adkm,cbnj->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||i,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(a,d,m,k)*t2_abab(c,b,n,j)
    doubles_res += -1.000000000000000 * einsum('nmil,lkcd,admk,cbnj->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||i,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,d,m,k)*t2_bbbb(c,b,j,n)
    doubles_res += -1.000000000000000 * einsum('mnil,kldc,admk,cbjn->abij', g_abab[o, o, o, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||l,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,i,m)*t2_abab(c,b,k,n)
    doubles_res +=  1.000000000000000 * einsum('mnlj,kldc,daim,cbkn->abij', g_abab[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||j,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,d,i,m)*t2_abab(c,b,k,n)
    doubles_res +=  1.000000000000000 * einsum('nmjl,klcd,adim,cbkn->abij', g_bbbb[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||l,j>_abab*l2_abab(l,k,d,c)*t2_aaaa(d,a,i,m)*t2_bbbb(c,b,k,n)
    doubles_res += -1.000000000000000 * einsum('mnlj,lkdc,daim,cbkn->abij', g_abab[o, o, o, o], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||l,j>_abab*l2_abab(l,k,c,d)*t2_abab(a,d,i,m)*t2_abab(c,b,n,k)
    doubles_res += -1.000000000000000 * einsum('nmlj,lkcd,adim,cbnk->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||j,l>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,d,i,m)*t2_bbbb(c,b,k,n)
    doubles_res += -1.000000000000000 * einsum('nmjl,kldc,adim,cbkn->abij', g_bbbb[o, o, o, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||i,l>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,m,j)*t2_abab(c,b,k,n)
    doubles_res += -1.000000000000000 * einsum('mnil,klcd,admj,cbkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||i,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(a,d,m,j)*t2_abab(c,b,n,k)
    doubles_res +=  1.000000000000000 * einsum('nmil,lkcd,admj,cbnk->abij', g_aaaa[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||i,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,d,m,j)*t2_bbbb(c,b,k,n)
    doubles_res +=  1.000000000000000 * einsum('mnil,kldc,admj,cbkn->abij', g_abab[o, o, o, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||i,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,l,m)*t2_abab(a,b,k,n)
    doubles_res +=  0.500000000000000 * einsum('mnij,kldc,dclm,abkn->abij', g_abab[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||i,j>_abab*l2_abab(k,l,d,c)*t2_abab(d,c,m,l)*t2_abab(a,b,k,n)
    doubles_res += -0.500000000000000 * einsum('mnij,kldc,dcml,abkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||i,j>_abab*l2_abab(k,l,c,d)*t2_abab(c,d,m,l)*t2_abab(a,b,k,n)
    doubles_res += -0.500000000000000 * einsum('mnij,klcd,cdml,abkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||i,j>_abab*l2_abab(l,k,d,c)*t2_abab(d,c,l,m)*t2_abab(a,b,n,k)
    doubles_res += -0.500000000000000 * einsum('nmij,lkdc,dclm,abnk->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||i,j>_abab*l2_abab(l,k,c,d)*t2_abab(c,d,l,m)*t2_abab(a,b,n,k)
    doubles_res += -0.500000000000000 * einsum('nmij,lkcd,cdlm,abnk->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,j>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,l,m)*t2_abab(a,b,n,k)
    doubles_res +=  0.500000000000000 * einsum('nmij,kldc,dclm,abnk->abij', g_abab[o, o, o, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||i,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,l,m)*t2_abab(c,b,k,n)
    doubles_res += -0.500000000000000 * einsum('mnij,kldc,dalm,cbkn->abij', g_abab[o, o, o, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||i,j>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,m,l)*t2_abab(c,b,k,n)
    doubles_res +=  0.500000000000000 * einsum('mnij,klcd,adml,cbkn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||i,j>_abab*l2_abab(l,k,d,c)*t2_aaaa(d,a,l,m)*t2_bbbb(c,b,k,n)
    doubles_res +=  0.500000000000000 * einsum('mnij,lkdc,dalm,cbkn->abij', g_abab[o, o, o, o], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,j>_abab*l2_abab(l,k,c,d)*t2_abab(a,d,l,m)*t2_abab(c,b,n,k)
    doubles_res +=  0.500000000000000 * einsum('nmij,lkcd,adlm,cbnk->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||i,j>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,d,m,l)*t2_bbbb(c,b,k,n)
    doubles_res += -0.500000000000000 * einsum('mnij,kldc,adml,cbkn->abij', g_abab[o, o, o, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||i,j>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,b,l,m)*t2_aaaa(c,a,k,n)
    doubles_res += -0.500000000000000 * einsum('nmij,kldc,dblm,cakn->abij', g_abab[o, o, o, o], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||i,j>_abab*l2_abab(k,l,d,c)*t2_abab(d,b,m,l)*t2_abab(a,c,k,n)
    doubles_res +=  0.500000000000000 * einsum('mnij,kldc,dbml,ackn->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,j>_abab*l2_abab(k,l,c,d)*t2_bbbb(d,b,l,m)*t2_aaaa(c,a,k,n)
    doubles_res +=  0.500000000000000 * einsum('nmij,klcd,dblm,cakn->abij', g_abab[o, o, o, o], l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||i,j>_abab*l2_abab(l,k,d,c)*t2_abab(d,b,l,m)*t2_abab(a,c,n,k)
    doubles_res +=  0.500000000000000 * einsum('nmij,lkdc,dblm,acnk->abij', g_abab[o, o, o, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||i,j>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,l,m)*t2_abab(a,c,n,k)
    doubles_res += -0.500000000000000 * einsum('nmij,kldc,dblm,acnk->abij', g_abab[o, o, o, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <d,m||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(c,e,k,j)*t2_abab(a,b,i,m)
    doubles_res +=  1.000000000000000 * einsum('dmle,kldc,cekj,abim->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||e,l>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,k,j)*t2_abab(a,b,i,m)
    doubles_res +=  1.000000000000000 * einsum('dmel,kldc,eckj,abim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,k,j)*t2_abab(a,b,i,m)
    doubles_res += -1.000000000000000 * einsum('mdel,klcd,cekj,abim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||l,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,c,j,k)*t2_abab(a,b,i,m)
    doubles_res += -1.000000000000000 * einsum('dmle,lkdc,ecjk,abim->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,k)*t2_abab(a,b,i,m)
    doubles_res +=  1.000000000000000 * einsum('mdel,kldc,ecjk,abim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,k)*t2_abab(a,b,m,j)
    doubles_res +=  1.000000000000000 * einsum('mdel,kldc,ecik,abmj->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,c,i,k)*t2_abab(a,b,m,j)
    doubles_res += -1.000000000000000 * einsum('mdel,klcd,ecik,abmj->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,i,k)*t2_abab(a,b,m,j)
    doubles_res += -1.000000000000000 * einsum('mdel,lkdc,ecik,abmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||l,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,i,k)*t2_abab(a,b,m,j)
    doubles_res +=  1.000000000000000 * einsum('mdle,lkcd,ceik,abmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,k)*t2_abab(a,b,m,j)
    doubles_res +=  1.000000000000000 * einsum('mdel,kldc,ecik,abmj->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(c,e,i,j)*t2_abab(a,b,k,m)
    doubles_res += -1.000000000000000 * einsum('dmle,kldc,ceij,abkm->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||e,l>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(a,b,k,m)
    doubles_res += -1.000000000000000 * einsum('dmel,kldc,ecij,abkm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,i,j)*t2_abab(a,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdel,klcd,ceij,abkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,i,j)*t2_abab(a,b,m,k)
    doubles_res +=  1.000000000000000 * einsum('mdel,lkdc,ecij,abmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||l,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,i,j)*t2_abab(a,b,m,k)
    doubles_res += -1.000000000000000 * einsum('mdle,lkcd,ceij,abmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(a,b,m,k)
    doubles_res += -1.000000000000000 * einsum('mdel,kldc,ecij,abmk->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,k,j)*t2_abab(c,b,i,m)
    doubles_res += -1.000000000000000 * einsum('dmle,kldc,aekj,cbim->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,k,j)*t2_abab(c,b,i,m)
    doubles_res +=  1.000000000000000 * einsum('mdel,klcd,aekj,cbim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(c,a,i,m)
    doubles_res += -1.000000000000000 * einsum('mdel,kldc,ebkj,caim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||e,l>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_abab(a,c,i,m)
    doubles_res += -1.000000000000000 * einsum('dmel,kldc,ebkj,acim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_aaaa(c,a,i,m)
    doubles_res +=  1.000000000000000 * einsum('mdel,klcd,ebkj,caim->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||l,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_abab(a,c,i,m)
    doubles_res +=  1.000000000000000 * einsum('dmle,lkdc,ebjk,acim->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_aaaa(c,a,i,m)
    doubles_res += -1.000000000000000 * einsum('mdle,lkcd,ebjk,caim->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_abab(a,c,i,m)
    doubles_res += -1.000000000000000 * einsum('mdel,kldc,ebjk,acim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,k)*t2_abab(c,b,m,j)
    doubles_res += -1.000000000000000 * einsum('mdel,kldc,eaik,cbmj->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,k)*t2_bbbb(c,b,j,m)
    doubles_res += -1.000000000000000 * einsum('dmel,kldc,eaik,cbjm->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,k)*t2_abab(c,b,m,j)
    doubles_res +=  1.000000000000000 * einsum('mdel,klcd,eaik,cbmj->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,k)*t2_bbbb(c,b,j,m)
    doubles_res +=  1.000000000000000 * einsum('dmle,lkdc,aeik,cbjm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||l,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,k)*t2_abab(c,b,m,j)
    doubles_res += -1.000000000000000 * einsum('mdle,lkcd,aeik,cbmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,k)*t2_bbbb(c,b,j,m)
    doubles_res += -1.000000000000000 * einsum('mdel,kldc,aeik,cbjm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,b,i,k)*t2_abab(a,c,m,j)
    doubles_res +=  1.000000000000000 * einsum('mdel,lkdc,ebik,acmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,k)*t2_abab(a,c,m,j)
    doubles_res += -1.000000000000000 * einsum('mdel,kldc,ebik,acmj->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,i,j)*t2_abab(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('dmle,kldc,aeij,cbkm->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,i,j)*t2_abab(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('mdel,klcd,aeij,cbkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <d,m||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,j)*t2_bbbb(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('dmle,lkdc,aeij,cbkm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,d||l,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,j)*t2_abab(c,b,m,k)
    doubles_res +=  1.000000000000000 * einsum('mdle,lkcd,aeij,cbmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,j)*t2_bbbb(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdel,kldc,aeij,cbkm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,i,j)*t2_aaaa(c,a,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdel,kldc,ebij,cakm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <d,m||e,l>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,i,j)*t2_abab(a,c,k,m)
    doubles_res +=  1.000000000000000 * einsum('dmel,kldc,ebij,ackm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,i,j)*t2_aaaa(c,a,k,m)
    doubles_res += -1.000000000000000 * einsum('mdel,klcd,ebij,cakm->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,b,i,j)*t2_abab(a,c,m,k)
    doubles_res += -1.000000000000000 * einsum('mdel,lkdc,ebij,acmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,j)*t2_abab(a,c,m,k)
    doubles_res +=  1.000000000000000 * einsum('mdel,kldc,ebij,acmk->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <d,m||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(a,b,i,m)
    doubles_res += -0.500000000000000 * einsum('dmej,kldc,eckl,abim->abij', g_abab[v, o, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <d,m||e,j>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(a,b,i,m)
    doubles_res += -0.500000000000000 * einsum('dmej,kldc,eckl,abim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,d||e,j>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(a,b,i,m)
    doubles_res +=  0.500000000000000 * einsum('mdej,klcd,cekl,abim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <d,m||e,j>_abab*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(a,b,i,m)
    doubles_res += -0.500000000000000 * einsum('dmej,lkdc,eclk,abim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,d||e,j>_bbbb*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(a,b,i,m)
    doubles_res +=  0.500000000000000 * einsum('mdej,lkcd,celk,abim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,d||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(a,b,i,m)
    doubles_res +=  0.500000000000000 * einsum('mdej,kldc,eckl,abim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,d||e,i>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(a,b,m,j)
    doubles_res +=  0.500000000000000 * einsum('mdei,kldc,eckl,abmj->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,d||e,i>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(a,b,m,j)
    doubles_res +=  0.500000000000000 * einsum('mdei,kldc,eckl,abmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||i,e>_abab*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(a,b,m,j)
    doubles_res += -0.500000000000000 * einsum('mdie,klcd,cekl,abmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,d||e,i>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(a,b,m,j)
    doubles_res +=  0.500000000000000 * einsum('mdei,lkdc,eclk,abmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||i,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(a,b,m,j)
    doubles_res += -0.500000000000000 * einsum('mdie,lkcd,celk,abmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(a,b,m,j)
    doubles_res += -0.500000000000000 * einsum('mdie,kldc,eckl,abmj->abij', g_abab[o, v, o, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,l)*t2_abab(a,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('dmej,kldc,ecil,abkm->abij', g_abab[v, o, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||e,j>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,i,l)*t2_abab(a,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('dmej,kldc,ecil,abkm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,i,l)*t2_abab(a,b,k,m)
    doubles_res += -1.000000000000000 * einsum('mdej,klcd,ceil,abkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,c,i,l)*t2_abab(a,b,m,k)
    doubles_res +=  1.000000000000000 * einsum('mdej,lkcd,ecil,abmk->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,l)*t2_abab(a,b,m,k)
    doubles_res +=  1.000000000000000 * einsum('mdej,kldc,ecil,abmk->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||i,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(c,e,l,j)*t2_abab(a,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('dmie,kldc,celj,abkm->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||i,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_abab(a,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('dmie,kldc,ecjl,abkm->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,i>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,l,j)*t2_abab(a,b,m,k)
    doubles_res += -1.000000000000000 * einsum('mdei,lkdc,eclj,abmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||i,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,l,j)*t2_abab(a,b,m,k)
    doubles_res +=  1.000000000000000 * einsum('mdie,lkcd,celj,abmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_abab(a,b,m,k)
    doubles_res +=  1.000000000000000 * einsum('mdie,kldc,ecjl,abmk->abij', g_abab[o, v, o, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_abab(c,b,i,m)
    doubles_res +=  0.500000000000000 * einsum('dmej,kldc,eakl,cbim->abij', g_abab[v, o, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(c,b,i,m)
    doubles_res += -0.500000000000000 * einsum('mdej,klcd,aekl,cbim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(c,b,i,m)
    doubles_res += -0.500000000000000 * einsum('mdej,lkcd,aelk,cbim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_abab(a,c,i,m)
    doubles_res +=  0.500000000000000 * einsum('dmej,kldc,ebkl,acim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,k,l)*t2_aaaa(c,a,i,m)
    doubles_res += -0.500000000000000 * einsum('mdej,klcd,ebkl,caim->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_abab(a,c,i,m)
    doubles_res +=  0.500000000000000 * einsum('dmej,lkdc,eblk,acim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,b,l,k)*t2_aaaa(c,a,i,m)
    doubles_res += -0.500000000000000 * einsum('mdej,lkcd,eblk,caim->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_abab(a,c,i,m)
    doubles_res += -0.500000000000000 * einsum('mdej,kldc,ebkl,acim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||e,i>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_abab(c,b,m,j)
    doubles_res += -0.500000000000000 * einsum('mdei,kldc,eakl,cbmj->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <d,m||i,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_bbbb(c,b,j,m)
    doubles_res += -0.500000000000000 * einsum('dmie,kldc,aekl,cbjm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,d||i,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(c,b,m,j)
    doubles_res +=  0.500000000000000 * einsum('mdie,klcd,aekl,cbmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <d,m||i,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_bbbb(c,b,j,m)
    doubles_res += -0.500000000000000 * einsum('dmie,lkdc,aelk,cbjm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,d||i,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(c,b,m,j)
    doubles_res +=  0.500000000000000 * einsum('mdie,lkcd,aelk,cbmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||e,i>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_abab(a,c,m,j)
    doubles_res += -0.500000000000000 * einsum('mdei,kldc,ebkl,acmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,d||e,i>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_abab(a,c,m,j)
    doubles_res += -0.500000000000000 * einsum('mdei,lkdc,eblk,acmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,d||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_abab(a,c,m,j)
    doubles_res +=  0.500000000000000 * einsum('mdie,kldc,ebkl,acmj->abij', g_abab[o, v, o, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,l)*t2_abab(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('dmej,kldc,eail,cbkm->abij', g_abab[v, o, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_abab(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdej,klcd,aeil,cbkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||e,j>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_bbbb(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('dmej,lkdc,eail,cbkm->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_abab(c,b,m,k)
    doubles_res += -1.000000000000000 * einsum('mdej,lkcd,eail,cbmk->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_bbbb(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('mdej,kldc,aeil,cbkm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||e,j>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,i,l)*t2_abab(a,c,k,m)
    doubles_res += -1.000000000000000 * einsum('dmej,kldc,ebil,ackm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,i,l)*t2_aaaa(c,a,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdej,klcd,ebil,cakm->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,l)*t2_abab(a,c,m,k)
    doubles_res += -1.000000000000000 * einsum('mdej,kldc,ebil,acmk->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||i,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,l,j)*t2_abab(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('dmie,kldc,aelj,cbkm->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <d,m||i,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,j)*t2_bbbb(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('dmie,lkdc,aelj,cbkm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||i,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,j)*t2_abab(c,b,m,k)
    doubles_res += -1.000000000000000 * einsum('mdie,lkcd,aelj,cbmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||e,i>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,l,j)*t2_aaaa(c,a,k,m)
    doubles_res += -1.000000000000000 * einsum('mdei,kldc,eblj,cakm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,m||i,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_abab(a,c,k,m)
    doubles_res += -1.000000000000000 * einsum('dmie,kldc,ebjl,ackm->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||i,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,b,j,l)*t2_aaaa(c,a,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdie,klcd,ebjl,cakm->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,d||e,i>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,b,l,j)*t2_abab(a,c,m,k)
    doubles_res +=  1.000000000000000 * einsum('mdei,lkdc,eblj,acmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,d||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_abab(a,c,m,k)
    doubles_res += -1.000000000000000 * einsum('mdie,kldc,ebjl,acmk->abij', g_abab[o, v, o, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <a,m||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,k,j)*t2_abab(c,b,i,m)
    doubles_res +=  1.000000000000000 * einsum('amle,kldc,dekj,cbim->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,m||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,j)*t2_abab(c,b,i,m)
    doubles_res += -1.000000000000000 * einsum('amel,klcd,edkj,cbim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <a,m||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,k)*t2_abab(c,b,i,m)
    doubles_res +=  1.000000000000000 * einsum('amle,lkcd,edjk,cbim->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,b||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,k,j)*t2_aaaa(c,a,i,m)
    doubles_res +=  1.000000000000000 * einsum('mble,kldc,dekj,caim->abij', g_abab[o, v, o, v], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,j)*t2_abab(a,c,i,m)
    doubles_res +=  1.000000000000000 * einsum('mbel,kldc,dekj,acim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,b||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,j)*t2_aaaa(c,a,i,m)
    doubles_res += -1.000000000000000 * einsum('mbel,klcd,edkj,caim->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,b||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,k)*t2_aaaa(c,a,i,m)
    doubles_res +=  1.000000000000000 * einsum('mble,lkcd,edjk,caim->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,k)*t2_abab(a,c,i,m)
    doubles_res +=  1.000000000000000 * einsum('mbel,kldc,edjk,acim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,a||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,k)*t2_abab(c,b,m,j)
    doubles_res +=  1.000000000000000 * einsum('mael,kldc,edik,cbmj->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <a,m||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,k)*t2_bbbb(c,b,j,m)
    doubles_res +=  1.000000000000000 * einsum('amel,kldc,edik,cbjm->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,m||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,k)*t2_bbbb(c,b,j,m)
    doubles_res += -1.000000000000000 * einsum('amle,lkdc,deik,cbjm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,a||e,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,i,k)*t2_abab(c,b,m,j)
    doubles_res +=  1.000000000000000 * einsum('mael,lkcd,edik,cbmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <a,m||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,k)*t2_bbbb(c,b,j,m)
    doubles_res +=  1.000000000000000 * einsum('amel,kldc,edik,cbjm->abij', g_abab[v, o, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,k)*t2_abab(a,c,m,j)
    doubles_res +=  1.000000000000000 * einsum('mbel,kldc,edik,acmj->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,b||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,k)*t2_abab(a,c,m,j)
    doubles_res += -1.000000000000000 * einsum('mble,lkdc,deik,acmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,k)*t2_abab(a,c,m,j)
    doubles_res +=  1.000000000000000 * einsum('mbel,kldc,edik,acmj->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,m||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('amle,kldc,deij,cbkm->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <a,m||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_abab(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('amel,klcd,edij,cbkm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <a,m||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_bbbb(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('amle,lkdc,deij,cbkm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,a||e,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,i,j)*t2_abab(c,b,m,k)
    doubles_res += -1.000000000000000 * einsum('mael,lkcd,edij,cbmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,m||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,j)*t2_bbbb(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('amel,kldc,edij,cbkm->abij', g_abab[v, o, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,b||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,j)*t2_aaaa(c,a,k,m)
    doubles_res += -1.000000000000000 * einsum('mble,kldc,deij,cakm->abij', g_abab[o, v, o, v], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,b||e,l>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(a,c,k,m)
    doubles_res += -1.000000000000000 * einsum('mbel,kldc,deij,ackm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_aaaa(c,a,k,m)
    doubles_res +=  1.000000000000000 * einsum('mbel,klcd,edij,cakm->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,b||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_abab(a,c,m,k)
    doubles_res +=  1.000000000000000 * einsum('mble,lkdc,deij,acmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,b||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,j)*t2_abab(a,c,m,k)
    doubles_res += -1.000000000000000 * einsum('mbel,kldc,edij,acmk->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,a||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(d,c,i,m)
    doubles_res += -0.500000000000000 * einsum('mael,kldc,ebkj,dcim->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||e,l>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_abab(d,c,i,m)
    doubles_res +=  0.500000000000000 * einsum('amel,kldc,ebkj,dcim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_abab(c,d,i,m)
    doubles_res +=  0.500000000000000 * einsum('amel,klcd,ebkj,cdim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,m||l,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_abab(d,c,i,m)
    doubles_res += -0.500000000000000 * einsum('amle,lkdc,ebjk,dcim->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,m||l,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_abab(c,d,i,m)
    doubles_res += -0.500000000000000 * einsum('amle,lkcd,ebjk,cdim->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,k,j)*t2_aaaa(d,c,i,m)
    doubles_res +=  0.500000000000000 * einsum('mble,kldc,aekj,dcim->abij', g_abab[o, v, o, v], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,k,j)*t2_abab(d,c,i,m)
    doubles_res += -0.500000000000000 * einsum('mbel,kldc,aekj,dcim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,k,j)*t2_abab(c,d,i,m)
    doubles_res += -0.500000000000000 * einsum('mbel,klcd,aekj,cdim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,a||e,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,b,i,k)*t2_abab(d,c,m,j)
    doubles_res += -0.500000000000000 * einsum('mael,lkdc,ebik,dcmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,a||e,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,b,i,k)*t2_abab(c,d,m,j)
    doubles_res += -0.500000000000000 * einsum('mael,lkcd,ebik,cdmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,k)*t2_bbbb(d,c,j,m)
    doubles_res +=  0.500000000000000 * einsum('amel,kldc,ebik,dcjm->abij', g_abab[v, o, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,k)*t2_abab(d,c,m,j)
    doubles_res += -0.500000000000000 * einsum('mbel,kldc,eaik,dcmj->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,k)*t2_abab(c,d,m,j)
    doubles_res += -0.500000000000000 * einsum('mbel,klcd,eaik,cdmj->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,k)*t2_abab(d,c,m,j)
    doubles_res +=  0.500000000000000 * einsum('mble,lkdc,aeik,dcmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||l,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,k)*t2_abab(c,d,m,j)
    doubles_res +=  0.500000000000000 * einsum('mble,lkcd,aeik,cdmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,k)*t2_bbbb(d,c,j,m)
    doubles_res += -0.500000000000000 * einsum('mbel,kldc,aeik,dcjm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,a||e,l>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,i,j)*t2_aaaa(d,c,k,m)
    doubles_res +=  0.500000000000000 * einsum('mael,kldc,ebij,dckm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,m||e,l>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,i,j)*t2_abab(d,c,k,m)
    doubles_res += -0.500000000000000 * einsum('amel,kldc,ebij,dckm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,m||e,l>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,i,j)*t2_abab(c,d,k,m)
    doubles_res += -0.500000000000000 * einsum('amel,klcd,ebij,cdkm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,a||e,l>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,b,i,j)*t2_abab(d,c,m,k)
    doubles_res +=  0.500000000000000 * einsum('mael,lkdc,ebij,dcmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,a||e,l>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,b,i,j)*t2_abab(c,d,m,k)
    doubles_res +=  0.500000000000000 * einsum('mael,lkcd,ebij,cdmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,m||e,l>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,j)*t2_bbbb(d,c,k,m)
    doubles_res += -0.500000000000000 * einsum('amel,kldc,ebij,dckm->abij', g_abab[v, o, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||l,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,i,j)*t2_aaaa(d,c,k,m)
    doubles_res += -0.500000000000000 * einsum('mble,kldc,aeij,dckm->abij', g_abab[o, v, o, v], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,i,j)*t2_abab(d,c,k,m)
    doubles_res +=  0.500000000000000 * einsum('mbel,kldc,aeij,dckm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,i,j)*t2_abab(c,d,k,m)
    doubles_res +=  0.500000000000000 * einsum('mbel,klcd,aeij,cdkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||l,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,j)*t2_abab(d,c,m,k)
    doubles_res += -0.500000000000000 * einsum('mble,lkdc,aeij,dcmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||l,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,j)*t2_abab(c,d,m,k)
    doubles_res += -0.500000000000000 * einsum('mble,lkcd,aeij,cdmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,j)*t2_bbbb(d,c,k,m)
    doubles_res +=  0.500000000000000 * einsum('mbel,kldc,aeij,dckm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,m||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,b,i,m)
    doubles_res += -0.500000000000000 * einsum('amej,kldc,edkl,cbim->abij', g_abab[v, o, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,b,i,m)
    doubles_res +=  0.500000000000000 * einsum('amej,klcd,edkl,cbim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,b,i,m)
    doubles_res +=  0.500000000000000 * einsum('amej,lkcd,edlk,cbim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(c,a,i,m)
    doubles_res += -0.500000000000000 * einsum('mbej,kldc,edkl,caim->abij', g_abab[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(a,c,i,m)
    doubles_res += -0.500000000000000 * einsum('mbej,kldc,dekl,acim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(c,a,i,m)
    doubles_res +=  0.500000000000000 * einsum('mbej,klcd,edkl,caim->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(a,c,i,m)
    doubles_res += -0.500000000000000 * einsum('mbej,lkdc,delk,acim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(c,a,i,m)
    doubles_res +=  0.500000000000000 * einsum('mbej,lkcd,edlk,caim->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(a,c,i,m)
    doubles_res +=  0.500000000000000 * einsum('mbej,kldc,edkl,acim->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,a||e,i>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,b,m,j)
    doubles_res +=  0.500000000000000 * einsum('maei,kldc,edkl,cbmj->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||i,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(c,b,j,m)
    doubles_res +=  0.500000000000000 * einsum('amie,kldc,dekl,cbjm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,a||e,i>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,b,m,j)
    doubles_res += -0.500000000000000 * einsum('maei,klcd,edkl,cbmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||i,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(c,b,j,m)
    doubles_res +=  0.500000000000000 * einsum('amie,lkdc,delk,cbjm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,a||e,i>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,b,m,j)
    doubles_res += -0.500000000000000 * einsum('maei,lkcd,edlk,cbmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,m||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(c,b,j,m)
    doubles_res += -0.500000000000000 * einsum('amie,kldc,edkl,cbjm->abij', g_abab[v, o, o, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||i,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(a,c,m,j)
    doubles_res +=  0.500000000000000 * einsum('mbie,kldc,dekl,acmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||i,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(a,c,m,j)
    doubles_res +=  0.500000000000000 * einsum('mbie,lkdc,delk,acmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(a,c,m,j)
    doubles_res += -0.500000000000000 * einsum('mbie,kldc,edkl,acmj->abij', g_abab[o, v, o, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <a,m||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('amej,kldc,edil,cbkm->abij', g_abab[v, o, v, o], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <a,m||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('amej,klcd,edil,cbkm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <a,m||e,j>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('amej,lkdc,edil,cbkm->abij', g_abab[v, o, v, o], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <a,m||e,j>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('amej,kldc,edil,cbkm->abij', g_abab[v, o, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_aaaa(c,a,k,m)
    doubles_res +=  1.000000000000000 * einsum('mbej,kldc,edil,cakm->abij', g_abab[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(a,c,k,m)
    doubles_res +=  1.000000000000000 * einsum('mbej,kldc,deil,ackm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_aaaa(c,a,k,m)
    doubles_res += -1.000000000000000 * einsum('mbej,klcd,edil,cakm->abij', g_abab[o, v, v, o], l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_abab(a,c,m,k)
    doubles_res += -1.000000000000000 * einsum('mbej,lkdc,edil,acmk->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_abab(a,c,m,k)
    doubles_res +=  1.000000000000000 * einsum('mbej,kldc,edil,acmk->abij', g_abab[o, v, v, o], l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <a,m||i,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('amie,kldc,delj,cbkm->abij', g_abab[v, o, o, v], l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <a,m||i,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('amie,klcd,edjl,cbkm->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <a,m||i,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(c,b,k,m)
    doubles_res += -1.000000000000000 * einsum('amie,lkdc,delj,cbkm->abij', g_abab[v, o, o, v], l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,a||e,i>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(c,b,m,k)
    doubles_res +=  1.000000000000000 * einsum('maei,lkcd,edlj,cbmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <a,m||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(c,b,k,m)
    doubles_res +=  1.000000000000000 * einsum('amie,kldc,edjl,cbkm->abij', g_abab[v, o, o, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,b||i,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(c,a,k,m)
    doubles_res +=  1.000000000000000 * einsum('mbie,kldc,delj,cakm->abij', g_abab[o, v, o, v], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,b||i,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(c,a,k,m)
    doubles_res += -1.000000000000000 * einsum('mbie,klcd,edjl,cakm->abij', g_abab[o, v, o, v], l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,b||i,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(a,c,m,k)
    doubles_res += -1.000000000000000 * einsum('mbie,lkdc,delj,acmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,b||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(a,c,m,k)
    doubles_res +=  1.000000000000000 * einsum('mbie,kldc,edjl,acmk->abij', g_abab[o, v, o, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||e,j>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_abab(d,c,i,m)
    doubles_res += -0.250000000000000 * einsum('amej,kldc,ebkl,dcim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,k,l)*t2_abab(c,d,i,m)
    doubles_res += -0.250000000000000 * einsum('amej,klcd,ebkl,cdim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||e,j>_abab*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_abab(d,c,i,m)
    doubles_res += -0.250000000000000 * einsum('amej,lkdc,eblk,dcim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||e,j>_abab*l2_abab(l,k,c,d)*t2_abab(e,b,l,k)*t2_abab(c,d,i,m)
    doubles_res += -0.250000000000000 * einsum('amej,lkcd,eblk,cdim->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <m,b||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_aaaa(d,c,i,m)
    doubles_res += -0.250000000000000 * einsum('mbej,kldc,eakl,dcim->abij', g_abab[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(d,c,i,m)
    doubles_res +=  0.250000000000000 * einsum('mbej,kldc,aekl,dcim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(c,d,i,m)
    doubles_res +=  0.250000000000000 * einsum('mbej,klcd,aekl,cdim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_bbbb*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(d,c,i,m)
    doubles_res +=  0.250000000000000 * einsum('mbej,lkdc,aelk,dcim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(c,d,i,m)
    doubles_res +=  0.250000000000000 * einsum('mbej,lkcd,aelk,cdim->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,a||e,i>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_abab(d,c,m,j)
    doubles_res +=  0.250000000000000 * einsum('maei,kldc,ebkl,dcmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,a||e,i>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,b,k,l)*t2_abab(c,d,m,j)
    doubles_res +=  0.250000000000000 * einsum('maei,klcd,ebkl,cdmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,a||e,i>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_abab(d,c,m,j)
    doubles_res +=  0.250000000000000 * einsum('maei,lkdc,eblk,dcmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,a||e,i>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,b,l,k)*t2_abab(c,d,m,j)
    doubles_res +=  0.250000000000000 * einsum('maei,lkcd,eblk,cdmj->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_bbbb(d,c,j,m)
    doubles_res += -0.250000000000000 * einsum('amie,kldc,ebkl,dcjm->abij', g_abab[v, o, o, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <m,b||i,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(d,c,m,j)
    doubles_res += -0.250000000000000 * einsum('mbie,kldc,aekl,dcmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <m,b||i,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(c,d,m,j)
    doubles_res += -0.250000000000000 * einsum('mbie,klcd,aekl,cdmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <m,b||i,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(d,c,m,j)
    doubles_res += -0.250000000000000 * einsum('mbie,lkdc,aelk,dcmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.2500 <m,b||i,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(c,d,m,j)
    doubles_res += -0.250000000000000 * einsum('mbie,lkcd,aelk,cdmj->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <a,m||e,j>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,i,l)*t2_abab(d,c,k,m)
    doubles_res +=  0.500000000000000 * einsum('amej,kldc,ebil,dckm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||e,j>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,i,l)*t2_abab(c,d,k,m)
    doubles_res +=  0.500000000000000 * einsum('amej,klcd,ebil,cdkm->abij', g_abab[v, o, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||e,j>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,l)*t2_bbbb(d,c,k,m)
    doubles_res +=  0.500000000000000 * einsum('amej,kldc,ebil,dckm->abij', g_abab[v, o, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,l)*t2_aaaa(d,c,k,m)
    doubles_res +=  0.500000000000000 * einsum('mbej,kldc,eail,dckm->abij', g_abab[o, v, v, o], l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(d,c,k,m)
    doubles_res += -0.500000000000000 * einsum('mbej,kldc,aeil,dckm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_abab(c,d,k,m)
    doubles_res += -0.500000000000000 * einsum('mbej,klcd,aeil,cdkm->abij', g_bbbb[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_abab(d,c,m,k)
    doubles_res +=  0.500000000000000 * einsum('mbej,lkdc,eail,dcmk->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_abab(c,d,m,k)
    doubles_res +=  0.500000000000000 * einsum('mbej,lkcd,eail,cdmk->abij', g_abab[o, v, v, o], l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_bbbb(d,c,k,m)
    doubles_res += -0.500000000000000 * einsum('mbej,kldc,aeil,dckm->abij', g_bbbb[o, v, v, o], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,a||e,i>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,l,j)*t2_aaaa(d,c,k,m)
    doubles_res += -0.500000000000000 * einsum('maei,kldc,eblj,dckm->abij', g_aaaa[o, v, v, o], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||i,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_abab(d,c,k,m)
    doubles_res +=  0.500000000000000 * einsum('amie,kldc,ebjl,dckm->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||i,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,b,j,l)*t2_abab(c,d,k,m)
    doubles_res +=  0.500000000000000 * einsum('amie,klcd,ebjl,cdkm->abij', g_abab[v, o, o, v], l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,a||e,i>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,b,l,j)*t2_abab(d,c,m,k)
    doubles_res += -0.500000000000000 * einsum('maei,lkdc,eblj,dcmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,a||e,i>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,b,l,j)*t2_abab(c,d,m,k)
    doubles_res += -0.500000000000000 * einsum('maei,lkcd,eblj,cdmk->abij', g_aaaa[o, v, v, o], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,m||i,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_bbbb(d,c,k,m)
    doubles_res +=  0.500000000000000 * einsum('amie,kldc,ebjl,dckm->abij', g_abab[v, o, o, v], l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||i,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,l,j)*t2_aaaa(d,c,k,m)
    doubles_res +=  0.500000000000000 * einsum('mbie,kldc,aelj,dckm->abij', g_abab[o, v, o, v], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||i,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,j)*t2_abab(d,c,m,k)
    doubles_res +=  0.500000000000000 * einsum('mbie,lkdc,aelj,dcmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,b||i,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,j)*t2_abab(c,d,m,k)
    doubles_res +=  0.500000000000000 * einsum('mbie,lkcd,aelj,cdmk->abij', g_abab[o, v, o, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <d,c||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_abab(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('dcef,kldc,eakl,fbij->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <d,c||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('dcfe,kldc,aekl,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <c,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('cdfe,klcd,aekl,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <d,c||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('dcfe,lkdc,aelk,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <c,d||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(f,b,i,j)
    doubles_res += -0.250000000000000 * einsum('cdfe,lkcd,aelk,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <d,c||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,j)*t2_abab(f,b,i,k)
    doubles_res +=  0.500000000000000 * einsum('dcfe,lkdc,aelj,fbik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <c,d||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,j)*t2_abab(f,b,i,k)
    doubles_res +=  0.500000000000000 * einsum('cdfe,lkcd,aelj,fbik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <d,c||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,l)*t2_abab(f,b,k,j)
    doubles_res +=  0.500000000000000 * einsum('dcef,kldc,eail,fbkj->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <d,c||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(f,b,k,j)
    doubles_res +=  0.500000000000000 * einsum('dcfe,kldc,aeil,fbkj->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <c,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_abab(f,b,k,j)
    doubles_res +=  0.500000000000000 * einsum('cdfe,klcd,aeil,fbkj->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <d,c||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,j,k)
    doubles_res +=  0.500000000000000 * einsum('dcef,lkdc,eail,fbjk->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <c,d||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,j,k)
    doubles_res +=  0.500000000000000 * einsum('cdef,lkcd,eail,fbjk->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <d,c||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_bbbb(f,b,j,k)
    doubles_res +=  0.500000000000000 * einsum('dcef,kldc,aeil,fbjk->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <d,c||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,k,l)
    doubles_res += -0.250000000000000 * einsum('dcfe,kldc,aeij,fbkl->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <c,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,j)*t2_abab(f,b,k,l)
    doubles_res += -0.250000000000000 * einsum('cdfe,klcd,aeij,fbkl->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <d,c||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,l,k)
    doubles_res += -0.250000000000000 * einsum('dcfe,lkdc,aeij,fblk->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <c,d||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,j)*t2_abab(f,b,l,k)
    doubles_res += -0.250000000000000 * einsum('cdfe,lkcd,aeij,fblk->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,j)*t2_bbbb(f,b,k,l)
    doubles_res +=  0.250000000000000 * einsum('dcef,kldc,aeij,fbkl->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <d,a||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(f,b,i,j)
    doubles_res +=  0.500000000000000 * einsum('daef,kldc,eckl,fbij->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,a||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(f,b,i,j)
    doubles_res +=  0.500000000000000 * einsum('daef,kldc,eckl,fbij->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(f,b,i,j)
    doubles_res +=  0.500000000000000 * einsum('adfe,klcd,cekl,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,a||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(f,b,i,j)
    doubles_res +=  0.500000000000000 * einsum('daef,lkdc,eclk,fbij->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,d||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(f,b,i,j)
    doubles_res +=  0.500000000000000 * einsum('adfe,lkcd,celk,fbij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,d||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(f,b,i,j)
    doubles_res +=  0.500000000000000 * einsum('adfe,kldc,eckl,fbij->abij', g_abab[v, v, v, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(a,f,i,j)
    doubles_res +=  0.500000000000000 * einsum('dbef,kldc,eckl,afij->abij', g_abab[v, v, v, v], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(a,f,i,j)
    doubles_res +=  0.500000000000000 * einsum('dbef,kldc,eckl,afij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(a,f,i,j)
    doubles_res +=  0.500000000000000 * einsum('dbef,klcd,cekl,afij->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(a,f,i,j)
    doubles_res +=  0.500000000000000 * einsum('dbef,lkdc,eclk,afij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(a,f,i,j)
    doubles_res +=  0.500000000000000 * einsum('dbef,lkcd,celk,afij->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(a,f,i,j)
    doubles_res +=  0.500000000000000 * einsum('dbef,kldc,eckl,afij->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,a||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,l,j)*t2_abab(f,b,i,k)
    doubles_res += -1.000000000000000 * einsum('daef,lkdc,eclj,fbik->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,d||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,l,j)*t2_abab(f,b,i,k)
    doubles_res += -1.000000000000000 * einsum('adfe,lkcd,celj,fbik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,d||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_abab(f,b,i,k)
    doubles_res += -1.000000000000000 * einsum('adfe,kldc,ecjl,fbik->abij', g_abab[v, v, v, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,b||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(c,e,l,j)*t2_aaaa(f,a,i,k)
    doubles_res += -1.000000000000000 * einsum('dbfe,kldc,celj,faik->abij', g_abab[v, v, v, v], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,b||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_aaaa(f,a,i,k)
    doubles_res += -1.000000000000000 * einsum('dbfe,kldc,ecjl,faik->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,b||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,c,l,j)*t2_abab(a,f,i,k)
    doubles_res += -1.000000000000000 * einsum('dbef,lkdc,eclj,afik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,b||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(c,e,l,j)*t2_abab(a,f,i,k)
    doubles_res += -1.000000000000000 * einsum('dbef,lkcd,celj,afik->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,b||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_abab(a,f,i,k)
    doubles_res += -1.000000000000000 * einsum('dbef,kldc,ecjl,afik->abij', g_bbbb[v, v, v, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,a||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,l)*t2_abab(f,b,k,j)
    doubles_res += -1.000000000000000 * einsum('daef,kldc,ecil,fbkj->abij', g_aaaa[v, v, v, v], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,a||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,c,i,l)*t2_abab(f,b,k,j)
    doubles_res += -1.000000000000000 * einsum('daef,kldc,ecil,fbkj->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(c,e,i,l)*t2_abab(f,b,k,j)
    doubles_res += -1.000000000000000 * einsum('adfe,klcd,ceil,fbkj->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,d||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,c,i,l)*t2_bbbb(f,b,j,k)
    doubles_res += -1.000000000000000 * einsum('adef,lkcd,ecil,fbjk->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,d||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,l)*t2_bbbb(f,b,j,k)
    doubles_res += -1.000000000000000 * einsum('adef,kldc,ecil,fbjk->abij', g_abab[v, v, v, v], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,b||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,l)*t2_abab(a,f,k,j)
    doubles_res += -1.000000000000000 * einsum('dbef,kldc,ecil,afkj->abij', g_abab[v, v, v, v], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,b||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,i,l)*t2_abab(a,f,k,j)
    doubles_res += -1.000000000000000 * einsum('dbef,kldc,ecil,afkj->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 <d,b||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,i,l)*t2_abab(a,f,k,j)
    doubles_res += -1.000000000000000 * einsum('dbef,klcd,ceil,afkj->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <d,a||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(f,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('daef,kldc,ecij,fbkl->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,d||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(c,e,i,j)*t2_abab(f,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('adfe,klcd,ceij,fbkl->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <d,a||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,c,i,j)*t2_abab(f,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('daef,lkdc,ecij,fblk->abij', g_aaaa[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,d||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(c,e,i,j)*t2_abab(f,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('adfe,lkcd,ceij,fblk->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,d||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,j)*t2_bbbb(f,b,k,l)
    doubles_res += -0.500000000000000 * einsum('adef,kldc,ecij,fbkl->abij', g_abab[v, v, v, v], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 <d,b||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(c,e,i,j)*t2_aaaa(f,a,k,l)
    doubles_res += -0.500000000000000 * einsum('dbfe,kldc,ceij,fakl->abij', g_abab[v, v, v, v], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(a,f,k,l)
    doubles_res +=  0.500000000000000 * einsum('dbef,kldc,ecij,afkl->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(c,e,i,j)*t2_abab(a,f,k,l)
    doubles_res +=  0.500000000000000 * einsum('dbef,klcd,ceij,afkl->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,c,i,j)*t2_abab(a,f,l,k)
    doubles_res +=  0.500000000000000 * einsum('dbef,lkdc,ecij,aflk->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(c,e,i,j)*t2_abab(a,f,l,k)
    doubles_res +=  0.500000000000000 * einsum('dbef,lkcd,ceij,aflk->abij', g_bbbb[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <a,b||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,f,i,j)
    doubles_res +=  0.500000000000000 * einsum('abef,kldc,edkl,cfij->abij', g_abab[v, v, v, v], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,b||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,c,i,j)
    doubles_res += -0.500000000000000 * einsum('abfe,kldc,dekl,fcij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,f,i,j)
    doubles_res += -0.500000000000000 * einsum('abef,klcd,edkl,cfij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,b||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,c,i,j)
    doubles_res += -0.500000000000000 * einsum('abfe,lkdc,delk,fcij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,f,i,j)
    doubles_res += -0.500000000000000 * einsum('abef,lkcd,edlk,cfij->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <a,b||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('abfe,kldc,edkl,fcij->abij', g_abab[v, v, v, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <a,b||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,c,i,k)
    doubles_res += -0.500000000000000 * einsum('abfe,kldc,delj,fcik->abij', g_abab[v, v, v, v], l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <a,b||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,c,i,k)
    doubles_res +=  0.500000000000000 * einsum('abfe,klcd,edjl,fcik->abij', g_abab[v, v, v, v], l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <a,b||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,c,i,k)
    doubles_res +=  0.500000000000000 * einsum('abfe,lkdc,delj,fcik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <a,b||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(c,f,i,k)
    doubles_res +=  0.500000000000000 * einsum('abef,lkcd,edlj,cfik->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,c,i,k)
    doubles_res += -0.500000000000000 * einsum('abfe,kldc,edjl,fcik->abij', g_abab[v, v, v, v], l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(c,f,k,j)
    doubles_res += -0.500000000000000 * einsum('abef,kldc,edil,cfkj->abij', g_abab[v, v, v, v], l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <a,b||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(f,c,k,j)
    doubles_res +=  0.500000000000000 * einsum('abfe,kldc,deil,fckj->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <a,b||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(c,f,k,j)
    doubles_res +=  0.500000000000000 * einsum('abef,klcd,edil,cfkj->abij', g_abab[v, v, v, v], l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <a,b||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(f,c,j,k)
    doubles_res +=  0.500000000000000 * einsum('abef,lkdc,edil,fcjk->abij', g_abab[v, v, v, v], l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(f,c,j,k)
    doubles_res += -0.500000000000000 * einsum('abef,kldc,edil,fcjk->abij', g_abab[v, v, v, v], l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,d||k,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,i,m,j)
    doubles_res += -0.500000000000000 * einsum('mdkl,kldc,cabimj->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||k,l>_abab*l2_abab(k,l,d,c)*t3_abbabb(a,c,b,i,j,m)
    doubles_res +=  0.500000000000000 * einsum('dmkl,kldc,acbijm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||k,l>_abab*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,i,m,j)
    doubles_res +=  0.500000000000000 * einsum('mdkl,klcd,cabimj->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,k>_abab*l2_abab(l,k,d,c)*t3_abbabb(a,c,b,i,j,m)
    doubles_res +=  0.500000000000000 * einsum('dmlk,lkdc,acbijm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||l,k>_abab*l2_abab(l,k,c,d)*t3_aabaab(c,a,b,i,m,j)
    doubles_res +=  0.500000000000000 * einsum('mdlk,lkcd,cabimj->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||k,l>_bbbb*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,b,i,j,m)
    doubles_res += -0.500000000000000 * einsum('mdkl,kldc,acbijm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||l,j>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,i,k,m)
    doubles_res += -1.000000000000000 * einsum('dmlj,kldc,cabikm->abij', g_abab[v, o, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||j,l>_bbbb*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,i,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdjl,klcd,cabikm->abij', g_bbbb[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||l,j>_abab*l2_abab(l,k,d,c)*t3_abbabb(a,c,b,i,k,m)
    doubles_res += -1.000000000000000 * einsum('dmlj,lkdc,acbikm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||l,j>_abab*l2_abab(l,k,c,d)*t3_aabaab(c,a,b,i,m,k)
    doubles_res += -1.000000000000000 * einsum('mdlj,lkcd,cabimk->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||j,l>_bbbb*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,b,i,k,m)
    doubles_res +=  1.000000000000000 * einsum('mdjl,kldc,acbikm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||i,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,m,k,j)
    doubles_res += -1.000000000000000 * einsum('mdil,kldc,cabmkj->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||i,l>_abab*l2_abab(k,l,d,c)*t3_abbabb(a,c,b,k,j,m)
    doubles_res += -1.000000000000000 * einsum('dmil,kldc,acbkjm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||i,l>_abab*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,m,k,j)
    doubles_res +=  1.000000000000000 * einsum('mdil,klcd,cabmkj->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||i,l>_aaaa*l2_abab(l,k,d,c)*t3_abbabb(a,c,b,m,k,j)
    doubles_res += -1.000000000000000 * einsum('mdil,lkdc,acbmkj->abij', g_aaaa[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||i,l>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,b,m,k,j)
    doubles_res +=  1.000000000000000 * einsum('mdil,kldc,acbmkj->abij', g_abab[o, v, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||i,j>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,k,l,m)
    doubles_res += -0.500000000000000 * einsum('dmij,kldc,cabklm->abij', g_abab[v, o, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <d,m||i,j>_abab*l2_abab(k,l,d,c)*t3_abbabb(a,c,b,k,l,m)
    doubles_res +=  0.500000000000000 * einsum('dmij,kldc,acbklm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <m,d||i,j>_abab*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,k,m,l)
    doubles_res +=  0.500000000000000 * einsum('mdij,klcd,cabkml->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <d,m||i,j>_abab*l2_abab(l,k,d,c)*t3_abbabb(a,c,b,l,k,m)
    doubles_res +=  0.500000000000000 * einsum('dmij,lkdc,acblkm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,d||i,j>_abab*l2_abab(l,k,c,d)*t3_aabaab(c,a,b,m,l,k)
    doubles_res += -0.500000000000000 * einsum('mdij,lkcd,cabmlk->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <m,d||i,j>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,b,m,l,k)
    doubles_res +=  0.500000000000000 * einsum('mdij,kldc,acbmlk->abij', g_abab[o, v, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <m,a||k,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,b,i,m,j)
    doubles_res += -0.250000000000000 * einsum('makl,kldc,dcbimj->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <a,m||k,l>_abab*l2_abab(k,l,d,c)*t3_abbabb(d,c,b,i,j,m)
    doubles_res += -0.250000000000000 * einsum('amkl,kldc,dcbijm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <a,m||k,l>_abab*l2_abab(k,l,c,d)*t3_abbabb(c,d,b,i,j,m)
    doubles_res += -0.250000000000000 * einsum('amkl,klcd,cdbijm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <a,m||l,k>_abab*l2_abab(l,k,d,c)*t3_abbabb(d,c,b,i,j,m)
    doubles_res += -0.250000000000000 * einsum('amlk,lkdc,dcbijm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <a,m||l,k>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,d,b,i,j,m)
    doubles_res += -0.250000000000000 * einsum('amlk,lkcd,cdbijm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||k,l>_abab*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,i,m,j)
    doubles_res += -0.250000000000000 * einsum('mbkl,kldc,dacimj->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||k,l>_abab*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,i,m,j)
    doubles_res +=  0.250000000000000 * einsum('mbkl,klcd,acdimj->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,k>_abab*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,i,m,j)
    doubles_res += -0.250000000000000 * einsum('mblk,lkdc,dacimj->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||l,k>_abab*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,i,m,j)
    doubles_res +=  0.250000000000000 * einsum('mblk,lkcd,acdimj->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||k,l>_bbbb*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,i,j,m)
    doubles_res +=  0.250000000000000 * einsum('mbkl,kldc,acdijm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,m||l,j>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,b,i,k,m)
    doubles_res += -0.500000000000000 * einsum('amlj,kldc,dcbikm->abij', g_abab[v, o, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,m||l,j>_abab*l2_abab(l,k,d,c)*t3_abbabb(d,c,b,i,k,m)
    doubles_res +=  0.500000000000000 * einsum('amlj,lkdc,dcbikm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,m||l,j>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,d,b,i,k,m)
    doubles_res +=  0.500000000000000 * einsum('amlj,lkcd,cdbikm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,b||l,j>_abab*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,a,i,k,m)
    doubles_res += -0.500000000000000 * einsum('mblj,kldc,dcaikm->abij', g_abab[o, v, o, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_bbbb*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,i,k,m)
    doubles_res += -0.500000000000000 * einsum('mbjl,kldc,dacikm->abij', g_bbbb[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <m,b||j,l>_bbbb*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,i,k,m)
    doubles_res +=  0.500000000000000 * einsum('mbjl,klcd,acdikm->abij', g_bbbb[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <m,b||l,j>_abab*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,i,m,k)
    doubles_res +=  0.500000000000000 * einsum('mblj,lkdc,dacimk->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,b||l,j>_abab*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,i,m,k)
    doubles_res += -0.500000000000000 * einsum('mblj,lkcd,acdimk->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_bbbb*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,i,k,m)
    doubles_res += -0.500000000000000 * einsum('mbjl,kldc,acdikm->abij', g_bbbb[o, v, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,a||i,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,b,m,k,j)
    doubles_res += -0.500000000000000 * einsum('mail,kldc,dcbmkj->abij', g_aaaa[o, v, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,m||i,l>_abab*l2_abab(k,l,d,c)*t3_abbabb(d,c,b,k,j,m)
    doubles_res +=  0.500000000000000 * einsum('amil,kldc,dcbkjm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,m||i,l>_abab*l2_abab(k,l,c,d)*t3_abbabb(c,d,b,k,j,m)
    doubles_res +=  0.500000000000000 * einsum('amil,klcd,cdbkjm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <m,a||i,l>_aaaa*l2_abab(l,k,d,c)*t3_abbabb(d,c,b,m,k,j)
    doubles_res +=  0.500000000000000 * einsum('mail,lkdc,dcbmkj->abij', g_aaaa[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <m,a||i,l>_aaaa*l2_abab(l,k,c,d)*t3_abbabb(c,d,b,m,k,j)
    doubles_res +=  0.500000000000000 * einsum('mail,lkcd,cdbmkj->abij', g_aaaa[o, v, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,m||i,l>_abab*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,b,j,k,m)
    doubles_res += -0.500000000000000 * einsum('amil,kldc,dcbjkm->abij', g_abab[v, o, o, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,b||i,l>_abab*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,m,k,j)
    doubles_res += -0.500000000000000 * einsum('mbil,kldc,dacmkj->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <m,b||i,l>_abab*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,m,k,j)
    doubles_res +=  0.500000000000000 * einsum('mbil,klcd,acdmkj->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <m,b||i,l>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,m,k,j)
    doubles_res += -0.500000000000000 * einsum('mbil,kldc,acdmkj->abij', g_abab[o, v, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||i,j>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,b,k,l,m)
    doubles_res += -0.250000000000000 * einsum('amij,kldc,dcbklm->abij', g_abab[v, o, o, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||i,j>_abab*l2_abab(k,l,d,c)*t3_abbabb(d,c,b,k,l,m)
    doubles_res += -0.250000000000000 * einsum('amij,kldc,dcbklm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||i,j>_abab*l2_abab(k,l,c,d)*t3_abbabb(c,d,b,k,l,m)
    doubles_res += -0.250000000000000 * einsum('amij,klcd,cdbklm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||i,j>_abab*l2_abab(l,k,d,c)*t3_abbabb(d,c,b,l,k,m)
    doubles_res += -0.250000000000000 * einsum('amij,lkdc,dcblkm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||i,j>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,d,b,l,k,m)
    doubles_res += -0.250000000000000 * einsum('amij,lkcd,cdblkm->abij', g_abab[v, o, o, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <a,m||i,j>_abab*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,b,k,l,m)
    doubles_res += -0.250000000000000 * einsum('amij,kldc,dcbklm->abij', g_abab[v, o, o, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <m,b||i,j>_abab*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,a,k,l,m)
    doubles_res += -0.250000000000000 * einsum('mbij,kldc,dcaklm->abij', g_abab[o, v, o, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <m,b||i,j>_abab*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,k,m,l)
    doubles_res += -0.250000000000000 * einsum('mbij,kldc,dackml->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <m,b||i,j>_abab*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,k,m,l)
    doubles_res +=  0.250000000000000 * einsum('mbij,klcd,acdkml->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <m,b||i,j>_abab*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,m,l,k)
    doubles_res +=  0.250000000000000 * einsum('mbij,lkdc,dacmlk->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <m,b||i,j>_abab*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,m,l,k)
    doubles_res += -0.250000000000000 * einsum('mbij,lkcd,acdmlk->abij', g_abab[o, v, o, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <m,b||i,j>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,m,l,k)
    doubles_res += -0.250000000000000 * einsum('mbij,kldc,acdmlk->abij', g_abab[o, v, o, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <d,c||e,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aabaab(e,a,b,i,k,j)
    doubles_res += -0.500000000000000 * einsum('dcel,kldc,eabikj->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <d,c||e,l>_abab*l2_abab(k,l,d,c)*t3_aabaab(e,a,b,i,k,j)
    doubles_res += -0.500000000000000 * einsum('dcel,kldc,eabikj->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <c,d||e,l>_abab*l2_abab(k,l,c,d)*t3_aabaab(e,a,b,i,k,j)
    doubles_res += -0.500000000000000 * einsum('cdel,klcd,eabikj->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <d,c||l,e>_abab*l2_abab(l,k,d,c)*t3_abbabb(a,e,b,i,j,k)
    doubles_res += -0.500000000000000 * einsum('dcle,lkdc,aebijk->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <c,d||l,e>_abab*l2_abab(l,k,c,d)*t3_abbabb(a,e,b,i,j,k)
    doubles_res += -0.500000000000000 * einsum('cdle,lkcd,aebijk->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <d,c||e,l>_bbbb*l2_bbbb(k,l,d,c)*t3_abbabb(a,e,b,i,j,k)
    doubles_res += -0.500000000000000 * einsum('dcel,kldc,aebijk->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,j>_abab*l2_abab(k,l,d,c)*t3_aabaab(e,a,b,i,k,l)
    doubles_res +=  0.250000000000000 * einsum('dcej,kldc,eabikl->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||e,j>_abab*l2_abab(k,l,c,d)*t3_aabaab(e,a,b,i,k,l)
    doubles_res +=  0.250000000000000 * einsum('cdej,klcd,eabikl->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,j>_abab*l2_abab(l,k,d,c)*t3_aabaab(e,a,b,i,l,k)
    doubles_res +=  0.250000000000000 * einsum('dcej,lkdc,eabilk->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||e,j>_abab*l2_abab(l,k,c,d)*t3_aabaab(e,a,b,i,l,k)
    doubles_res +=  0.250000000000000 * einsum('cdej,lkcd,eabilk->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <d,c||e,j>_bbbb*l2_bbbb(k,l,d,c)*t3_abbabb(a,e,b,i,k,l)
    doubles_res += -0.250000000000000 * einsum('dcej,kldc,aebikl->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,i>_aaaa*l2_aaaa(k,l,d,c)*t3_aabaab(e,a,b,l,k,j)
    doubles_res +=  0.250000000000000 * einsum('dcei,kldc,eablkj->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||i,e>_abab*l2_abab(k,l,d,c)*t3_abbabb(a,e,b,k,j,l)
    doubles_res +=  0.250000000000000 * einsum('dcie,kldc,aebkjl->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||i,e>_abab*l2_abab(k,l,c,d)*t3_abbabb(a,e,b,k,j,l)
    doubles_res +=  0.250000000000000 * einsum('cdie,klcd,aebkjl->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <d,c||i,e>_abab*l2_abab(l,k,d,c)*t3_abbabb(a,e,b,l,k,j)
    doubles_res += -0.250000000000000 * einsum('dcie,lkdc,aeblkj->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <c,d||i,e>_abab*l2_abab(l,k,c,d)*t3_abbabb(a,e,b,l,k,j)
    doubles_res += -0.250000000000000 * einsum('cdie,lkcd,aeblkj->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <d,a||e,l>_aaaa*l2_aaaa(k,l,d,c)*t3_aabaab(e,c,b,i,k,j)
    doubles_res +=  1.000000000000000 * einsum('dael,kldc,ecbikj->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,d||e,l>_abab*l2_abab(k,l,c,d)*t3_aabaab(e,c,b,i,k,j)
    doubles_res +=  1.000000000000000 * einsum('adel,klcd,ecbikj->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <d,a||e,l>_aaaa*l2_abab(l,k,d,c)*t3_abbabb(e,c,b,i,j,k)
    doubles_res +=  1.000000000000000 * einsum('dael,lkdc,ecbijk->abij', g_aaaa[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,d||l,e>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,e,b,i,j,k)
    doubles_res +=  1.000000000000000 * einsum('adle,lkcd,cebijk->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,d||e,l>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(e,c,b,i,j,k)
    doubles_res +=  1.000000000000000 * einsum('adel,kldc,ecbijk->abij', g_abab[v, v, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <d,b||l,e>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(a,c,e,i,k,j)
    doubles_res += -1.000000000000000 * einsum('dble,kldc,aceikj->abij', g_abab[v, v, o, v], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <d,b||e,l>_abab*l2_abab(k,l,d,c)*t3_aabaab(e,a,c,i,k,j)
    doubles_res +=  1.000000000000000 * einsum('dbel,kldc,eacikj->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <d,b||e,l>_bbbb*l2_abab(k,l,c,d)*t3_aabaab(a,c,e,i,k,j)
    doubles_res += -1.000000000000000 * einsum('dbel,klcd,aceikj->abij', g_bbbb[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <d,b||l,e>_abab*l2_abab(l,k,d,c)*t3_abbabb(a,c,e,i,j,k)
    doubles_res += -1.000000000000000 * einsum('dble,lkdc,aceijk->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <d,b||e,l>_bbbb*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,e,i,j,k)
    doubles_res += -1.000000000000000 * einsum('dbel,kldc,aceijk->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,d||e,j>_abab*l2_abab(k,l,c,d)*t3_aabaab(e,c,b,i,k,l)
    doubles_res += -0.500000000000000 * einsum('adej,klcd,ecbikl->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,d||e,j>_abab*l2_abab(l,k,c,d)*t3_aabaab(e,c,b,i,l,k)
    doubles_res += -0.500000000000000 * einsum('adej,lkcd,ecbilk->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,d||e,j>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(e,c,b,i,k,l)
    doubles_res +=  0.500000000000000 * einsum('adej,kldc,ecbikl->abij', g_abab[v, v, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_abab*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,c,a,i,k,l)
    doubles_res +=  0.500000000000000 * einsum('dbej,kldc,ecaikl->abij', g_abab[v, v, v, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_abab*l2_abab(k,l,d,c)*t3_aabaab(e,a,c,i,k,l)
    doubles_res += -0.500000000000000 * einsum('dbej,kldc,eacikl->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_bbbb*l2_abab(k,l,c,d)*t3_aabaab(a,c,e,i,k,l)
    doubles_res +=  0.500000000000000 * einsum('dbej,klcd,aceikl->abij', g_bbbb[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_abab*l2_abab(l,k,d,c)*t3_aabaab(e,a,c,i,l,k)
    doubles_res += -0.500000000000000 * einsum('dbej,lkdc,eacilk->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_bbbb*l2_abab(l,k,c,d)*t3_aabaab(a,c,e,i,l,k)
    doubles_res +=  0.500000000000000 * einsum('dbej,lkcd,aceilk->abij', g_bbbb[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_bbbb*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,e,i,k,l)
    doubles_res += -0.500000000000000 * einsum('dbej,kldc,aceikl->abij', g_bbbb[v, v, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <d,a||e,i>_aaaa*l2_aaaa(k,l,d,c)*t3_aabaab(e,c,b,l,k,j)
    doubles_res += -0.500000000000000 * einsum('daei,kldc,ecblkj->abij', g_aaaa[v, v, v, o], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <d,a||e,i>_aaaa*l2_abab(k,l,d,c)*t3_abbabb(e,c,b,k,j,l)
    doubles_res += -0.500000000000000 * einsum('daei,kldc,ecbkjl->abij', g_aaaa[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,d||i,e>_abab*l2_abab(k,l,c,d)*t3_abbabb(c,e,b,k,j,l)
    doubles_res += -0.500000000000000 * einsum('adie,klcd,cebkjl->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <d,a||e,i>_aaaa*l2_abab(l,k,d,c)*t3_abbabb(e,c,b,l,k,j)
    doubles_res +=  0.500000000000000 * einsum('daei,lkdc,ecblkj->abij', g_aaaa[v, v, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,d||i,e>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,e,b,l,k,j)
    doubles_res +=  0.500000000000000 * einsum('adie,lkcd,ceblkj->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,d||i,e>_abab*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,c,b,j,k,l)
    doubles_res +=  0.500000000000000 * einsum('adie,kldc,ecbjkl->abij', g_abab[v, v, o, v], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <d,b||i,e>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(a,c,e,l,k,j)
    doubles_res +=  0.500000000000000 * einsum('dbie,kldc,acelkj->abij', g_abab[v, v, o, v], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <d,b||i,e>_abab*l2_abab(k,l,d,c)*t3_abbabb(a,c,e,k,j,l)
    doubles_res +=  0.500000000000000 * einsum('dbie,kldc,acekjl->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <d,b||i,e>_abab*l2_abab(l,k,d,c)*t3_abbabb(a,c,e,l,k,j)
    doubles_res += -0.500000000000000 * einsum('dbie,lkdc,acelkj->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||l,e>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,i,k,j)
    doubles_res += -0.500000000000000 * einsum('able,kldc,cdeikj->abij', g_abab[v, v, o, v], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,l>_abab*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,i,k,j)
    doubles_res += -0.500000000000000 * einsum('abel,kldc,edcikj->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||e,l>_abab*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,i,k,j)
    doubles_res += -0.500000000000000 * einsum('abel,klcd,ecdikj->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <a,b||l,e>_abab*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,i,j,k)
    doubles_res += -0.500000000000000 * einsum('able,lkdc,decijk->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,b||l,e>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('able,lkcd,cdeijk->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <a,b||e,l>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('abel,kldc,edcijk->abij', g_abab[v, v, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||e,j>_abab*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,d,c,i,k,l)
    doubles_res +=  0.250000000000000 * einsum('abej,kldc,edcikl->abij', g_abab[v, v, v, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||e,j>_abab*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,i,k,l)
    doubles_res +=  0.250000000000000 * einsum('abej,kldc,edcikl->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||e,j>_abab*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,i,k,l)
    doubles_res +=  0.250000000000000 * einsum('abej,klcd,ecdikl->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||e,j>_abab*l2_abab(l,k,d,c)*t3_aabaab(e,d,c,i,l,k)
    doubles_res +=  0.250000000000000 * einsum('abej,lkdc,edcilk->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||e,j>_abab*l2_abab(l,k,c,d)*t3_aabaab(e,c,d,i,l,k)
    doubles_res +=  0.250000000000000 * einsum('abej,lkcd,ecdilk->abij', g_abab[v, v, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||e,j>_abab*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,i,k,l)
    doubles_res +=  0.250000000000000 * einsum('abej,kldc,edcikl->abij', g_abab[v, v, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||i,e>_abab*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,l,k,j)
    doubles_res +=  0.250000000000000 * einsum('abie,kldc,cdelkj->abij', g_abab[v, v, o, v], l2_aaaa, t3_aabaab, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||i,e>_abab*l2_abab(k,l,d,c)*t3_abbabb(d,e,c,k,j,l)
    doubles_res +=  0.250000000000000 * einsum('abie,kldc,deckjl->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <a,b||i,e>_abab*l2_abab(k,l,c,d)*t3_abbabb(c,d,e,k,j,l)
    doubles_res += -0.250000000000000 * einsum('abie,klcd,cdekjl->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.2500 <a,b||i,e>_abab*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,l,k,j)
    doubles_res += -0.250000000000000 * einsum('abie,lkdc,declkj->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||i,e>_abab*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,l,k,j)
    doubles_res +=  0.250000000000000 * einsum('abie,lkcd,cdelkj->abij', g_abab[v, v, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 <a,b||i,e>_abab*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,d,c,j,k,l)
    doubles_res +=  0.250000000000000 * einsum('abie,kldc,edcjkl->abij', g_abab[v, v, o, v], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (1, 2), (0, 1)])
    
    return doubles_res

