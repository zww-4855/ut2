import numpy
from numpy import einsum



def residQf2_aaaa(g,l2,t2,o,v):

    g_aaaa=g["aaaa"]
    g_bbbb=g["bbbb"]
    g_abab=g["abab"]

    l2_aaaa=l2["aaaa"]
    l2_bbbb=l2["bbbb"]
    l2_abab=l2["abab"]

    t2_aaaa=t2["aaaa"]
    t2_bbbb=t2["bbbb"]
    t2_abab=t2["abab"]
    #	 -0.1250 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,k,l)*t2_aaaa(d,c,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,kldc,efkl,dcjm,abin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,k,l)*t2_abab(d,c,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,kldc,efkl,dcjm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,k,l)*t2_abab(d,c,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmfe,kldc,fekl,dcjm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,k,l)*t2_abab(c,d,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,klcd,efkl,cdjm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,k,l)*t2_abab(c,d,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmfe,klcd,fekl,cdjm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,k)*t2_abab(d,c,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,lkdc,eflk,dcjm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,k)*t2_abab(d,c,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmfe,lkdc,felk,dcjm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,k)*t2_abab(c,d,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,lkcd,eflk,cdjm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,k)*t2_abab(c,d,j,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmfe,lkcd,felk,cdjm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,k,l)*t2_aaaa(d,a,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efkl,dajm,cbin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,k,l)*t2_aaaa(d,a,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,kldc,efkl,dajm,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,k,l)*t2_aaaa(d,a,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,kldc,fekl,dajm,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,k,l)*t2_abab(a,d,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,efkl,adjm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,k,l)*t2_abab(a,d,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,klcd,fekl,adjm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,k)*t2_aaaa(d,a,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkdc,eflk,dajm,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,k)*t2_aaaa(d,a,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,lkdc,felk,dajm,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,k)*t2_abab(a,d,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkcd,eflk,adjm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,k)*t2_abab(a,d,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,lkcd,felk,adjm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,k,l)*t2_abab(a,d,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efkl,adjm,bcin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,j,l)*t2_aaaa(d,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efjl,dckm,abin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,j,l)*t2_abab(d,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efjl,dckm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,j,l)*t2_abab(d,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,kldc,fejl,dckm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,j,l)*t2_abab(c,d,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,efjl,cdkm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,j,l)*t2_abab(c,d,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,klcd,fejl,cdkm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,f,j,l)*t2_abab(d,c,m,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkdc,efjl,dcmk,abin->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,f,j,l)*t2_abab(c,d,m,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkcd,efjl,cdmk,abin->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,j,l)*t2_bbbb(d,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efjl,dckm,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,j,l)*t2_bbbb(d,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,kldc,fejl,dckm,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,j,l)*t2_aaaa(d,c,i,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,kldc,efjl,dcim,abkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,j,l)*t2_abab(d,c,i,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,kldc,efjl,dcim,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,j,l)*t2_abab(d,c,i,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmfe,kldc,fejl,dcim,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,j,l)*t2_abab(c,d,i,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,klcd,efjl,cdim,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,j,l)*t2_abab(c,d,i,m)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmfe,klcd,fejl,cdim,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,j,l)*t2_aaaa(d,a,k,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,efjl,dakm,cbin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,j,l)*t2_aaaa(d,a,k,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,kldc,efjl,dakm,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,j,l)*t2_aaaa(d,a,k,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,fejl,dakm,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,j,l)*t2_abab(a,d,k,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,efjl,adkm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,j,l)*t2_abab(a,d,k,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,klcd,fejl,adkm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,f,j,l)*t2_abab(a,d,m,k)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,efjl,admk,cbin->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,j,l)*t2_abab(a,d,m,k)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,kldc,efjl,admk,bcin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,j,l)*t2_abab(a,d,m,k)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,fejl,admk,bcin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,j,l)*t2_aaaa(d,a,i,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,efjl,daim,cbkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,j,l)*t2_aaaa(d,a,i,m)*t2_abab(b,c,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,kldc,efjl,daim,bckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,j,l)*t2_aaaa(d,a,i,m)*t2_abab(b,c,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,fejl,daim,bckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,j,l)*t2_abab(a,d,i,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,efjl,adim,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,j,l)*t2_abab(a,d,i,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,klcd,fejl,adim,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,f,j,l)*t2_aaaa(d,a,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkdc,efjl,daim,bcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,j,l)*t2_abab(a,d,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,efjl,adim,bcnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,j,l)*t2_abab(a,d,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,kldc,fejl,adim,bcnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,i,j)*t2_aaaa(d,c,l,m)*t2_aaaa(a,b,k,n)
    double_res += -0.250000000000000 * einsum('nmef,kldc,efij,dclm,abkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,f,i,j)*t2_abab(d,c,m,l)*t2_aaaa(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,efij,dcml,abkn->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_aaaa(e,f,i,j)*t2_abab(c,d,m,l)*t2_aaaa(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,efij,cdml,abkn->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,i,j)*t2_aaaa(d,a,l,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efij,dalm,cbkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_aaaa(e,f,i,j)*t2_abab(a,d,m,l)*t2_aaaa(c,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,klcd,efij,adml,cbkn->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,f,i,j)*t2_aaaa(d,a,l,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkdc,efij,dalm,bcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_aaaa(e,f,i,j)*t2_abab(a,d,m,l)*t2_abab(b,c,n,k)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efij,adml,bcnk->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_aaaa(f,c,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edlm,fcjk,abin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_aaaa(f,c,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,kldc,delm,fcjk,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_aaaa(f,c,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,edml,fcjk,abin->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_aaaa(f,c,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,klcd,edlm,fcjk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_abab(f,c,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkdc,edlm,fcjk,abin->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_abab(f,c,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,lkdc,delm,fcjk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,m)*t2_abab(c,f,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,edlm,cfjk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_abab(f,c,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edml,fcjk,abin->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_abab(f,c,j,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,kldc,edlm,fcjk,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_aaaa(f,c,i,j)*t2_aaaa(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edlm,fcij,abkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_aaaa(f,c,i,j)*t2_aaaa(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmfe,kldc,delm,fcij,abkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_aaaa(f,c,i,j)*t2_aaaa(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,klcd,edml,fcij,abkn->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_aaaa(f,c,i,j)*t2_aaaa(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmfe,klcd,edlm,fcij,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_aaaa(f,a,j,k)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('nmef,kldc,edlm,fajk,cbin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_aaaa(f,a,j,k)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('nmfe,kldc,delm,fajk,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.7500 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,l)*t2_aaaa(f,a,j,k)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.750000000000000 * einsum('mnfe,kldc,deml,fajk,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_aaaa(f,a,j,k)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('nmef,klcd,edml,fajk,cbin->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_aaaa(f,a,j,k)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('nmfe,klcd,edlm,fajk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_abab(a,f,j,k)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('mnef,lkdc,edlm,afjk,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_abab(a,f,j,k)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('nmef,lkdc,delm,afjk,bcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.7500 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,m)*t2_abab(a,f,j,k)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.750000000000000 * einsum('nmef,lkcd,edlm,afjk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_abab(a,f,j,k)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('mnef,kldc,edml,afjk,bcin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_abab(a,f,j,k)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('nmef,kldc,edlm,afjk,bcin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_aaaa(f,a,i,j)*t2_aaaa(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edlm,faij,cbkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_aaaa(f,a,i,j)*t2_aaaa(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,kldc,delm,faij,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,l)*t2_aaaa(f,a,i,j)*t2_abab(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,kldc,deml,faij,bckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_aaaa(f,a,i,j)*t2_aaaa(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,klcd,edml,faij,cbkn->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_aaaa(f,a,i,j)*t2_aaaa(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,klcd,edlm,faij,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_aaaa(f,a,i,j)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,lkdc,edlm,faij,bcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_aaaa(f,a,i,j)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,lkdc,delm,faij,bcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_aaaa(f,a,i,j)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,kldc,edml,faij,bcnk->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_aaaa(f,a,i,j)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,kldc,edlm,faij,bcnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,m)*t2_aaaa(f,c,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edjm,fckl,abin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,j,m)*t2_aaaa(f,c,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,kldc,dejm,fckl,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,d,j,m)*t2_abab(f,c,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edjm,fckl,abin->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,j,m)*t2_abab(f,c,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,kldc,dejm,fckl,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,j,m)*t2_abab(c,f,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,edjm,cfkl,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,m)*t2_abab(f,c,l,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,edjm,fclk,abin->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,j,m)*t2_abab(f,c,l,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,lkdc,dejm,fclk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,j,m)*t2_abab(c,f,l,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,edjm,cflk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,m)*t2_bbbb(f,c,k,l)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edjm,fckl,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,m)*t2_aaaa(f,c,i,l)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edjm,fcil,abkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,j,m)*t2_aaaa(f,c,i,l)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,kldc,dejm,fcil,abkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,d,j,m)*t2_abab(f,c,i,l)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edjm,fcil,abkn->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,j,m)*t2_abab(f,c,i,l)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,kldc,dejm,fcil,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,j,m)*t2_abab(c,f,i,l)*t2_aaaa(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,klcd,edjm,cfil,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,m)*t2_aaaa(f,a,k,l)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjm,fakl,cbin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,j,m)*t2_aaaa(f,a,k,l)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,kldc,dejm,fakl,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,j,m)*t2_abab(a,f,k,l)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,kldc,edjm,afkl,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,j,m)*t2_abab(a,f,k,l)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,dejm,afkl,bcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,j,m)*t2_abab(a,f,k,l)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,edjm,afkl,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,m)*t2_abab(a,f,l,k)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,lkdc,edjm,aflk,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,j,m)*t2_abab(a,f,l,k)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,dejm,aflk,bcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,j,m)*t2_abab(a,f,l,k)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,edjm,aflk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,m)*t2_aaaa(f,a,i,l)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,kldc,edjm,fail,cbkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,j,m)*t2_aaaa(f,a,i,l)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmfe,kldc,dejm,fail,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,j,m)*t2_abab(a,f,i,l)*t2_abab(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,kldc,edjm,afil,bckn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,j,m)*t2_abab(a,f,i,l)*t2_abab(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,dejm,afil,bckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,j,m)*t2_abab(a,f,i,l)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,klcd,edjm,afil,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,m)*t2_aaaa(f,a,i,l)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,lkdc,edjm,fail,bcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,j,m)*t2_aaaa(f,a,i,l)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,lkdc,dejm,fail,bcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,m)*t2_abab(a,f,i,l)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edjm,afil,bcnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(f,c,i,j)*t2_aaaa(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmef,kldc,edkl,fcij,abnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(f,c,i,j)*t2_aaaa(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,edkl,fcij,abnm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(f,c,i,j)*t2_aaaa(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,edlk,fcij,abnm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(f,a,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edkl,fajm,cbin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(a,f,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edkl,afjm,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_aaaa(f,a,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,dekl,fajm,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(a,f,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,dekl,afjm,bcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(f,a,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,edkl,fajm,cbin->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(a,f,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,edkl,afjm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_aaaa(f,a,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,lkdc,delk,fajm,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(a,f,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,delk,afjm,bcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(f,a,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,edlk,fajm,cbin->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(a,f,j,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,edlk,afjm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_aaaa(f,a,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,edkl,fajm,bcin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(a,f,j,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edkl,afjm,bcin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(f,a,i,j)*t2_aaaa(c,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edkl,faij,cbnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_aaaa(f,a,i,j)*t2_abab(b,c,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmfe,kldc,dekl,faij,bcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_aaaa(f,a,i,j)*t2_abab(b,c,m,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnfe,kldc,dekl,faij,bcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(f,a,i,j)*t2_aaaa(c,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,klcd,edkl,faij,cbnm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_aaaa(f,a,i,j)*t2_abab(b,c,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmfe,lkdc,delk,faij,bcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_aaaa(f,a,i,j)*t2_abab(b,c,m,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnfe,lkdc,delk,faij,bcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(f,a,i,j)*t2_aaaa(c,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkcd,edlk,faij,cbnm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_aaaa(f,a,i,j)*t2_abab(b,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,kldc,edkl,faij,bcnm->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_aaaa(f,a,i,j)*t2_abab(b,c,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,kldc,edkl,faij,bcmn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_aaaa(f,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjl,fckm,abin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_abab(c,f,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edjl,cfkm,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,j,l)*t2_abab(f,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,kldc,dejl,fckm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_aaaa(f,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,edjl,fckm,abin->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_abab(c,f,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,edjl,cfkm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_abab(f,c,m,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkdc,edjl,fcmk,abin->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_bbbb(f,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,edjl,fckm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_abab(f,c,m,k)*t2_aaaa(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edjl,fcmk,abin->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_bbbb(f,c,k,m)*t2_aaaa(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjl,fckm,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_aaaa(f,c,i,k)*t2_aaaa(a,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edjl,fcik,abnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_aaaa(f,c,i,k)*t2_aaaa(a,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,klcd,edjl,fcik,abnm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_abab(f,c,i,k)*t2_aaaa(a,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkdc,edjl,fcik,abnm->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_abab(f,c,i,k)*t2_aaaa(a,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edjl,fcik,abnm->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.9167 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_aaaa(f,a,k,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.916666666666670 * einsum('nmef,kldc,edjl,fakm,cbin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.9167 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_abab(a,f,k,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.916666666666670 * einsum('nmef,kldc,edjl,afkm,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.9167 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,j,l)*t2_aaaa(f,a,k,m)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.916666666666670 * einsum('mnfe,kldc,dejl,fakm,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.9167 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,j,l)*t2_abab(a,f,k,m)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.916666666666670 * einsum('nmef,kldc,dejl,afkm,bcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.9167 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_aaaa(f,a,k,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate = -0.916666666666670 * einsum('nmef,klcd,edjl,fakm,cbin->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.9167 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_abab(a,f,k,m)*t2_aaaa(c,b,i,n)
    contracted_intermediate =  0.916666666666670 * einsum('nmef,klcd,edjl,afkm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.9167 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_abab(a,f,m,k)*t2_abab(b,c,i,n)
    contracted_intermediate = -0.916666666666670 * einsum('mnef,lkdc,edjl,afmk,bcin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.9167 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_abab(a,f,m,k)*t2_abab(b,c,i,n)
    contracted_intermediate =  0.916666666666670 * einsum('mnef,kldc,edjl,afmk,bcin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_aaaa(f,a,i,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edjl,faim,cbkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_abab(a,f,i,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,kldc,edjl,afim,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,j,l)*t2_aaaa(f,a,i,m)*t2_abab(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,kldc,dejl,faim,bckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,j,l)*t2_abab(a,f,i,m)*t2_abab(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,kldc,dejl,afim,bckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_aaaa(f,a,i,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,klcd,edjl,faim,cbkn->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_abab(a,f,i,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,klcd,edjl,afim,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_aaaa(f,a,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,lkdc,edjl,faim,bcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_abab(a,f,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,lkdc,edjl,afim,bcnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_aaaa(f,a,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edjl,faim,bcnk->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_abab(a,f,i,m)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,kldc,edjl,afim,bcnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_aaaa(f,a,i,k)*t2_aaaa(c,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjl,faik,cbnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,j,l)*t2_aaaa(f,a,i,k)*t2_abab(b,c,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,kldc,dejl,faik,bcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,j,l)*t2_aaaa(f,a,i,k)*t2_abab(b,c,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,dejl,faik,bcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_aaaa(f,a,i,k)*t2_aaaa(c,b,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,edjl,faik,cbnm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_abab(a,f,i,k)*t2_abab(b,c,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,edjl,afik,bcnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_abab(a,f,i,k)*t2_abab(b,c,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,lkdc,edjl,afik,bcmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_abab(a,f,i,k)*t2_abab(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjl,afik,bcnm->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_abab(a,f,i,k)*t2_abab(b,c,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,kldc,edjl,afik,bcmn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_aaaa(f,a,l,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edij,falm,cbkn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_abab(a,f,l,m)*t2_aaaa(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,kldc,edij,aflm,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_abab(a,f,m,l)*t2_abab(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,kldc,edij,afml,bckn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,j)*t2_aaaa(f,a,l,m)*t2_abab(b,c,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,lkdc,edij,falm,bcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,j)*t2_abab(a,f,l,m)*t2_abab(b,c,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,lkdc,edij,aflm,bcnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_aaaa(f,a,k,l)*t2_aaaa(c,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edij,fakl,cbnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_abab(a,f,k,l)*t2_abab(b,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edij,afkl,bcnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_abab(a,f,k,l)*t2_abab(b,c,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,kldc,edij,afkl,bcmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,j)*t2_abab(a,f,l,k)*t2_abab(b,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkdc,edij,aflk,bcnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,j)*t2_abab(a,f,l,k)*t2_abab(b,c,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkdc,edij,aflk,bcmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,l,m)*t2_aaaa(f,b,j,k)*t2_aaaa(d,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,ealm,fbjk,dcin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,l,m)*t2_aaaa(f,b,j,k)*t2_aaaa(d,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,kldc,aelm,fbjk,dcin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,m,l)*t2_aaaa(f,b,j,k)*t2_abab(d,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,aeml,fbjk,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,m,l)*t2_aaaa(f,b,j,k)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,klcd,aeml,fbjk,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,l,m)*t2_abab(b,f,j,k)*t2_abab(d,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,lkdc,ealm,bfjk,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(a,e,l,m)*t2_abab(b,f,j,k)*t2_abab(d,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkdc,aelm,bfjk,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,l,m)*t2_abab(b,f,j,k)*t2_abab(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,lkcd,ealm,bfjk,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,l,m)*t2_abab(b,f,j,k)*t2_abab(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,aelm,bfjk,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.3750 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,l,m)*t2_aaaa(f,b,i,j)*t2_aaaa(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('nmef,kldc,ealm,fbij,dckn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,l,m)*t2_aaaa(f,b,i,j)*t2_aaaa(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('nmfe,kldc,aelm,fbij,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,m,l)*t2_aaaa(f,b,i,j)*t2_abab(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('mnfe,kldc,aeml,fbij,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,m,l)*t2_aaaa(f,b,i,j)*t2_abab(c,d,k,n)
    double_res +=  0.375000000000000 * einsum('mnfe,klcd,aeml,fbij,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,l,m)*t2_aaaa(f,b,i,j)*t2_abab(d,c,n,k)
    double_res +=  0.375000000000000 * einsum('nmef,lkdc,ealm,fbij,dcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,m)*t2_aaaa(f,b,i,j)*t2_abab(d,c,n,k)
    double_res +=  0.375000000000000 * einsum('nmfe,lkdc,aelm,fbij,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,l,m)*t2_aaaa(f,b,i,j)*t2_abab(c,d,n,k)
    double_res +=  0.375000000000000 * einsum('nmef,lkcd,ealm,fbij,cdnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,m)*t2_aaaa(f,b,i,j)*t2_abab(c,d,n,k)
    double_res +=  0.375000000000000 * einsum('nmfe,lkcd,aelm,fbij,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,m,l)*t2_aaaa(f,b,i,j)*t2_bbbb(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('mnfe,kldc,aeml,fbij,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,m)*t2_aaaa(f,b,k,l)*t2_aaaa(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eajm,fbkl,dcin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,j,m)*t2_aaaa(f,b,k,l)*t2_aaaa(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,kldc,aejm,fbkl,dcin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,j,m)*t2_abab(b,f,k,l)*t2_abab(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,kldc,eajm,bfkl,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,j,m)*t2_abab(b,f,k,l)*t2_abab(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,aejm,bfkl,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,j,m)*t2_abab(b,f,k,l)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,klcd,eajm,bfkl,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,j,m)*t2_abab(b,f,k,l)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,aejm,bfkl,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,j,m)*t2_abab(b,f,l,k)*t2_abab(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkdc,eajm,bflk,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(a,e,j,m)*t2_abab(b,f,l,k)*t2_abab(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkdc,aejm,bflk,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,j,m)*t2_abab(b,f,l,k)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkcd,eajm,bflk,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,j,m)*t2_abab(b,f,l,k)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkcd,aejm,bflk,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,m)*t2_aaaa(f,b,i,l)*t2_aaaa(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,kldc,eajm,fbil,dckn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,j,m)*t2_aaaa(f,b,i,l)*t2_aaaa(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmfe,kldc,aejm,fbil,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,j,m)*t2_abab(b,f,i,l)*t2_abab(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('mnef,kldc,eajm,bfil,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,j,m)*t2_abab(b,f,i,l)*t2_abab(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,kldc,aejm,bfil,dckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,j,m)*t2_abab(b,f,i,l)*t2_abab(c,d,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('mnef,klcd,eajm,bfil,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,j,m)*t2_abab(b,f,i,l)*t2_abab(c,d,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,klcd,aejm,bfil,cdkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,j,m)*t2_aaaa(f,b,i,l)*t2_abab(d,c,n,k)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,lkdc,eajm,fbil,dcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,j,m)*t2_aaaa(f,b,i,l)*t2_abab(d,c,n,k)
    contracted_intermediate = -0.375000000000000 * einsum('nmfe,lkdc,aejm,fbil,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,j,m)*t2_aaaa(f,b,i,l)*t2_abab(c,d,n,k)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,lkcd,eajm,fbil,cdnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,j,m)*t2_aaaa(f,b,i,l)*t2_abab(c,d,n,k)
    contracted_intermediate = -0.375000000000000 * einsum('nmfe,lkcd,aejm,fbil,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_aaaa(e,a,j,m)*t2_abab(b,f,i,l)*t2_bbbb(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('mnef,kldc,eajm,bfil,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,m)*t2_abab(b,f,i,l)*t2_bbbb(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,kldc,aejm,bfil,dckn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_aaaa(f,b,j,m)*t2_aaaa(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eakl,fbjm,dcin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_abab(b,f,j,m)*t2_aaaa(d,c,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,kldc,eakl,bfjm,dcin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_aaaa(f,b,j,m)*t2_abab(d,c,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnfe,kldc,aekl,fbjm,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(b,f,j,m)*t2_abab(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,aekl,bfjm,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_aaaa(f,b,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnfe,klcd,aekl,fbjm,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(b,f,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,aekl,bfjm,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_aaaa(f,b,j,m)*t2_abab(d,c,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnfe,lkdc,aelk,fbjm,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(b,f,j,m)*t2_abab(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkdc,aelk,bfjm,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_aaaa(f,b,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnfe,lkcd,aelk,fbjm,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(b,f,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkcd,aelk,bfjm,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_aaaa(f,b,i,j)*t2_aaaa(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmef,kldc,eakl,fbij,dcnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_aaaa(f,b,i,j)*t2_abab(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,kldc,aekl,fbij,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_aaaa(f,b,i,j)*t2_abab(d,c,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,kldc,aekl,fbij,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_aaaa(f,b,i,j)*t2_abab(c,d,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,klcd,aekl,fbij,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_aaaa(f,b,i,j)*t2_abab(c,d,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,klcd,aekl,fbij,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_aaaa(f,b,i,j)*t2_abab(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,lkdc,aelk,fbij,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_aaaa(f,b,i,j)*t2_abab(d,c,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,lkdc,aelk,fbij,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_aaaa(f,b,i,j)*t2_abab(c,d,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,lkcd,aelk,fbij,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_aaaa(f,b,i,j)*t2_abab(c,d,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,lkcd,aelk,fbij,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,l)*t2_aaaa(f,b,k,m)*t2_aaaa(d,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,eajl,fbkm,dcin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,l)*t2_abab(b,f,k,m)*t2_aaaa(d,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,eajl,bfkm,dcin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,j,l)*t2_aaaa(f,b,k,m)*t2_abab(d,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,aejl,fbkm,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,j,l)*t2_abab(b,f,k,m)*t2_abab(d,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,aejl,bfkm,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,j,l)*t2_aaaa(f,b,k,m)*t2_abab(c,d,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,klcd,aejl,fbkm,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,j,l)*t2_abab(b,f,k,m)*t2_abab(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,aejl,bfkm,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,j,l)*t2_abab(b,f,m,k)*t2_abab(d,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,lkdc,eajl,bfmk,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,j,l)*t2_abab(b,f,m,k)*t2_abab(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,lkcd,eajl,bfmk,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,l)*t2_aaaa(f,b,i,m)*t2_aaaa(d,c,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,eajl,fbim,dckn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,l)*t2_abab(b,f,i,m)*t2_aaaa(d,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,eajl,bfim,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,m)*t2_abab(d,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,aejl,fbim,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,j,l)*t2_abab(b,f,i,m)*t2_abab(d,c,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,aejl,bfim,dckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,m)*t2_abab(c,d,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,klcd,aejl,fbim,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,j,l)*t2_abab(b,f,i,m)*t2_abab(c,d,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,aejl,bfim,cdkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,j,l)*t2_aaaa(f,b,i,m)*t2_abab(d,c,n,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,eajl,fbim,dcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,j,l)*t2_abab(b,f,i,m)*t2_abab(d,c,n,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkdc,eajl,bfim,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,j,l)*t2_aaaa(f,b,i,m)*t2_abab(c,d,n,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,eajl,fbim,cdnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,j,l)*t2_abab(b,f,i,m)*t2_abab(c,d,n,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,eajl,bfim,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,m)*t2_bbbb(d,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,aejl,fbim,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,l)*t2_abab(b,f,i,m)*t2_bbbb(d,c,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,aejl,bfim,dckn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,l)*t2_aaaa(f,b,i,k)*t2_aaaa(d,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eajl,fbik,dcnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,k)*t2_abab(d,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,kldc,aejl,fbik,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,k)*t2_abab(d,c,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,kldc,aejl,fbik,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,k)*t2_abab(c,d,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,klcd,aejl,fbik,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,j,l)*t2_aaaa(f,b,i,k)*t2_abab(c,d,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,klcd,aejl,fbik,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,j,l)*t2_abab(b,f,i,k)*t2_abab(d,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkdc,eajl,bfik,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,j,l)*t2_abab(b,f,i,k)*t2_abab(d,c,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkdc,eajl,bfik,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,j,l)*t2_abab(b,f,i,k)*t2_abab(c,d,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkcd,eajl,bfik,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,j,l)*t2_abab(b,f,i,k)*t2_abab(c,d,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkcd,eajl,bfik,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,l)*t2_abab(b,f,i,k)*t2_bbbb(d,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,aejl,bfik,dcnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,k)*t2_aaaa(d,b,i,m)*t2_aaaa(f,c,l,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eajk,dbim,fcln->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,k)*t2_aaaa(d,b,i,m)*t2_abab(c,f,l,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,kldc,eajk,dbim,cfln->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,a,j,k)*t2_aaaa(d,b,i,m)*t2_abab(f,c,n,l)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,kldc,eajk,dbim,fcnl->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,j,k)*t2_aaaa(d,b,i,m)*t2_bbbb(f,c,l,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnef,kldc,eajk,dbim,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,j,k)*t2_abab(b,d,i,m)*t2_abab(c,f,n,l)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,eajk,bdim,cfnl->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,j,k)*t2_aaaa(d,b,i,m)*t2_abab(f,c,l,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,lkdc,aejk,dbim,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,j,k)*t2_abab(b,d,i,m)*t2_aaaa(f,c,l,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmfe,lkcd,aejk,bdim,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,j,k)*t2_abab(b,d,i,m)*t2_abab(c,f,l,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkcd,aejk,bdim,cfln->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,k)*t2_abab(b,d,i,m)*t2_abab(f,c,n,l)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,kldc,aejk,bdim,fcnl->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,k)*t2_abab(b,d,i,m)*t2_bbbb(f,c,l,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,aejk,bdim,fcln->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_aaaa(f,b,l,m)*t2_aaaa(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,eaij,fblm,dckn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,m)*t2_aaaa(d,c,k,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,eaij,bflm,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,m,l)*t2_abab(d,c,k,n)
    double_res += -0.500000000000000 * einsum('mnef,kldc,eaij,bfml,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,j)*t2_abab(b,f,m,l)*t2_abab(c,d,k,n)
    double_res += -0.500000000000000 * einsum('mnef,klcd,eaij,bfml,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,j)*t2_aaaa(f,b,l,m)*t2_abab(d,c,n,k)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,eaij,fblm,dcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,m)*t2_abab(d,c,n,k)
    double_res += -0.500000000000000 * einsum('nmef,lkdc,eaij,bflm,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,j)*t2_aaaa(f,b,l,m)*t2_abab(c,d,n,k)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,eaij,fblm,cdnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,m)*t2_abab(c,d,n,k)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,eaij,bflm,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,m,l)*t2_bbbb(d,c,k,n)
    double_res += -0.500000000000000 * einsum('mnef,kldc,eaij,bfml,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_aaaa(f,b,k,l)*t2_aaaa(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmef,kldc,eaij,fbkl,dcnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,k,l)*t2_abab(d,c,n,m)
    double_res +=  0.125000000000000 * einsum('nmef,kldc,eaij,bfkl,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,k,l)*t2_abab(d,c,m,n)
    double_res +=  0.125000000000000 * einsum('mnef,kldc,eaij,bfkl,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,j)*t2_abab(b,f,k,l)*t2_abab(c,d,n,m)
    double_res +=  0.125000000000000 * einsum('nmef,klcd,eaij,bfkl,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,j)*t2_abab(b,f,k,l)*t2_abab(c,d,m,n)
    double_res +=  0.125000000000000 * einsum('mnef,klcd,eaij,bfkl,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,k)*t2_abab(d,c,n,m)
    double_res +=  0.125000000000000 * einsum('nmef,lkdc,eaij,bflk,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,k)*t2_abab(d,c,m,n)
    double_res +=  0.125000000000000 * einsum('mnef,lkdc,eaij,bflk,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,k)*t2_abab(c,d,n,m)
    double_res +=  0.125000000000000 * einsum('nmef,lkcd,eaij,bflk,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,j)*t2_abab(b,f,l,k)*t2_abab(c,d,m,n)
    double_res +=  0.125000000000000 * einsum('mnef,lkcd,eaij,bflk,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_aaaa(d,c,l,m)*t2_aaaa(f,a,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,kldc,ebjk,dclm,fain->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_aaaa(d,c,l,m)*t2_abab(a,f,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnef,kldc,ebjk,dclm,afin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_abab(d,c,m,l)*t2_aaaa(f,a,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('nmef,kldc,ebjk,dcml,fain->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_abab(d,c,m,l)*t2_abab(a,f,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('mnef,kldc,ebjk,dcml,afin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_aaaa(e,b,j,k)*t2_abab(c,d,m,l)*t2_aaaa(f,a,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('nmef,klcd,ebjk,cdml,fain->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,b,j,k)*t2_abab(c,d,m,l)*t2_abab(a,f,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('mnef,klcd,ebjk,cdml,afin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(b,e,j,k)*t2_abab(d,c,l,m)*t2_aaaa(f,a,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('nmfe,lkdc,bejk,dclm,fain->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(b,e,j,k)*t2_abab(d,c,l,m)*t2_abab(a,f,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('nmef,lkdc,bejk,dclm,afin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(b,e,j,k)*t2_abab(c,d,l,m)*t2_aaaa(f,a,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('nmfe,lkcd,bejk,cdlm,fain->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(b,e,j,k)*t2_abab(c,d,l,m)*t2_abab(a,f,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('nmef,lkcd,bejk,cdlm,afin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(b,e,j,k)*t2_bbbb(d,c,l,m)*t2_aaaa(f,a,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmfe,kldc,bejk,dclm,fain->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(b,e,j,k)*t2_bbbb(d,c,l,m)*t2_abab(a,f,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,kldc,bejk,dclm,afin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_aaaa(d,c,l,m)*t2_aaaa(f,a,k,n)
    double_res += -0.125000000000000 * einsum('nmef,kldc,ebij,dclm,fakn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_aaaa(d,c,l,m)*t2_abab(a,f,k,n)
    double_res += -0.125000000000000 * einsum('mnef,kldc,ebij,dclm,afkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_abab(d,c,m,l)*t2_aaaa(f,a,k,n)
    double_res +=  0.125000000000000 * einsum('nmef,kldc,ebij,dcml,fakn->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_abab(d,c,m,l)*t2_abab(a,f,k,n)
    double_res +=  0.125000000000000 * einsum('mnef,kldc,ebij,dcml,afkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_aaaa(e,b,i,j)*t2_abab(c,d,m,l)*t2_aaaa(f,a,k,n)
    double_res +=  0.125000000000000 * einsum('nmef,klcd,ebij,cdml,fakn->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,b,i,j)*t2_abab(c,d,m,l)*t2_abab(a,f,k,n)
    double_res +=  0.125000000000000 * einsum('mnef,klcd,ebij,cdml,afkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,b,i,j)*t2_abab(d,c,l,m)*t2_abab(a,f,n,k)
    double_res +=  0.125000000000000 * einsum('nmef,lkdc,ebij,dclm,afnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,b,i,j)*t2_abab(c,d,l,m)*t2_abab(a,f,n,k)
    double_res +=  0.125000000000000 * einsum('nmef,lkcd,ebij,cdlm,afnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_bbbb(d,c,l,m)*t2_abab(a,f,n,k)
    double_res += -0.125000000000000 * einsum('nmef,kldc,ebij,dclm,afnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.0833 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,j,m)*t2_aaaa(e,b,k,n)*t2_aaaa(f,c,i,l)
    contracted_intermediate = -0.083333333333330 * einsum('nmef,kldc,dajm,ebkn,fcil->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.0833 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,j,m)*t2_abab(b,e,k,n)*t2_aaaa(f,c,i,l)
    contracted_intermediate =  0.083333333333330 * einsum('mnfe,kldc,dajm,bekn,fcil->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.0833 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(d,a,j,m)*t2_aaaa(e,b,k,n)*t2_abab(f,c,i,l)
    contracted_intermediate = -0.083333333333330 * einsum('nmef,kldc,dajm,ebkn,fcil->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.0833 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_aaaa(d,a,j,m)*t2_abab(b,e,k,n)*t2_abab(f,c,i,l)
    contracted_intermediate =  0.083333333333330 * einsum('mnfe,kldc,dajm,bekn,fcil->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.0833 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,j,m)*t2_aaaa(e,b,k,n)*t2_abab(c,f,i,l)
    contracted_intermediate =  0.083333333333330 * einsum('nmef,klcd,adjm,ebkn,cfil->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.0833 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,d,j,m)*t2_abab(b,e,k,n)*t2_abab(c,f,i,l)
    contracted_intermediate = -0.083333333333330 * einsum('nmef,klcd,adjm,bekn,cfil->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.0833 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,d,j,m)*t2_abab(b,e,n,k)*t2_aaaa(f,c,i,l)
    contracted_intermediate = -0.083333333333330 * einsum('nmfe,lkcd,adjm,benk,fcil->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.0833 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,d,j,m)*t2_abab(b,e,n,k)*t2_abab(f,c,i,l)
    contracted_intermediate = -0.083333333333330 * einsum('nmfe,kldc,adjm,benk,fcil->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
   
    return double_res 


def residQf2_bbbb(g,l2,t2,o,v):

    g_aaaa=g["aaaa"]
    g_bbbb=g["bbbb"]
    g_abab=g["abab"]

    l2_aaaa=l2["aaaa"]
    l2_bbbb=l2["bbbb"]
    l2_abab=l2["abab"]

    t2_aaaa=t2["aaaa"]
    t2_bbbb=t2["bbbb"]
    t2_abab=t2["abab"]

    #	 -0.1250 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,k,l)*t2_abab(d,c,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnef,kldc,efkl,dcmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,k,l)*t2_abab(d,c,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnfe,kldc,fekl,dcmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,k,l)*t2_abab(c,d,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnef,klcd,efkl,cdmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,k,l)*t2_abab(c,d,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnfe,klcd,fekl,cdmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,k)*t2_abab(d,c,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnef,lkdc,eflk,dcmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,k)*t2_abab(d,c,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnfe,lkdc,felk,dcmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,k)*t2_abab(c,d,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnef,lkcd,eflk,cdmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,k)*t2_abab(c,d,m,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnfe,lkcd,felk,cdmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,k,l)*t2_bbbb(d,c,j,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,kldc,efkl,dcjm,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,k,l)*t2_abab(d,a,m,j)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efkl,damj,cbni->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,k,l)*t2_abab(d,a,m,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,kldc,efkl,damj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,k,l)*t2_abab(d,a,m,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,kldc,fekl,damj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,k,l)*t2_bbbb(d,a,j,m)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,efkl,dajm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,k,l)*t2_bbbb(d,a,j,m)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,klcd,fekl,dajm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,k)*t2_abab(d,a,m,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkdc,eflk,damj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,k)*t2_abab(d,a,m,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,lkdc,felk,damj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,k)*t2_bbbb(d,a,j,m)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkcd,eflk,dajm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,k)*t2_bbbb(d,a,j,m)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,lkcd,felk,dajm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,k,l)*t2_bbbb(d,a,j,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efkl,dajm,cbin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,l,j)*t2_aaaa(d,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,kldc,eflj,dckm,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,l,j)*t2_aaaa(d,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,kldc,felj,dckm,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_abab(d,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efjl,dckm,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,f,j,l)*t2_abab(c,d,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,efjl,cdkm,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,j)*t2_abab(d,c,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkdc,eflj,dcmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,j)*t2_abab(d,c,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,lkdc,felj,dcmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,j)*t2_abab(c,d,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkcd,eflj,cdmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,j)*t2_abab(c,d,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,lkcd,felj,cdmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_bbbb(d,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efjl,dckm,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,j)*t2_abab(d,c,m,i)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnef,lkdc,eflj,dcmi,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,j)*t2_abab(d,c,m,i)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnfe,lkdc,felj,dcmi,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,j)*t2_abab(c,d,m,i)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnef,lkcd,eflj,cdmi,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,j)*t2_abab(c,d,m,i)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnfe,lkcd,felj,cdmi,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_bbbb(d,c,i,m)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,kldc,efjl,dcim,abkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,l,j)*t2_abab(d,a,k,m)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,eflj,dakm,cbni->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,l,j)*t2_abab(d,a,k,m)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,kldc,felj,dakm,cbni->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_abab(d,a,k,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,efjl,dakm,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,j)*t2_abab(d,a,m,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,lkdc,eflj,damk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,j)*t2_abab(d,a,m,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,lkdc,felj,damk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,j)*t2_bbbb(d,a,k,m)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,eflj,dakm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,j)*t2_bbbb(d,a,k,m)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,lkcd,felj,dakm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_bbbb(d,a,k,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,efjl,dakm,cbin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,l,j)*t2_abab(d,a,m,i)*t2_abab(c,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,kldc,eflj,dami,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,l,j)*t2_abab(d,a,m,i)*t2_abab(c,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,felj,dami,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,f,j,l)*t2_bbbb(d,a,i,m)*t2_abab(c,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,efjl,daim,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,j)*t2_abab(d,a,m,i)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,lkdc,eflj,dami,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,j)*t2_abab(d,a,m,i)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,lkdc,felj,dami,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,j)*t2_bbbb(d,a,i,m)*t2_abab(c,b,n,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,eflj,daim,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,j)*t2_bbbb(d,a,i,m)*t2_abab(c,b,n,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,lkcd,felj,daim,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_bbbb(d,a,i,m)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,efjl,daim,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_bbbb(e,f,i,j)*t2_abab(d,c,l,m)*t2_bbbb(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,efij,dclm,abkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,f,i,j)*t2_abab(c,d,l,m)*t2_bbbb(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,efij,cdlm,abkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,i,j)*t2_bbbb(d,c,l,m)*t2_bbbb(a,b,k,n)
    double_res += -0.250000000000000 * einsum('nmef,kldc,efij,dclm,abkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 P(a,b)<n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_bbbb(e,f,i,j)*t2_abab(d,a,l,m)*t2_abab(c,b,k,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efij,dalm,cbkn->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,f,i,j)*t2_bbbb(d,a,l,m)*t2_abab(c,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,klcd,efij,dalm,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_bbbb(e,f,i,j)*t2_abab(d,a,l,m)*t2_bbbb(c,b,k,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkdc,efij,dalm,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,i,j)*t2_bbbb(d,a,l,m)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,efij,dalm,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_abab(c,f,k,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,kldc,edlm,cfkj,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_abab(c,f,k,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,delm,cfkj,abin->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,l)*t2_abab(f,c,k,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,deml,fckj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_abab(c,f,k,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,klcd,edml,cfkj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_abab(c,f,k,j)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,edlm,cfkj,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_bbbb(f,c,j,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,lkdc,edlm,fcjk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_bbbb(f,c,j,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,delm,fcjk,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_bbbb(f,c,j,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,kldc,edml,fcjk,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_bbbb(f,c,j,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edlm,fcjk,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_bbbb(f,c,i,j)*t2_bbbb(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,lkdc,edlm,fcij,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_bbbb(f,c,i,j)*t2_bbbb(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,lkdc,delm,fcij,abkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_bbbb(f,c,i,j)*t2_bbbb(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,kldc,edml,fcij,abkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_bbbb(f,c,i,j)*t2_bbbb(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edlm,fcij,abkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_abab(f,a,k,j)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.750000000000000 * einsum('nmef,kldc,edlm,fakj,cbni->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_abab(f,a,k,j)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.750000000000000 * einsum('nmfe,kldc,delm,fakj,cbni->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.7500 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,l)*t2_abab(f,a,k,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.750000000000000 * einsum('mnfe,kldc,deml,fakj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_abab(f,a,k,j)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.750000000000000 * einsum('nmef,klcd,edml,fakj,cbni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_abab(f,a,k,j)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.750000000000000 * einsum('nmfe,klcd,edlm,fakj,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_bbbb(f,a,j,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('mnef,lkdc,edlm,fajk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_bbbb(f,a,j,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('nmef,lkdc,delm,fajk,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.7500 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,m)*t2_bbbb(f,a,j,k)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.750000000000000 * einsum('nmef,lkcd,edlm,fajk,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_bbbb(f,a,j,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('mnef,kldc,edml,fajk,cbin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.7500 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_bbbb(f,a,j,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.750000000000000 * einsum('nmef,kldc,edlm,fajk,cbin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_bbbb(f,a,i,j)*t2_abab(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,kldc,edlm,faij,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_bbbb(f,a,i,j)*t2_abab(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,kldc,delm,faij,cbkn->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_bbbb(f,a,i,j)*t2_abab(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,klcd,edml,faij,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_bbbb(f,a,i,j)*t2_abab(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,klcd,edlm,faij,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_bbbb(f,a,i,j)*t2_bbbb(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,lkdc,edlm,faij,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_bbbb(f,a,i,j)*t2_bbbb(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,lkdc,delm,faij,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,m)*t2_bbbb(f,a,i,j)*t2_abab(c,b,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,lkcd,edlm,faij,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_bbbb(f,a,i,j)*t2_bbbb(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,kldc,edml,faij,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_bbbb(f,a,i,j)*t2_bbbb(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edlm,faij,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,m,j)*t2_aaaa(f,c,k,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,demj,fckl,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,j)*t2_abab(f,c,k,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,demj,fckl,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,j)*t2_abab(c,f,k,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,klcd,edmj,cfkl,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,m)*t2_abab(c,f,k,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,edjm,cfkl,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,m,j)*t2_abab(f,c,l,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,lkdc,demj,fclk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,m,j)*t2_abab(c,f,l,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,lkcd,edmj,cflk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,m)*t2_abab(c,f,l,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,edjm,cflk,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,j)*t2_bbbb(f,c,k,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,kldc,edmj,fckl,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,m)*t2_bbbb(f,c,k,l)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edjm,fckl,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,m,j)*t2_abab(f,c,l,i)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,lkdc,demj,fcli,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,m,j)*t2_abab(c,f,l,i)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,lkcd,edmj,cfli,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,m)*t2_abab(c,f,l,i)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,lkcd,edjm,cfli,abkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,j)*t2_bbbb(f,c,i,l)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,kldc,edmj,fcil,abkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,m)*t2_bbbb(f,c,i,l)*t2_bbbb(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edjm,fcil,abkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,j)*t2_abab(f,a,k,l)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,demj,fakl,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,j)*t2_abab(f,a,k,l)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,edmj,fakl,cbni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,m)*t2_abab(f,a,k,l)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,klcd,edjm,fakl,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,m,j)*t2_abab(f,a,l,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,lkdc,demj,falk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,m,j)*t2_abab(f,a,l,k)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,edmj,falk,cbni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,m)*t2_abab(f,a,l,k)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,lkcd,edjm,falk,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,j)*t2_bbbb(f,a,k,l)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,kldc,edmj,fakl,cbin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,m)*t2_bbbb(f,a,k,l)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjm,fakl,cbin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,m,j)*t2_abab(f,a,l,i)*t2_abab(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,kldc,demj,fali,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,j)*t2_bbbb(f,a,i,l)*t2_abab(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnef,klcd,edmj,fail,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,m)*t2_bbbb(f,a,i,l)*t2_abab(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,klcd,edjm,fail,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,m,j)*t2_abab(f,a,l,i)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,lkdc,demj,fali,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,m,j)*t2_abab(f,a,l,i)*t2_abab(c,b,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,lkcd,edmj,fali,cbnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,m)*t2_abab(f,a,l,i)*t2_abab(c,b,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,lkcd,edjm,fali,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,j)*t2_bbbb(f,a,i,l)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnef,kldc,edmj,fail,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,m)*t2_bbbb(f,a,i,l)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,kldc,edjm,fail,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(f,c,i,j)*t2_bbbb(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,dekl,fcij,abnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(f,c,i,j)*t2_bbbb(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,delk,fcij,abnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(f,c,i,j)*t2_bbbb(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmef,kldc,edkl,fcij,abnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(f,a,m,j)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edkl,famj,cbni->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_bbbb(f,a,j,m)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,edkl,fajm,cbni->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,a,m,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,dekl,famj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(f,a,j,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,dekl,fajm,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(f,a,m,j)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,edkl,famj,cbni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_bbbb(f,a,j,m)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,edkl,fajm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,a,m,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,lkdc,delk,famj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(f,a,j,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,delk,fajm,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(f,a,m,j)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,edlk,famj,cbni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_bbbb(f,a,j,m)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,edlk,fajm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,a,m,j)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,edkl,famj,cbin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(f,a,j,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edkl,fajm,cbin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_bbbb(f,a,i,j)*t2_abab(c,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edkl,faij,cbnm->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_bbbb(f,a,i,j)*t2_abab(c,b,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,kldc,edkl,faij,cbmn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(f,a,i,j)*t2_bbbb(c,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,kldc,dekl,faij,cbnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_bbbb(f,a,i,j)*t2_abab(c,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,klcd,edkl,faij,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_bbbb(f,a,i,j)*t2_abab(c,b,m,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnef,klcd,edkl,faij,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(f,a,i,j)*t2_bbbb(c,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkdc,delk,faij,cbnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_bbbb(f,a,i,j)*t2_abab(c,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkcd,edlk,faij,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_bbbb(f,a,i,j)*t2_abab(c,b,m,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnef,lkcd,edlk,faij,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(f,a,i,j)*t2_bbbb(c,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edkl,faij,cbnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,delj,fckm,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(c,f,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,delj,cfkm,abin->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,klcd,edjl,fckm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(c,f,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,edjl,cfkm,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,c,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,lkdc,delj,fcmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,delj,fckm,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(c,f,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnef,lkcd,edlj,cfmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,c,m,k)*t2_bbbb(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,edjl,fcmk,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(f,c,k,m)*t2_bbbb(a,b,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjl,fckm,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(c,f,k,i)*t2_bbbb(a,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,delj,cfki,abnm->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(c,f,k,i)*t2_bbbb(a,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,klcd,edjl,cfki,abnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,c,i,k)*t2_bbbb(a,b,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkdc,delj,fcik,abnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(f,c,i,k)*t2_bbbb(a,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edjl,fcik,abnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.9167 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(f,a,k,m)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.916666666666670 * einsum('nmfe,kldc,delj,fakm,cbni->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.9167 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(f,a,k,m)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.916666666666670 * einsum('nmfe,klcd,edjl,fakm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.9167 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,a,m,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.916666666666670 * einsum('mnfe,lkdc,delj,famk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.9167 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,a,k,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.916666666666670 * einsum('nmef,lkdc,delj,fakm,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.9167 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(f,a,m,k)*t2_abab(c,b,n,i)
    contracted_intermediate = -0.916666666666670 * einsum('nmef,lkcd,edlj,famk,cbni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.9167 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_bbbb(f,a,k,m)*t2_abab(c,b,n,i)
    contracted_intermediate =  0.916666666666670 * einsum('nmef,lkcd,edlj,fakm,cbni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.9167 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,a,m,k)*t2_bbbb(c,b,i,n)
    contracted_intermediate = -0.916666666666670 * einsum('mnfe,kldc,edjl,famk,cbin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.9167 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(f,a,k,m)*t2_bbbb(c,b,i,n)
    contracted_intermediate =  0.916666666666670 * einsum('nmef,kldc,edjl,fakm,cbin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(f,a,m,i)*t2_abab(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,kldc,delj,fami,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,a,i,m)*t2_abab(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,delj,faim,cbkn->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(f,a,m,i)*t2_abab(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,klcd,edjl,fami,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_bbbb(f,a,i,m)*t2_abab(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,klcd,edjl,faim,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,a,m,i)*t2_bbbb(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,lkdc,delj,fami,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,a,i,m)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,lkdc,delj,faim,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(f,a,m,i)*t2_abab(c,b,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,lkcd,edlj,fami,cbnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_bbbb(f,a,i,m)*t2_abab(c,b,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,lkcd,edlj,faim,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,a,m,i)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,kldc,edjl,fami,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(f,a,i,m)*t2_bbbb(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edjl,faim,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(f,a,k,i)*t2_abab(c,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,kldc,delj,faki,cbnm->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(f,a,k,i)*t2_abab(c,b,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,delj,faki,cbmn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(f,a,k,i)*t2_abab(c,b,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmfe,klcd,edjl,faki,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(f,a,k,i)*t2_abab(c,b,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,klcd,edjl,faki,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,a,i,k)*t2_bbbb(c,b,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,delj,faik,cbnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_bbbb(f,a,i,k)*t2_abab(c,b,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,edlj,faik,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_bbbb(f,a,i,k)*t2_abab(c,b,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,lkcd,edlj,faik,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(f,a,i,k)*t2_bbbb(c,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,edjl,faik,cbnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,j)*t2_abab(f,a,m,l)*t2_abab(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('mnfe,klcd,edij,faml,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,j)*t2_bbbb(f,a,l,m)*t2_abab(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,klcd,edij,falm,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,i,j)*t2_abab(f,a,l,m)*t2_abab(c,b,n,k)
    contracted_intermediate = -1.000000000000000 * einsum('nmfe,lkcd,edij,falm,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,i,j)*t2_abab(f,a,m,l)*t2_bbbb(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnfe,kldc,edij,faml,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,i,j)*t2_bbbb(f,a,l,m)*t2_bbbb(c,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,kldc,edij,falm,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,j)*t2_abab(f,a,k,l)*t2_abab(c,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,klcd,edij,fakl,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,j)*t2_abab(f,a,k,l)*t2_abab(c,b,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,klcd,edij,fakl,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,i,j)*t2_abab(f,a,l,k)*t2_abab(c,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,lkcd,edij,falk,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,i,j)*t2_abab(f,a,l,k)*t2_abab(c,b,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,lkcd,edij,falk,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,i,j)*t2_bbbb(f,a,k,l)*t2_bbbb(c,b,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,edij,fakl,cbnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,a,m,l)*t2_abab(f,b,k,j)*t2_abab(d,c,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,eaml,fbkj,dcni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,l,m)*t2_abab(f,b,k,j)*t2_abab(d,c,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,kldc,ealm,fbkj,dcni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,a,m,l)*t2_abab(f,b,k,j)*t2_abab(c,d,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,klcd,eaml,fbkj,cdni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,l,m)*t2_abab(f,b,k,j)*t2_abab(c,d,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,klcd,ealm,fbkj,cdni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,m)*t2_bbbb(f,b,j,k)*t2_abab(d,c,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,ealm,fbjk,dcni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,m)*t2_bbbb(f,b,j,k)*t2_abab(c,d,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,ealm,fbjk,cdni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,a,m,l)*t2_bbbb(f,b,j,k)*t2_bbbb(d,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnef,kldc,eaml,fbjk,dcin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,l,m)*t2_bbbb(f,b,j,k)*t2_bbbb(d,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,ealm,fbjk,dcin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.3750 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,a,l,m)*t2_bbbb(f,b,i,j)*t2_aaaa(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('nmef,kldc,ealm,fbij,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,a,m,l)*t2_bbbb(f,b,i,j)*t2_abab(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('mnef,kldc,eaml,fbij,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,a,l,m)*t2_bbbb(f,b,i,j)*t2_abab(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('nmef,kldc,ealm,fbij,dckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,a,m,l)*t2_bbbb(f,b,i,j)*t2_abab(c,d,k,n)
    double_res +=  0.375000000000000 * einsum('mnef,klcd,eaml,fbij,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,a,l,m)*t2_bbbb(f,b,i,j)*t2_abab(c,d,k,n)
    double_res +=  0.375000000000000 * einsum('nmef,klcd,ealm,fbij,cdkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,m)*t2_bbbb(f,b,i,j)*t2_abab(d,c,n,k)
    double_res +=  0.375000000000000 * einsum('nmef,lkdc,ealm,fbij,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,m)*t2_bbbb(f,b,i,j)*t2_abab(c,d,n,k)
    double_res +=  0.375000000000000 * einsum('nmef,lkcd,ealm,fbij,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,a,m,l)*t2_bbbb(f,b,i,j)*t2_bbbb(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('mnef,kldc,eaml,fbij,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,l,m)*t2_bbbb(f,b,i,j)*t2_bbbb(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('nmef,kldc,ealm,fbij,dckn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,a,m,j)*t2_abab(f,b,k,l)*t2_abab(d,c,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eamj,fbkl,dcni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,j,m)*t2_abab(f,b,k,l)*t2_abab(d,c,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,kldc,eajm,fbkl,dcni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,a,m,j)*t2_abab(f,b,k,l)*t2_abab(c,d,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,eamj,fbkl,cdni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,j,m)*t2_abab(f,b,k,l)*t2_abab(c,d,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,klcd,eajm,fbkl,cdni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,a,m,j)*t2_abab(f,b,l,k)*t2_abab(d,c,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkdc,eamj,fblk,dcni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,a,j,m)*t2_abab(f,b,l,k)*t2_abab(d,c,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,lkdc,eajm,fblk,dcni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,a,m,j)*t2_abab(f,b,l,k)*t2_abab(c,d,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkcd,eamj,fblk,cdni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,a,j,m)*t2_abab(f,b,l,k)*t2_abab(c,d,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,lkcd,eajm,fblk,cdni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,a,m,j)*t2_bbbb(f,b,k,l)*t2_bbbb(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,kldc,eamj,fbkl,dcin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,m)*t2_bbbb(f,b,k,l)*t2_bbbb(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eajm,fbkl,dcin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,a,m,j)*t2_abab(f,b,l,i)*t2_aaaa(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,kldc,eamj,fbli,dckn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_bbbb(e,a,j,m)*t2_abab(f,b,l,i)*t2_aaaa(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmfe,kldc,eajm,fbli,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,a,m,j)*t2_bbbb(f,b,i,l)*t2_abab(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('mnef,kldc,eamj,fbil,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,a,j,m)*t2_bbbb(f,b,i,l)*t2_abab(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,kldc,eajm,fbil,dckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,a,m,j)*t2_bbbb(f,b,i,l)*t2_abab(c,d,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('mnef,klcd,eamj,fbil,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,a,j,m)*t2_bbbb(f,b,i,l)*t2_abab(c,d,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,klcd,eajm,fbil,cdkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,a,m,j)*t2_abab(f,b,l,i)*t2_abab(d,c,n,k)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,lkdc,eamj,fbli,dcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,a,j,m)*t2_abab(f,b,l,i)*t2_abab(d,c,n,k)
    contracted_intermediate = -0.375000000000000 * einsum('nmfe,lkdc,eajm,fbli,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,a,m,j)*t2_abab(f,b,l,i)*t2_abab(c,d,n,k)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,lkcd,eamj,fbli,cdnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,a,j,m)*t2_abab(f,b,l,i)*t2_abab(c,d,n,k)
    contracted_intermediate = -0.375000000000000 * einsum('nmfe,lkcd,eajm,fbli,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,a,m,j)*t2_bbbb(f,b,i,l)*t2_bbbb(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('mnef,kldc,eamj,fbil,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3750 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,m)*t2_bbbb(f,b,i,l)*t2_bbbb(d,c,k,n)
    contracted_intermediate = -0.375000000000000 * einsum('nmef,kldc,eajm,fbil,dckn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,a,k,l)*t2_abab(f,b,m,j)*t2_abab(d,c,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eakl,fbmj,dcni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,a,k,l)*t2_bbbb(f,b,j,m)*t2_abab(d,c,n,i)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,kldc,eakl,fbjm,dcni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,a,k,l)*t2_abab(f,b,m,j)*t2_abab(c,d,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,eakl,fbmj,cdni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,a,k,l)*t2_bbbb(f,b,j,m)*t2_abab(c,d,n,i)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,klcd,eakl,fbjm,cdni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,a,l,k)*t2_abab(f,b,m,j)*t2_abab(d,c,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkdc,ealk,fbmj,dcni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,k)*t2_bbbb(f,b,j,m)*t2_abab(d,c,n,i)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkdc,ealk,fbjm,dcni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,a,l,k)*t2_abab(f,b,m,j)*t2_abab(c,d,n,i)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkcd,ealk,fbmj,cdni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,k)*t2_bbbb(f,b,j,m)*t2_abab(c,d,n,i)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkcd,ealk,fbjm,cdni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,k,l)*t2_abab(f,b,m,j)*t2_bbbb(d,c,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnfe,kldc,eakl,fbmj,dcin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,k,l)*t2_bbbb(f,b,j,m)*t2_bbbb(d,c,i,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eakl,fbjm,dcin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,a,k,l)*t2_bbbb(f,b,i,j)*t2_abab(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmef,kldc,eakl,fbij,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,a,k,l)*t2_bbbb(f,b,i,j)*t2_abab(d,c,m,n)
    double_res += -0.125000000000000 * einsum('mnef,kldc,eakl,fbij,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,a,k,l)*t2_bbbb(f,b,i,j)*t2_abab(c,d,n,m)
    double_res += -0.125000000000000 * einsum('nmef,klcd,eakl,fbij,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,a,k,l)*t2_bbbb(f,b,i,j)*t2_abab(c,d,m,n)
    double_res += -0.125000000000000 * einsum('mnef,klcd,eakl,fbij,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,k)*t2_bbbb(f,b,i,j)*t2_abab(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmef,lkdc,ealk,fbij,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,k)*t2_bbbb(f,b,i,j)*t2_abab(d,c,m,n)
    double_res += -0.125000000000000 * einsum('mnef,lkdc,ealk,fbij,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,k)*t2_bbbb(f,b,i,j)*t2_abab(c,d,n,m)
    double_res += -0.125000000000000 * einsum('nmef,lkcd,ealk,fbij,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,k)*t2_bbbb(f,b,i,j)*t2_abab(c,d,m,n)
    double_res += -0.125000000000000 * einsum('mnef,lkcd,ealk,fbij,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,k,l)*t2_bbbb(f,b,i,j)*t2_bbbb(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmef,kldc,eakl,fbij,dcnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_abab(f,b,k,m)*t2_abab(d,c,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,kldc,eajl,fbkm,dcni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,j,l)*t2_abab(f,b,k,m)*t2_abab(c,d,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmfe,klcd,eajl,fbkm,cdni->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,a,l,j)*t2_abab(f,b,m,k)*t2_abab(d,c,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkdc,ealj,fbmk,dcni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,j)*t2_bbbb(f,b,k,m)*t2_abab(d,c,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,ealj,fbkm,dcni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,a,l,j)*t2_abab(f,b,m,k)*t2_abab(c,d,n,i)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,ealj,fbmk,cdni->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,j)*t2_bbbb(f,b,k,m)*t2_abab(c,d,n,i)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,ealj,fbkm,cdni->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_abab(f,b,m,k)*t2_bbbb(d,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('mnfe,kldc,eajl,fbmk,dcin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_bbbb(f,b,k,m)*t2_bbbb(d,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,eajl,fbkm,dcin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,a,l,j)*t2_abab(f,b,m,i)*t2_aaaa(d,c,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,ealj,fbmi,dckn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,a,l,j)*t2_bbbb(f,b,i,m)*t2_aaaa(d,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,kldc,ealj,fbim,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_abab(f,b,m,i)*t2_abab(d,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,eajl,fbmi,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_bbbb(f,b,i,m)*t2_abab(d,c,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,eajl,fbim,dckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,j,l)*t2_abab(f,b,m,i)*t2_abab(c,d,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,klcd,eajl,fbmi,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,a,j,l)*t2_bbbb(f,b,i,m)*t2_abab(c,d,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,klcd,eajl,fbim,cdkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(e,a,l,j)*t2_abab(f,b,m,i)*t2_abab(d,c,n,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkdc,ealj,fbmi,dcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,j)*t2_bbbb(f,b,i,m)*t2_abab(d,c,n,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkdc,ealj,fbim,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,a,l,j)*t2_abab(f,b,m,i)*t2_abab(c,d,n,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,lkcd,ealj,fbmi,cdnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,j)*t2_bbbb(f,b,i,m)*t2_abab(c,d,n,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,lkcd,ealj,fbim,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_abab(f,b,m,i)*t2_bbbb(d,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('mnfe,kldc,eajl,fbmi,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_bbbb(f,b,i,m)*t2_bbbb(d,c,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,kldc,eajl,fbim,dckn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,a,l,j)*t2_abab(f,b,k,i)*t2_aaaa(d,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,ealj,fbki,dcnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_abab(f,b,k,i)*t2_abab(d,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,kldc,eajl,fbki,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_abab(f,b,k,i)*t2_abab(d,c,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,kldc,eajl,fbki,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,j,l)*t2_abab(f,b,k,i)*t2_abab(c,d,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,klcd,eajl,fbki,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,j,l)*t2_abab(f,b,k,i)*t2_abab(c,d,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,klcd,eajl,fbki,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,j)*t2_bbbb(f,b,i,k)*t2_abab(d,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkdc,ealj,fbik,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,a,l,j)*t2_bbbb(f,b,i,k)*t2_abab(d,c,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkdc,ealj,fbik,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,j)*t2_bbbb(f,b,i,k)*t2_abab(c,d,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,lkcd,ealj,fbik,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,a,l,j)*t2_bbbb(f,b,i,k)*t2_abab(c,d,m,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,lkcd,ealj,fbik,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,l)*t2_bbbb(f,b,i,k)*t2_bbbb(d,c,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eajl,fbik,dcnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,a,k,j)*t2_abab(d,b,m,i)*t2_aaaa(f,c,l,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eakj,dbmi,fcln->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,a,k,j)*t2_abab(d,b,m,i)*t2_abab(c,f,l,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnef,kldc,eakj,dbmi,cfln->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,a,k,j)*t2_abab(d,b,m,i)*t2_abab(f,c,n,l)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,kldc,eakj,dbmi,fcnl->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,a,k,j)*t2_abab(d,b,m,i)*t2_bbbb(f,c,l,n)
    contracted_intermediate = -0.250000000000000 * einsum('mnef,kldc,eakj,dbmi,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,a,k,j)*t2_bbbb(d,b,i,m)*t2_abab(c,f,n,l)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,klcd,eakj,dbim,cfnl->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,a,j,k)*t2_abab(d,b,m,i)*t2_abab(f,c,l,n)
    contracted_intermediate =  0.250000000000000 * einsum('mnfe,lkdc,eajk,dbmi,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,a,j,k)*t2_bbbb(d,b,i,m)*t2_aaaa(f,c,l,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmfe,lkcd,eajk,dbim,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,a,j,k)*t2_bbbb(d,b,i,m)*t2_abab(c,f,l,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmef,lkcd,eajk,dbim,cfln->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,k)*t2_bbbb(d,b,i,m)*t2_abab(f,c,n,l)
    contracted_intermediate =  0.250000000000000 * einsum('nmfe,kldc,eajk,dbim,fcnl->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,k)*t2_bbbb(d,b,i,m)*t2_bbbb(f,c,l,n)
    contracted_intermediate =  0.250000000000000 * einsum('nmef,kldc,eajk,dbim,fcln->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,l,m)*t2_aaaa(d,c,k,n)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,eaij,fblm,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,m,l)*t2_abab(d,c,k,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,eaij,fbml,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_bbbb(f,b,l,m)*t2_abab(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,eaij,fblm,dckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,j)*t2_abab(f,b,m,l)*t2_abab(c,d,k,n)
    double_res += -0.500000000000000 * einsum('mnfe,klcd,eaij,fbml,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,j)*t2_bbbb(f,b,l,m)*t2_abab(c,d,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,eaij,fblm,cdkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,l,m)*t2_abab(d,c,n,k)
    double_res += -0.500000000000000 * einsum('nmfe,lkdc,eaij,fblm,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,a,i,j)*t2_abab(f,b,l,m)*t2_abab(c,d,n,k)
    double_res += -0.500000000000000 * einsum('nmfe,lkcd,eaij,fblm,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,m,l)*t2_bbbb(d,c,k,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,eaij,fbml,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_bbbb(f,b,l,m)*t2_bbbb(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,eaij,fblm,dckn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,k,l)*t2_abab(d,c,n,m)
    double_res +=  0.125000000000000 * einsum('nmfe,kldc,eaij,fbkl,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,k,l)*t2_abab(d,c,m,n)
    double_res +=  0.125000000000000 * einsum('mnfe,kldc,eaij,fbkl,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,j)*t2_abab(f,b,k,l)*t2_abab(c,d,n,m)
    double_res +=  0.125000000000000 * einsum('nmfe,klcd,eaij,fbkl,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,j)*t2_abab(f,b,k,l)*t2_abab(c,d,m,n)
    double_res +=  0.125000000000000 * einsum('mnfe,klcd,eaij,fbkl,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,l,k)*t2_abab(d,c,n,m)
    double_res +=  0.125000000000000 * einsum('nmfe,lkdc,eaij,fblk,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,a,i,j)*t2_abab(f,b,l,k)*t2_abab(d,c,m,n)
    double_res +=  0.125000000000000 * einsum('mnfe,lkdc,eaij,fblk,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,a,i,j)*t2_abab(f,b,l,k)*t2_abab(c,d,n,m)
    double_res +=  0.125000000000000 * einsum('nmfe,lkcd,eaij,fblk,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,a,i,j)*t2_abab(f,b,l,k)*t2_abab(c,d,m,n)
    double_res +=  0.125000000000000 * einsum('mnfe,lkcd,eaij,fblk,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_bbbb(f,b,k,l)*t2_bbbb(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmef,kldc,eaij,fbkl,dcnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 P(i,j)<n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(d,c,l,m)*t2_abab(f,a,n,i)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,kldc,ebkj,dclm,fani->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(d,c,l,m)*t2_bbbb(f,a,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('mnef,kldc,ebkj,dclm,fain->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_abab(d,c,m,l)*t2_abab(f,a,n,i)
    contracted_intermediate =  0.125000000000000 * einsum('nmef,kldc,ebkj,dcml,fani->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_abab(d,c,m,l)*t2_bbbb(f,a,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('mnef,kldc,ebkj,dcml,fain->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_abab(c,d,m,l)*t2_abab(f,a,n,i)
    contracted_intermediate =  0.125000000000000 * einsum('nmef,klcd,ebkj,cdml,fani->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_abab(c,d,m,l)*t2_bbbb(f,a,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('mnef,klcd,ebkj,cdml,fain->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_abab(d,c,l,m)*t2_abab(f,a,n,i)
    contracted_intermediate =  0.125000000000000 * einsum('nmfe,lkdc,ebjk,dclm,fani->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_abab(d,c,l,m)*t2_bbbb(f,a,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('nmef,lkdc,ebjk,dclm,fain->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_abab(c,d,l,m)*t2_abab(f,a,n,i)
    contracted_intermediate =  0.125000000000000 * einsum('nmfe,lkcd,ebjk,cdlm,fani->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_abab(c,d,l,m)*t2_bbbb(f,a,i,n)
    contracted_intermediate =  0.125000000000000 * einsum('nmef,lkcd,ebjk,cdlm,fain->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_bbbb(d,c,l,m)*t2_abab(f,a,n,i)
    contracted_intermediate = -0.125000000000000 * einsum('nmfe,kldc,ebjk,dclm,fani->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_bbbb(d,c,l,m)*t2_bbbb(f,a,i,n)
    contracted_intermediate = -0.125000000000000 * einsum('nmef,kldc,ebjk,dclm,fain->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_bbbb(e,b,i,j)*t2_aaaa(d,c,l,m)*t2_abab(f,a,k,n)
    double_res += -0.125000000000000 * einsum('mnfe,kldc,ebij,dclm,fakn->abij', g_abab[o, o, v, v], l2_aaaa, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_bbbb(e,b,i,j)*t2_abab(d,c,m,l)*t2_abab(f,a,k,n)
    double_res +=  0.125000000000000 * einsum('mnfe,kldc,ebij,dcml,fakn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,b,i,j)*t2_abab(c,d,m,l)*t2_abab(f,a,k,n)
    double_res +=  0.125000000000000 * einsum('mnfe,klcd,ebij,cdml,fakn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,b,i,j)*t2_abab(d,c,l,m)*t2_abab(f,a,n,k)
    double_res +=  0.125000000000000 * einsum('nmfe,lkdc,ebij,dclm,fank->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_bbbb(e,b,i,j)*t2_abab(d,c,l,m)*t2_bbbb(f,a,k,n)
    double_res +=  0.125000000000000 * einsum('nmef,lkdc,ebij,dclm,fakn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,b,i,j)*t2_abab(c,d,l,m)*t2_abab(f,a,n,k)
    double_res +=  0.125000000000000 * einsum('nmfe,lkcd,ebij,cdlm,fank->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,b,i,j)*t2_abab(c,d,l,m)*t2_bbbb(f,a,k,n)
    double_res +=  0.125000000000000 * einsum('nmef,lkcd,ebij,cdlm,fakn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,i,j)*t2_bbbb(d,c,l,m)*t2_abab(f,a,n,k)
    double_res += -0.125000000000000 * einsum('nmfe,kldc,ebij,dclm,fank->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,i,j)*t2_bbbb(d,c,l,m)*t2_bbbb(f,a,k,n)
    double_res += -0.125000000000000 * einsum('nmef,kldc,ebij,dclm,fakn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.0833 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,a,m,j)*t2_abab(e,b,k,n)*t2_abab(c,f,l,i)
    contracted_intermediate = -0.083333333333330 * einsum('mnef,kldc,damj,ebkn,cfli->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.0833 P(i,j)*P(a,b)<m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(d,a,m,j)*t2_abab(e,b,k,n)*t2_bbbb(f,c,i,l)
    contracted_intermediate = -0.083333333333330 * einsum('mnef,kldc,damj,ebkn,fcil->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.0833 P(i,j)*P(a,b)<n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_abab(d,a,m,j)*t2_abab(e,b,n,k)*t2_abab(f,c,l,i)
    contracted_intermediate = -0.083333333333330 * einsum('nmef,lkdc,damj,ebnk,fcli->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.0833 P(i,j)*P(a,b)<m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,a,m,j)*t2_bbbb(e,b,k,n)*t2_abab(f,c,l,i)
    contracted_intermediate =  0.083333333333330 * einsum('mnfe,lkdc,damj,ebkn,fcli->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.0833 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_bbbb(d,a,j,m)*t2_abab(e,b,n,k)*t2_abab(c,f,l,i)
    contracted_intermediate =  0.083333333333330 * einsum('nmef,lkcd,dajm,ebnk,cfli->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.0833 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(d,a,j,m)*t2_bbbb(e,b,k,n)*t2_abab(c,f,l,i)
    contracted_intermediate = -0.083333333333330 * einsum('nmef,lkcd,dajm,ebkn,cfli->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.0833 P(i,j)*P(a,b)<n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(d,a,j,m)*t2_abab(e,b,n,k)*t2_bbbb(f,c,i,l)
    contracted_intermediate =  0.083333333333330 * einsum('nmef,kldc,dajm,ebnk,fcil->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.0833 P(i,j)*P(a,b)<n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(d,a,j,m)*t2_bbbb(e,b,k,n)*t2_bbbb(f,c,i,l)
    contracted_intermediate = -0.083333333333330 * einsum('nmef,kldc,dajm,ebkn,fcil->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
   
    return double_res 





def residQf2_abab(g,l2,t2,o,v):

    g_aaaa=g["aaaa"]
    g_bbbb=g["bbbb"]
    g_abab=g["abab"]

    l2_aaaa=l2["aaaa"]
    l2_bbbb=l2["bbbb"]
    l2_abab=l2["abab"]

    t2_aaaa=t2["aaaa"]
    t2_bbbb=t2["bbbb"]
    t2_abab=t2["abab"]

    #	 -0.1250 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,k,l)*t2_abab(d,c,m,j)*t2_abab(a,b,i,n)
    double_res = -0.125000000000000 * einsum('mnef,kldc,efkl,dcmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,k,l)*t2_abab(d,c,m,j)*t2_abab(a,b,i,n)
    double_res += -0.125000000000000 * einsum('mnfe,kldc,fekl,dcmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,k,l)*t2_abab(c,d,m,j)*t2_abab(a,b,i,n)
    double_res += -0.125000000000000 * einsum('mnef,klcd,efkl,cdmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,k,l)*t2_abab(c,d,m,j)*t2_abab(a,b,i,n)
    double_res += -0.125000000000000 * einsum('mnfe,klcd,fekl,cdmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,k)*t2_abab(d,c,m,j)*t2_abab(a,b,i,n)
    double_res += -0.125000000000000 * einsum('mnef,lkdc,eflk,dcmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,k)*t2_abab(d,c,m,j)*t2_abab(a,b,i,n)
    double_res += -0.125000000000000 * einsum('mnfe,lkdc,felk,dcmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,k)*t2_abab(c,d,m,j)*t2_abab(a,b,i,n)
    double_res += -0.125000000000000 * einsum('mnef,lkcd,eflk,cdmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,k)*t2_abab(c,d,m,j)*t2_abab(a,b,i,n)
    double_res += -0.125000000000000 * einsum('mnfe,lkcd,felk,cdmj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,k,l)*t2_bbbb(d,c,j,m)*t2_abab(a,b,i,n)
    double_res += -0.125000000000000 * einsum('nmef,kldc,efkl,dcjm,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,k,l)*t2_aaaa(d,c,i,m)*t2_abab(a,b,n,j)
    double_res += -0.125000000000000 * einsum('nmef,kldc,efkl,dcim,abnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,k,l)*t2_abab(d,c,i,m)*t2_abab(a,b,n,j)
    double_res += -0.125000000000000 * einsum('nmef,kldc,efkl,dcim,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,k,l)*t2_abab(d,c,i,m)*t2_abab(a,b,n,j)
    double_res += -0.125000000000000 * einsum('nmfe,kldc,fekl,dcim,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,k,l)*t2_abab(c,d,i,m)*t2_abab(a,b,n,j)
    double_res += -0.125000000000000 * einsum('nmef,klcd,efkl,cdim,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,k,l)*t2_abab(c,d,i,m)*t2_abab(a,b,n,j)
    double_res += -0.125000000000000 * einsum('nmfe,klcd,fekl,cdim,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,k)*t2_abab(d,c,i,m)*t2_abab(a,b,n,j)
    double_res += -0.125000000000000 * einsum('nmef,lkdc,eflk,dcim,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,k)*t2_abab(d,c,i,m)*t2_abab(a,b,n,j)
    double_res += -0.125000000000000 * einsum('nmfe,lkdc,felk,dcim,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,k)*t2_abab(c,d,i,m)*t2_abab(a,b,n,j)
    double_res += -0.125000000000000 * einsum('nmef,lkcd,eflk,cdim,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,k)*t2_abab(c,d,i,m)*t2_abab(a,b,n,j)
    double_res += -0.125000000000000 * einsum('nmfe,lkcd,felk,cdim,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,k,l)*t2_abab(a,d,m,j)*t2_abab(c,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnef,klcd,efkl,admj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,k,l)*t2_abab(a,d,m,j)*t2_abab(c,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,klcd,fekl,admj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,k)*t2_abab(a,d,m,j)*t2_abab(c,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkcd,eflk,admj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,k)*t2_abab(a,d,m,j)*t2_abab(c,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkcd,felk,admj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,k,l)*t2_aaaa(d,a,i,m)*t2_abab(c,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,efkl,daim,cbnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,k,l)*t2_aaaa(d,a,i,m)*t2_bbbb(c,b,j,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,efkl,daim,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,k,l)*t2_aaaa(d,a,i,m)*t2_bbbb(c,b,j,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,fekl,daim,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,k,l)*t2_abab(a,d,i,m)*t2_abab(c,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,efkl,adim,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,k,l)*t2_abab(a,d,i,m)*t2_abab(c,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,klcd,fekl,adim,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,k)*t2_aaaa(d,a,i,m)*t2_bbbb(c,b,j,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkdc,eflk,daim,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,k)*t2_aaaa(d,a,i,m)*t2_bbbb(c,b,j,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,felk,daim,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,k)*t2_abab(a,d,i,m)*t2_abab(c,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,eflk,adim,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,k)*t2_abab(a,d,i,m)*t2_abab(c,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,lkcd,felk,adim,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,k,l)*t2_abab(a,d,i,m)*t2_bbbb(c,b,j,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,efkl,adim,cbjn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,l,j)*t2_aaaa(d,c,k,m)*t2_abab(a,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,eflj,dckm,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,l,j)*t2_aaaa(d,c,k,m)*t2_abab(a,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,felj,dckm,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_abab(d,c,k,m)*t2_abab(a,b,i,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,efjl,dckm,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,f,j,l)*t2_abab(c,d,k,m)*t2_abab(a,b,i,n)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,efjl,cdkm,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,j)*t2_abab(d,c,m,k)*t2_abab(a,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkdc,eflj,dcmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,j)*t2_abab(d,c,m,k)*t2_abab(a,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,felj,dcmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,j)*t2_abab(c,d,m,k)*t2_abab(a,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkcd,eflj,cdmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,j)*t2_abab(c,d,m,k)*t2_abab(a,b,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkcd,felj,cdmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_bbbb(d,c,k,m)*t2_abab(a,b,i,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,efjl,dckm,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,i,l)*t2_aaaa(d,c,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,efil,dckm,abnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,i,l)*t2_abab(d,c,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,efil,dckm,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,i,l)*t2_abab(d,c,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,feil,dckm,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,i,l)*t2_abab(c,d,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,efil,cdkm,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,i,l)*t2_abab(c,d,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,klcd,feil,cdkm,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,f,i,l)*t2_abab(d,c,m,k)*t2_abab(a,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,efil,dcmk,abnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,f,i,l)*t2_abab(c,d,m,k)*t2_abab(a,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,efil,cdmk,abnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,i,l)*t2_bbbb(d,c,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,efil,dckm,abnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,i,l)*t2_bbbb(d,c,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,feil,dckm,abnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,l,j)*t2_aaaa(d,c,i,m)*t2_abab(a,b,k,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,eflj,dcim,abkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,l,j)*t2_aaaa(d,c,i,m)*t2_abab(a,b,k,n)
    double_res += -0.250000000000000 * einsum('mnfe,kldc,felj,dcim,abkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_abab(d,c,i,m)*t2_abab(a,b,k,n)
    double_res += -0.250000000000000 * einsum('nmef,kldc,efjl,dcim,abkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,f,j,l)*t2_abab(c,d,i,m)*t2_abab(a,b,k,n)
    double_res += -0.250000000000000 * einsum('nmef,klcd,efjl,cdim,abkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,j)*t2_abab(d,c,i,m)*t2_abab(a,b,n,k)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,eflj,dcim,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,j)*t2_abab(d,c,i,m)*t2_abab(a,b,n,k)
    double_res +=  0.250000000000000 * einsum('nmfe,lkdc,felj,dcim,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,j)*t2_abab(c,d,i,m)*t2_abab(a,b,n,k)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,eflj,cdim,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,j)*t2_abab(c,d,i,m)*t2_abab(a,b,n,k)
    double_res +=  0.250000000000000 * einsum('nmfe,lkcd,felj,cdim,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,i,l)*t2_abab(d,c,m,j)*t2_abab(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,efil,dcmj,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,i,l)*t2_abab(d,c,m,j)*t2_abab(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,feil,dcmj,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,i,l)*t2_abab(c,d,m,j)*t2_abab(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnef,klcd,efil,cdmj,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,i,l)*t2_abab(c,d,m,j)*t2_abab(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnfe,klcd,feil,cdmj,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,f,i,l)*t2_abab(d,c,m,j)*t2_abab(a,b,n,k)
    double_res += -0.250000000000000 * einsum('nmef,lkdc,efil,dcmj,abnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,f,i,l)*t2_abab(c,d,m,j)*t2_abab(a,b,n,k)
    double_res += -0.250000000000000 * einsum('nmef,lkcd,efil,cdmj,abnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,i,l)*t2_bbbb(d,c,j,m)*t2_abab(a,b,n,k)
    double_res += -0.250000000000000 * einsum('nmef,kldc,efil,dcjm,abnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,i,l)*t2_bbbb(d,c,j,m)*t2_abab(a,b,n,k)
    double_res += -0.250000000000000 * einsum('nmfe,kldc,feil,dcjm,abnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,l,j)*t2_aaaa(d,a,k,m)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('mnef,kldc,eflj,dakm,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,l,j)*t2_aaaa(d,a,k,m)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,felj,dakm,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,f,j,l)*t2_abab(a,d,k,m)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('nmef,klcd,efjl,adkm,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,j)*t2_abab(a,d,m,k)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('mnef,lkcd,eflj,admk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,j)*t2_abab(a,d,m,k)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,lkcd,felj,admk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,f,i,l)*t2_aaaa(d,a,k,m)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,kldc,efil,dakm,cbnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,i,l)*t2_aaaa(d,a,k,m)*t2_bbbb(c,b,j,n)
    double_res += -0.500000000000000 * einsum('mnef,kldc,efil,dakm,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,i,l)*t2_aaaa(d,a,k,m)*t2_bbbb(c,b,j,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,feil,dakm,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,i,l)*t2_abab(a,d,k,m)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,klcd,efil,adkm,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,i,l)*t2_abab(a,d,k,m)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,klcd,feil,adkm,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,f,i,l)*t2_abab(a,d,m,k)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,efil,admk,cbnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,i,l)*t2_abab(a,d,m,k)*t2_bbbb(c,b,j,n)
    double_res += -0.500000000000000 * einsum('mnef,kldc,efil,admk,cbjn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,i,l)*t2_abab(a,d,m,k)*t2_bbbb(c,b,j,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,feil,admk,cbjn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,l,j)*t2_aaaa(d,a,i,m)*t2_abab(c,b,k,n)
    double_res +=  0.500000000000000 * einsum('mnef,kldc,eflj,daim,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,l,j)*t2_aaaa(d,a,i,m)*t2_abab(c,b,k,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,felj,daim,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,f,j,l)*t2_abab(a,d,i,m)*t2_abab(c,b,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,efjl,adim,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,l,j)*t2_aaaa(d,a,i,m)*t2_bbbb(c,b,k,n)
    double_res += -0.500000000000000 * einsum('mnef,lkdc,eflj,daim,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,l,j)*t2_aaaa(d,a,i,m)*t2_bbbb(c,b,k,n)
    double_res += -0.500000000000000 * einsum('mnfe,lkdc,felj,daim,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,l,j)*t2_abab(a,d,i,m)*t2_abab(c,b,n,k)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,eflj,adim,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,l,j)*t2_abab(a,d,i,m)*t2_abab(c,b,n,k)
    double_res += -0.500000000000000 * einsum('nmfe,lkcd,felj,adim,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,f,j,l)*t2_abab(a,d,i,m)*t2_bbbb(c,b,k,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,efjl,adim,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,i,l)*t2_abab(a,d,m,j)*t2_abab(c,b,k,n)
    double_res += -0.500000000000000 * einsum('mnef,klcd,efil,admj,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,i,l)*t2_abab(a,d,m,j)*t2_abab(c,b,k,n)
    double_res += -0.500000000000000 * einsum('mnfe,klcd,feil,admj,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,f,i,l)*t2_abab(a,d,m,j)*t2_abab(c,b,n,k)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,efil,admj,cbnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,i,l)*t2_abab(a,d,m,j)*t2_bbbb(c,b,k,n)
    double_res +=  0.500000000000000 * einsum('mnef,kldc,efil,admj,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,i,l)*t2_abab(a,d,m,j)*t2_bbbb(c,b,k,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,feil,admj,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,i,j)*t2_aaaa(d,c,l,m)*t2_abab(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,efij,dclm,abkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,i,j)*t2_aaaa(d,c,l,m)*t2_abab(a,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,feij,dclm,abkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,i,j)*t2_abab(d,c,m,l)*t2_abab(a,b,k,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,efij,dcml,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,i,j)*t2_abab(d,c,m,l)*t2_abab(a,b,k,n)
    double_res += -0.250000000000000 * einsum('mnfe,kldc,feij,dcml,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,i,j)*t2_abab(c,d,m,l)*t2_abab(a,b,k,n)
    double_res += -0.250000000000000 * einsum('mnef,klcd,efij,cdml,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,i,j)*t2_abab(c,d,m,l)*t2_abab(a,b,k,n)
    double_res += -0.250000000000000 * einsum('mnfe,klcd,feij,cdml,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,i,j)*t2_abab(d,c,l,m)*t2_abab(a,b,n,k)
    double_res += -0.250000000000000 * einsum('nmef,lkdc,efij,dclm,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,i,j)*t2_abab(d,c,l,m)*t2_abab(a,b,n,k)
    double_res += -0.250000000000000 * einsum('nmfe,lkdc,feij,dclm,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,i,j)*t2_abab(c,d,l,m)*t2_abab(a,b,n,k)
    double_res += -0.250000000000000 * einsum('nmef,lkcd,efij,cdlm,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,i,j)*t2_abab(c,d,l,m)*t2_abab(a,b,n,k)
    double_res += -0.250000000000000 * einsum('nmfe,lkcd,feij,cdlm,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,i,j)*t2_bbbb(d,c,l,m)*t2_abab(a,b,n,k)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,efij,dclm,abnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,i,j)*t2_bbbb(d,c,l,m)*t2_abab(a,b,n,k)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,feij,dclm,abnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,i,j)*t2_aaaa(d,a,l,m)*t2_abab(c,b,k,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,efij,dalm,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,i,j)*t2_aaaa(d,a,l,m)*t2_abab(c,b,k,n)
    double_res += -0.250000000000000 * einsum('mnfe,kldc,feij,dalm,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,i,j)*t2_abab(a,d,m,l)*t2_abab(c,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnef,klcd,efij,adml,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,i,j)*t2_abab(a,d,m,l)*t2_abab(c,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnfe,klcd,feij,adml,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,i,j)*t2_aaaa(d,a,l,m)*t2_bbbb(c,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkdc,efij,dalm,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,i,j)*t2_aaaa(d,a,l,m)*t2_bbbb(c,b,k,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,feij,dalm,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,f,i,j)*t2_abab(a,d,l,m)*t2_abab(c,b,n,k)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,efij,adlm,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(f,e,i,j)*t2_abab(a,d,l,m)*t2_abab(c,b,n,k)
    double_res +=  0.250000000000000 * einsum('nmfe,lkcd,feij,adlm,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,i,j)*t2_abab(a,d,m,l)*t2_bbbb(c,b,k,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,efij,adml,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,i,j)*t2_abab(a,d,m,l)*t2_bbbb(c,b,k,n)
    double_res += -0.250000000000000 * einsum('mnfe,kldc,feij,adml,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,f,i,j)*t2_abab(d,b,l,m)*t2_aaaa(c,a,k,n)
    double_res += -0.250000000000000 * einsum('nmef,kldc,efij,dblm,cakn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(f,e,i,j)*t2_abab(d,b,l,m)*t2_aaaa(c,a,k,n)
    double_res += -0.250000000000000 * einsum('nmfe,kldc,feij,dblm,cakn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,f,i,j)*t2_abab(d,b,m,l)*t2_abab(a,c,k,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,efij,dbml,ackn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(f,e,i,j)*t2_abab(d,b,m,l)*t2_abab(a,c,k,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,feij,dbml,ackn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,f,i,j)*t2_bbbb(d,b,l,m)*t2_aaaa(c,a,k,n)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,efij,dblm,cakn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(f,e,i,j)*t2_bbbb(d,b,l,m)*t2_aaaa(c,a,k,n)
    double_res +=  0.250000000000000 * einsum('nmfe,klcd,feij,dblm,cakn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,f,i,j)*t2_abab(d,b,l,m)*t2_abab(a,c,n,k)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,efij,dblm,acnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(f,e,i,j)*t2_abab(d,b,l,m)*t2_abab(a,c,n,k)
    double_res +=  0.250000000000000 * einsum('nmfe,lkdc,feij,dblm,acnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,f,i,j)*t2_bbbb(d,b,l,m)*t2_abab(a,c,n,k)
    double_res += -0.250000000000000 * einsum('nmef,kldc,efij,dblm,acnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(f,e,i,j)*t2_bbbb(d,b,l,m)*t2_abab(a,c,n,k)
    double_res += -0.250000000000000 * einsum('nmfe,kldc,feij,dblm,acnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_abab(c,f,k,j)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('mnef,kldc,edlm,cfkj,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_abab(c,f,k,j)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,delm,cfkj,abin->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,l)*t2_abab(f,c,k,j)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,deml,fckj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_abab(c,f,k,j)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('mnef,klcd,edml,cfkj,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_abab(c,f,k,j)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('nmef,klcd,edlm,cfkj,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_bbbb(f,c,j,k)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnef,lkdc,edlm,fcjk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_bbbb(f,c,j,k)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,delm,fcjk,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_bbbb(f,c,j,k)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnef,kldc,edml,fcjk,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_bbbb(f,c,j,k)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edlm,fcjk,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_aaaa(f,c,i,k)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edlm,fcik,abnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_aaaa(f,c,i,k)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmfe,kldc,delm,fcik,abnj->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_aaaa(f,c,i,k)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,edml,fcik,abnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_aaaa(f,c,i,k)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmfe,klcd,edlm,fcik,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_abab(f,c,i,k)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkdc,edlm,fcik,abnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_abab(f,c,i,k)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,lkdc,delm,fcik,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,m)*t2_abab(c,f,i,k)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,edlm,cfik,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_abab(f,c,i,k)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edml,fcik,abnj->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_abab(f,c,i,k)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,edlm,fcik,abnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_abab(c,f,i,j)*t2_abab(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,kldc,edlm,cfij,abkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_abab(c,f,i,j)*t2_abab(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,delm,cfij,abkn->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,l)*t2_abab(f,c,i,j)*t2_abab(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnfe,kldc,deml,fcij,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_abab(c,f,i,j)*t2_abab(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,klcd,edml,cfij,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_abab(c,f,i,j)*t2_abab(a,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,klcd,edlm,cfij,abkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_abab(f,c,i,j)*t2_abab(a,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,lkdc,edlm,fcij,abnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_abab(f,c,i,j)*t2_abab(a,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmfe,lkdc,delm,fcij,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,m)*t2_abab(c,f,i,j)*t2_abab(a,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,lkcd,edlm,cfij,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_abab(f,c,i,j)*t2_abab(a,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edml,fcij,abnk->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_abab(f,c,i,j)*t2_abab(a,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmfe,kldc,edlm,fcij,abnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_abab(a,f,k,j)*t2_abab(c,b,i,n)
    double_res +=  0.750000000000000 * einsum('mnef,kldc,edlm,afkj,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_abab(a,f,k,j)*t2_abab(c,b,i,n)
    double_res +=  0.750000000000000 * einsum('nmef,kldc,delm,afkj,cbin->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_abab(a,f,k,j)*t2_abab(c,b,i,n)
    double_res +=  0.750000000000000 * einsum('mnef,klcd,edml,afkj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_abab(a,f,k,j)*t2_abab(c,b,i,n)
    double_res +=  0.750000000000000 * einsum('nmef,klcd,edlm,afkj,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_abab(f,b,k,j)*t2_aaaa(c,a,i,n)
    double_res += -0.750000000000000 * einsum('nmef,kldc,edlm,fbkj,cain->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_abab(f,b,k,j)*t2_aaaa(c,a,i,n)
    double_res += -0.750000000000000 * einsum('nmfe,kldc,delm,fbkj,cain->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,l)*t2_abab(f,b,k,j)*t2_abab(a,c,i,n)
    double_res +=  0.750000000000000 * einsum('mnfe,kldc,deml,fbkj,acin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_abab(f,b,k,j)*t2_aaaa(c,a,i,n)
    double_res += -0.750000000000000 * einsum('nmef,klcd,edml,fbkj,cain->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_abab(f,b,k,j)*t2_aaaa(c,a,i,n)
    double_res += -0.750000000000000 * einsum('nmfe,klcd,edlm,fbkj,cain->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_bbbb(f,b,j,k)*t2_abab(a,c,i,n)
    double_res += -0.750000000000000 * einsum('mnef,lkdc,edlm,fbjk,acin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_bbbb(f,b,j,k)*t2_abab(a,c,i,n)
    double_res += -0.750000000000000 * einsum('nmef,lkdc,delm,fbjk,acin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,m)*t2_bbbb(f,b,j,k)*t2_aaaa(c,a,i,n)
    double_res +=  0.750000000000000 * einsum('nmef,lkcd,edlm,fbjk,cain->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_bbbb(f,b,j,k)*t2_abab(a,c,i,n)
    double_res += -0.750000000000000 * einsum('mnef,kldc,edml,fbjk,acin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_bbbb(f,b,j,k)*t2_abab(a,c,i,n)
    double_res += -0.750000000000000 * einsum('nmef,kldc,edlm,fbjk,acin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_aaaa(f,a,i,k)*t2_abab(c,b,n,j)
    double_res += -0.750000000000000 * einsum('nmef,kldc,edlm,faik,cbnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_aaaa(f,a,i,k)*t2_abab(c,b,n,j)
    double_res += -0.750000000000000 * einsum('nmfe,kldc,delm,faik,cbnj->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,l)*t2_aaaa(f,a,i,k)*t2_bbbb(c,b,j,n)
    double_res +=  0.750000000000000 * einsum('mnfe,kldc,deml,faik,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_aaaa(f,a,i,k)*t2_abab(c,b,n,j)
    double_res += -0.750000000000000 * einsum('nmef,klcd,edml,faik,cbnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_aaaa(f,a,i,k)*t2_abab(c,b,n,j)
    double_res += -0.750000000000000 * einsum('nmfe,klcd,edlm,faik,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_abab(a,f,i,k)*t2_bbbb(c,b,j,n)
    double_res += -0.750000000000000 * einsum('mnef,lkdc,edlm,afik,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_abab(a,f,i,k)*t2_bbbb(c,b,j,n)
    double_res += -0.750000000000000 * einsum('nmef,lkdc,delm,afik,cbjn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,m)*t2_abab(a,f,i,k)*t2_abab(c,b,n,j)
    double_res +=  0.750000000000000 * einsum('nmef,lkcd,edlm,afik,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_abab(a,f,i,k)*t2_bbbb(c,b,j,n)
    double_res += -0.750000000000000 * einsum('mnef,kldc,edml,afik,cbjn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.7500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_abab(a,f,i,k)*t2_bbbb(c,b,j,n)
    double_res += -0.750000000000000 * einsum('nmef,kldc,edlm,afik,cbjn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_abab(f,b,i,k)*t2_abab(a,c,n,j)
    double_res +=  0.750000000000000 * einsum('nmef,lkdc,edlm,fbik,acnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_abab(f,b,i,k)*t2_abab(a,c,n,j)
    double_res +=  0.750000000000000 * einsum('nmfe,lkdc,delm,fbik,acnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_abab(f,b,i,k)*t2_abab(a,c,n,j)
    double_res +=  0.750000000000000 * einsum('nmef,kldc,edml,fbik,acnj->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	  0.7500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_abab(f,b,i,k)*t2_abab(a,c,n,j)
    double_res +=  0.750000000000000 * einsum('nmfe,kldc,edlm,fbik,acnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_abab(a,f,i,j)*t2_abab(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,kldc,edlm,afij,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_abab(a,f,i,j)*t2_abab(c,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,kldc,delm,afij,cbkn->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_abab(a,f,i,j)*t2_abab(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,klcd,edml,afij,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_abab(a,f,i,j)*t2_abab(c,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,klcd,edlm,afij,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_abab(a,f,i,j)*t2_bbbb(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,lkdc,edlm,afij,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_abab(a,f,i,j)*t2_bbbb(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,lkdc,delm,afij,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,m)*t2_abab(a,f,i,j)*t2_abab(c,b,n,k)
    double_res += -1.000000000000000 * einsum('nmef,lkcd,edlm,afij,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_abab(a,f,i,j)*t2_bbbb(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,kldc,edml,afij,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_abab(a,f,i,j)*t2_bbbb(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edlm,afij,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,l,m)*t2_abab(f,b,i,j)*t2_aaaa(c,a,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edlm,fbij,cakn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,m)*t2_abab(f,b,i,j)*t2_aaaa(c,a,k,n)
    double_res +=  1.000000000000000 * einsum('nmfe,kldc,delm,fbij,cakn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,l)*t2_abab(f,b,i,j)*t2_abab(a,c,k,n)
    double_res += -1.000000000000000 * einsum('mnfe,kldc,deml,fbij,ackn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,l)*t2_abab(f,b,i,j)*t2_aaaa(c,a,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,klcd,edml,fbij,cakn->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,l,m)*t2_abab(f,b,i,j)*t2_aaaa(c,a,k,n)
    double_res +=  1.000000000000000 * einsum('nmfe,klcd,edlm,fbij,cakn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,l,m)*t2_abab(f,b,i,j)*t2_abab(a,c,n,k)
    double_res += -1.000000000000000 * einsum('nmef,lkdc,edlm,fbij,acnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,m)*t2_abab(f,b,i,j)*t2_abab(a,c,n,k)
    double_res += -1.000000000000000 * einsum('nmfe,lkdc,delm,fbij,acnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,l)*t2_abab(f,b,i,j)*t2_abab(a,c,n,k)
    double_res += -1.000000000000000 * einsum('nmef,kldc,edml,fbij,acnk->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,l,m)*t2_abab(f,b,i,j)*t2_abab(a,c,n,k)
    double_res += -1.000000000000000 * einsum('nmfe,kldc,edlm,fbij,acnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,m,j)*t2_aaaa(f,c,k,l)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,demj,fckl,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,j)*t2_abab(f,c,k,l)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,demj,fckl,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,j)*t2_abab(c,f,k,l)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnef,klcd,edmj,cfkl,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,m)*t2_abab(c,f,k,l)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,edjm,cfkl,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,m,j)*t2_abab(f,c,l,k)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnfe,lkdc,demj,fclk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,m,j)*t2_abab(c,f,l,k)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnef,lkcd,edmj,cflk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,m)*t2_abab(c,f,l,k)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,edjm,cflk,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,j)*t2_bbbb(f,c,k,l)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnef,kldc,edmj,fckl,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,m)*t2_bbbb(f,c,k,l)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edjm,fckl,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_aaaa(f,c,k,l)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edim,fckl,abnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,m)*t2_aaaa(f,c,k,l)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmfe,kldc,deim,fckl,abnj->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_abab(f,c,k,l)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edim,fckl,abnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,m)*t2_abab(f,c,k,l)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmfe,kldc,deim,fckl,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,m)*t2_abab(c,f,k,l)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,edim,cfkl,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,m)*t2_abab(f,c,l,k)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,edim,fclk,abnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,m)*t2_abab(f,c,l,k)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmfe,lkdc,deim,fclk,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,i,m)*t2_abab(c,f,l,k)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,edim,cflk,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,m)*t2_bbbb(f,c,k,l)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edim,fckl,abnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,m,j)*t2_aaaa(f,c,i,l)*t2_abab(a,b,k,n)
    double_res += -1.000000000000000 * einsum('mnfe,kldc,demj,fcil,abkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,j)*t2_abab(f,c,i,l)*t2_abab(a,b,k,n)
    double_res += -1.000000000000000 * einsum('mnfe,kldc,demj,fcil,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,j)*t2_abab(c,f,i,l)*t2_abab(a,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,klcd,edmj,cfil,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,m)*t2_abab(c,f,i,l)*t2_abab(a,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,klcd,edjm,cfil,abkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,m,j)*t2_aaaa(f,c,i,l)*t2_abab(a,b,n,k)
    double_res += -1.000000000000000 * einsum('nmef,lkcd,edmj,fcil,abnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,m)*t2_aaaa(f,c,i,l)*t2_abab(a,b,n,k)
    double_res += -1.000000000000000 * einsum('nmfe,lkcd,edjm,fcil,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,j)*t2_abab(f,c,i,l)*t2_abab(a,b,n,k)
    double_res += -1.000000000000000 * einsum('nmef,kldc,edmj,fcil,abnk->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,m)*t2_abab(f,c,i,l)*t2_abab(a,b,n,k)
    double_res += -1.000000000000000 * einsum('nmfe,kldc,edjm,fcil,abnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_abab(c,f,l,j)*t2_abab(a,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,kldc,edim,cflj,abkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,m)*t2_abab(c,f,l,j)*t2_abab(a,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,kldc,deim,cflj,abkn->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_bbbb(f,c,j,l)*t2_abab(a,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,kldc,edim,fcjl,abkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,m)*t2_bbbb(f,c,j,l)*t2_abab(a,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,kldc,deim,fcjl,abkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,m)*t2_abab(f,c,l,j)*t2_abab(a,b,n,k)
    double_res += -1.000000000000000 * einsum('nmef,lkdc,edim,fclj,abnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,m)*t2_abab(f,c,l,j)*t2_abab(a,b,n,k)
    double_res += -1.000000000000000 * einsum('nmfe,lkdc,deim,fclj,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,i,m)*t2_abab(c,f,l,j)*t2_abab(a,b,n,k)
    double_res += -1.000000000000000 * einsum('nmef,lkcd,edim,cflj,abnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,m)*t2_bbbb(f,c,j,l)*t2_abab(a,b,n,k)
    double_res += -1.000000000000000 * einsum('nmef,kldc,edim,fcjl,abnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,m,j)*t2_aaaa(f,a,k,l)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,demj,fakl,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,j)*t2_abab(a,f,k,l)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('mnef,klcd,edmj,afkl,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,m)*t2_abab(a,f,k,l)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('nmef,klcd,edjm,afkl,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,m,j)*t2_abab(a,f,l,k)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('mnef,lkcd,edmj,aflk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,m)*t2_abab(a,f,l,k)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,edjm,aflk,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,j)*t2_abab(f,b,k,l)*t2_abab(a,c,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,demj,fbkl,acin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,j)*t2_abab(f,b,k,l)*t2_aaaa(c,a,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,edmj,fbkl,cain->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,m)*t2_abab(f,b,k,l)*t2_aaaa(c,a,i,n)
    double_res +=  0.500000000000000 * einsum('nmfe,klcd,edjm,fbkl,cain->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,m,j)*t2_abab(f,b,l,k)*t2_abab(a,c,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,lkdc,demj,fblk,acin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,m,j)*t2_abab(f,b,l,k)*t2_aaaa(c,a,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,edmj,fblk,cain->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,m)*t2_abab(f,b,l,k)*t2_aaaa(c,a,i,n)
    double_res +=  0.500000000000000 * einsum('nmfe,lkcd,edjm,fblk,cain->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,j)*t2_bbbb(f,b,k,l)*t2_abab(a,c,i,n)
    double_res += -0.500000000000000 * einsum('mnef,kldc,edmj,fbkl,acin->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,m)*t2_bbbb(f,b,k,l)*t2_abab(a,c,i,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edjm,fbkl,acin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_aaaa(f,a,k,l)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edim,fakl,cbnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,m)*t2_aaaa(f,a,k,l)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,deim,fakl,cbnj->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_abab(a,f,k,l)*t2_bbbb(c,b,j,n)
    double_res +=  0.500000000000000 * einsum('mnef,kldc,edim,afkl,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,m)*t2_abab(a,f,k,l)*t2_bbbb(c,b,j,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,deim,afkl,cbjn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,m)*t2_abab(a,f,k,l)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,klcd,edim,afkl,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,m)*t2_abab(a,f,l,k)*t2_bbbb(c,b,j,n)
    double_res +=  0.500000000000000 * einsum('mnef,lkdc,edim,aflk,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,i,m)*t2_abab(a,f,l,k)*t2_bbbb(c,b,j,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,deim,aflk,cbjn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,i,m)*t2_abab(a,f,l,k)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,edim,aflk,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_abab(f,b,k,l)*t2_abab(a,c,n,j)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edim,fbkl,acnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,m)*t2_abab(f,b,k,l)*t2_abab(a,c,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,deim,fbkl,acnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,m)*t2_abab(f,b,l,k)*t2_abab(a,c,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkdc,edim,fblk,acnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,m)*t2_abab(f,b,l,k)*t2_abab(a,c,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,lkdc,deim,fblk,acnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,m)*t2_bbbb(f,b,k,l)*t2_abab(a,c,n,j)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edim,fbkl,acnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,m,j)*t2_aaaa(f,a,i,l)*t2_abab(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnfe,kldc,demj,fail,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,m,j)*t2_abab(a,f,i,l)*t2_abab(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,klcd,edmj,afil,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,m)*t2_abab(a,f,i,l)*t2_abab(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,klcd,edjm,afil,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,m,j)*t2_aaaa(f,a,i,l)*t2_bbbb(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnfe,lkdc,demj,fail,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,m,j)*t2_aaaa(f,a,i,l)*t2_abab(c,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,lkcd,edmj,fail,cbnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,m)*t2_aaaa(f,a,i,l)*t2_abab(c,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmfe,lkcd,edjm,fail,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,j)*t2_abab(a,f,i,l)*t2_bbbb(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,kldc,edmj,afil,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,m)*t2_abab(a,f,i,l)*t2_bbbb(c,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,kldc,edjm,afil,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,m,j)*t2_abab(f,b,i,l)*t2_abab(a,c,k,n)
    double_res +=  1.000000000000000 * einsum('mnfe,kldc,demj,fbil,ackn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,m,j)*t2_abab(f,b,i,l)*t2_aaaa(c,a,k,n)
    double_res += -1.000000000000000 * einsum('nmef,klcd,edmj,fbil,cakn->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,m)*t2_abab(f,b,i,l)*t2_aaaa(c,a,k,n)
    double_res += -1.000000000000000 * einsum('nmfe,klcd,edjm,fbil,cakn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,m,j)*t2_abab(f,b,i,l)*t2_abab(a,c,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edmj,fbil,acnk->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,m)*t2_abab(f,b,i,l)*t2_abab(a,c,n,k)
    double_res +=  1.000000000000000 * einsum('nmfe,kldc,edjm,fbil,acnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_abab(a,f,l,j)*t2_abab(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,kldc,edim,aflj,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,m)*t2_abab(a,f,l,j)*t2_abab(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,deim,aflj,cbkn->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,m)*t2_abab(a,f,l,j)*t2_bbbb(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,lkdc,edim,aflj,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,i,m)*t2_abab(a,f,l,j)*t2_bbbb(c,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,lkdc,deim,aflj,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,i,m)*t2_abab(a,f,l,j)*t2_abab(c,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,lkcd,edim,aflj,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_abab(f,b,l,j)*t2_aaaa(c,a,k,n)
    double_res += -1.000000000000000 * einsum('nmef,kldc,edim,fblj,cakn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,m)*t2_abab(f,b,l,j)*t2_aaaa(c,a,k,n)
    double_res += -1.000000000000000 * einsum('nmfe,kldc,deim,fblj,cakn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,m)*t2_bbbb(f,b,j,l)*t2_abab(a,c,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,kldc,edim,fbjl,ackn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,m)*t2_bbbb(f,b,j,l)*t2_abab(a,c,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,deim,fbjl,ackn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,m)*t2_bbbb(f,b,j,l)*t2_aaaa(c,a,k,n)
    double_res += -1.000000000000000 * einsum('nmef,klcd,edim,fbjl,cakn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,m)*t2_abab(f,b,l,j)*t2_abab(a,c,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,lkdc,edim,fblj,acnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,m)*t2_abab(f,b,l,j)*t2_abab(a,c,n,k)
    double_res +=  1.000000000000000 * einsum('nmfe,lkdc,deim,fblj,acnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,m)*t2_bbbb(f,b,j,l)*t2_abab(a,c,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edim,fbjl,acnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,f,i,j)*t2_abab(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,edkl,cfij,abnm->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,f,i,j)*t2_abab(a,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,edkl,cfij,abmn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,c,i,j)*t2_abab(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmfe,kldc,dekl,fcij,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,c,i,j)*t2_abab(a,b,m,n)
    double_res += -0.250000000000000 * einsum('mnfe,kldc,dekl,fcij,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,f,i,j)*t2_abab(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmef,klcd,edkl,cfij,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,f,i,j)*t2_abab(a,b,m,n)
    double_res += -0.250000000000000 * einsum('mnef,klcd,edkl,cfij,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,c,i,j)*t2_abab(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmfe,lkdc,delk,fcij,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,c,i,j)*t2_abab(a,b,m,n)
    double_res += -0.250000000000000 * einsum('mnfe,lkdc,delk,fcij,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,f,i,j)*t2_abab(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmef,lkcd,edlk,cfij,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,f,i,j)*t2_abab(a,b,m,n)
    double_res += -0.250000000000000 * einsum('mnef,lkcd,edlk,cfij,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,c,i,j)*t2_abab(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,edkl,fcij,abnm->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,c,i,j)*t2_abab(a,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,edkl,fcij,abmn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(a,f,m,j)*t2_abab(c,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnef,kldc,edkl,afmj,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(a,f,m,j)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('mnef,klcd,edkl,afmj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(a,f,m,j)*t2_abab(c,b,i,n)
    double_res += -0.500000000000000 * einsum('mnef,lkcd,edlk,afmj,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(f,b,m,j)*t2_aaaa(c,a,i,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edkl,fbmj,cain->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_bbbb(f,b,j,m)*t2_aaaa(c,a,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edkl,fbjm,cain->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,b,m,j)*t2_abab(a,c,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,dekl,fbmj,acin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(f,b,j,m)*t2_abab(a,c,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,dekl,fbjm,acin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(f,b,m,j)*t2_aaaa(c,a,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,edkl,fbmj,cain->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_bbbb(f,b,j,m)*t2_aaaa(c,a,i,n)
    double_res += -0.500000000000000 * einsum('nmef,klcd,edkl,fbjm,cain->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,b,m,j)*t2_abab(a,c,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,lkdc,delk,fbmj,acin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(f,b,j,m)*t2_abab(a,c,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,delk,fbjm,acin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(f,b,m,j)*t2_aaaa(c,a,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,edlk,fbmj,cain->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_bbbb(f,b,j,m)*t2_aaaa(c,a,i,n)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,edlk,fbjm,cain->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,b,m,j)*t2_abab(a,c,i,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,edkl,fbmj,acin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(f,b,j,m)*t2_abab(a,c,i,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edkl,fbjm,acin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(f,a,i,m)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edkl,faim,cbnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(a,f,i,m)*t2_abab(c,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edkl,afim,cbnj->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_aaaa(f,a,i,m)*t2_bbbb(c,b,j,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,dekl,faim,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(a,f,i,m)*t2_bbbb(c,b,j,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,dekl,afim,cbjn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(f,a,i,m)*t2_abab(c,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,edkl,faim,cbnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(a,f,i,m)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,klcd,edkl,afim,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_aaaa(f,a,i,m)*t2_bbbb(c,b,j,n)
    double_res += -0.500000000000000 * einsum('mnfe,lkdc,delk,faim,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(a,f,i,m)*t2_bbbb(c,b,j,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,delk,afim,cbjn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(f,a,i,m)*t2_abab(c,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,edlk,faim,cbnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(a,f,i,m)*t2_abab(c,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,edlk,afim,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_aaaa(f,a,i,m)*t2_bbbb(c,b,j,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,edkl,faim,cbjn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(a,f,i,m)*t2_bbbb(c,b,j,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edkl,afim,cbjn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,b,i,m)*t2_abab(a,c,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,dekl,fbim,acnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,b,i,m)*t2_abab(a,c,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,lkdc,delk,fbim,acnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,b,i,m)*t2_abab(a,c,n,j)
    double_res +=  0.500000000000000 * einsum('nmfe,kldc,edkl,fbim,acnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(a,f,i,j)*t2_abab(c,b,n,m)
    double_res += -0.250000000000000 * einsum('nmef,kldc,edkl,afij,cbnm->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(a,f,i,j)*t2_abab(c,b,m,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,edkl,afij,cbmn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(a,f,i,j)*t2_bbbb(c,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,dekl,afij,cbnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(a,f,i,j)*t2_abab(c,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,edkl,afij,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(a,f,i,j)*t2_abab(c,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,klcd,edkl,afij,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(a,f,i,j)*t2_bbbb(c,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,delk,afij,cbnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(a,f,i,j)*t2_abab(c,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,edlk,afij,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(a,f,i,j)*t2_abab(c,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkcd,edlk,afij,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(a,f,i,j)*t2_bbbb(c,b,n,m)
    double_res += -0.250000000000000 * einsum('nmef,kldc,edkl,afij,cbnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(f,b,i,j)*t2_aaaa(c,a,n,m)
    double_res += -0.250000000000000 * einsum('nmef,kldc,edkl,fbij,canm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,b,i,j)*t2_abab(a,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,dekl,fbij,acnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,b,i,j)*t2_abab(a,c,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,dekl,fbij,acmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(f,b,i,j)*t2_aaaa(c,a,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,edkl,fbij,canm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,b,i,j)*t2_abab(a,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,lkdc,delk,fbij,acnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,b,i,j)*t2_abab(a,c,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,delk,fbij,acmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(f,b,i,j)*t2_aaaa(c,a,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,edlk,fbij,canm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,b,i,j)*t2_abab(a,c,n,m)
    double_res += -0.250000000000000 * einsum('nmfe,kldc,edkl,fbij,acnm->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,b,i,j)*t2_abab(a,c,m,n)
    double_res += -0.250000000000000 * einsum('mnfe,kldc,edkl,fbij,acmn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 2), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,c,k,m)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,delj,fckm,abin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(c,f,k,m)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,delj,cfkm,abin->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,c,k,m)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnfe,klcd,edjl,fckm,abin->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(c,f,k,m)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('nmef,klcd,edjl,cfkm,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,c,m,k)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,lkdc,delj,fcmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,c,k,m)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,delj,fckm,abin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(c,f,m,k)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('mnef,lkcd,edlj,cfmk,abin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,c,m,k)*t2_abab(a,b,i,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,edjl,fcmk,abin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(f,c,k,m)*t2_abab(a,b,i,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edjl,fckm,abin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_aaaa(f,c,k,m)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edil,fckm,abnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(c,f,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edil,cfkm,abnj->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(f,c,k,m)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,deil,fckm,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_aaaa(f,c,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,edil,fckm,abnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(c,f,k,m)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,klcd,edil,cfkm,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_abab(f,c,m,k)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkdc,edil,fcmk,abnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(f,c,k,m)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,edil,fckm,abnj->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_abab(f,c,m,k)*t2_abab(a,b,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edil,fcmk,abnj->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(f,c,k,m)*t2_abab(a,b,n,j)
    double_res += -0.500000000000000 * einsum('nmef,kldc,edil,fckm,abnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,c,i,k)*t2_abab(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmfe,kldc,delj,fcik,abnm->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,c,i,k)*t2_abab(a,b,m,n)
    double_res += -0.250000000000000 * einsum('mnfe,kldc,delj,fcik,abmn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,c,i,k)*t2_abab(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,klcd,edjl,fcik,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,c,i,k)*t2_abab(a,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,klcd,edjl,fcik,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,c,i,k)*t2_abab(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,lkdc,delj,fcik,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,c,i,k)*t2_abab(a,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,delj,fcik,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(c,f,i,k)*t2_abab(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,edlj,cfik,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(c,f,i,k)*t2_abab(a,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkcd,edlj,cfik,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,c,i,k)*t2_abab(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmfe,kldc,edjl,fcik,abnm->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,c,i,k)*t2_abab(a,b,m,n)
    double_res += -0.250000000000000 * einsum('mnfe,kldc,edjl,fcik,abmn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(c,f,k,j)*t2_abab(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmef,kldc,edil,cfkj,abnm->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(c,f,k,j)*t2_abab(a,b,m,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,edil,cfkj,abmn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(f,c,k,j)*t2_abab(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,deil,fckj,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(f,c,k,j)*t2_abab(a,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,deil,fckj,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(c,f,k,j)*t2_abab(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,edil,cfkj,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(c,f,k,j)*t2_abab(a,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,klcd,edil,cfkj,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(f,c,j,k)*t2_abab(a,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,edil,fcjk,abnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(f,c,j,k)*t2_abab(a,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkdc,edil,fcjk,abmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(f,c,j,k)*t2_abab(a,b,n,m)
    double_res += -0.250000000000000 * einsum('nmef,kldc,edil,fcjk,abnm->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(f,c,j,k)*t2_abab(a,b,m,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,edil,fcjk,abmn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (0, 2), (0, 1)])
    
    #	  0.9167 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,a,k,m)*t2_abab(c,b,i,n)
    double_res +=  0.916666666666670 * einsum('mnfe,kldc,delj,fakm,cbin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(a,f,k,m)*t2_abab(c,b,i,n)
    double_res += -0.916666666666670 * einsum('nmef,kldc,delj,afkm,cbin->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,a,k,m)*t2_abab(c,b,i,n)
    double_res += -0.916666666666670 * einsum('mnfe,klcd,edjl,fakm,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(a,f,k,m)*t2_abab(c,b,i,n)
    double_res +=  0.916666666666670 * einsum('nmef,klcd,edjl,afkm,cbin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(a,f,m,k)*t2_abab(c,b,i,n)
    double_res +=  0.916666666666670 * einsum('mnef,lkcd,edlj,afmk,cbin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(f,b,k,m)*t2_aaaa(c,a,i,n)
    double_res +=  0.916666666666670 * einsum('nmfe,kldc,delj,fbkm,cain->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(f,b,k,m)*t2_aaaa(c,a,i,n)
    double_res += -0.916666666666670 * einsum('nmfe,klcd,edjl,fbkm,cain->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,b,m,k)*t2_abab(a,c,i,n)
    double_res +=  0.916666666666670 * einsum('mnfe,lkdc,delj,fbmk,acin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,b,k,m)*t2_abab(a,c,i,n)
    double_res += -0.916666666666670 * einsum('nmef,lkdc,delj,fbkm,acin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(f,b,m,k)*t2_aaaa(c,a,i,n)
    double_res += -0.916666666666670 * einsum('nmef,lkcd,edlj,fbmk,cain->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_bbbb(f,b,k,m)*t2_aaaa(c,a,i,n)
    double_res +=  0.916666666666670 * einsum('nmef,lkcd,edlj,fbkm,cain->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,b,m,k)*t2_abab(a,c,i,n)
    double_res += -0.916666666666670 * einsum('mnfe,kldc,edjl,fbmk,acin->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(f,b,k,m)*t2_abab(a,c,i,n)
    double_res +=  0.916666666666670 * einsum('nmef,kldc,edjl,fbkm,acin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_aaaa(f,a,k,m)*t2_abab(c,b,n,j)
    double_res +=  0.916666666666670 * einsum('nmef,kldc,edil,fakm,cbnj->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(a,f,k,m)*t2_abab(c,b,n,j)
    double_res += -0.916666666666670 * einsum('nmef,kldc,edil,afkm,cbnj->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_aaaa(f,a,k,m)*t2_bbbb(c,b,j,n)
    double_res +=  0.916666666666670 * einsum('mnfe,kldc,deil,fakm,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(a,f,k,m)*t2_bbbb(c,b,j,n)
    double_res += -0.916666666666670 * einsum('nmef,kldc,deil,afkm,cbjn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_aaaa(f,a,k,m)*t2_abab(c,b,n,j)
    double_res += -0.916666666666670 * einsum('nmef,klcd,edil,fakm,cbnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(a,f,k,m)*t2_abab(c,b,n,j)
    double_res +=  0.916666666666670 * einsum('nmef,klcd,edil,afkm,cbnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_abab(a,f,m,k)*t2_bbbb(c,b,j,n)
    double_res += -0.916666666666670 * einsum('mnef,lkdc,edil,afmk,cbjn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_abab(a,f,m,k)*t2_bbbb(c,b,j,n)
    double_res +=  0.916666666666670 * einsum('mnef,kldc,edil,afmk,cbjn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(f,b,k,m)*t2_abab(a,c,n,j)
    double_res +=  0.916666666666670 * einsum('nmfe,kldc,deil,fbkm,acnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_abab(f,b,m,k)*t2_abab(a,c,n,j)
    double_res +=  0.916666666666670 * einsum('nmef,lkdc,edil,fbmk,acnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(f,b,k,m)*t2_abab(a,c,n,j)
    double_res += -0.916666666666670 * einsum('nmef,lkdc,edil,fbkm,acnj->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.9167 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_abab(f,b,m,k)*t2_abab(a,c,n,j)
    double_res += -0.916666666666670 * einsum('nmef,kldc,edil,fbmk,acnj->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	  0.9167 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(f,b,k,m)*t2_abab(a,c,n,j)
    double_res +=  0.916666666666670 * einsum('nmef,kldc,edil,fbkm,acnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,a,i,m)*t2_abab(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnfe,kldc,delj,faim,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(a,f,i,m)*t2_abab(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,delj,afim,cbkn->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,a,i,m)*t2_abab(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnfe,klcd,edjl,faim,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(a,f,i,m)*t2_abab(c,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,klcd,edjl,afim,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,a,i,m)*t2_bbbb(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnfe,lkdc,delj,faim,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(a,f,i,m)*t2_bbbb(c,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,lkdc,delj,afim,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_aaaa(f,a,i,m)*t2_abab(c,b,n,k)
    double_res += -1.000000000000000 * einsum('nmef,lkcd,edlj,faim,cbnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(a,f,i,m)*t2_abab(c,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,lkcd,edlj,afim,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_aaaa(f,a,i,m)*t2_bbbb(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnfe,kldc,edjl,faim,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(a,f,i,m)*t2_bbbb(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edjl,afim,cbkn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(f,b,i,m)*t2_aaaa(c,a,k,n)
    double_res += -1.000000000000000 * einsum('nmfe,kldc,delj,fbim,cakn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(f,b,i,m)*t2_aaaa(c,a,k,n)
    double_res +=  1.000000000000000 * einsum('nmfe,klcd,edjl,fbim,cakn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,b,i,m)*t2_abab(a,c,n,k)
    double_res +=  1.000000000000000 * einsum('nmfe,lkdc,delj,fbim,acnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,b,i,m)*t2_abab(a,c,n,k)
    double_res += -1.000000000000000 * einsum('nmfe,kldc,edjl,fbim,acnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(a,f,m,j)*t2_abab(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,kldc,edil,afmj,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(a,f,m,j)*t2_abab(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,klcd,edil,afmj,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_abab(a,f,m,j)*t2_bbbb(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,lkdc,edil,afmj,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_abab(a,f,m,j)*t2_bbbb(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,kldc,edil,afmj,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(f,b,m,j)*t2_aaaa(c,a,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edil,fbmj,cakn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(f,b,j,m)*t2_aaaa(c,a,k,n)
    double_res += -1.000000000000000 * einsum('nmef,kldc,edil,fbjm,cakn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(f,b,m,j)*t2_abab(a,c,k,n)
    double_res +=  1.000000000000000 * einsum('mnfe,kldc,deil,fbmj,ackn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_bbbb(f,b,j,m)*t2_abab(a,c,k,n)
    double_res += -1.000000000000000 * einsum('nmef,kldc,deil,fbjm,ackn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(f,b,m,j)*t2_aaaa(c,a,k,n)
    double_res += -1.000000000000000 * einsum('nmef,klcd,edil,fbmj,cakn->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_bbbb(f,b,j,m)*t2_aaaa(c,a,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,klcd,edil,fbjm,cakn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_abab(f,b,m,j)*t2_abab(a,c,n,k)
    double_res += -1.000000000000000 * einsum('nmef,lkdc,edil,fbmj,acnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(f,b,j,m)*t2_abab(a,c,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,lkdc,edil,fbjm,acnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_abab(f,b,m,j)*t2_abab(a,c,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edil,fbmj,acnk->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(f,b,j,m)*t2_abab(a,c,n,k)
    double_res += -1.000000000000000 * einsum('nmef,kldc,edil,fbjm,acnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,a,i,k)*t2_abab(c,b,n,m)
    double_res +=  0.500000000000000 * einsum('nmfe,kldc,delj,faik,cbnm->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,a,i,k)*t2_abab(c,b,m,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,delj,faik,cbmn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,a,i,k)*t2_abab(c,b,n,m)
    double_res += -0.500000000000000 * einsum('nmfe,klcd,edjl,faik,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,a,i,k)*t2_abab(c,b,m,n)
    double_res += -0.500000000000000 * einsum('mnfe,klcd,edjl,faik,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(a,f,i,k)*t2_bbbb(c,b,n,m)
    double_res += -0.500000000000000 * einsum('nmef,lkdc,delj,afik,cbnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(a,f,i,k)*t2_abab(c,b,n,m)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,edlj,afik,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(a,f,i,k)*t2_abab(c,b,m,n)
    double_res += -0.500000000000000 * einsum('mnef,lkcd,edlj,afik,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(a,f,i,k)*t2_bbbb(c,b,n,m)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edjl,afik,cbnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,b,i,k)*t2_abab(a,c,n,m)
    double_res += -0.500000000000000 * einsum('nmfe,lkdc,delj,fbik,acnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,b,i,k)*t2_abab(a,c,m,n)
    double_res += -0.500000000000000 * einsum('mnfe,lkdc,delj,fbik,acmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(f,b,i,k)*t2_aaaa(c,a,n,m)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,edlj,fbik,canm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,b,i,k)*t2_abab(a,c,n,m)
    double_res +=  0.500000000000000 * einsum('nmfe,kldc,edjl,fbik,acnm->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,b,i,k)*t2_abab(a,c,m,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,edjl,fbik,acmn->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(a,f,k,j)*t2_abab(c,b,n,m)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edil,afkj,cbnm->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(a,f,k,j)*t2_abab(c,b,m,n)
    double_res +=  0.500000000000000 * einsum('mnef,kldc,edil,afkj,cbmn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(a,f,k,j)*t2_bbbb(c,b,n,m)
    double_res += -0.500000000000000 * einsum('nmef,kldc,deil,afkj,cbnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(a,f,k,j)*t2_abab(c,b,n,m)
    double_res += -0.500000000000000 * einsum('nmef,klcd,edil,afkj,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(a,f,k,j)*t2_abab(c,b,m,n)
    double_res += -0.500000000000000 * einsum('mnef,klcd,edil,afkj,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(f,b,k,j)*t2_aaaa(c,a,n,m)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edil,fbkj,canm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(f,b,k,j)*t2_abab(a,c,n,m)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,deil,fbkj,acnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(f,b,k,j)*t2_abab(a,c,m,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,deil,fbkj,acmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(f,b,k,j)*t2_aaaa(c,a,n,m)
    double_res += -0.500000000000000 * einsum('nmef,klcd,edil,fbkj,canm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(f,b,j,k)*t2_abab(a,c,n,m)
    double_res += -0.500000000000000 * einsum('nmef,lkdc,edil,fbjk,acnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(f,b,j,k)*t2_abab(a,c,m,n)
    double_res += -0.500000000000000 * einsum('mnef,lkdc,edil,fbjk,acmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(f,b,j,k)*t2_abab(a,c,n,m)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,edil,fbjk,acnm->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(f,b,j,k)*t2_abab(a,c,m,n)
    double_res +=  0.500000000000000 * einsum('mnef,kldc,edil,fbjk,acmn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,j)*t2_aaaa(f,a,l,m)*t2_abab(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnfe,kldc,deij,falm,cbkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(a,f,l,m)*t2_abab(c,b,k,n)
    double_res += -1.000000000000000 * einsum('nmef,kldc,deij,aflm,cbkn->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_abab(a,f,m,l)*t2_abab(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnef,klcd,edij,afml,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_aaaa(f,a,l,m)*t2_bbbb(c,b,k,n)
    double_res += -1.000000000000000 * einsum('mnfe,lkdc,deij,falm,cbkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_abab(a,f,l,m)*t2_bbbb(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,lkdc,deij,aflm,cbkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,i,j)*t2_aaaa(f,a,l,m)*t2_abab(c,b,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,lkcd,edij,falm,cbnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,i,j)*t2_abab(a,f,l,m)*t2_abab(c,b,n,k)
    double_res += -1.000000000000000 * einsum('nmef,lkcd,edij,aflm,cbnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,j)*t2_abab(a,f,m,l)*t2_bbbb(c,b,k,n)
    double_res +=  1.000000000000000 * einsum('mnef,kldc,edij,afml,cbkn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(f,b,l,m)*t2_aaaa(c,a,k,n)
    double_res +=  1.000000000000000 * einsum('nmfe,kldc,deij,fblm,cakn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(f,b,m,l)*t2_abab(a,c,k,n)
    double_res += -1.000000000000000 * einsum('mnfe,kldc,deij,fbml,ackn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,j)*t2_bbbb(f,b,l,m)*t2_abab(a,c,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,deij,fblm,ackn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_abab(f,b,m,l)*t2_aaaa(c,a,k,n)
    double_res +=  1.000000000000000 * einsum('nmef,klcd,edij,fbml,cakn->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_bbbb(f,b,l,m)*t2_aaaa(c,a,k,n)
    double_res += -1.000000000000000 * einsum('nmef,klcd,edij,fblm,cakn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_abab(f,b,l,m)*t2_abab(a,c,n,k)
    double_res += -1.000000000000000 * einsum('nmfe,lkdc,deij,fblm,acnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,j)*t2_abab(f,b,m,l)*t2_abab(a,c,n,k)
    double_res += -1.000000000000000 * einsum('nmef,kldc,edij,fbml,acnk->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,j)*t2_bbbb(f,b,l,m)*t2_abab(a,c,n,k)
    double_res +=  1.000000000000000 * einsum('nmef,kldc,edij,fblm,acnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,j)*t2_aaaa(f,a,k,l)*t2_abab(c,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,deij,fakl,cbnm->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,j)*t2_aaaa(f,a,k,l)*t2_abab(c,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,deij,fakl,cbmn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(a,f,k,l)*t2_bbbb(c,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,deij,afkl,cbnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_abab(a,f,k,l)*t2_abab(c,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,edij,afkl,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_abab(a,f,k,l)*t2_abab(c,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,klcd,edij,afkl,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_abab(a,f,l,k)*t2_bbbb(c,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,deij,aflk,cbnm->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,i,j)*t2_abab(a,f,l,k)*t2_abab(c,b,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,edij,aflk,cbnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,d,i,j)*t2_abab(a,f,l,k)*t2_abab(c,b,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkcd,edij,aflk,cbmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(f,b,k,l)*t2_abab(a,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,deij,fbkl,acnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(f,b,k,l)*t2_abab(a,c,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,deij,fbkl,acmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_abab(f,b,k,l)*t2_aaaa(c,a,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,edij,fbkl,canm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_abab(f,b,l,k)*t2_abab(a,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,lkdc,deij,fblk,acnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_abab(f,b,l,k)*t2_abab(a,c,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,deij,fblk,acmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,d,i,j)*t2_abab(f,b,l,k)*t2_aaaa(c,a,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,edij,fblk,canm->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,j)*t2_bbbb(f,b,k,l)*t2_abab(a,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,edij,fbkl,acnm->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,j)*t2_bbbb(f,b,k,l)*t2_abab(a,c,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,edij,fbkl,acmn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,l,m)*t2_abab(f,b,k,j)*t2_aaaa(d,c,i,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,ealm,fbkj,dcin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,l,m)*t2_abab(f,b,k,j)*t2_aaaa(d,c,i,n)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,aelm,fbkj,dcin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,m,l)*t2_abab(f,b,k,j)*t2_abab(d,c,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,aeml,fbkj,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,m,l)*t2_abab(f,b,k,j)*t2_abab(c,d,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,klcd,aeml,fbkj,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,l,m)*t2_bbbb(f,b,j,k)*t2_abab(d,c,i,n)
    double_res +=  0.500000000000000 * einsum('mnef,lkdc,ealm,fbjk,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(a,e,l,m)*t2_bbbb(f,b,j,k)*t2_abab(d,c,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,aelm,fbjk,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,l,m)*t2_bbbb(f,b,j,k)*t2_abab(c,d,i,n)
    double_res +=  0.500000000000000 * einsum('mnef,lkcd,ealm,fbjk,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,l,m)*t2_bbbb(f,b,j,k)*t2_abab(c,d,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,aelm,fbjk,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,l,m)*t2_abab(f,b,i,k)*t2_abab(d,c,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkdc,ealm,fbik,dcnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,m)*t2_abab(f,b,i,k)*t2_abab(d,c,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,lkdc,aelm,fbik,dcnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,l,m)*t2_abab(f,b,i,k)*t2_abab(c,d,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,ealm,fbik,cdnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,m)*t2_abab(f,b,i,k)*t2_abab(c,d,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,lkcd,aelm,fbik,cdnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,m,l)*t2_abab(f,b,i,k)*t2_bbbb(d,c,j,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,aeml,fbik,dcjn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (1, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,l,m)*t2_abab(f,b,i,j)*t2_aaaa(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('nmef,kldc,ealm,fbij,dckn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,l,m)*t2_abab(f,b,i,j)*t2_aaaa(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('nmfe,kldc,aelm,fbij,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,m,l)*t2_abab(f,b,i,j)*t2_abab(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('mnfe,kldc,aeml,fbij,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,m,l)*t2_abab(f,b,i,j)*t2_abab(c,d,k,n)
    double_res +=  0.375000000000000 * einsum('mnfe,klcd,aeml,fbij,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,l,m)*t2_abab(f,b,i,j)*t2_abab(d,c,n,k)
    double_res +=  0.375000000000000 * einsum('nmef,lkdc,ealm,fbij,dcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,m)*t2_abab(f,b,i,j)*t2_abab(d,c,n,k)
    double_res +=  0.375000000000000 * einsum('nmfe,lkdc,aelm,fbij,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,l,m)*t2_abab(f,b,i,j)*t2_abab(c,d,n,k)
    double_res +=  0.375000000000000 * einsum('nmef,lkcd,ealm,fbij,cdnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,m)*t2_abab(f,b,i,j)*t2_abab(c,d,n,k)
    double_res +=  0.375000000000000 * einsum('nmfe,lkcd,aelm,fbij,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.3750 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,m,l)*t2_abab(f,b,i,j)*t2_bbbb(d,c,k,n)
    double_res +=  0.375000000000000 * einsum('mnfe,kldc,aeml,fbij,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,m,j)*t2_abab(f,b,k,l)*t2_abab(d,c,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,aemj,fbkl,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,m,j)*t2_abab(f,b,k,l)*t2_abab(c,d,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,klcd,aemj,fbkl,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,m,j)*t2_abab(f,b,l,k)*t2_abab(d,c,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,aemj,fblk,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,m,j)*t2_abab(f,b,l,k)*t2_abab(c,d,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkcd,aemj,fblk,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,m)*t2_abab(f,b,k,l)*t2_abab(d,c,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,eaim,fbkl,dcnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,m)*t2_abab(f,b,k,l)*t2_abab(d,c,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,aeim,fbkl,dcnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,m)*t2_abab(f,b,k,l)*t2_abab(c,d,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,eaim,fbkl,cdnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,m)*t2_abab(f,b,k,l)*t2_abab(c,d,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,klcd,aeim,fbkl,cdnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,m)*t2_abab(f,b,l,k)*t2_abab(d,c,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,eaim,fblk,dcnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,m)*t2_abab(f,b,l,k)*t2_abab(d,c,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,lkdc,aeim,fblk,dcnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,m)*t2_abab(f,b,l,k)*t2_abab(c,d,n,j)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,eaim,fblk,cdnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,m)*t2_abab(f,b,l,k)*t2_abab(c,d,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,lkcd,aeim,fblk,cdnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_aaaa(e,a,i,m)*t2_bbbb(f,b,k,l)*t2_bbbb(d,c,j,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,eaim,fbkl,dcjn->abij', g_abab[o, o, v, v], l2_bbbb, t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,m)*t2_bbbb(f,b,k,l)*t2_bbbb(d,c,j,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,aeim,fbkl,dcjn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.3750 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,m,j)*t2_abab(f,b,i,l)*t2_abab(d,c,k,n)
    double_res += -0.375000000000000 * einsum('mnfe,kldc,aemj,fbil,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,m,j)*t2_abab(f,b,i,l)*t2_abab(c,d,k,n)
    double_res += -0.375000000000000 * einsum('mnfe,klcd,aemj,fbil,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,m,j)*t2_abab(f,b,i,l)*t2_bbbb(d,c,k,n)
    double_res += -0.375000000000000 * einsum('mnfe,kldc,aemj,fbil,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,m)*t2_abab(f,b,l,j)*t2_aaaa(d,c,k,n)
    double_res += -0.375000000000000 * einsum('nmef,kldc,eaim,fblj,dckn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,i,m)*t2_abab(f,b,l,j)*t2_aaaa(d,c,k,n)
    double_res += -0.375000000000000 * einsum('nmfe,kldc,aeim,fblj,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,m)*t2_bbbb(f,b,j,l)*t2_abab(d,c,k,n)
    double_res += -0.375000000000000 * einsum('mnef,kldc,eaim,fbjl,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,i,m)*t2_bbbb(f,b,j,l)*t2_abab(d,c,k,n)
    double_res += -0.375000000000000 * einsum('nmef,kldc,aeim,fbjl,dckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,m)*t2_bbbb(f,b,j,l)*t2_abab(c,d,k,n)
    double_res += -0.375000000000000 * einsum('mnef,klcd,eaim,fbjl,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,i,m)*t2_bbbb(f,b,j,l)*t2_abab(c,d,k,n)
    double_res += -0.375000000000000 * einsum('nmef,klcd,aeim,fbjl,cdkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,m)*t2_abab(f,b,l,j)*t2_abab(d,c,n,k)
    double_res += -0.375000000000000 * einsum('nmef,lkdc,eaim,fblj,dcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,m)*t2_abab(f,b,l,j)*t2_abab(d,c,n,k)
    double_res += -0.375000000000000 * einsum('nmfe,lkdc,aeim,fblj,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,m)*t2_abab(f,b,l,j)*t2_abab(c,d,n,k)
    double_res += -0.375000000000000 * einsum('nmef,lkcd,eaim,fblj,cdnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,m)*t2_abab(f,b,l,j)*t2_abab(c,d,n,k)
    double_res += -0.375000000000000 * einsum('nmfe,lkcd,aeim,fblj,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_aaaa(e,a,i,m)*t2_bbbb(f,b,j,l)*t2_bbbb(d,c,k,n)
    double_res += -0.375000000000000 * einsum('mnef,kldc,eaim,fbjl,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_aaaa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3750 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,m)*t2_bbbb(f,b,j,l)*t2_bbbb(d,c,k,n)
    double_res += -0.375000000000000 * einsum('nmef,kldc,aeim,fbjl,dckn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_abab(f,b,m,j)*t2_aaaa(d,c,i,n)
    double_res += -0.250000000000000 * einsum('nmef,kldc,eakl,fbmj,dcin->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_bbbb(f,b,j,m)*t2_aaaa(d,c,i,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,eakl,fbjm,dcin->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(f,b,m,j)*t2_abab(d,c,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,aekl,fbmj,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_bbbb(f,b,j,m)*t2_abab(d,c,i,n)
    double_res += -0.250000000000000 * einsum('nmef,kldc,aekl,fbjm,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(f,b,m,j)*t2_abab(c,d,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,klcd,aekl,fbmj,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_bbbb(f,b,j,m)*t2_abab(c,d,i,n)
    double_res += -0.250000000000000 * einsum('nmef,klcd,aekl,fbjm,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(f,b,m,j)*t2_abab(d,c,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,aelk,fbmj,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_bbbb(f,b,j,m)*t2_abab(d,c,i,n)
    double_res += -0.250000000000000 * einsum('nmef,lkdc,aelk,fbjm,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(f,b,m,j)*t2_abab(c,d,i,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkcd,aelk,fbmj,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_bbbb(f,b,j,m)*t2_abab(c,d,i,n)
    double_res += -0.250000000000000 * einsum('nmef,lkcd,aelk,fbjm,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(f,b,i,m)*t2_abab(d,c,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,aekl,fbim,dcnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(f,b,i,m)*t2_abab(c,d,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,klcd,aekl,fbim,cdnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(f,b,i,m)*t2_abab(d,c,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,lkdc,aelk,fbim,dcnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(f,b,i,m)*t2_abab(c,d,n,j)
    double_res +=  0.250000000000000 * einsum('nmfe,lkcd,aelk,fbim,cdnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_abab(f,b,i,j)*t2_aaaa(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmef,kldc,eakl,fbij,dcnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(f,b,i,j)*t2_abab(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,kldc,aekl,fbij,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(f,b,i,j)*t2_abab(d,c,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,kldc,aekl,fbij,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(f,b,i,j)*t2_abab(c,d,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,klcd,aekl,fbij,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(f,b,i,j)*t2_abab(c,d,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,klcd,aekl,fbij,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(f,b,i,j)*t2_abab(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,lkdc,aelk,fbij,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(f,b,i,j)*t2_abab(d,c,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,lkdc,aelk,fbij,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(f,b,i,j)*t2_abab(c,d,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,lkcd,aelk,fbij,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(f,b,i,j)*t2_abab(c,d,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,lkcd,aelk,fbij,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,l,j)*t2_abab(f,b,k,m)*t2_aaaa(d,c,i,n)
    double_res +=  0.500000000000000 * einsum('nmfe,kldc,aelj,fbkm,dcin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,j)*t2_abab(f,b,m,k)*t2_abab(d,c,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,lkdc,aelj,fbmk,dcin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_abab(a,e,l,j)*t2_bbbb(f,b,k,m)*t2_abab(d,c,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,aelj,fbkm,dcin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,j)*t2_abab(f,b,m,k)*t2_abab(c,d,i,n)
    double_res += -0.500000000000000 * einsum('mnfe,lkcd,aelj,fbmk,cdin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,l,j)*t2_bbbb(f,b,k,m)*t2_abab(c,d,i,n)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,aelj,fbkm,cdin->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(f,b,k,m)*t2_abab(d,c,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,aeil,fbkm,dcnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_abab(f,b,k,m)*t2_abab(c,d,n,j)
    double_res += -0.500000000000000 * einsum('nmfe,klcd,aeil,fbkm,cdnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_abab(f,b,m,k)*t2_abab(d,c,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkdc,eail,fbmk,dcnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,k,m)*t2_abab(d,c,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,eail,fbkm,dcnj->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_abab(f,b,m,k)*t2_abab(c,d,n,j)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,eail,fbmk,cdnj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,k,m)*t2_abab(c,d,n,j)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,eail,fbkm,cdnj->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(f,b,m,k)*t2_bbbb(d,c,j,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,aeil,fbmk,dcjn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_bbbb(f,b,k,m)*t2_bbbb(d,c,j,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,aeil,fbkm,dcjn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,l,j)*t2_abab(f,b,i,m)*t2_aaaa(d,c,k,n)
    double_res += -0.500000000000000 * einsum('nmfe,kldc,aelj,fbim,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,j)*t2_abab(f,b,i,m)*t2_abab(d,c,n,k)
    double_res += -0.500000000000000 * einsum('nmfe,lkdc,aelj,fbim,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,j)*t2_abab(f,b,i,m)*t2_abab(c,d,n,k)
    double_res += -0.500000000000000 * einsum('nmfe,lkcd,aelj,fbim,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,l)*t2_abab(f,b,m,j)*t2_aaaa(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,eail,fbmj,dckn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,j,m)*t2_aaaa(d,c,k,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,eail,fbjm,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(f,b,m,j)*t2_abab(d,c,k,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,aeil,fbmj,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_bbbb(f,b,j,m)*t2_abab(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,aeil,fbjm,dckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_abab(f,b,m,j)*t2_abab(c,d,k,n)
    double_res += -0.500000000000000 * einsum('mnfe,klcd,aeil,fbmj,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_bbbb(f,b,j,m)*t2_abab(c,d,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,klcd,aeil,fbjm,cdkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_abab(f,b,m,j)*t2_abab(d,c,n,k)
    double_res +=  0.500000000000000 * einsum('nmef,lkdc,eail,fbmj,dcnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,j,m)*t2_abab(d,c,n,k)
    double_res += -0.500000000000000 * einsum('nmef,lkdc,eail,fbjm,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_abab(f,b,m,j)*t2_abab(c,d,n,k)
    double_res +=  0.500000000000000 * einsum('nmef,lkcd,eail,fbmj,cdnk->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,j,m)*t2_abab(c,d,n,k)
    double_res += -0.500000000000000 * einsum('nmef,lkcd,eail,fbjm,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(f,b,m,j)*t2_bbbb(d,c,k,n)
    double_res += -0.500000000000000 * einsum('mnfe,kldc,aeil,fbmj,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_bbbb(f,b,j,m)*t2_bbbb(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('nmef,kldc,aeil,fbjm,dckn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,j)*t2_abab(f,b,i,k)*t2_abab(d,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,lkdc,aelj,fbik,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,l,j)*t2_abab(f,b,i,k)*t2_abab(d,c,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,aelj,fbik,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,j)*t2_abab(f,b,i,k)*t2_abab(c,d,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,lkcd,aelj,fbik,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,l,j)*t2_abab(f,b,i,k)*t2_abab(c,d,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkcd,aelj,fbik,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,l)*t2_abab(f,b,k,j)*t2_aaaa(d,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,eail,fbkj,dcnm->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(f,b,k,j)*t2_abab(d,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,aeil,fbkj,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(f,b,k,j)*t2_abab(d,c,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,kldc,aeil,fbkj,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_abab(f,b,k,j)*t2_abab(c,d,n,m)
    double_res +=  0.250000000000000 * einsum('nmfe,klcd,aeil,fbkj,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_abab(f,b,k,j)*t2_abab(c,d,m,n)
    double_res +=  0.250000000000000 * einsum('mnfe,klcd,aeil,fbkj,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,j,k)*t2_abab(d,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkdc,eail,fbjk,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,j,k)*t2_abab(d,c,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkdc,eail,fbjk,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,j,k)*t2_abab(c,d,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,eail,fbjk,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_bbbb(f,b,j,k)*t2_abab(c,d,m,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkcd,eail,fbjk,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_bbbb(f,b,j,k)*t2_bbbb(d,c,n,m)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,aeil,fbjk,dcnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,k,j)*t2_abab(d,b,i,m)*t2_aaaa(f,c,l,n)
    double_res += -0.250000000000000 * einsum('nmfe,kldc,aekj,dbim,fcln->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(a,e,k,j)*t2_abab(d,b,i,m)*t2_abab(c,f,l,n)
    double_res += -0.250000000000000 * einsum('nmef,kldc,aekj,dbim,cfln->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,k,j)*t2_abab(d,b,i,m)*t2_abab(f,c,n,l)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,aekj,dbim,fcnl->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,k,j)*t2_abab(d,b,i,m)*t2_bbbb(f,c,l,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,aekj,dbim,fcln->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(d,a,i,m)*t2_aaaa(f,c,l,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,ebkj,daim,fcln->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(d,a,i,m)*t2_abab(c,f,l,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,ebkj,daim,cfln->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(d,a,i,m)*t2_abab(f,c,n,l)
    double_res += -0.250000000000000 * einsum('nmef,kldc,ebkj,daim,fcnl->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(d,a,i,m)*t2_bbbb(f,c,l,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,ebkj,daim,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_abab(a,d,i,m)*t2_abab(c,f,n,l)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,ebkj,adim,cfnl->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_aaaa(d,a,i,m)*t2_abab(f,c,l,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,ebjk,daim,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_abab(a,d,i,m)*t2_aaaa(f,c,l,n)
    double_res += -0.250000000000000 * einsum('nmfe,lkcd,ebjk,adim,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_abab(a,d,i,m)*t2_abab(c,f,l,n)
    double_res += -0.250000000000000 * einsum('nmef,lkcd,ebjk,adim,cfln->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_abab(a,d,i,m)*t2_abab(f,c,n,l)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,ebjk,adim,fcnl->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_abab(a,d,i,m)*t2_bbbb(f,c,l,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,ebjk,adim,fcln->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,k)*t2_abab(d,b,m,j)*t2_aaaa(f,c,l,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,eaik,dbmj,fcln->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,k)*t2_abab(d,b,m,j)*t2_abab(c,f,l,n)
    double_res +=  0.250000000000000 * einsum('mnef,kldc,eaik,dbmj,cfln->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,k)*t2_abab(d,b,m,j)*t2_abab(f,c,n,l)
    double_res += -0.250000000000000 * einsum('nmef,kldc,eaik,dbmj,fcnl->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,k)*t2_abab(d,b,m,j)*t2_bbbb(f,c,l,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,eaik,dbmj,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,k)*t2_bbbb(d,b,j,m)*t2_abab(c,f,n,l)
    double_res +=  0.250000000000000 * einsum('nmef,klcd,eaik,dbjm,cfnl->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,k)*t2_abab(d,b,m,j)*t2_abab(f,c,l,n)
    double_res +=  0.250000000000000 * einsum('mnfe,lkdc,aeik,dbmj,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,k)*t2_bbbb(d,b,j,m)*t2_aaaa(f,c,l,n)
    double_res += -0.250000000000000 * einsum('nmfe,lkcd,aeik,dbjm,fcln->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,e,i,k)*t2_bbbb(d,b,j,m)*t2_abab(c,f,l,n)
    double_res += -0.250000000000000 * einsum('nmef,lkcd,aeik,dbjm,cfln->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,k)*t2_bbbb(d,b,j,m)*t2_abab(f,c,n,l)
    double_res +=  0.250000000000000 * einsum('nmfe,kldc,aeik,dbjm,fcnl->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,k)*t2_bbbb(d,b,j,m)*t2_bbbb(f,c,l,n)
    double_res +=  0.250000000000000 * einsum('nmef,kldc,aeik,dbjm,fcln->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(e,b,i,k)*t2_abab(a,d,m,j)*t2_aaaa(f,c,l,n)
    double_res +=  0.250000000000000 * einsum('nmef,lkcd,ebik,admj,fcln->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 <m,n||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,b,i,k)*t2_abab(a,d,m,j)*t2_abab(c,f,l,n)
    double_res +=  0.250000000000000 * einsum('mnef,lkcd,ebik,admj,cfln->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,k)*t2_abab(a,d,m,j)*t2_abab(f,c,n,l)
    double_res += -0.250000000000000 * einsum('nmef,kldc,ebik,admj,fcnl->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 <m,n||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,k)*t2_abab(a,d,m,j)*t2_bbbb(f,c,l,n)
    double_res += -0.250000000000000 * einsum('mnef,kldc,ebik,admj,fcln->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,l,m)*t2_aaaa(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('nmfe,kldc,aeij,fblm,dckn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,m,l)*t2_abab(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,aeij,fbml,dckn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(a,e,i,j)*t2_bbbb(f,b,l,m)*t2_abab(d,c,k,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,aeij,fblm,dckn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,j)*t2_abab(f,b,m,l)*t2_abab(c,d,k,n)
    double_res +=  0.500000000000000 * einsum('mnfe,klcd,aeij,fbml,cdkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_abab(a,e,i,j)*t2_bbbb(f,b,l,m)*t2_abab(c,d,k,n)
    double_res += -0.500000000000000 * einsum('nmef,klcd,aeij,fblm,cdkn->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,l,m)*t2_abab(d,c,n,k)
    double_res +=  0.500000000000000 * einsum('nmfe,lkdc,aeij,fblm,dcnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,j)*t2_abab(f,b,l,m)*t2_abab(c,d,n,k)
    double_res +=  0.500000000000000 * einsum('nmfe,lkcd,aeij,fblm,cdnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	  0.5000 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,m,l)*t2_bbbb(d,c,k,n)
    double_res +=  0.500000000000000 * einsum('mnfe,kldc,aeij,fbml,dckn->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,j)*t2_bbbb(f,b,l,m)*t2_bbbb(d,c,k,n)
    double_res += -0.500000000000000 * einsum('nmef,kldc,aeij,fblm,dckn->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,k,l)*t2_abab(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,kldc,aeij,fbkl,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,k,l)*t2_abab(d,c,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,kldc,aeij,fbkl,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,j)*t2_abab(f,b,k,l)*t2_abab(c,d,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,klcd,aeij,fbkl,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(k,l,c,d)*t2_abab(a,e,i,j)*t2_abab(f,b,k,l)*t2_abab(c,d,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,klcd,aeij,fbkl,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,l,k)*t2_abab(d,c,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,lkdc,aeij,fblk,dcnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(a,e,i,j)*t2_abab(f,b,l,k)*t2_abab(d,c,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,lkdc,aeij,fblk,dcmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,j)*t2_abab(f,b,l,k)*t2_abab(c,d,n,m)
    double_res += -0.125000000000000 * einsum('nmfe,lkcd,aeij,fblk,cdnm->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,e,i,j)*t2_abab(f,b,l,k)*t2_abab(c,d,m,n)
    double_res += -0.125000000000000 * einsum('mnfe,lkcd,aeij,fblk,cdmn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,j)*t2_bbbb(f,b,k,l)*t2_bbbb(d,c,n,m)
    double_res +=  0.125000000000000 * einsum('nmef,kldc,aeij,fbkl,dcnm->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 4), (0, 2), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(d,c,l,m)*t2_aaaa(f,a,i,n)
    double_res +=  0.125000000000000 * einsum('nmef,kldc,ebkj,dclm,fain->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	  0.1250 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(d,c,l,m)*t2_abab(a,f,i,n)
    double_res +=  0.125000000000000 * einsum('mnef,kldc,ebkj,dclm,afin->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_abab(d,c,m,l)*t2_aaaa(f,a,i,n)
    double_res += -0.125000000000000 * einsum('nmef,kldc,ebkj,dcml,fain->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_abab(d,c,m,l)*t2_abab(a,f,i,n)
    double_res += -0.125000000000000 * einsum('mnef,kldc,ebkj,dcml,afin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_abab(c,d,m,l)*t2_aaaa(f,a,i,n)
    double_res += -0.125000000000000 * einsum('nmef,klcd,ebkj,cdml,fain->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_abab(c,d,m,l)*t2_abab(a,f,i,n)
    double_res += -0.125000000000000 * einsum('mnef,klcd,ebkj,cdml,afin->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_abab(d,c,l,m)*t2_aaaa(f,a,i,n)
    double_res += -0.125000000000000 * einsum('nmfe,lkdc,ebjk,dclm,fain->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_bbbb*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_abab(d,c,l,m)*t2_abab(a,f,i,n)
    double_res += -0.125000000000000 * einsum('nmef,lkdc,ebjk,dclm,afin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_abab(c,d,l,m)*t2_aaaa(f,a,i,n)
    double_res += -0.125000000000000 * einsum('nmfe,lkcd,ebjk,cdlm,fain->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_abab(c,d,l,m)*t2_abab(a,f,i,n)
    double_res += -0.125000000000000 * einsum('nmef,lkcd,ebjk,cdlm,afin->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	  0.1250 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_bbbb(d,c,l,m)*t2_aaaa(f,a,i,n)
    double_res +=  0.125000000000000 * einsum('nmfe,kldc,ebjk,dclm,fain->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_bbbb(d,c,l,m)*t2_abab(a,f,i,n)
    double_res +=  0.125000000000000 * einsum('nmef,kldc,ebjk,dclm,afin->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,b,i,k)*t2_abab(d,c,l,m)*t2_abab(a,f,n,j)
    double_res += -0.125000000000000 * einsum('nmef,lkdc,ebik,dclm,afnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,b,i,k)*t2_abab(c,d,l,m)*t2_abab(a,f,n,j)
    double_res += -0.125000000000000 * einsum('nmef,lkcd,ebik,cdlm,afnj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,k)*t2_bbbb(d,c,l,m)*t2_abab(a,f,n,j)
    double_res +=  0.125000000000000 * einsum('nmef,kldc,ebik,dclm,afnj->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 4), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(e,b,i,j)*t2_aaaa(d,c,l,m)*t2_aaaa(f,a,k,n)
    double_res += -0.125000000000000 * einsum('nmef,kldc,ebij,dclm,fakn->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.1250 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(e,b,i,j)*t2_aaaa(d,c,l,m)*t2_abab(a,f,k,n)
    double_res += -0.125000000000000 * einsum('mnef,kldc,ebij,dclm,afkn->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(e,b,i,j)*t2_abab(d,c,m,l)*t2_aaaa(f,a,k,n)
    double_res +=  0.125000000000000 * einsum('nmef,kldc,ebij,dcml,fakn->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(e,b,i,j)*t2_abab(d,c,m,l)*t2_abab(a,f,k,n)
    double_res +=  0.125000000000000 * einsum('mnef,kldc,ebij,dcml,afkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_aaaa*l2_abab(k,l,c,d)*t2_abab(e,b,i,j)*t2_abab(c,d,m,l)*t2_aaaa(f,a,k,n)
    double_res +=  0.125000000000000 * einsum('nmef,klcd,ebij,cdml,fakn->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(e,b,i,j)*t2_abab(c,d,m,l)*t2_abab(a,f,k,n)
    double_res +=  0.125000000000000 * einsum('mnef,klcd,ebij,cdml,afkn->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_abab*l2_abab(l,k,d,c)*t2_abab(e,b,i,j)*t2_abab(d,c,l,m)*t2_abab(a,f,n,k)
    double_res +=  0.125000000000000 * einsum('nmef,lkdc,ebij,dclm,afnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(e,b,i,j)*t2_abab(c,d,l,m)*t2_abab(a,f,n,k)
    double_res +=  0.125000000000000 * einsum('nmef,lkcd,ebij,cdlm,afnk->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	 -0.1250 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,j)*t2_bbbb(d,c,l,m)*t2_abab(a,f,n,k)
    double_res += -0.125000000000000 * einsum('nmef,kldc,ebij,dclm,afnk->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (1, 2), (0, 1)])
    
    #	  0.0833 <m,n||e,f>_abab*l2_abab(k,l,c,d)*t2_abab(a,d,m,j)*t2_abab(e,b,k,n)*t2_abab(c,f,i,l)
    double_res +=  0.083333333333330 * einsum('mnef,klcd,admj,ebkn,cfil->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <n,m||e,f>_aaaa*l2_abab(l,k,c,d)*t2_abab(a,d,m,j)*t2_abab(e,b,n,k)*t2_aaaa(f,c,i,l)
    double_res +=  0.083333333333330 * einsum('nmef,lkcd,admj,ebnk,fcil->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <m,n||f,e>_abab*l2_abab(l,k,c,d)*t2_abab(a,d,m,j)*t2_bbbb(e,b,k,n)*t2_aaaa(f,c,i,l)
    double_res += -0.083333333333330 * einsum('mnfe,lkcd,admj,ebkn,fcil->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <n,m||e,f>_aaaa*l2_bbbb(k,l,d,c)*t2_abab(a,d,m,j)*t2_abab(e,b,n,k)*t2_abab(f,c,i,l)
    double_res +=  0.083333333333330 * einsum('nmef,kldc,admj,ebnk,fcil->abij', g_aaaa[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <m,n||f,e>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,d,m,j)*t2_bbbb(e,b,k,n)*t2_abab(f,c,i,l)
    double_res += -0.083333333333330 * einsum('mnfe,kldc,admj,ebkn,fcil->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||e,f>_aaaa*l2_aaaa(k,l,d,c)*t2_abab(d,b,m,j)*t2_aaaa(e,a,k,n)*t2_aaaa(f,c,i,l)
    double_res += -0.083333333333330 * einsum('nmef,kldc,dbmj,eakn,fcil->abij', g_aaaa[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <m,n||f,e>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,b,m,j)*t2_abab(a,e,k,n)*t2_aaaa(f,c,i,l)
    double_res +=  0.083333333333330 * einsum('mnfe,kldc,dbmj,aekn,fcil->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||e,f>_aaaa*l2_abab(k,l,d,c)*t2_abab(d,b,m,j)*t2_aaaa(e,a,k,n)*t2_abab(f,c,i,l)
    double_res += -0.083333333333330 * einsum('nmef,kldc,dbmj,eakn,fcil->abij', g_aaaa[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <m,n||f,e>_abab*l2_abab(k,l,d,c)*t2_abab(d,b,m,j)*t2_abab(a,e,k,n)*t2_abab(f,c,i,l)
    double_res +=  0.083333333333330 * einsum('mnfe,kldc,dbmj,aekn,fcil->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <n,m||e,f>_abab*l2_abab(k,l,c,d)*t2_bbbb(d,b,j,m)*t2_aaaa(e,a,k,n)*t2_abab(c,f,i,l)
    double_res +=  0.083333333333330 * einsum('nmef,klcd,dbjm,eakn,cfil->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||e,f>_bbbb*l2_abab(k,l,c,d)*t2_bbbb(d,b,j,m)*t2_abab(a,e,k,n)*t2_abab(c,f,i,l)
    double_res += -0.083333333333330 * einsum('nmef,klcd,dbjm,aekn,cfil->abij', g_bbbb[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||f,e>_abab*l2_abab(l,k,c,d)*t2_bbbb(d,b,j,m)*t2_abab(a,e,n,k)*t2_aaaa(f,c,i,l)
    double_res += -0.083333333333330 * einsum('nmfe,lkcd,dbjm,aenk,fcil->abij', g_abab[o, o, v, v], l2_abab, t2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||f,e>_abab*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,j,m)*t2_abab(a,e,n,k)*t2_abab(f,c,i,l)
    double_res += -0.083333333333330 * einsum('nmfe,kldc,dbjm,aenk,fcil->abij', g_abab[o, o, v, v], l2_bbbb, t2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <m,n||e,f>_abab*l2_aaaa(k,l,d,c)*t2_aaaa(d,a,i,m)*t2_abab(e,b,k,n)*t2_abab(c,f,l,j)
    double_res += -0.083333333333330 * einsum('mnef,kldc,daim,ebkn,cflj->abij', g_abab[o, o, v, v], l2_aaaa, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <m,n||e,f>_abab*l2_abab(k,l,d,c)*t2_aaaa(d,a,i,m)*t2_abab(e,b,k,n)*t2_bbbb(f,c,j,l)
    double_res += -0.083333333333330 * einsum('mnef,kldc,daim,ebkn,fcjl->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||e,f>_aaaa*l2_abab(l,k,d,c)*t2_aaaa(d,a,i,m)*t2_abab(e,b,n,k)*t2_abab(f,c,l,j)
    double_res += -0.083333333333330 * einsum('nmef,lkdc,daim,ebnk,fclj->abij', g_aaaa[o, o, v, v], l2_abab, t2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <m,n||f,e>_abab*l2_abab(l,k,d,c)*t2_aaaa(d,a,i,m)*t2_bbbb(e,b,k,n)*t2_abab(f,c,l,j)
    double_res +=  0.083333333333330 * einsum('mnfe,lkdc,daim,ebkn,fclj->abij', g_abab[o, o, v, v], l2_abab, t2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <n,m||e,f>_abab*l2_abab(l,k,c,d)*t2_abab(a,d,i,m)*t2_abab(e,b,n,k)*t2_abab(c,f,l,j)
    double_res +=  0.083333333333330 * einsum('nmef,lkcd,adim,ebnk,cflj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||e,f>_bbbb*l2_abab(l,k,c,d)*t2_abab(a,d,i,m)*t2_bbbb(e,b,k,n)*t2_abab(c,f,l,j)
    double_res += -0.083333333333330 * einsum('nmef,lkcd,adim,ebkn,cflj->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <n,m||e,f>_abab*l2_bbbb(k,l,d,c)*t2_abab(a,d,i,m)*t2_abab(e,b,n,k)*t2_bbbb(f,c,j,l)
    double_res +=  0.083333333333330 * einsum('nmef,kldc,adim,ebnk,fcjl->abij', g_abab[o, o, v, v], l2_bbbb, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||e,f>_bbbb*l2_bbbb(k,l,d,c)*t2_abab(a,d,i,m)*t2_bbbb(e,b,k,n)*t2_bbbb(f,c,j,l)
    double_res += -0.083333333333330 * einsum('nmef,kldc,adim,ebkn,fcjl->abij', g_bbbb[o, o, v, v], l2_bbbb, t2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||e,f>_abab*l2_aaaa(k,l,d,c)*t2_abab(d,b,i,m)*t2_aaaa(e,a,k,n)*t2_abab(c,f,l,j)
    double_res += -0.083333333333330 * einsum('nmef,kldc,dbim,eakn,cflj->abij', g_abab[o, o, v, v], l2_aaaa, t2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <n,m||e,f>_bbbb*l2_aaaa(k,l,d,c)*t2_abab(d,b,i,m)*t2_abab(a,e,k,n)*t2_abab(c,f,l,j)
    double_res +=  0.083333333333330 * einsum('nmef,kldc,dbim,aekn,cflj->abij', g_bbbb[o, o, v, v], l2_aaaa, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	 -0.0833 <n,m||e,f>_abab*l2_abab(k,l,d,c)*t2_abab(d,b,i,m)*t2_aaaa(e,a,k,n)*t2_bbbb(f,c,j,l)
    double_res += -0.083333333333330 * einsum('nmef,kldc,dbim,eakn,fcjl->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <n,m||e,f>_bbbb*l2_abab(k,l,d,c)*t2_abab(d,b,i,m)*t2_abab(a,e,k,n)*t2_bbbb(f,c,j,l)
    double_res +=  0.083333333333330 * einsum('nmef,kldc,dbim,aekn,fcjl->abij', g_bbbb[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    
    #	  0.0833 <n,m||f,e>_abab*l2_abab(l,k,d,c)*t2_abab(d,b,i,m)*t2_abab(a,e,n,k)*t2_abab(f,c,l,j)
    double_res +=  0.083333333333330 * einsum('nmfe,lkdc,dbim,aenk,fclj->abij', g_abab[o, o, v, v], l2_abab, t2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
   
    return double_res 
