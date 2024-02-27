from numpy import einsum

def t4_test_residual(t2,g,oa,va):
    
    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||k,l>*t2(a,b,j,m)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,abjm,cdin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||k,l>*t2(a,d,j,m)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,adjm,bcin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||j,l>*t2(a,b,k,m)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,abkm,cdin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||j,l>*t2(a,d,k,m)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,adkm,bcin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||i,l>*t2(a,b,k,m)*t2(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,abkm,cdjn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||i,l>*t2(a,d,k,m)*t2(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,adkm,bcjn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)*P(b,c)<n,m||j,k>*t2(a,b,l,m)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,ablm,cdin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)<n,m||j,k>*t2(a,d,l,m)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,adlm,bcin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,c)<n,m||i,k>*t2(a,b,l,m)*t2(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmik,ablm,cdjn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||i,k>*t2(a,d,l,m)*t2(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmik,adlm,bcjn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||i,j>*t2(a,b,l,m)*t2(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,ablm,cdkn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||i,j>*t2(a,d,l,m)*t2(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,adlm,bckn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
   
    t4_resid_oo=quadruples_res
 
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,l>*t2(e,b,j,k)*t2(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ebjk,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<m,a||e,l>*t2(e,b,i,j)*t2(c,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ebij,cdkm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,l>*t2(e,d,j,k)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,edjk,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,l>*t2(e,d,i,j)*t2(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,edij,bckm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,a||e,k>*t2(e,b,j,l)*t2(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,ebjl,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,k>*t2(e,d,j,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,edjl,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<m,a||e,j>*t2(e,b,k,l)*t2(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ebkl,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,j>*t2(e,b,i,k)*t2(c,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ebik,cdlm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,a||e,j>*t2(e,d,k,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edkl,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>*t2(e,d,i,k)*t2(b,c,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edik,bclm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,a||e,i>*t2(e,b,k,l)*t2(c,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,ebkl,cdjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,a||e,i>*t2(e,d,k,l)*t2(b,c,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,edkl,bcjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,l>*t2(e,a,j,k)*t2(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,eajk,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,c)<m,b||e,l>*t2(e,a,i,j)*t2(c,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,eaij,cdkm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<m,b||e,k>*t2(e,a,j,l)*t2(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbek,eajl,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,c)<m,b||e,j>*t2(e,a,k,l)*t2(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,eakl,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,j>*t2(e,a,i,k)*t2(c,d,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,eaik,cdlm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,c)<m,b||e,i>*t2(e,a,k,l)*t2(c,d,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbei,eakl,cdjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,l>*t2(e,a,j,k)*t2(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,eajk,bdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,c||e,l>*t2(e,a,i,j)*t2(b,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,eaij,bdkm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,l>*t2(e,d,j,k)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edjk,abim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,l>*t2(e,d,i,j)*t2(a,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edij,abkm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,k>*t2(e,a,j,l)*t2(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,eajl,bdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,k>*t2(e,d,j,l)*t2(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,edjl,abim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,c||e,j>*t2(e,a,k,l)*t2(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,eakl,bdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,j>*t2(e,a,i,k)*t2(b,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,eaik,bdlm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)<m,c||e,j>*t2(e,d,k,l)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edkl,abim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,j>*t2(e,d,i,k)*t2(a,b,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edik,ablm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,c||e,i>*t2(e,a,k,l)*t2(b,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,eakl,bdjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<m,c||e,i>*t2(e,d,k,l)*t2(a,b,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,edkl,abjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,l>*t2(e,a,j,k)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eajk,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,d||e,l>*t2(e,a,i,j)*t2(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eaij,bckm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,k>*t2(e,a,j,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdek,eajl,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,d||e,j>*t2(e,a,k,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,eakl,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,j>*t2(e,a,i,k)*t2(b,c,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,eaik,bclm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,d||e,i>*t2(e,a,k,l)*t2(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdei,eakl,bcjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<a,b||e,f>*t2(e,c,k,l)*t2(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abef,eckl,fdij->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<a,b||e,f>*t2(e,c,i,l)*t2(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecil,fdjk->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<a,b||e,f>*t2(e,c,j,k)*t2(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecjk,fdil->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<a,d||e,f>*t2(e,b,k,l)*t2(f,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebkl,fcij->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<a,d||e,f>*t2(e,b,i,l)*t2(f,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebil,fcjk->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<a,d||e,f>*t2(e,b,j,k)*t2(f,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebjk,fcil->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,d)<b,c||e,f>*t2(e,a,k,l)*t2(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eakl,fdij->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,d)<b,c||e,f>*t2(e,a,i,l)*t2(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eail,fdjk->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,d)<b,c||e,f>*t2(e,a,j,k)*t2(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eajk,fdil->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbkjil', contracted_intermediate) 
    
    return t4_resid_oo,quadruples_res

