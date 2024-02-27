import numpy
from numpy import einsum

def square_brakT(g_aaaa,g_abab,g_bbbb,o,v,l2_aaaa,l2_abab,l2_bbbb,t3_aaaaaa,t3_aabaab,t3_abbabb,t3_bbbbbb):
    
    #	  0.2500 <l,k||c,j>_aaaa*l2_aaaa(i,j,b,a)*t3_aaaaaa(c,b,a,i,l,k)
    energy =  0.250000000000000 * einsum('lkcj,ijba,cbailk', g_aaaa[o, o, v, o], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <l,k||j,c>_abab*l2_aaaa(i,j,b,a)*t3_aabaab(a,b,c,i,l,k)
    energy +=  0.250000000000000 * einsum('lkjc,ijba,abcilk', g_abab[o, o, o, v], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,l||j,c>_abab*l2_aaaa(i,j,b,a)*t3_aabaab(a,b,c,i,k,l)
    energy +=  0.250000000000000 * einsum('kljc,ijba,abcikl', g_abab[o, o, o, v], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <l,k||c,j>_abab*l2_abab(i,j,b,a)*t3_aabaab(c,b,a,i,l,k)
    energy +=  0.250000000000000 * einsum('lkcj,ijba,cbailk', g_abab[o, o, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,l||c,j>_abab*l2_abab(i,j,b,a)*t3_aabaab(c,b,a,i,k,l)
    energy +=  0.250000000000000 * einsum('klcj,ijba,cbaikl', g_abab[o, o, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <l,k||c,j>_bbbb*l2_abab(i,j,b,a)*t3_abbabb(b,c,a,i,l,k)
    energy += -0.250000000000000 * einsum('lkcj,ijba,bcailk', g_bbbb[o, o, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <l,k||c,j>_abab*l2_abab(i,j,a,b)*t3_aabaab(c,a,b,i,l,k)
    energy +=  0.250000000000000 * einsum('lkcj,ijab,cabilk', g_abab[o, o, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,l||c,j>_abab*l2_abab(i,j,a,b)*t3_aabaab(c,a,b,i,k,l)
    energy +=  0.250000000000000 * einsum('klcj,ijab,cabikl', g_abab[o, o, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <l,k||c,j>_bbbb*l2_abab(i,j,a,b)*t3_abbabb(a,b,c,i,l,k)
    energy +=  0.250000000000000 * einsum('lkcj,ijab,abcilk', g_bbbb[o, o, v, o], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <l,k||c,j>_aaaa*l2_abab(j,i,b,a)*t3_aabaab(c,b,a,k,l,i)
    energy +=  0.250000000000000 * einsum('lkcj,jiba,cbakli', g_aaaa[o, o, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <l,k||j,c>_abab*l2_abab(j,i,b,a)*t3_abbabb(b,c,a,l,i,k)
    energy +=  0.250000000000000 * einsum('lkjc,jiba,bcalik', g_abab[o, o, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <k,l||j,c>_abab*l2_abab(j,i,b,a)*t3_abbabb(b,c,a,k,l,i)
    energy += -0.250000000000000 * einsum('kljc,jiba,bcakli', g_abab[o, o, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <l,k||c,j>_aaaa*l2_abab(j,i,a,b)*t3_aabaab(c,a,b,k,l,i)
    energy +=  0.250000000000000 * einsum('lkcj,jiab,cabkli', g_aaaa[o, o, v, o], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <l,k||j,c>_abab*l2_abab(j,i,a,b)*t3_abbabb(a,b,c,l,i,k)
    energy += -0.250000000000000 * einsum('lkjc,jiab,abclik', g_abab[o, o, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,l||j,c>_abab*l2_abab(j,i,a,b)*t3_abbabb(a,b,c,k,l,i)
    energy +=  0.250000000000000 * einsum('kljc,jiab,abckli', g_abab[o, o, o, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <l,k||c,j>_abab*l2_bbbb(i,j,b,a)*t3_abbabb(c,b,a,l,i,k)
    energy += -0.250000000000000 * einsum('lkcj,ijba,cbalik', g_abab[o, o, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,l||c,j>_abab*l2_bbbb(i,j,b,a)*t3_abbabb(c,b,a,k,l,i)
    energy +=  0.250000000000000 * einsum('klcj,ijba,cbakli', g_abab[o, o, v, o], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <l,k||c,j>_bbbb*l2_bbbb(i,j,b,a)*t3_bbbbbb(c,b,a,i,l,k)
    energy +=  0.250000000000000 * einsum('lkcj,ijba,cbailk', g_bbbb[o, o, v, o], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,b||c,d>_aaaa*l2_aaaa(i,j,b,a)*t3_aaaaaa(c,d,a,i,j,k)
    energy +=  0.250000000000000 * einsum('kbcd,ijba,cdaijk', g_aaaa[o, v, v, v], l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <b,k||c,d>_abab*l2_aaaa(i,j,b,a)*t3_aabaab(c,a,d,i,j,k)
    energy +=  0.250000000000000 * einsum('bkcd,ijba,cadijk', g_abab[v, o, v, v], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <b,k||d,c>_abab*l2_aaaa(i,j,b,a)*t3_aabaab(a,d,c,i,j,k)
    energy += -0.250000000000000 * einsum('bkdc,ijba,adcijk', g_abab[v, o, v, v], l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <k,b||c,d>_aaaa*l2_abab(i,j,b,a)*t3_aabaab(c,d,a,i,k,j)
    energy += -0.250000000000000 * einsum('kbcd,ijba,cdaikj', g_aaaa[o, v, v, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <b,k||c,d>_abab*l2_abab(i,j,b,a)*t3_abbabb(c,d,a,i,j,k)
    energy += -0.250000000000000 * einsum('bkcd,ijba,cdaijk', g_abab[v, o, v, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <b,k||d,c>_abab*l2_abab(i,j,b,a)*t3_abbabb(d,c,a,i,j,k)
    energy += -0.250000000000000 * einsum('bkdc,ijba,dcaijk', g_abab[v, o, v, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <k,b||c,d>_abab*l2_abab(i,j,a,b)*t3_aabaab(c,a,d,i,k,j)
    energy += -0.250000000000000 * einsum('kbcd,ijab,cadikj', g_abab[o, v, v, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,b||d,c>_abab*l2_abab(i,j,a,b)*t3_aabaab(a,d,c,i,k,j)
    energy +=  0.250000000000000 * einsum('kbdc,ijab,adcikj', g_abab[o, v, v, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,b||c,d>_bbbb*l2_abab(i,j,a,b)*t3_abbabb(a,d,c,i,j,k)
    energy +=  0.250000000000000 * einsum('kbcd,ijab,adcijk', g_bbbb[o, v, v, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,b||c,d>_aaaa*l2_abab(j,i,b,a)*t3_aabaab(c,d,a,k,j,i)
    energy +=  0.250000000000000 * einsum('kbcd,jiba,cdakji', g_aaaa[o, v, v, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <b,k||c,d>_abab*l2_abab(j,i,b,a)*t3_abbabb(c,d,a,j,i,k)
    energy += -0.250000000000000 * einsum('bkcd,jiba,cdajik', g_abab[v, o, v, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <b,k||d,c>_abab*l2_abab(j,i,b,a)*t3_abbabb(d,c,a,j,i,k)
    energy += -0.250000000000000 * einsum('bkdc,jiba,dcajik', g_abab[v, o, v, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,b||c,d>_abab*l2_abab(j,i,a,b)*t3_aabaab(c,a,d,k,j,i)
    energy +=  0.250000000000000 * einsum('kbcd,jiab,cadkji', g_abab[o, v, v, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <k,b||d,c>_abab*l2_abab(j,i,a,b)*t3_aabaab(a,d,c,k,j,i)
    energy += -0.250000000000000 * einsum('kbdc,jiab,adckji', g_abab[o, v, v, v], l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,b||c,d>_bbbb*l2_abab(j,i,a,b)*t3_abbabb(a,d,c,j,i,k)
    energy +=  0.250000000000000 * einsum('kbcd,jiab,adcjik', g_bbbb[o, v, v, v], l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <k,b||c,d>_abab*l2_bbbb(i,j,b,a)*t3_abbabb(c,d,a,k,j,i)
    energy += -0.250000000000000 * einsum('kbcd,ijba,cdakji', g_abab[o, v, v, v], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <k,b||d,c>_abab*l2_bbbb(i,j,b,a)*t3_abbabb(d,c,a,k,j,i)
    energy += -0.250000000000000 * einsum('kbdc,ijba,dcakji', g_abab[o, v, v, v], l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <k,b||c,d>_bbbb*l2_bbbb(i,j,b,a)*t3_bbbbbb(c,d,a,i,j,k)
    energy +=  0.250000000000000 * einsum('kbcd,ijba,cdaijk', g_bbbb[o, v, v, v], l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
     
    return energy
