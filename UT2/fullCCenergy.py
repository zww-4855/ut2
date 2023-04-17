import numpy as np
from numpy import einsum


def fullCC_energyMain(ccd_kernel,get_perturbCorr=False):
    """
    Drives the determination of spin-integrated, CCD energy. This includes unmodified energy, as well as calling subsequent modules to extract perturbative corrections. 
    
    :param ccd_kernel: Object of the UltT2CC class. 
    :param get_perturbCorr: Boolean flag to determine if perturbative corrections to the energy are called for
    
    :return: Returns either the baseline CCD energy, or factorization based perturbative energy corrections 
    """

    sliceInfo=ccd_kernel.sliceInfo
    oa=sliceInfo["occ_aa"]
    ob=sliceInfo["occ_bb"]
    va=sliceInfo["virt_aa"]
    vb=sliceInfo["virt_bb"]

    t1_aa = ccd_kernel.tamps["t1aa"]
    t1_bb = ccd_kernel.tamps["t1bb"]

    t2_aaaa=ccd_kernel.tamps["t2aa"]
    t2_bbbb=ccd_kernel.tamps["t2bb"]
    t2_abab=ccd_kernel.tamps["t2ab"]

    t3_aaaaaa=ccd_kernel.tamps["t3aaa"]
    t3_bbbbbb=ccd_kernel.tamps["t3bbb"]
    t3_aabaab=ccd_kernel.tamps["t3aab"]
    t3_abbabb=ccd_kernel.tamps["t3abb"]

    fock=ccd_kernel.ints["oei"]
    tei=ccd_kernel.ints["tei"]

    f_aa=fock["faa"]
    f_bb=fock["fbb"]


    g_aaaa=tei["g_aaaa"]
    g_bbbb=tei["g_bbbb"]
    g_abab=tei["g_abab"]


    if get_perturbCorr==True:
        l2dic=ccd_kernel.get_l2amps()
        l2_aaaa=l2dic["l2aa"]
        l2_bbbb=l2dic["l2bb"]
        l2_abab=l2dic["l2ab"]

        qf_corr=energy_pertQf(g_aaaa,g_abab,g_bbbb,t2_aaaa,t2_abab,t2_bbbb,t3_aaaaaa,t3_bbbbbb,t3_aabaab,t3_abbabb,l2_aaaa,l2_bbbb,l2_abab,oa,va)
        return qf_corr
    else:    
        return ccsd_energy_with_spin(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)


def ccsd_energy_with_spin(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | e(-T) H e(T) | 0> :

    o = oa
    v = va

    #     1.0000 f_aa(i,i)
    energy =  1.000000000000000 * einsum('ii', f_aa[o, o])

    #     1.0000 f_bb(i,i)
    energy +=  1.000000000000000 * einsum('ii', f_bb[o, o])

    #     1.0000 f_aa(i,a)*t1_aa(a,i)
    energy +=  1.000000000000000 * einsum('ia,ai', f_aa[o, v], t1_aa)

    #     1.0000 f_bb(i,a)*t1_bb(a,i)
    energy +=  1.000000000000000 * einsum('ia,ai', f_bb[o, v], t1_bb)

    #    -0.5000 <j,i||j,i>_aaaa
    energy += -0.500000000000000 * einsum('jiji', g_aaaa[o, o, o, o])

    #    -0.5000 <j,i||j,i>_abab
    energy += -0.500000000000000 * einsum('jiji', g_abab[o, o, o, o])

    #    -0.5000 <i,j||i,j>_abab
    energy += -0.500000000000000 * einsum('ijij', g_abab[o, o, o, o])

    #    -0.5000 <j,i||j,i>_bbbb
    energy += -0.500000000000000 * einsum('jiji', g_bbbb[o, o, o, o])

    #     0.2500 <j,i||a,b>_aaaa*t2_aaaa(a,b,j,i)
    energy +=  0.250000000000000 * einsum('jiab,abji', g_aaaa[o, o, v, v], t2_aaaa)

    #     0.2500 <j,i||a,b>_abab*t2_abab(a,b,j,i)
    energy +=  0.250000000000000 * einsum('jiab,abji', g_abab[o, o, v, v], t2_abab)

    #     0.2500 <i,j||a,b>_abab*t2_abab(a,b,i,j)
    energy +=  0.250000000000000 * einsum('ijab,abij', g_abab[o, o, v, v], t2_abab)

    #     0.2500 <j,i||b,a>_abab*t2_abab(b,a,j,i)
    energy +=  0.250000000000000 * einsum('jiba,baji', g_abab[o, o, v, v], t2_abab)

    #     0.2500 <i,j||b,a>_abab*t2_abab(b,a,i,j)
    energy +=  0.250000000000000 * einsum('ijba,baij', g_abab[o, o, v, v], t2_abab)

    #     0.2500 <j,i||a,b>_bbbb*t2_bbbb(a,b,j,i)
    energy +=  0.250000000000000 * einsum('jiab,abji', g_bbbb[o, o, v, v], t2_bbbb)

    #    -0.5000 <j,i||a,b>_aaaa*t1_aa(a,i)*t1_aa(b,j)
    energy += -0.500000000000000 * einsum('jiab,ai,bj', g_aaaa[o, o, v, v], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #     0.5000 <i,j||a,b>_abab*t1_aa(a,i)*t1_bb(b,j)
    energy +=  0.500000000000000 * einsum('ijab,ai,bj', g_abab[o, o, v, v], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #     0.5000 <j,i||b,a>_abab*t1_bb(a,i)*t1_aa(b,j)
    energy +=  0.500000000000000 * einsum('jiba,ai,bj', g_abab[o, o, v, v], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #    -0.5000 <j,i||a,b>_bbbb*t1_bb(a,i)*t1_bb(b,j)
    energy += -0.500000000000000 * einsum('jiab,ai,bj', g_bbbb[o, o, v, v], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    return energy



def energy_pertQf(g_aaaa,g_abab,g_bbbb,t2_aaaa,t2_abab,t2_bbbb,t3_aaaaaa,t3_bbbbbb,t3_aabaab,t3_abbabb,l2_aaaa,l2_bbbb,l2_abab,o,v):
    #	 -0.1250 <n,m||k,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,j,m)*t2_aaaa(b,a,i,n)
    energy = -0.125000000000000 * einsum('nmkl,ijba,kldc,dcjm,bain', g_aaaa[o, o, o, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,j,m)*t2_aaaa(b,a,i,n)
    energy += -0.125000000000000 * einsum('nmkl,ijba,kldc,dcjm,bain', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,j,m)*t2_aaaa(b,a,i,n)
    energy += -0.125000000000000 * einsum('nmkl,ijba,klcd,cdjm,bain', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||l,k>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,j,m)*t2_aaaa(b,a,i,n)
    energy += -0.125000000000000 * einsum('nmlk,ijba,lkdc,dcjm,bain', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||l,k>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,j,m)*t2_aaaa(b,a,i,n)
    energy += -0.125000000000000 * einsum('nmlk,ijba,lkcd,cdjm,bain', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||k,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,m,j)*t2_abab(b,a,i,n)
    energy += -0.125000000000000 * einsum('mnkl,ijba,kldc,dcmj,bain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||k,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,m,j)*t2_abab(b,a,i,n)
    energy += -0.125000000000000 * einsum('mnkl,ijba,klcd,cdmj,bain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||l,k>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,m,j)*t2_abab(b,a,i,n)
    energy += -0.125000000000000 * einsum('mnlk,ijba,lkdc,dcmj,bain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||l,k>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,m,j)*t2_abab(b,a,i,n)
    energy += -0.125000000000000 * einsum('mnlk,ijba,lkcd,cdmj,bain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,j,m)*t2_abab(b,a,i,n)
    energy += -0.125000000000000 * einsum('nmkl,ijba,kldc,dcjm,bain', g_bbbb[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||k,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,c,m,j)*t2_abab(a,b,i,n)
    energy += -0.125000000000000 * einsum('mnkl,ijab,kldc,dcmj,abin', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||k,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,d,m,j)*t2_abab(a,b,i,n)
    energy += -0.125000000000000 * einsum('mnkl,ijab,klcd,cdmj,abin', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||l,k>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,c,m,j)*t2_abab(a,b,i,n)
    energy += -0.125000000000000 * einsum('mnlk,ijab,lkdc,dcmj,abin', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||l,k>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(c,d,m,j)*t2_abab(a,b,i,n)
    energy += -0.125000000000000 * einsum('mnlk,ijab,lkcd,cdmj,abin', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,j,m)*t2_abab(a,b,i,n)
    energy += -0.125000000000000 * einsum('nmkl,ijab,kldc,dcjm,abin', g_bbbb[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,j,m)*t2_abab(b,a,n,i)
    energy += -0.125000000000000 * einsum('nmkl,jiba,kldc,dcjm,bani', g_aaaa[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,j,m)*t2_abab(b,a,n,i)
    energy += -0.125000000000000 * einsum('nmkl,jiba,kldc,dcjm,bani', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,j,m)*t2_abab(b,a,n,i)
    energy += -0.125000000000000 * einsum('nmkl,jiba,klcd,cdjm,bani', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||l,k>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,j,m)*t2_abab(b,a,n,i)
    energy += -0.125000000000000 * einsum('nmlk,jiba,lkdc,dcjm,bani', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||l,k>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,j,m)*t2_abab(b,a,n,i)
    energy += -0.125000000000000 * einsum('nmlk,jiba,lkcd,cdjm,bani', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,j,m)*t2_abab(a,b,n,i)
    energy += -0.125000000000000 * einsum('nmkl,jiab,kldc,dcjm,abni', g_aaaa[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,c,j,m)*t2_abab(a,b,n,i)
    energy += -0.125000000000000 * einsum('nmkl,jiab,kldc,dcjm,abni', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(c,d,j,m)*t2_abab(a,b,n,i)
    energy += -0.125000000000000 * einsum('nmkl,jiab,klcd,cdjm,abni', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||l,k>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,c,j,m)*t2_abab(a,b,n,i)
    energy += -0.125000000000000 * einsum('nmlk,jiab,lkdc,dcjm,abni', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||l,k>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,d,j,m)*t2_abab(a,b,n,i)
    energy += -0.125000000000000 * einsum('nmlk,jiab,lkcd,cdjm,abni', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||k,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,m,j)*t2_bbbb(b,a,i,n)
    energy += -0.125000000000000 * einsum('mnkl,ijba,kldc,dcmj,bain', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||k,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,m,j)*t2_bbbb(b,a,i,n)
    energy += -0.125000000000000 * einsum('mnkl,ijba,klcd,cdmj,bain', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||l,k>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,m,j)*t2_bbbb(b,a,i,n)
    energy += -0.125000000000000 * einsum('mnlk,ijba,lkdc,dcmj,bain', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||l,k>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,m,j)*t2_bbbb(b,a,i,n)
    energy += -0.125000000000000 * einsum('mnlk,ijba,lkcd,cdmj,bain', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||k,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,j,m)*t2_bbbb(b,a,i,n)
    energy += -0.125000000000000 * einsum('nmkl,ijba,kldc,dcjm,bain', g_bbbb[o, o, o, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,j,m)*t2_aaaa(c,a,i,n)
    energy +=  0.250000000000000 * einsum('nmkl,ijba,kldc,dbjm,cain', g_aaaa[o, o, o, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||k,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_aaaa(d,b,j,m)*t2_abab(a,c,i,n)
    energy +=  0.250000000000000 * einsum('mnkl,ijba,kldc,dbjm,acin', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,j,m)*t2_aaaa(c,a,i,n)
    energy +=  0.250000000000000 * einsum('nmkl,ijba,klcd,bdjm,cain', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,k>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(d,b,j,m)*t2_abab(a,c,i,n)
    energy +=  0.250000000000000 * einsum('mnlk,ijba,lkdc,dbjm,acin', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||l,k>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,j,m)*t2_aaaa(c,a,i,n)
    energy +=  0.250000000000000 * einsum('nmlk,ijba,lkcd,bdjm,cain', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_bbbb*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,j,m)*t2_abab(a,c,i,n)
    energy +=  0.250000000000000 * einsum('nmkl,ijba,kldc,bdjm,acin', g_bbbb[o, o, o, o], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||k,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,m,j)*t2_abab(c,a,i,n)
    energy +=  0.250000000000000 * einsum('mnkl,ijba,klcd,bdmj,cain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,k>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,m,j)*t2_abab(c,a,i,n)
    energy +=  0.250000000000000 * einsum('mnlk,ijba,lkcd,bdmj,cain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_aaaa*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,b,m,j)*t2_aaaa(c,a,i,n)
    energy +=  0.250000000000000 * einsum('nmkl,ijab,kldc,dbmj,cain', g_aaaa[o, o, o, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||k,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,b,m,j)*t2_abab(a,c,i,n)
    energy +=  0.250000000000000 * einsum('mnkl,ijab,kldc,dbmj,acin', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_bbbb(d,b,j,m)*t2_aaaa(c,a,i,n)
    energy +=  0.250000000000000 * einsum('nmkl,ijab,klcd,dbjm,cain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,k>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,b,m,j)*t2_abab(a,c,i,n)
    energy +=  0.250000000000000 * einsum('mnlk,ijab,lkdc,dbmj,acin', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||l,k>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_bbbb(d,b,j,m)*t2_aaaa(c,a,i,n)
    energy +=  0.250000000000000 * einsum('nmlk,ijab,lkcd,dbjm,cain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,j,m)*t2_abab(a,c,i,n)
    energy +=  0.250000000000000 * einsum('nmkl,ijab,kldc,dbjm,acin', g_bbbb[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,j,m)*t2_abab(c,a,n,i)
    energy +=  0.250000000000000 * einsum('nmkl,jiba,kldc,dbjm,cani', g_aaaa[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||k,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_aaaa(d,b,j,m)*t2_bbbb(c,a,i,n)
    energy +=  0.250000000000000 * einsum('mnkl,jiba,kldc,dbjm,cain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,j,m)*t2_abab(c,a,n,i)
    energy +=  0.250000000000000 * einsum('nmkl,jiba,klcd,bdjm,cani', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,k>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_aaaa(d,b,j,m)*t2_bbbb(c,a,i,n)
    energy +=  0.250000000000000 * einsum('mnlk,jiba,lkdc,dbjm,cain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||l,k>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,j,m)*t2_abab(c,a,n,i)
    energy +=  0.250000000000000 * einsum('nmlk,jiba,lkcd,bdjm,cani', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_bbbb*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,j,m)*t2_bbbb(c,a,i,n)
    energy +=  0.250000000000000 * einsum('nmkl,jiba,kldc,bdjm,cain', g_bbbb[o, o, o, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,b,j,m)*t2_abab(a,c,n,i)
    energy +=  0.250000000000000 * einsum('nmkl,jiab,kldc,dbjm,acni', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||l,k>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,b,j,m)*t2_abab(a,c,n,i)
    energy +=  0.250000000000000 * einsum('nmlk,jiab,lkdc,dbjm,acni', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_aaaa*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,b,m,j)*t2_abab(c,a,n,i)
    energy +=  0.250000000000000 * einsum('nmkl,ijba,kldc,dbmj,cani', g_aaaa[o, o, o, o], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||k,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,b,m,j)*t2_bbbb(c,a,i,n)
    energy +=  0.250000000000000 * einsum('mnkl,ijba,kldc,dbmj,cain', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(d,b,j,m)*t2_abab(c,a,n,i)
    energy +=  0.250000000000000 * einsum('nmkl,ijba,klcd,dbjm,cani', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,k>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,b,m,j)*t2_bbbb(c,a,i,n)
    energy +=  0.250000000000000 * einsum('mnlk,ijba,lkdc,dbmj,cain', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||l,k>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(d,b,j,m)*t2_abab(c,a,n,i)
    energy +=  0.250000000000000 * einsum('nmlk,ijba,lkcd,dbjm,cani', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||k,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,j,m)*t2_bbbb(c,a,i,n)
    energy +=  0.250000000000000 * einsum('nmkl,ijba,kldc,dbjm,cain', g_bbbb[o, o, o, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_aaaa(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,kldc,dckm,bain', g_aaaa[o, o, o, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_aaaa(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,kldc,dckm,bain', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_aaaa(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,klcd,cdkm,bain', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_aaaa(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,lkdc,dcmk,bain', g_aaaa[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_aaaa(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,lkcd,cdmk,bain', g_aaaa[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_aaaa(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,kldc,dckm,bain', g_abab[o, o, o, o], l2_aaaa, l2_bbbb, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_abab(b,a,i,n)
    energy +=  0.250000000000000 * einsum('mnlj,ijba,kldc,dckm,bain', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_abab(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,kldc,dckm,bain', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_abab(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,klcd,cdkm,bain', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_abab(b,a,i,n)
    energy +=  0.250000000000000 * einsum('mnlj,ijba,lkdc,dcmk,bain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_abab(b,a,i,n)
    energy +=  0.250000000000000 * einsum('mnlj,ijba,lkcd,cdmk,bain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_abab(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,kldc,dckm,bain', g_bbbb[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_abab(a,b,i,n)
    energy +=  0.250000000000000 * einsum('mnlj,ijab,kldc,dckm,abin', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_abab(a,b,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijab,kldc,dckm,abin', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_abab(a,b,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijab,klcd,cdkm,abin', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_abab(a,b,i,n)
    energy +=  0.250000000000000 * einsum('mnlj,ijab,lkdc,dcmk,abin', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_abab(a,b,i,n)
    energy +=  0.250000000000000 * einsum('mnlj,ijab,lkcd,cdmk,abin', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_abab(a,b,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijab,kldc,dckm,abin', g_bbbb[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_abab(b,a,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiba,kldc,dckm,bani', g_aaaa[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_abab(b,a,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiba,kldc,dckm,bani', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_abab(b,a,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiba,klcd,cdkm,bani', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_abab(b,a,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiba,lkdc,dcmk,bani', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_abab(b,a,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiba,lkcd,cdmk,bani', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_abab(b,a,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiba,kldc,dckm,bani', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_abab(a,b,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiab,kldc,dckm,abni', g_aaaa[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_abab(a,b,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiab,kldc,dckm,abni', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_abab(a,b,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiab,klcd,cdkm,abni', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_abab(a,b,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiab,lkdc,dcmk,abni', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_abab(a,b,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiab,lkcd,cdmk,abni', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_abab(a,b,n,i)
    energy +=  0.250000000000000 * einsum('nmjl,jiab,kldc,dckm,abni', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,k,m)*t2_bbbb(b,a,i,n)
    energy +=  0.250000000000000 * einsum('mnlj,ijba,kldc,dckm,bain', g_abab[o, o, o, o], l2_bbbb, l2_aaaa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,k,m)*t2_bbbb(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,kldc,dckm,bain', g_bbbb[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,k,m)*t2_bbbb(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,klcd,cdkm,bain', g_bbbb[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,m,k)*t2_bbbb(b,a,i,n)
    energy +=  0.250000000000000 * einsum('mnlj,ijba,lkdc,dcmk,bain', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,m,k)*t2_bbbb(b,a,i,n)
    energy +=  0.250000000000000 * einsum('mnlj,ijba,lkcd,cdmk,bain', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,k,m)*t2_bbbb(b,a,i,n)
    energy +=  0.250000000000000 * einsum('nmjl,ijba,kldc,dckm,bain', g_bbbb[o, o, o, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,i,m)*t2_aaaa(b,a,k,n)
    energy += -0.250000000000000 * einsum('nmjl,ijba,kldc,dcim,bakn', g_aaaa[o, o, o, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,i,m)*t2_aaaa(b,a,k,n)
    energy += -0.250000000000000 * einsum('nmjl,ijba,kldc,dcim,bakn', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,i,m)*t2_aaaa(b,a,k,n)
    energy += -0.250000000000000 * einsum('nmjl,ijba,klcd,cdim,bakn', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||l,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,i,m)*t2_abab(b,a,k,n)
    energy += -0.250000000000000 * einsum('mnlj,ijba,kldc,dcim,bakn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,i,m)*t2_abab(b,a,k,n)
    energy += -0.250000000000000 * einsum('nmjl,ijba,kldc,dcim,bakn', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,i,m)*t2_abab(b,a,k,n)
    energy += -0.250000000000000 * einsum('nmjl,ijba,klcd,cdim,bakn', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,i,m)*t2_abab(b,a,n,k)
    energy +=  0.250000000000000 * einsum('nmlj,ijba,lkdc,dcim,bank', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,i,m)*t2_abab(b,a,n,k)
    energy +=  0.250000000000000 * einsum('nmlj,ijba,lkcd,cdim,bank', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||l,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,i,m)*t2_abab(a,b,k,n)
    energy += -0.250000000000000 * einsum('mnlj,ijab,kldc,dcim,abkn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,c,i,m)*t2_abab(a,b,k,n)
    energy += -0.250000000000000 * einsum('nmjl,ijab,kldc,dcim,abkn', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,d,i,m)*t2_abab(a,b,k,n)
    energy += -0.250000000000000 * einsum('nmjl,ijab,klcd,cdim,abkn', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,c,i,m)*t2_abab(a,b,n,k)
    energy +=  0.250000000000000 * einsum('nmlj,ijab,lkdc,dcim,abnk', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(c,d,i,m)*t2_abab(a,b,n,k)
    energy +=  0.250000000000000 * einsum('nmlj,ijab,lkcd,cdim,abnk', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,m,i)*t2_abab(b,a,k,n)
    energy +=  0.250000000000000 * einsum('mnjl,jiba,kldc,dcmi,bakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,m,i)*t2_abab(b,a,k,n)
    energy +=  0.250000000000000 * einsum('mnjl,jiba,klcd,cdmi,bakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,m,i)*t2_abab(b,a,n,k)
    energy += -0.250000000000000 * einsum('nmjl,jiba,lkdc,dcmi,bank', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,m,i)*t2_abab(b,a,n,k)
    energy += -0.250000000000000 * einsum('nmjl,jiba,lkcd,cdmi,bank', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,i,m)*t2_abab(b,a,n,k)
    energy += -0.250000000000000 * einsum('nmjl,jiba,kldc,dcim,bank', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,c,m,i)*t2_abab(a,b,k,n)
    energy +=  0.250000000000000 * einsum('mnjl,jiab,kldc,dcmi,abkn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(c,d,m,i)*t2_abab(a,b,k,n)
    energy +=  0.250000000000000 * einsum('mnjl,jiab,klcd,cdmi,abkn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,c,m,i)*t2_abab(a,b,n,k)
    energy += -0.250000000000000 * einsum('nmjl,jiab,lkdc,dcmi,abnk', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,d,m,i)*t2_abab(a,b,n,k)
    energy += -0.250000000000000 * einsum('nmjl,jiab,lkcd,cdmi,abnk', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,i,m)*t2_abab(a,b,n,k)
    energy += -0.250000000000000 * einsum('nmjl,jiab,kldc,dcim,abnk', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,m,i)*t2_bbbb(b,a,k,n)
    energy += -0.250000000000000 * einsum('mnlj,ijba,lkdc,dcmi,bakn', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,m,i)*t2_bbbb(b,a,k,n)
    energy += -0.250000000000000 * einsum('mnlj,ijba,lkcd,cdmi,bakn', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,i,m)*t2_bbbb(b,a,k,n)
    energy += -0.250000000000000 * einsum('nmjl,ijba,kldc,dcim,bakn', g_bbbb[o, o, o, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,k,m)*t2_aaaa(c,a,i,n)
    energy += -0.500000000000000 * einsum('nmjl,ijba,kldc,dbkm,cain', g_aaaa[o, o, o, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_aaaa(d,b,k,m)*t2_abab(a,c,i,n)
    energy += -0.500000000000000 * einsum('mnjl,ijba,kldc,dbkm,acin', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,k,m)*t2_aaaa(c,a,i,n)
    energy += -0.500000000000000 * einsum('nmjl,ijba,klcd,bdkm,cain', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,m,k)*t2_aaaa(c,a,i,n)
    energy += -0.500000000000000 * einsum('nmjl,ijba,lkcd,bdmk,cain', g_aaaa[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||j,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,m,k)*t2_abab(a,c,i,n)
    energy += -0.500000000000000 * einsum('mnjl,ijba,kldc,bdmk,acin', g_abab[o, o, o, o], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||l,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,k,m)*t2_abab(c,a,i,n)
    energy += -0.500000000000000 * einsum('mnlj,ijba,kldc,dbkm,cain', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,k,m)*t2_abab(c,a,i,n)
    energy += -0.500000000000000 * einsum('nmjl,ijba,klcd,bdkm,cain', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,m,k)*t2_abab(c,a,i,n)
    energy += -0.500000000000000 * einsum('mnlj,ijba,lkcd,bdmk,cain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||l,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,b,k,m)*t2_aaaa(c,a,i,n)
    energy += -0.500000000000000 * einsum('nmlj,ijab,kldc,dbkm,cain', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,b,k,m)*t2_abab(a,c,i,n)
    energy += -0.500000000000000 * einsum('nmjl,ijab,kldc,dbkm,acin', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,b,m,k)*t2_abab(a,c,i,n)
    energy += -0.500000000000000 * einsum('mnlj,ijab,lkdc,dbmk,acin', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_bbbb(d,b,k,m)*t2_aaaa(c,a,i,n)
    energy += -0.500000000000000 * einsum('nmlj,ijab,lkcd,dbkm,cain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,k,m)*t2_abab(a,c,i,n)
    energy += -0.500000000000000 * einsum('nmjl,ijab,kldc,dbkm,acin', g_bbbb[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,k,m)*t2_abab(c,a,n,i)
    energy += -0.500000000000000 * einsum('nmjl,jiba,kldc,dbkm,cani', g_aaaa[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_aaaa(d,b,k,m)*t2_bbbb(c,a,i,n)
    energy += -0.500000000000000 * einsum('mnjl,jiba,kldc,dbkm,cain', g_abab[o, o, o, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,k,m)*t2_abab(c,a,n,i)
    energy += -0.500000000000000 * einsum('nmjl,jiba,klcd,bdkm,cani', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,m,k)*t2_abab(c,a,n,i)
    energy += -0.500000000000000 * einsum('nmjl,jiba,lkcd,bdmk,cani', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||j,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,m,k)*t2_bbbb(c,a,i,n)
    energy += -0.500000000000000 * einsum('mnjl,jiba,kldc,bdmk,cain', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,b,k,m)*t2_abab(a,c,n,i)
    energy += -0.500000000000000 * einsum('nmjl,jiab,kldc,dbkm,acni', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,b,m,k)*t2_abab(a,c,n,i)
    energy += -0.500000000000000 * einsum('nmjl,jiab,lkdc,dbmk,acni', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,k,m)*t2_abab(a,c,n,i)
    energy += -0.500000000000000 * einsum('nmjl,jiab,kldc,dbkm,acni', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||l,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,b,k,m)*t2_abab(c,a,n,i)
    energy += -0.500000000000000 * einsum('nmlj,ijba,kldc,dbkm,cani', g_abab[o, o, o, o], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,b,k,m)*t2_bbbb(c,a,i,n)
    energy += -0.500000000000000 * einsum('nmjl,ijba,kldc,dbkm,cain', g_bbbb[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,b,m,k)*t2_bbbb(c,a,i,n)
    energy += -0.500000000000000 * einsum('mnlj,ijba,lkdc,dbmk,cain', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(d,b,k,m)*t2_abab(c,a,n,i)
    energy += -0.500000000000000 * einsum('nmlj,ijba,lkcd,dbkm,cani', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,k,m)*t2_bbbb(c,a,i,n)
    energy += -0.500000000000000 * einsum('nmjl,ijba,kldc,dbkm,cain', g_bbbb[o, o, o, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,i,m)*t2_aaaa(c,a,k,n)
    energy +=  0.500000000000000 * einsum('nmjl,ijba,kldc,dbim,cakn', g_aaaa[o, o, o, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,n||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_aaaa(d,b,i,m)*t2_abab(a,c,k,n)
    energy +=  0.500000000000000 * einsum('mnjl,ijba,kldc,dbim,ackn', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,i,m)*t2_aaaa(c,a,k,n)
    energy +=  0.500000000000000 * einsum('nmjl,ijba,klcd,bdim,cakn', g_abab[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(d,b,i,m)*t2_abab(a,c,n,k)
    energy += -0.500000000000000 * einsum('nmjl,ijba,lkdc,dbim,acnk', g_aaaa[o, o, o, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,i,m)*t2_abab(a,c,n,k)
    energy += -0.500000000000000 * einsum('nmjl,ijba,kldc,bdim,acnk', g_abab[o, o, o, o], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,n||l,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,i,m)*t2_abab(c,a,k,n)
    energy +=  0.500000000000000 * einsum('mnlj,ijba,kldc,dbim,cakn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,i,m)*t2_abab(c,a,k,n)
    energy +=  0.500000000000000 * einsum('nmjl,ijba,klcd,bdim,cakn', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(d,b,i,m)*t2_bbbb(c,a,k,n)
    energy += -0.500000000000000 * einsum('mnlj,ijba,lkdc,dbim,cakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,i,m)*t2_abab(c,a,n,k)
    energy += -0.500000000000000 * einsum('nmlj,ijba,lkcd,bdim,cank', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,i,m)*t2_bbbb(c,a,k,n)
    energy += -0.500000000000000 * einsum('nmjl,ijba,kldc,bdim,cakn', g_bbbb[o, o, o, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||l,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,b,i,m)*t2_aaaa(c,a,k,n)
    energy +=  0.500000000000000 * einsum('nmlj,ijab,kldc,dbim,cakn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,b,i,m)*t2_abab(a,c,k,n)
    energy +=  0.500000000000000 * einsum('nmjl,ijab,kldc,dbim,ackn', g_bbbb[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,b,i,m)*t2_abab(a,c,n,k)
    energy += -0.500000000000000 * einsum('nmlj,ijab,lkdc,dbim,acnk', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,m,i)*t2_abab(c,a,k,n)
    energy += -0.500000000000000 * einsum('mnjl,jiba,klcd,bdmi,cakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,m,i)*t2_abab(c,a,n,k)
    energy +=  0.500000000000000 * einsum('nmjl,jiba,lkcd,bdmi,cank', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,n||j,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,m,i)*t2_bbbb(c,a,k,n)
    energy +=  0.500000000000000 * einsum('mnjl,jiba,kldc,bdmi,cakn', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,b,m,i)*t2_aaaa(c,a,k,n)
    energy += -0.500000000000000 * einsum('nmjl,jiab,kldc,dbmi,cakn', g_aaaa[o, o, o, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,b,m,i)*t2_abab(a,c,k,n)
    energy += -0.500000000000000 * einsum('mnjl,jiab,kldc,dbmi,ackn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_bbbb(d,b,i,m)*t2_aaaa(c,a,k,n)
    energy += -0.500000000000000 * einsum('nmjl,jiab,klcd,dbim,cakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,b,m,i)*t2_abab(a,c,n,k)
    energy +=  0.500000000000000 * einsum('nmjl,jiab,lkdc,dbmi,acnk', g_aaaa[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,i,m)*t2_abab(a,c,n,k)
    energy +=  0.500000000000000 * einsum('nmjl,jiab,kldc,dbim,acnk', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,n||l,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,b,m,i)*t2_abab(c,a,k,n)
    energy += -0.500000000000000 * einsum('mnlj,ijba,kldc,dbmi,cakn', g_abab[o, o, o, o], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <n,m||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(d,b,i,m)*t2_abab(c,a,k,n)
    energy += -0.500000000000000 * einsum('nmjl,ijba,klcd,dbim,cakn', g_bbbb[o, o, o, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,n||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,b,m,i)*t2_bbbb(c,a,k,n)
    energy +=  0.500000000000000 * einsum('mnlj,ijba,lkdc,dbmi,cakn', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(d,b,i,m)*t2_abab(c,a,n,k)
    energy +=  0.500000000000000 * einsum('nmlj,ijba,lkcd,dbim,cank', g_abab[o, o, o, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <n,m||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,i,m)*t2_bbbb(c,a,k,n)
    energy +=  0.500000000000000 * einsum('nmjl,ijba,kldc,dbim,cakn', g_bbbb[o, o, o, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,l,m)*t2_aaaa(b,a,k,n)
    energy += -0.125000000000000 * einsum('nmij,ijba,kldc,dclm,bakn', g_aaaa[o, o, o, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <n,m||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,m,l)*t2_aaaa(b,a,k,n)
    energy +=  0.125000000000000 * einsum('nmij,ijba,kldc,dcml,bakn', g_aaaa[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <n,m||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,m,l)*t2_aaaa(b,a,k,n)
    energy +=  0.125000000000000 * einsum('nmij,ijba,klcd,cdml,bakn', g_aaaa[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,n||i,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,l,m)*t2_abab(b,a,k,n)
    energy +=  0.125000000000000 * einsum('mnij,ijba,kldc,dclm,bakn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,m,l)*t2_abab(b,a,k,n)
    energy += -0.125000000000000 * einsum('mnij,ijba,kldc,dcml,bakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,m,l)*t2_abab(b,a,k,n)
    energy += -0.125000000000000 * einsum('mnij,ijba,klcd,cdml,bakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,l,m)*t2_abab(b,a,n,k)
    energy += -0.125000000000000 * einsum('nmij,ijba,lkdc,dclm,bank', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,l,m)*t2_abab(b,a,n,k)
    energy += -0.125000000000000 * einsum('nmij,ijba,lkcd,cdlm,bank', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <n,m||i,j>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,l,m)*t2_abab(b,a,n,k)
    energy +=  0.125000000000000 * einsum('nmij,ijba,kldc,dclm,bank', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,n||i,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,l,m)*t2_abab(a,b,k,n)
    energy +=  0.125000000000000 * einsum('mnij,ijab,kldc,dclm,abkn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,c,m,l)*t2_abab(a,b,k,n)
    energy += -0.125000000000000 * einsum('mnij,ijab,kldc,dcml,abkn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,d,m,l)*t2_abab(a,b,k,n)
    energy += -0.125000000000000 * einsum('mnij,ijab,klcd,cdml,abkn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,c,l,m)*t2_abab(a,b,n,k)
    energy += -0.125000000000000 * einsum('nmij,ijab,lkdc,dclm,abnk', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(c,d,l,m)*t2_abab(a,b,n,k)
    energy += -0.125000000000000 * einsum('nmij,ijab,lkcd,cdlm,abnk', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <n,m||i,j>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,l,m)*t2_abab(a,b,n,k)
    energy +=  0.125000000000000 * einsum('nmij,ijab,kldc,dclm,abnk', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,n||j,i>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,l,m)*t2_abab(b,a,k,n)
    energy +=  0.125000000000000 * einsum('mnji,jiba,kldc,dclm,bakn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(d,c,m,l)*t2_abab(b,a,k,n)
    energy += -0.125000000000000 * einsum('mnji,jiba,kldc,dcml,bakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(c,d,m,l)*t2_abab(b,a,k,n)
    energy += -0.125000000000000 * einsum('mnji,jiba,klcd,cdml,bakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,l,m)*t2_abab(b,a,n,k)
    energy += -0.125000000000000 * einsum('nmji,jiba,lkdc,dclm,bank', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,l,m)*t2_abab(b,a,n,k)
    energy += -0.125000000000000 * einsum('nmji,jiba,lkcd,cdlm,bank', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <n,m||j,i>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,l,m)*t2_abab(b,a,n,k)
    energy +=  0.125000000000000 * einsum('nmji,jiba,kldc,dclm,bank', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,n||j,i>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(d,c,l,m)*t2_abab(a,b,k,n)
    energy +=  0.125000000000000 * einsum('mnji,jiab,kldc,dclm,abkn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,c,m,l)*t2_abab(a,b,k,n)
    energy += -0.125000000000000 * einsum('mnji,jiab,kldc,dcml,abkn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,n||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(c,d,m,l)*t2_abab(a,b,k,n)
    energy += -0.125000000000000 * einsum('mnji,jiab,klcd,cdml,abkn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,c,l,m)*t2_abab(a,b,n,k)
    energy += -0.125000000000000 * einsum('nmji,jiab,lkdc,dclm,abnk', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,d,l,m)*t2_abab(a,b,n,k)
    energy += -0.125000000000000 * einsum('nmji,jiab,lkcd,cdlm,abnk', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <n,m||j,i>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,l,m)*t2_abab(a,b,n,k)
    energy +=  0.125000000000000 * einsum('nmji,jiab,kldc,dclm,abnk', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <n,m||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,c,l,m)*t2_bbbb(b,a,k,n)
    energy +=  0.125000000000000 * einsum('nmij,ijba,lkdc,dclm,bakn', g_bbbb[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <n,m||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,d,l,m)*t2_bbbb(b,a,k,n)
    energy +=  0.125000000000000 * einsum('nmij,ijba,lkcd,cdlm,bakn', g_bbbb[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <n,m||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,c,l,m)*t2_bbbb(b,a,k,n)
    energy += -0.125000000000000 * einsum('nmij,ijba,kldc,dclm,bakn', g_bbbb[o, o, o, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,l,m)*t2_aaaa(c,a,k,n)
    energy +=  0.250000000000000 * einsum('nmij,ijba,kldc,dblm,cakn', g_aaaa[o, o, o, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,m,l)*t2_aaaa(c,a,k,n)
    energy += -0.250000000000000 * einsum('nmij,ijba,klcd,bdml,cakn', g_aaaa[o, o, o, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(d,b,l,m)*t2_abab(a,c,n,k)
    energy += -0.250000000000000 * einsum('nmij,ijba,lkdc,dblm,acnk', g_aaaa[o, o, o, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,m,l)*t2_abab(a,c,n,k)
    energy +=  0.250000000000000 * einsum('nmij,ijba,kldc,bdml,acnk', g_aaaa[o, o, o, o], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||i,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,l,m)*t2_abab(c,a,k,n)
    energy += -0.250000000000000 * einsum('mnij,ijba,kldc,dblm,cakn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,m,l)*t2_abab(c,a,k,n)
    energy +=  0.250000000000000 * einsum('mnij,ijba,klcd,bdml,cakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(d,b,l,m)*t2_bbbb(c,a,k,n)
    energy +=  0.250000000000000 * einsum('mnij,ijba,lkdc,dblm,cakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,l,m)*t2_abab(c,a,n,k)
    energy +=  0.250000000000000 * einsum('nmij,ijba,lkcd,bdlm,cank', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||i,j>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,m,l)*t2_bbbb(c,a,k,n)
    energy += -0.250000000000000 * einsum('mnij,ijba,kldc,bdml,cakn', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||i,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,b,l,m)*t2_aaaa(c,a,k,n)
    energy += -0.250000000000000 * einsum('nmij,ijab,kldc,dblm,cakn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,b,m,l)*t2_abab(a,c,k,n)
    energy +=  0.250000000000000 * einsum('mnij,ijab,kldc,dbml,ackn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_bbbb(d,b,l,m)*t2_aaaa(c,a,k,n)
    energy +=  0.250000000000000 * einsum('nmij,ijab,klcd,dblm,cakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,b,l,m)*t2_abab(a,c,n,k)
    energy +=  0.250000000000000 * einsum('nmij,ijab,lkdc,dblm,acnk', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||i,j>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,l,m)*t2_abab(a,c,n,k)
    energy += -0.250000000000000 * einsum('nmij,ijab,kldc,dblm,acnk', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||j,i>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(d,b,l,m)*t2_abab(c,a,k,n)
    energy += -0.250000000000000 * einsum('mnji,jiba,kldc,dblm,cakn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(b,d,m,l)*t2_abab(c,a,k,n)
    energy +=  0.250000000000000 * einsum('mnji,jiba,klcd,bdml,cakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_aaaa(d,b,l,m)*t2_bbbb(c,a,k,n)
    energy +=  0.250000000000000 * einsum('mnji,jiba,lkdc,dblm,cakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(b,d,l,m)*t2_abab(c,a,n,k)
    energy +=  0.250000000000000 * einsum('nmji,jiba,lkcd,bdlm,cank', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,n||j,i>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,d,m,l)*t2_bbbb(c,a,k,n)
    energy += -0.250000000000000 * einsum('mnji,jiba,kldc,bdml,cakn', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,i>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,b,l,m)*t2_aaaa(c,a,k,n)
    energy += -0.250000000000000 * einsum('nmji,jiab,kldc,dblm,cakn', g_abab[o, o, o, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,n||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,b,m,l)*t2_abab(a,c,k,n)
    energy +=  0.250000000000000 * einsum('mnji,jiab,kldc,dbml,ackn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_bbbb(d,b,l,m)*t2_aaaa(c,a,k,n)
    energy +=  0.250000000000000 * einsum('nmji,jiab,klcd,dblm,cakn', g_abab[o, o, o, o], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,b,l,m)*t2_abab(a,c,n,k)
    energy +=  0.250000000000000 * einsum('nmji,jiab,lkdc,dblm,acnk', g_abab[o, o, o, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||j,i>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,l,m)*t2_abab(a,c,n,k)
    energy += -0.250000000000000 * einsum('nmji,jiab,kldc,dblm,acnk', g_abab[o, o, o, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,b,l,m)*t2_abab(c,a,k,n)
    energy +=  0.250000000000000 * einsum('nmij,ijba,kldc,dblm,cakn', g_bbbb[o, o, o, o], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(d,b,l,m)*t2_abab(c,a,k,n)
    energy += -0.250000000000000 * einsum('nmij,ijba,klcd,dblm,cakn', g_bbbb[o, o, o, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <n,m||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,b,l,m)*t2_bbbb(c,a,k,n)
    energy += -0.250000000000000 * einsum('nmij,ijba,lkdc,dblm,cakn', g_bbbb[o, o, o, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <n,m||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(d,b,l,m)*t2_bbbb(c,a,k,n)
    energy +=  0.250000000000000 * einsum('nmij,ijba,kldc,dblm,cakn', g_bbbb[o, o, o, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,j,k)*t2_aaaa(b,a,i,m)
    energy +=  0.500000000000000 * einsum('mdel,ijba,kldc,ecjk,baim', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_aaaa(e,c,j,k)*t2_aaaa(b,a,i,m)
    energy += -0.500000000000000 * einsum('mdel,ijba,klcd,ecjk,baim', g_abab[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,j,k)*t2_aaaa(b,a,i,m)
    energy += -0.500000000000000 * einsum('mdel,ijba,lkdc,ecjk,baim', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||l,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,j,k)*t2_aaaa(b,a,i,m)
    energy +=  0.500000000000000 * einsum('mdle,ijba,lkcd,cejk,baim', g_abab[o, v, o, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,k)*t2_aaaa(b,a,i,m)
    energy +=  0.500000000000000 * einsum('mdel,ijba,kldc,ecjk,baim', g_abab[o, v, v, o], l2_aaaa, l2_bbbb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,e>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(c,e,k,j)*t2_abab(b,a,i,m)
    energy +=  0.500000000000000 * einsum('dmle,ijba,kldc,cekj,baim', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,j)*t2_abab(b,a,i,m)
    energy +=  0.500000000000000 * einsum('dmel,ijba,kldc,eckj,baim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,j)*t2_abab(b,a,i,m)
    energy += -0.500000000000000 * einsum('mdel,ijba,klcd,cekj,baim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_bbbb(e,c,j,k)*t2_abab(b,a,i,m)
    energy += -0.500000000000000 * einsum('dmle,ijba,lkdc,ecjk,baim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,k)*t2_abab(b,a,i,m)
    energy +=  0.500000000000000 * einsum('mdel,ijba,kldc,ecjk,baim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(c,e,k,j)*t2_abab(a,b,i,m)
    energy +=  0.500000000000000 * einsum('dmle,ijab,kldc,cekj,abim', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,k,j)*t2_abab(a,b,i,m)
    energy +=  0.500000000000000 * einsum('dmel,ijab,kldc,eckj,abim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,k,j)*t2_abab(a,b,i,m)
    energy += -0.500000000000000 * einsum('mdel,ijab,klcd,cekj,abim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_bbbb(e,c,j,k)*t2_abab(a,b,i,m)
    energy += -0.500000000000000 * einsum('dmle,ijab,lkdc,ecjk,abim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,k)*t2_abab(a,b,i,m)
    energy +=  0.500000000000000 * einsum('mdel,ijab,kldc,ecjk,abim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,j,k)*t2_abab(b,a,m,i)
    energy +=  0.500000000000000 * einsum('mdel,jiba,kldc,ecjk,bami', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_aaaa(e,c,j,k)*t2_abab(b,a,m,i)
    energy += -0.500000000000000 * einsum('mdel,jiba,klcd,ecjk,bami', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,j,k)*t2_abab(b,a,m,i)
    energy += -0.500000000000000 * einsum('mdel,jiba,lkdc,ecjk,bami', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,j,k)*t2_abab(b,a,m,i)
    energy +=  0.500000000000000 * einsum('mdle,jiba,lkcd,cejk,bami', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,k)*t2_abab(b,a,m,i)
    energy +=  0.500000000000000 * einsum('mdel,jiba,kldc,ecjk,bami', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,j,k)*t2_abab(a,b,m,i)
    energy +=  0.500000000000000 * einsum('mdel,jiab,kldc,ecjk,abmi', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_aaaa(e,c,j,k)*t2_abab(a,b,m,i)
    energy += -0.500000000000000 * einsum('mdel,jiab,klcd,ecjk,abmi', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,j,k)*t2_abab(a,b,m,i)
    energy += -0.500000000000000 * einsum('mdel,jiab,lkdc,ecjk,abmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,j,k)*t2_abab(a,b,m,i)
    energy +=  0.500000000000000 * einsum('mdle,jiab,lkcd,cejk,abmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,k)*t2_abab(a,b,m,i)
    energy +=  0.500000000000000 * einsum('mdel,jiab,kldc,ecjk,abmi', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,e>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(c,e,k,j)*t2_bbbb(b,a,i,m)
    energy +=  0.500000000000000 * einsum('dmle,ijba,kldc,cekj,baim', g_abab[v, o, o, v], l2_bbbb, l2_aaaa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,j)*t2_bbbb(b,a,i,m)
    energy +=  0.500000000000000 * einsum('dmel,ijba,kldc,eckj,baim', g_abab[v, o, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,j)*t2_bbbb(b,a,i,m)
    energy += -0.500000000000000 * einsum('mdel,ijba,klcd,cekj,baim', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_bbbb(e,c,j,k)*t2_bbbb(b,a,i,m)
    energy += -0.500000000000000 * einsum('dmle,ijba,lkdc,ecjk,baim', g_abab[v, o, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,k)*t2_bbbb(b,a,i,m)
    energy +=  0.500000000000000 * einsum('mdel,ijba,kldc,ecjk,baim', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,j)*t2_aaaa(b,a,k,m)
    energy +=  0.250000000000000 * einsum('mdel,ijba,kldc,ecij,bakm', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_aaaa(e,c,i,j)*t2_aaaa(b,a,k,m)
    energy += -0.250000000000000 * einsum('mdel,ijba,klcd,ecij,bakm', g_abab[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||l,e>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(c,e,i,j)*t2_abab(b,a,k,m)
    energy += -0.250000000000000 * einsum('dmle,ijba,kldc,ceij,bakm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(b,a,k,m)
    energy += -0.250000000000000 * einsum('dmel,ijba,kldc,ecij,bakm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,i,j)*t2_abab(b,a,k,m)
    energy +=  0.250000000000000 * einsum('mdel,ijba,klcd,ceij,bakm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_aaaa*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,i,j)*t2_abab(b,a,m,k)
    energy +=  0.250000000000000 * einsum('mdel,ijba,lkdc,ecij,bamk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,i,j)*t2_abab(b,a,m,k)
    energy += -0.250000000000000 * einsum('mdle,ijba,lkcd,ceij,bamk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||e,l>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(b,a,m,k)
    energy += -0.250000000000000 * einsum('mdel,ijba,kldc,ecij,bamk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||l,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(c,e,i,j)*t2_abab(a,b,k,m)
    energy += -0.250000000000000 * einsum('dmle,ijab,kldc,ceij,abkm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(a,b,k,m)
    energy += -0.250000000000000 * einsum('dmel,ijab,kldc,ecij,abkm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,i,j)*t2_abab(a,b,k,m)
    energy +=  0.250000000000000 * einsum('mdel,ijab,klcd,ceij,abkm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_aaaa*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,i,j)*t2_abab(a,b,m,k)
    energy +=  0.250000000000000 * einsum('mdel,ijab,lkdc,ecij,abmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,i,j)*t2_abab(a,b,m,k)
    energy += -0.250000000000000 * einsum('mdle,ijab,lkcd,ceij,abmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||e,l>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(a,b,m,k)
    energy += -0.250000000000000 * einsum('mdel,ijab,kldc,ecij,abmk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||l,e>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_abab(c,e,j,i)*t2_abab(b,a,k,m)
    energy += -0.250000000000000 * einsum('dmle,jiba,kldc,ceji,bakm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,j,i)*t2_abab(b,a,k,m)
    energy += -0.250000000000000 * einsum('dmel,jiba,kldc,ecji,bakm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_bbbb*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,j,i)*t2_abab(b,a,k,m)
    energy +=  0.250000000000000 * einsum('mdel,jiba,klcd,ceji,bakm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,j,i)*t2_abab(b,a,m,k)
    energy +=  0.250000000000000 * einsum('mdel,jiba,lkdc,ecji,bamk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,j,i)*t2_abab(b,a,m,k)
    energy += -0.250000000000000 * einsum('mdle,jiba,lkcd,ceji,bamk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||e,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,i)*t2_abab(b,a,m,k)
    energy += -0.250000000000000 * einsum('mdel,jiba,kldc,ecji,bamk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||l,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(c,e,j,i)*t2_abab(a,b,k,m)
    energy += -0.250000000000000 * einsum('dmle,jiab,kldc,ceji,abkm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,j,i)*t2_abab(a,b,k,m)
    energy += -0.250000000000000 * einsum('dmel,jiab,kldc,ecji,abkm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_bbbb*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,j,i)*t2_abab(a,b,k,m)
    energy +=  0.250000000000000 * einsum('mdel,jiab,klcd,ceji,abkm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,j,i)*t2_abab(a,b,m,k)
    energy +=  0.250000000000000 * einsum('mdel,jiab,lkdc,ecji,abmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,j,i)*t2_abab(a,b,m,k)
    energy += -0.250000000000000 * einsum('mdle,jiab,lkcd,ceji,abmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||e,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,i)*t2_abab(a,b,m,k)
    energy += -0.250000000000000 * einsum('mdel,jiab,kldc,ecji,abmk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_bbbb(e,c,i,j)*t2_bbbb(b,a,k,m)
    energy += -0.250000000000000 * einsum('dmle,ijba,lkdc,ecij,bakm', g_abab[v, o, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,i,j)*t2_bbbb(b,a,k,m)
    energy +=  0.250000000000000 * einsum('mdel,ijba,kldc,ecij,bakm', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_aaaa(c,a,i,m)
    energy += -1.000000000000000 * einsum('mdel,ijba,kldc,ebjk,caim', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_abab(a,c,i,m)
    energy += -1.000000000000000 * einsum('dmel,ijba,kldc,ebjk,acim', g_abab[v, o, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_aaaa(e,b,j,k)*t2_aaaa(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mdel,ijba,klcd,ebjk,caim', g_abab[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <d,m||l,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,j,k)*t2_abab(a,c,i,m)
    energy +=  1.000000000000000 * einsum('dmle,ijba,lkdc,bejk,acim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||l,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,j,k)*t2_aaaa(c,a,i,m)
    energy += -1.000000000000000 * einsum('mdle,ijba,lkcd,bejk,caim', g_abab[o, v, o, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_bbbb*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,j,k)*t2_abab(a,c,i,m)
    energy += -1.000000000000000 * einsum('mdel,ijba,kldc,bejk,acim', g_bbbb[o, v, v, o], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||l,e>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(b,e,k,j)*t2_abab(c,a,i,m)
    energy += -1.000000000000000 * einsum('dmle,ijba,kldc,bekj,caim', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,k,j)*t2_abab(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mdel,ijba,klcd,bekj,caim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_aaaa*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_aaaa(c,a,i,m)
    energy += -1.000000000000000 * einsum('mdel,ijab,kldc,ebkj,caim', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_abab(a,c,i,m)
    energy += -1.000000000000000 * einsum('dmel,ijab,kldc,ebkj,acim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_aaaa(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mdel,ijab,klcd,ebkj,caim', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <d,m||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_abab(a,c,i,m)
    energy +=  1.000000000000000 * einsum('dmle,ijab,lkdc,ebjk,acim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_aaaa(c,a,i,m)
    energy += -1.000000000000000 * einsum('mdle,ijab,lkcd,ebjk,caim', g_abab[o, v, o, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_abab(a,c,i,m)
    energy += -1.000000000000000 * einsum('mdel,ijab,kldc,ebjk,acim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_abab(c,a,m,i)
    energy += -1.000000000000000 * einsum('mdel,jiba,kldc,ebjk,cami', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_aaaa(e,b,j,k)*t2_bbbb(c,a,i,m)
    energy += -1.000000000000000 * einsum('dmel,jiba,kldc,ebjk,caim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_aaaa(e,b,j,k)*t2_abab(c,a,m,i)
    energy +=  1.000000000000000 * einsum('mdel,jiba,klcd,ebjk,cami', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <d,m||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,j,k)*t2_bbbb(c,a,i,m)
    energy +=  1.000000000000000 * einsum('dmle,jiba,lkdc,bejk,caim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,j,k)*t2_abab(c,a,m,i)
    energy += -1.000000000000000 * einsum('mdle,jiba,lkcd,bejk,cami', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_bbbb*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,j,k)*t2_bbbb(c,a,i,m)
    energy += -1.000000000000000 * einsum('mdel,jiba,kldc,bejk,caim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,j,k)*t2_abab(a,c,m,i)
    energy +=  1.000000000000000 * einsum('mdel,jiab,lkdc,ebjk,acmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,b,j,k)*t2_abab(a,c,m,i)
    energy += -1.000000000000000 * einsum('mdel,jiab,kldc,ebjk,acmi', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_aaaa*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(e,b,k,j)*t2_abab(c,a,m,i)
    energy += -1.000000000000000 * einsum('mdel,ijba,kldc,ebkj,cami', g_aaaa[o, v, v, o], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||e,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,b,k,j)*t2_bbbb(c,a,i,m)
    energy += -1.000000000000000 * einsum('dmel,ijba,kldc,ebkj,caim', g_abab[v, o, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,b,k,j)*t2_abab(c,a,m,i)
    energy +=  1.000000000000000 * einsum('mdel,ijba,klcd,ebkj,cami', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <d,m||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_bbbb(e,b,j,k)*t2_bbbb(c,a,i,m)
    energy +=  1.000000000000000 * einsum('dmle,ijba,lkdc,ebjk,caim', g_abab[v, o, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(e,b,j,k)*t2_abab(c,a,m,i)
    energy += -1.000000000000000 * einsum('mdle,ijba,lkcd,ebjk,cami', g_abab[o, v, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,k)*t2_bbbb(c,a,i,m)
    energy += -1.000000000000000 * einsum('mdel,ijba,kldc,ebjk,caim', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_aaaa(c,a,k,m)
    energy += -0.500000000000000 * einsum('mdel,ijba,kldc,ebij,cakm', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_abab(a,c,k,m)
    energy += -0.500000000000000 * einsum('dmel,ijba,kldc,ebij,ackm', g_abab[v, o, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_aaaa(e,b,i,j)*t2_aaaa(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mdel,ijba,klcd,ebij,cakm', g_abab[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,b,i,j)*t2_abab(a,c,m,k)
    energy +=  0.500000000000000 * einsum('mdel,ijba,lkdc,ebij,acmk', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_aaaa(e,b,i,j)*t2_abab(a,c,m,k)
    energy += -0.500000000000000 * einsum('mdel,ijba,kldc,ebij,acmk', g_abab[o, v, v, o], l2_aaaa, l2_bbbb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,e>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(b,e,i,j)*t2_abab(c,a,k,m)
    energy +=  0.500000000000000 * einsum('dmle,ijba,kldc,beij,cakm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,i,j)*t2_abab(c,a,k,m)
    energy += -0.500000000000000 * einsum('mdel,ijba,klcd,beij,cakm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,i,j)*t2_bbbb(c,a,k,m)
    energy += -0.500000000000000 * einsum('dmle,ijba,lkdc,beij,cakm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,i,j)*t2_abab(c,a,m,k)
    energy +=  0.500000000000000 * einsum('mdle,ijba,lkcd,beij,camk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,i,j)*t2_bbbb(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mdel,ijba,kldc,beij,cakm', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_aaaa*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(e,b,i,j)*t2_aaaa(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mdel,ijab,kldc,ebij,cakm', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,i,j)*t2_abab(a,c,k,m)
    energy +=  0.500000000000000 * einsum('dmel,ijab,kldc,ebij,ackm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,b,i,j)*t2_aaaa(c,a,k,m)
    energy += -0.500000000000000 * einsum('mdel,ijab,klcd,ebij,cakm', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_aaaa*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,i,j)*t2_abab(a,c,m,k)
    energy += -0.500000000000000 * einsum('mdel,ijab,lkdc,ebij,acmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,j)*t2_abab(a,c,m,k)
    energy +=  0.500000000000000 * einsum('mdel,ijab,kldc,ebij,acmk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,e>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_abab(b,e,j,i)*t2_abab(c,a,k,m)
    energy +=  0.500000000000000 * einsum('dmle,jiba,kldc,beji,cakm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_bbbb*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,j,i)*t2_abab(c,a,k,m)
    energy += -0.500000000000000 * einsum('mdel,jiba,klcd,beji,cakm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,j,i)*t2_bbbb(c,a,k,m)
    energy += -0.500000000000000 * einsum('dmle,jiba,lkdc,beji,cakm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,j,i)*t2_abab(c,a,m,k)
    energy +=  0.500000000000000 * einsum('mdle,jiba,lkcd,beji,camk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_bbbb*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,j,i)*t2_bbbb(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mdel,jiba,kldc,beji,cakm', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(e,b,j,i)*t2_aaaa(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mdel,jiab,kldc,ebji,cakm', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,j,i)*t2_abab(a,c,k,m)
    energy +=  0.500000000000000 * einsum('dmel,jiab,kldc,ebji,ackm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(e,b,j,i)*t2_aaaa(c,a,k,m)
    energy += -0.500000000000000 * einsum('mdel,jiab,klcd,ebji,cakm', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,j,i)*t2_abab(a,c,m,k)
    energy += -0.500000000000000 * einsum('mdel,jiab,lkdc,ebji,acmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,b,j,i)*t2_abab(a,c,m,k)
    energy +=  0.500000000000000 * einsum('mdel,jiab,kldc,ebji,acmk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,e>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_bbbb(e,b,i,j)*t2_abab(c,a,k,m)
    energy += -0.500000000000000 * einsum('dmle,ijba,kldc,ebij,cakm', g_abab[v, o, o, v], l2_bbbb, l2_aaaa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,b,i,j)*t2_abab(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mdel,ijba,klcd,ebij,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_bbbb(e,b,i,j)*t2_bbbb(c,a,k,m)
    energy +=  0.500000000000000 * einsum('dmle,ijba,lkdc,ebij,cakm', g_abab[v, o, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(e,b,i,j)*t2_abab(c,a,m,k)
    energy += -0.500000000000000 * einsum('mdle,ijba,lkcd,ebij,camk', g_abab[o, v, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,i,j)*t2_bbbb(c,a,k,m)
    energy += -0.500000000000000 * einsum('mdel,ijba,kldc,ebij,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_aaaa(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,kldc,eckl,baim', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_aaaa(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,kldc,eckl,baim', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_aaaa(b,a,i,m)
    energy += -0.250000000000000 * einsum('mdje,ijba,klcd,cekl,baim', g_abab[o, v, o, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_aaaa(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,lkdc,eclk,baim', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_aaaa(b,a,i,m)
    energy += -0.250000000000000 * einsum('mdje,ijba,lkcd,celk,baim', g_abab[o, v, o, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_aaaa(b,a,i,m)
    energy += -0.250000000000000 * einsum('mdje,ijba,kldc,eckl,baim', g_abab[o, v, o, v], l2_aaaa, l2_bbbb, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(b,a,i,m)
    energy += -0.250000000000000 * einsum('dmej,ijba,kldc,eckl,baim', g_abab[v, o, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(b,a,i,m)
    energy += -0.250000000000000 * einsum('dmej,ijba,kldc,eckl,baim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,klcd,cekl,baim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(b,a,i,m)
    energy += -0.250000000000000 * einsum('dmej,ijba,lkdc,eclk,baim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_bbbb*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,lkcd,celk,baim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,kldc,eckl,baim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(a,b,i,m)
    energy += -0.250000000000000 * einsum('dmej,ijab,kldc,eckl,abim', g_abab[v, o, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(a,b,i,m)
    energy += -0.250000000000000 * einsum('dmej,ijab,kldc,eckl,abim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(a,b,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijab,klcd,cekl,abim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(a,b,i,m)
    energy += -0.250000000000000 * einsum('dmej,ijab,lkdc,eclk,abim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(a,b,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijab,lkcd,celk,abim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(a,b,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijab,kldc,eckl,abim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(b,a,m,i)
    energy +=  0.250000000000000 * einsum('mdej,jiba,kldc,eckl,bami', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(b,a,m,i)
    energy +=  0.250000000000000 * einsum('mdej,jiba,kldc,eckl,bami', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(b,a,m,i)
    energy += -0.250000000000000 * einsum('mdje,jiba,klcd,cekl,bami', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(b,a,m,i)
    energy +=  0.250000000000000 * einsum('mdej,jiba,lkdc,eclk,bami', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(b,a,m,i)
    energy += -0.250000000000000 * einsum('mdje,jiba,lkcd,celk,bami', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||j,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(b,a,m,i)
    energy += -0.250000000000000 * einsum('mdje,jiba,kldc,eckl,bami', g_abab[o, v, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(a,b,m,i)
    energy +=  0.250000000000000 * einsum('mdej,jiab,kldc,eckl,abmi', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_aaaa*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(a,b,m,i)
    energy +=  0.250000000000000 * einsum('mdej,jiab,kldc,eckl,abmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(a,b,m,i)
    energy += -0.250000000000000 * einsum('mdje,jiab,klcd,cekl,abmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(a,b,m,i)
    energy +=  0.250000000000000 * einsum('mdej,jiab,lkdc,eclk,abmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(a,b,m,i)
    energy += -0.250000000000000 * einsum('mdje,jiab,lkcd,celk,abmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,d||j,e>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(a,b,m,i)
    energy += -0.250000000000000 * einsum('mdje,jiab,kldc,eckl,abmi', g_abab[o, v, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_bbbb(b,a,i,m)
    energy += -0.250000000000000 * einsum('dmej,ijba,kldc,eckl,baim', g_abab[v, o, v, o], l2_bbbb, l2_aaaa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_bbbb(b,a,i,m)
    energy += -0.250000000000000 * einsum('dmej,ijba,kldc,eckl,baim', g_abab[v, o, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_bbbb(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,klcd,cekl,baim', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,m||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_bbbb(b,a,i,m)
    energy += -0.250000000000000 * einsum('dmej,ijba,lkdc,eclk,baim', g_abab[v, o, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_bbbb(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,lkcd,celk,baim', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,d||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_bbbb(b,a,i,m)
    energy +=  0.250000000000000 * einsum('mdej,ijba,kldc,eckl,baim', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,l)*t2_aaaa(b,a,k,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,kldc,ecil,bakm', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,i,l)*t2_aaaa(b,a,k,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,kldc,ecil,bakm', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,i,l)*t2_aaaa(b,a,k,m)
    energy +=  0.500000000000000 * einsum('mdje,ijba,klcd,ceil,bakm', g_abab[o, v, o, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,l)*t2_abab(b,a,k,m)
    energy +=  0.500000000000000 * einsum('dmej,ijba,kldc,ecil,bakm', g_abab[v, o, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,i,l)*t2_abab(b,a,k,m)
    energy +=  0.500000000000000 * einsum('dmej,ijba,kldc,ecil,bakm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,i,l)*t2_abab(b,a,k,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,klcd,ceil,bakm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,c,i,l)*t2_abab(b,a,m,k)
    energy +=  0.500000000000000 * einsum('mdej,ijba,lkcd,ecil,bamk', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,j>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,l)*t2_abab(b,a,m,k)
    energy +=  0.500000000000000 * einsum('mdej,ijba,kldc,ecil,bamk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,l)*t2_abab(a,b,k,m)
    energy +=  0.500000000000000 * einsum('dmej,ijab,kldc,ecil,abkm', g_abab[v, o, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,i,l)*t2_abab(a,b,k,m)
    energy +=  0.500000000000000 * einsum('dmej,ijab,kldc,ecil,abkm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,i,l)*t2_abab(a,b,k,m)
    energy += -0.500000000000000 * einsum('mdej,ijab,klcd,ceil,abkm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_aaaa(e,c,i,l)*t2_abab(a,b,m,k)
    energy +=  0.500000000000000 * einsum('mdej,ijab,lkcd,ecil,abmk', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||e,j>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,l)*t2_abab(a,b,m,k)
    energy +=  0.500000000000000 * einsum('mdej,ijab,kldc,ecil,abmk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||j,e>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_abab(c,e,l,i)*t2_abab(b,a,k,m)
    energy +=  0.500000000000000 * einsum('dmje,jiba,kldc,celi,bakm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_bbbb(e,c,i,l)*t2_abab(b,a,k,m)
    energy +=  0.500000000000000 * einsum('dmje,jiba,kldc,ecil,bakm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,i)*t2_abab(b,a,m,k)
    energy += -0.500000000000000 * einsum('mdej,jiba,lkdc,ecli,bamk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,i)*t2_abab(b,a,m,k)
    energy +=  0.500000000000000 * einsum('mdje,jiba,lkcd,celi,bamk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,i,l)*t2_abab(b,a,m,k)
    energy +=  0.500000000000000 * einsum('mdje,jiba,kldc,ecil,bamk', g_abab[o, v, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||j,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(c,e,l,i)*t2_abab(a,b,k,m)
    energy +=  0.500000000000000 * einsum('dmje,jiab,kldc,celi,abkm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_bbbb(e,c,i,l)*t2_abab(a,b,k,m)
    energy +=  0.500000000000000 * einsum('dmje,jiab,kldc,ecil,abkm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,l,i)*t2_abab(a,b,m,k)
    energy += -0.500000000000000 * einsum('mdej,jiab,lkdc,ecli,abmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,l,i)*t2_abab(a,b,m,k)
    energy +=  0.500000000000000 * einsum('mdje,jiab,lkcd,celi,abmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,i,l)*t2_abab(a,b,m,k)
    energy +=  0.500000000000000 * einsum('mdje,jiab,kldc,ecil,abmk', g_abab[o, v, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,i)*t2_bbbb(b,a,k,m)
    energy +=  0.500000000000000 * einsum('dmej,ijba,lkdc,ecli,bakm', g_abab[v, o, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,i)*t2_bbbb(b,a,k,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,lkcd,celi,bakm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,i,l)*t2_bbbb(b,a,k,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,kldc,ecil,bakm', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,k,l)*t2_aaaa(c,a,i,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,kldc,ebkl,caim', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(b,e,k,l)*t2_abab(a,c,i,m)
    energy += -0.500000000000000 * einsum('dmje,ijba,kldc,bekl,acim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,k,l)*t2_aaaa(c,a,i,m)
    energy +=  0.500000000000000 * einsum('mdje,ijba,klcd,bekl,caim', g_abab[o, v, o, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,l,k)*t2_abab(a,c,i,m)
    energy += -0.500000000000000 * einsum('dmje,ijba,lkdc,belk,acim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,l,k)*t2_aaaa(c,a,i,m)
    energy +=  0.500000000000000 * einsum('mdje,ijba,lkcd,belk,caim', g_abab[o, v, o, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,k,l)*t2_abab(c,a,i,m)
    energy +=  0.500000000000000 * einsum('dmej,ijba,kldc,ebkl,caim', g_abab[v, o, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,k,l)*t2_abab(c,a,i,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,klcd,bekl,caim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,l,k)*t2_abab(c,a,i,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,lkcd,belk,caim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_abab(a,c,i,m)
    energy +=  0.500000000000000 * einsum('dmej,ijab,kldc,ebkl,acim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,b,k,l)*t2_aaaa(c,a,i,m)
    energy += -0.500000000000000 * einsum('mdej,ijab,klcd,ebkl,caim', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_abab(a,c,i,m)
    energy +=  0.500000000000000 * einsum('dmej,ijab,lkdc,eblk,acim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(e,b,l,k)*t2_aaaa(c,a,i,m)
    energy += -0.500000000000000 * einsum('mdej,ijab,lkcd,eblk,caim', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_abab(a,c,i,m)
    energy += -0.500000000000000 * einsum('mdej,ijab,kldc,ebkl,acim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,k,l)*t2_abab(c,a,m,i)
    energy += -0.500000000000000 * einsum('mdej,jiba,kldc,ebkl,cami', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(b,e,k,l)*t2_bbbb(c,a,i,m)
    energy += -0.500000000000000 * einsum('dmje,jiba,kldc,bekl,caim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,k,l)*t2_abab(c,a,m,i)
    energy +=  0.500000000000000 * einsum('mdje,jiba,klcd,bekl,cami', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,l,k)*t2_bbbb(c,a,i,m)
    energy += -0.500000000000000 * einsum('dmje,jiba,lkdc,belk,caim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,l,k)*t2_abab(c,a,m,i)
    energy +=  0.500000000000000 * einsum('mdje,jiba,lkcd,belk,cami', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_aaaa*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_abab(a,c,m,i)
    energy += -0.500000000000000 * einsum('mdej,jiab,kldc,ebkl,acmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_abab(a,c,m,i)
    energy += -0.500000000000000 * einsum('mdej,jiab,lkdc,eblk,acmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,e>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_abab(a,c,m,i)
    energy +=  0.500000000000000 * einsum('mdje,jiab,kldc,ebkl,acmi', g_abab[o, v, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_bbbb(c,a,i,m)
    energy +=  0.500000000000000 * einsum('dmej,ijba,kldc,ebkl,caim', g_abab[v, o, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,b,k,l)*t2_abab(c,a,m,i)
    energy += -0.500000000000000 * einsum('mdej,ijba,klcd,ebkl,cami', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_bbbb(c,a,i,m)
    energy +=  0.500000000000000 * einsum('dmej,ijba,lkdc,eblk,caim', g_abab[v, o, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,b,l,k)*t2_abab(c,a,m,i)
    energy += -0.500000000000000 * einsum('mdej,ijba,lkcd,eblk,cami', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_bbbb(c,a,i,m)
    energy += -0.500000000000000 * einsum('mdej,ijba,kldc,ebkl,caim', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,i,l)*t2_aaaa(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mdej,ijba,kldc,ebil,cakm', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <d,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(b,e,i,l)*t2_abab(a,c,k,m)
    energy +=  1.000000000000000 * einsum('dmje,ijba,kldc,beil,ackm', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,i,l)*t2_aaaa(c,a,k,m)
    energy += -1.000000000000000 * einsum('mdje,ijba,klcd,beil,cakm', g_abab[o, v, o, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,b,i,l)*t2_abab(a,c,m,k)
    energy += -1.000000000000000 * einsum('mdej,ijba,lkdc,ebil,acmk', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,i,l)*t2_abab(a,c,m,k)
    energy +=  1.000000000000000 * einsum('mdje,ijba,kldc,beil,acmk', g_abab[o, v, o, v], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||e,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,i,l)*t2_abab(c,a,k,m)
    energy += -1.000000000000000 * einsum('dmej,ijba,kldc,ebil,cakm', g_abab[v, o, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,i,l)*t2_abab(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mdej,ijba,klcd,beil,cakm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <d,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,b,i,l)*t2_bbbb(c,a,k,m)
    energy +=  1.000000000000000 * einsum('dmej,ijba,lkdc,ebil,cakm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,b,i,l)*t2_abab(c,a,m,k)
    energy += -1.000000000000000 * einsum('mdej,ijba,lkcd,ebil,camk', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,i,l)*t2_bbbb(c,a,k,m)
    energy += -1.000000000000000 * einsum('mdej,ijba,kldc,beil,cakm', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,i,l)*t2_abab(a,c,k,m)
    energy += -1.000000000000000 * einsum('dmej,ijab,kldc,ebil,ackm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,b,i,l)*t2_aaaa(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mdej,ijab,klcd,ebil,cakm', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,b,i,l)*t2_abab(a,c,m,k)
    energy += -1.000000000000000 * einsum('mdej,ijab,kldc,ebil,acmk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||j,e>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_abab(b,e,l,i)*t2_abab(c,a,k,m)
    energy += -1.000000000000000 * einsum('dmje,jiba,kldc,beli,cakm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <d,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,l,i)*t2_bbbb(c,a,k,m)
    energy +=  1.000000000000000 * einsum('dmje,jiba,lkdc,beli,cakm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,l,i)*t2_abab(c,a,m,k)
    energy += -1.000000000000000 * einsum('mdje,jiba,lkcd,beli,camk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(e,b,l,i)*t2_aaaa(c,a,k,m)
    energy += -1.000000000000000 * einsum('mdej,jiab,kldc,ebli,cakm', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_bbbb(e,b,i,l)*t2_abab(a,c,k,m)
    energy += -1.000000000000000 * einsum('dmje,jiab,kldc,ebil,ackm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_bbbb(e,b,i,l)*t2_aaaa(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mdje,jiab,klcd,ebil,cakm', g_abab[o, v, o, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,l,i)*t2_abab(a,c,m,k)
    energy +=  1.000000000000000 * einsum('mdej,jiab,lkdc,ebli,acmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||j,e>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,i,l)*t2_abab(a,c,m,k)
    energy += -1.000000000000000 * einsum('mdje,jiab,kldc,ebil,acmk', g_abab[o, v, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <d,m||e,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(e,b,l,i)*t2_abab(c,a,k,m)
    energy +=  1.000000000000000 * einsum('dmej,ijba,kldc,ebli,cakm', g_abab[v, o, v, o], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,d||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,b,i,l)*t2_abab(c,a,k,m)
    energy += -1.000000000000000 * einsum('mdej,ijba,klcd,ebil,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <d,m||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,b,l,i)*t2_bbbb(c,a,k,m)
    energy += -1.000000000000000 * einsum('dmej,ijba,lkdc,ebli,cakm', g_abab[v, o, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,b,l,i)*t2_abab(c,a,m,k)
    energy +=  1.000000000000000 * einsum('mdej,ijba,lkcd,ebli,camk', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,d||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,i,l)*t2_bbbb(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mdej,ijba,kldc,ebil,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,k)*t2_aaaa(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mbel,ijba,kldc,edjk,caim', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_aaaa(e,d,j,k)*t2_abab(a,c,i,m)
    energy +=  1.000000000000000 * einsum('bmel,ijba,kldc,edjk,acim', g_abab[v, o, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <b,m||l,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,j,k)*t2_abab(a,c,i,m)
    energy += -1.000000000000000 * einsum('bmle,ijba,lkdc,dejk,acim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,j,k)*t2_aaaa(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mbel,ijba,lkcd,edjk,caim', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,k)*t2_abab(a,c,i,m)
    energy +=  1.000000000000000 * einsum('bmel,ijba,kldc,edjk,acim', g_abab[v, o, v, o], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||l,e>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,e,k,j)*t2_abab(c,a,i,m)
    energy +=  1.000000000000000 * einsum('bmle,ijba,kldc,dekj,caim', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <b,m||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,k,j)*t2_abab(c,a,i,m)
    energy += -1.000000000000000 * einsum('bmel,ijba,klcd,edkj,caim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,k)*t2_abab(c,a,i,m)
    energy +=  1.000000000000000 * einsum('bmle,ijba,lkcd,edjk,caim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||l,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,e,k,j)*t2_aaaa(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mble,ijab,kldc,dekj,caim', g_abab[o, v, o, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,e,k,j)*t2_abab(a,c,i,m)
    energy +=  1.000000000000000 * einsum('mbel,ijab,kldc,dekj,acim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,d,k,j)*t2_aaaa(c,a,i,m)
    energy += -1.000000000000000 * einsum('mbel,ijab,klcd,edkj,caim', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,k)*t2_aaaa(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mble,ijab,lkcd,edjk,caim', g_abab[o, v, o, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,k)*t2_abab(a,c,i,m)
    energy +=  1.000000000000000 * einsum('mbel,ijab,kldc,edjk,acim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,k)*t2_abab(c,a,m,i)
    energy +=  1.000000000000000 * einsum('mbel,jiba,kldc,edjk,cami', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_aaaa(e,d,j,k)*t2_bbbb(c,a,i,m)
    energy +=  1.000000000000000 * einsum('bmel,jiba,kldc,edjk,caim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <b,m||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,j,k)*t2_bbbb(c,a,i,m)
    energy += -1.000000000000000 * einsum('bmle,jiba,lkdc,dejk,caim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,j,k)*t2_abab(c,a,m,i)
    energy +=  1.000000000000000 * einsum('mbel,jiba,lkcd,edjk,cami', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||e,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,k)*t2_bbbb(c,a,i,m)
    energy +=  1.000000000000000 * einsum('bmel,jiba,kldc,edjk,caim', g_abab[v, o, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_aaaa(e,d,j,k)*t2_abab(a,c,m,i)
    energy +=  1.000000000000000 * einsum('mbel,jiab,kldc,edjk,acmi', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,e,j,k)*t2_abab(a,c,m,i)
    energy += -1.000000000000000 * einsum('mble,jiab,lkdc,dejk,acmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,k)*t2_abab(a,c,m,i)
    energy +=  1.000000000000000 * einsum('mbel,jiab,kldc,edjk,acmi', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,e,k,j)*t2_abab(c,a,m,i)
    energy +=  1.000000000000000 * einsum('mble,ijba,kldc,dekj,cami', g_abab[o, v, o, v], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,e,k,j)*t2_bbbb(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mbel,ijba,kldc,dekj,caim', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,k,j)*t2_abab(c,a,m,i)
    energy += -1.000000000000000 * einsum('mbel,ijba,klcd,edkj,cami', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(e,d,j,k)*t2_abab(c,a,m,i)
    energy +=  1.000000000000000 * einsum('mble,ijba,lkcd,edjk,cami', g_abab[o, v, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,k)*t2_bbbb(c,a,i,m)
    energy +=  1.000000000000000 * einsum('mbel,ijba,kldc,edjk,caim', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_aaaa(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mbel,ijba,kldc,edij,cakm', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_aaaa(e,d,i,j)*t2_abab(a,c,k,m)
    energy +=  0.500000000000000 * einsum('bmel,ijba,kldc,edij,ackm', g_abab[v, o, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,j)*t2_abab(a,c,m,k)
    energy += -0.500000000000000 * einsum('mbel,ijba,lkdc,edij,acmk', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||l,e>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(c,a,k,m)
    energy += -0.500000000000000 * einsum('bmle,ijba,kldc,deij,cakm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_abab(c,a,k,m)
    energy +=  0.500000000000000 * einsum('bmel,ijba,klcd,edij,cakm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_bbbb(c,a,k,m)
    energy +=  0.500000000000000 * einsum('bmle,ijba,lkdc,deij,cakm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_aaaa*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,i,j)*t2_abab(c,a,m,k)
    energy += -0.500000000000000 * einsum('mbel,ijba,lkcd,edij,camk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||e,l>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,j)*t2_bbbb(c,a,k,m)
    energy += -0.500000000000000 * einsum('bmel,ijba,kldc,edij,cakm', g_abab[v, o, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||l,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,e,i,j)*t2_aaaa(c,a,k,m)
    energy += -0.500000000000000 * einsum('mble,ijab,kldc,deij,cakm', g_abab[o, v, o, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,e,i,j)*t2_abab(a,c,k,m)
    energy += -0.500000000000000 * einsum('mbel,ijab,kldc,deij,ackm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,d,i,j)*t2_aaaa(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mbel,ijab,klcd,edij,cakm', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,e,i,j)*t2_abab(a,c,m,k)
    energy +=  0.500000000000000 * einsum('mble,ijab,lkdc,deij,acmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,j)*t2_abab(a,c,m,k)
    energy += -0.500000000000000 * einsum('mbel,ijab,kldc,edij,acmk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||l,e>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,e,j,i)*t2_abab(c,a,k,m)
    energy += -0.500000000000000 * einsum('bmle,jiba,kldc,deji,cakm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,j,i)*t2_abab(c,a,k,m)
    energy +=  0.500000000000000 * einsum('bmel,jiba,klcd,edji,cakm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,j,i)*t2_bbbb(c,a,k,m)
    energy +=  0.500000000000000 * einsum('bmle,jiba,lkdc,deji,cakm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,j,i)*t2_abab(c,a,m,k)
    energy += -0.500000000000000 * einsum('mbel,jiba,lkcd,edji,camk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||e,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,i)*t2_bbbb(c,a,k,m)
    energy += -0.500000000000000 * einsum('bmel,jiba,kldc,edji,cakm', g_abab[v, o, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||l,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,e,j,i)*t2_aaaa(c,a,k,m)
    energy += -0.500000000000000 * einsum('mble,jiab,kldc,deji,cakm', g_abab[o, v, o, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_bbbb*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,e,j,i)*t2_abab(a,c,k,m)
    energy += -0.500000000000000 * einsum('mbel,jiab,kldc,deji,ackm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(e,d,j,i)*t2_aaaa(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mbel,jiab,klcd,edji,cakm', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,e,j,i)*t2_abab(a,c,m,k)
    energy +=  0.500000000000000 * einsum('mble,jiab,lkdc,deji,acmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,i)*t2_abab(a,c,m,k)
    energy += -0.500000000000000 * einsum('mbel,jiab,kldc,edji,acmk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,j)*t2_abab(c,a,k,m)
    energy += -0.500000000000000 * einsum('mbel,ijba,klcd,edij,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(e,d,i,j)*t2_abab(c,a,m,k)
    energy +=  0.500000000000000 * einsum('mble,ijba,lkcd,edij,camk', g_abab[o, v, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,i,j)*t2_bbbb(c,a,k,m)
    energy +=  0.500000000000000 * einsum('mbel,ijba,kldc,edij,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,j,k)*t2_aaaa(d,c,i,m)
    energy +=  0.500000000000000 * einsum('mbel,ijba,kldc,eajk,dcim', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_aaaa(e,a,j,k)*t2_abab(d,c,i,m)
    energy += -0.500000000000000 * einsum('bmel,ijba,kldc,eajk,dcim', g_abab[v, o, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_aaaa(e,a,j,k)*t2_abab(c,d,i,m)
    energy += -0.500000000000000 * einsum('bmel,ijba,klcd,eajk,cdim', g_abab[v, o, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||l,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(a,e,j,k)*t2_abab(d,c,i,m)
    energy +=  0.500000000000000 * einsum('bmle,ijba,lkdc,aejk,dcim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||l,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(a,e,j,k)*t2_abab(c,d,i,m)
    energy +=  0.500000000000000 * einsum('bmle,ijba,lkcd,aejk,cdim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_aaaa*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(e,a,k,j)*t2_aaaa(d,c,i,m)
    energy += -0.500000000000000 * einsum('mbel,ijba,kldc,eakj,dcim', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,a,k,j)*t2_abab(d,c,i,m)
    energy +=  0.500000000000000 * einsum('bmel,ijba,kldc,eakj,dcim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,a,k,j)*t2_abab(c,d,i,m)
    energy +=  0.500000000000000 * einsum('bmel,ijba,klcd,eakj,cdim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_bbbb(e,a,j,k)*t2_abab(d,c,i,m)
    energy += -0.500000000000000 * einsum('bmle,ijba,lkdc,eajk,dcim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(e,a,j,k)*t2_abab(c,d,i,m)
    energy += -0.500000000000000 * einsum('bmle,ijba,lkcd,eajk,cdim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||l,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(a,e,k,j)*t2_aaaa(d,c,i,m)
    energy +=  0.500000000000000 * einsum('mble,ijab,kldc,aekj,dcim', g_abab[o, v, o, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(a,e,k,j)*t2_abab(d,c,i,m)
    energy += -0.500000000000000 * einsum('mbel,ijab,kldc,aekj,dcim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(a,e,k,j)*t2_abab(c,d,i,m)
    energy += -0.500000000000000 * einsum('mbel,ijab,klcd,aekj,cdim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,a,j,k)*t2_abab(d,c,m,i)
    energy += -0.500000000000000 * einsum('mbel,jiba,lkdc,eajk,dcmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(e,a,j,k)*t2_abab(c,d,m,i)
    energy += -0.500000000000000 * einsum('mbel,jiba,lkcd,eajk,cdmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,a,j,k)*t2_bbbb(d,c,i,m)
    energy +=  0.500000000000000 * einsum('bmel,jiba,kldc,eajk,dcim', g_abab[v, o, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_aaaa(e,a,j,k)*t2_abab(d,c,m,i)
    energy += -0.500000000000000 * einsum('mbel,jiab,kldc,eajk,dcmi', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_aaaa(e,a,j,k)*t2_abab(c,d,m,i)
    energy += -0.500000000000000 * einsum('mbel,jiab,klcd,eajk,cdmi', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(a,e,j,k)*t2_abab(d,c,m,i)
    energy +=  0.500000000000000 * einsum('mble,jiab,lkdc,aejk,dcmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(a,e,j,k)*t2_abab(c,d,m,i)
    energy +=  0.500000000000000 * einsum('mble,jiab,lkcd,aejk,cdmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,l>_bbbb*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,k)*t2_bbbb(d,c,i,m)
    energy += -0.500000000000000 * einsum('mbel,jiab,kldc,aejk,dcim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,a,k,j)*t2_abab(d,c,m,i)
    energy +=  0.500000000000000 * einsum('mbel,ijba,kldc,eakj,dcmi', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,a,k,j)*t2_abab(c,d,m,i)
    energy +=  0.500000000000000 * einsum('mbel,ijba,klcd,eakj,cdmi', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_bbbb(e,a,j,k)*t2_abab(d,c,m,i)
    energy += -0.500000000000000 * einsum('mble,ijba,lkdc,eajk,dcmi', g_abab[o, v, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(e,a,j,k)*t2_abab(c,d,m,i)
    energy += -0.500000000000000 * einsum('mble,ijba,lkcd,eajk,cdmi', g_abab[o, v, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,j,k)*t2_bbbb(d,c,i,m)
    energy +=  0.500000000000000 * einsum('mbel,ijba,kldc,eajk,dcim', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_aaaa(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,ijba,kldc,eaij,dckm', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_abab(d,c,k,m)
    energy += -0.250000000000000 * einsum('bmel,ijba,kldc,eaij,dckm', g_abab[v, o, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_aaaa(e,a,i,j)*t2_abab(c,d,k,m)
    energy += -0.250000000000000 * einsum('bmel,ijba,klcd,eaij,cdkm', g_abab[v, o, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,j)*t2_abab(d,c,m,k)
    energy +=  0.250000000000000 * einsum('mbel,ijba,lkdc,eaij,dcmk', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,j)*t2_abab(c,d,m,k)
    energy +=  0.250000000000000 * einsum('mbel,ijba,lkcd,eaij,cdmk', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_aaaa(e,a,i,j)*t2_bbbb(d,c,k,m)
    energy += -0.250000000000000 * einsum('bmel,ijba,kldc,eaij,dckm', g_abab[v, o, v, o], l2_aaaa, l2_bbbb, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_aaaa*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(e,a,i,j)*t2_aaaa(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,ijba,kldc,eaij,dckm', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,a,i,j)*t2_abab(d,c,k,m)
    energy += -0.250000000000000 * einsum('bmel,ijba,kldc,eaij,dckm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,a,i,j)*t2_abab(c,d,k,m)
    energy += -0.250000000000000 * einsum('bmel,ijba,klcd,eaij,cdkm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_aaaa*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,a,i,j)*t2_abab(d,c,m,k)
    energy +=  0.250000000000000 * einsum('mbel,ijba,lkdc,eaij,dcmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_aaaa*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,a,i,j)*t2_abab(c,d,m,k)
    energy +=  0.250000000000000 * einsum('mbel,ijba,lkcd,eaij,cdmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,l>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,a,i,j)*t2_bbbb(d,c,k,m)
    energy += -0.250000000000000 * einsum('bmel,ijba,kldc,eaij,dckm', g_abab[v, o, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(a,e,i,j)*t2_aaaa(d,c,k,m)
    energy += -0.250000000000000 * einsum('mble,ijab,kldc,aeij,dckm', g_abab[o, v, o, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(a,e,i,j)*t2_abab(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,ijab,kldc,aeij,dckm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(a,e,i,j)*t2_abab(c,d,k,m)
    energy +=  0.250000000000000 * einsum('mbel,ijab,klcd,aeij,cdkm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(a,e,i,j)*t2_abab(d,c,m,k)
    energy += -0.250000000000000 * einsum('mble,ijab,lkdc,aeij,dcmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(a,e,i,j)*t2_abab(c,d,m,k)
    energy += -0.250000000000000 * einsum('mble,ijab,lkcd,aeij,cdmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,j)*t2_bbbb(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,ijab,kldc,aeij,dckm', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_abab(e,a,j,i)*t2_aaaa(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,jiba,kldc,eaji,dckm', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(e,a,j,i)*t2_abab(d,c,k,m)
    energy += -0.250000000000000 * einsum('bmel,jiba,kldc,eaji,dckm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(e,a,j,i)*t2_abab(c,d,k,m)
    energy += -0.250000000000000 * einsum('bmel,jiba,klcd,eaji,cdkm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,a,j,i)*t2_abab(d,c,m,k)
    energy +=  0.250000000000000 * einsum('mbel,jiba,lkdc,eaji,dcmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(e,a,j,i)*t2_abab(c,d,m,k)
    energy +=  0.250000000000000 * einsum('mbel,jiba,lkcd,eaji,cdmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,a,j,i)*t2_bbbb(d,c,k,m)
    energy += -0.250000000000000 * einsum('bmel,jiba,kldc,eaji,dckm', g_abab[v, o, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(a,e,j,i)*t2_aaaa(d,c,k,m)
    energy += -0.250000000000000 * einsum('mble,jiab,kldc,aeji,dckm', g_abab[o, v, o, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_bbbb*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(a,e,j,i)*t2_abab(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,jiab,kldc,aeji,dckm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_bbbb*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(a,e,j,i)*t2_abab(c,d,k,m)
    energy +=  0.250000000000000 * einsum('mbel,jiab,klcd,aeji,cdkm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(a,e,j,i)*t2_abab(d,c,m,k)
    energy += -0.250000000000000 * einsum('mble,jiab,lkdc,aeji,dcmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(a,e,j,i)*t2_abab(c,d,m,k)
    energy += -0.250000000000000 * einsum('mble,jiab,lkcd,aeji,cdmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_bbbb*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(a,e,j,i)*t2_bbbb(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,jiab,kldc,aeji,dckm', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_aaaa(d,c,k,m)
    energy += -0.250000000000000 * einsum('mble,ijba,kldc,eaij,dckm', g_abab[o, v, o, v], l2_bbbb, l2_aaaa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_abab(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,ijba,kldc,eaij,dckm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,j)*t2_abab(c,d,k,m)
    energy +=  0.250000000000000 * einsum('mbel,ijba,klcd,eaij,cdkm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_bbbb(e,a,i,j)*t2_abab(d,c,m,k)
    energy += -0.250000000000000 * einsum('mble,ijba,lkdc,eaij,dcmk', g_abab[o, v, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_bbbb(e,a,i,j)*t2_abab(c,d,m,k)
    energy += -0.250000000000000 * einsum('mble,ijba,lkcd,eaij,cdmk', g_abab[o, v, o, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,i,j)*t2_bbbb(d,c,k,m)
    energy +=  0.250000000000000 * einsum('mbel,ijba,kldc,eaij,dckm', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(c,a,i,m)
    energy +=  0.500000000000000 * einsum('mbej,ijba,kldc,edkl,caim', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(a,c,i,m)
    energy +=  0.500000000000000 * einsum('bmje,ijba,kldc,dekl,acim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(c,a,i,m)
    energy += -0.500000000000000 * einsum('mbej,ijba,klcd,edkl,caim', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(a,c,i,m)
    energy +=  0.500000000000000 * einsum('bmje,ijba,lkdc,delk,acim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(c,a,i,m)
    energy += -0.500000000000000 * einsum('mbej,ijba,lkcd,edlk,caim', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(a,c,i,m)
    energy += -0.500000000000000 * einsum('bmje,ijba,kldc,edkl,acim', g_abab[v, o, o, v], l2_aaaa, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,a,i,m)
    energy += -0.500000000000000 * einsum('bmej,ijba,kldc,edkl,caim', g_abab[v, o, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,a,i,m)
    energy +=  0.500000000000000 * einsum('bmej,ijba,klcd,edkl,caim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,a,i,m)
    energy +=  0.500000000000000 * einsum('bmej,ijba,lkcd,edlk,caim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(c,a,i,m)
    energy += -0.500000000000000 * einsum('mbej,ijab,kldc,edkl,caim', g_abab[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(a,c,i,m)
    energy += -0.500000000000000 * einsum('mbej,ijab,kldc,dekl,acim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(c,a,i,m)
    energy +=  0.500000000000000 * einsum('mbej,ijab,klcd,edkl,caim', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(a,c,i,m)
    energy += -0.500000000000000 * einsum('mbej,ijab,lkdc,delk,acim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(c,a,i,m)
    energy +=  0.500000000000000 * einsum('mbej,ijab,lkcd,edlk,caim', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(a,c,i,m)
    energy +=  0.500000000000000 * einsum('mbej,ijab,kldc,edkl,acim', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,a,m,i)
    energy +=  0.500000000000000 * einsum('mbej,jiba,kldc,edkl,cami', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(c,a,i,m)
    energy +=  0.500000000000000 * einsum('bmje,jiba,kldc,dekl,caim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,a,m,i)
    energy += -0.500000000000000 * einsum('mbej,jiba,klcd,edkl,cami', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(c,a,i,m)
    energy +=  0.500000000000000 * einsum('bmje,jiba,lkdc,delk,caim', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,a,m,i)
    energy += -0.500000000000000 * einsum('mbej,jiba,lkcd,edlk,cami', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(c,a,i,m)
    energy += -0.500000000000000 * einsum('bmje,jiba,kldc,edkl,caim', g_abab[v, o, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(a,c,m,i)
    energy +=  0.500000000000000 * einsum('mbje,jiab,kldc,dekl,acmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(a,c,m,i)
    energy +=  0.500000000000000 * einsum('mbje,jiab,lkdc,delk,acmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(a,c,m,i)
    energy += -0.500000000000000 * einsum('mbje,jiab,kldc,edkl,acmi', g_abab[o, v, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,a,m,i)
    energy += -0.500000000000000 * einsum('mbej,ijba,kldc,edkl,cami', g_abab[o, v, v, o], l2_bbbb, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(c,a,i,m)
    energy += -0.500000000000000 * einsum('mbej,ijba,kldc,dekl,caim', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,a,m,i)
    energy +=  0.500000000000000 * einsum('mbej,ijba,klcd,edkl,cami', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(c,a,i,m)
    energy += -0.500000000000000 * einsum('mbej,ijba,lkdc,delk,caim', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,a,m,i)
    energy +=  0.500000000000000 * einsum('mbej,ijba,lkcd,edlk,cami', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(c,a,i,m)
    energy +=  0.500000000000000 * einsum('mbej,ijba,kldc,edkl,caim', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_aaaa(c,a,k,m)
    energy += -1.000000000000000 * einsum('mbej,ijba,kldc,edil,cakm', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(a,c,k,m)
    energy += -1.000000000000000 * einsum('bmje,ijba,kldc,deil,ackm', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_aaaa(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mbej,ijba,klcd,edil,cakm', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_abab(a,c,m,k)
    energy +=  1.000000000000000 * einsum('mbej,ijba,lkdc,edil,acmk', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_abab(a,c,m,k)
    energy += -1.000000000000000 * einsum('mbej,ijba,kldc,edil,acmk', g_aaaa[o, v, v, o], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_abab(c,a,k,m)
    energy +=  1.000000000000000 * einsum('bmej,ijba,kldc,edil,cakm', g_abab[v, o, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_abab(c,a,k,m)
    energy += -1.000000000000000 * einsum('bmej,ijba,klcd,edil,cakm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_bbbb(c,a,k,m)
    energy += -1.000000000000000 * einsum('bmej,ijba,lkdc,edil,cakm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_bbbb(c,a,k,m)
    energy +=  1.000000000000000 * einsum('bmej,ijba,kldc,edil,cakm', g_abab[v, o, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,i,l)*t2_aaaa(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mbej,ijab,kldc,edil,cakm', g_abab[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,e,i,l)*t2_abab(a,c,k,m)
    energy +=  1.000000000000000 * einsum('mbej,ijab,kldc,deil,ackm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,d,i,l)*t2_aaaa(c,a,k,m)
    energy += -1.000000000000000 * einsum('mbej,ijab,klcd,edil,cakm', g_abab[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_aaaa(e,d,i,l)*t2_abab(a,c,m,k)
    energy += -1.000000000000000 * einsum('mbej,ijab,lkdc,edil,acmk', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,d,i,l)*t2_abab(a,c,m,k)
    energy +=  1.000000000000000 * einsum('mbej,ijab,kldc,edil,acmk', g_abab[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,i)*t2_abab(c,a,k,m)
    energy +=  1.000000000000000 * einsum('bmje,jiba,kldc,deli,cakm', g_abab[v, o, o, v], l2_abab, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,l)*t2_abab(c,a,k,m)
    energy += -1.000000000000000 * einsum('bmje,jiba,klcd,edil,cakm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,i)*t2_bbbb(c,a,k,m)
    energy += -1.000000000000000 * einsum('bmje,jiba,lkdc,deli,cakm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,i)*t2_abab(c,a,m,k)
    energy +=  1.000000000000000 * einsum('mbej,jiba,lkcd,edli,camk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,i,l)*t2_bbbb(c,a,k,m)
    energy +=  1.000000000000000 * einsum('bmje,jiba,kldc,edil,cakm', g_abab[v, o, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,i)*t2_aaaa(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mbje,jiab,kldc,deli,cakm', g_abab[o, v, o, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,l)*t2_aaaa(c,a,k,m)
    energy += -1.000000000000000 * einsum('mbje,jiab,klcd,edil,cakm', g_abab[o, v, o, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,e,l,i)*t2_abab(a,c,m,k)
    energy += -1.000000000000000 * einsum('mbje,jiab,lkdc,deli,acmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,i,l)*t2_abab(a,c,m,k)
    energy +=  1.000000000000000 * einsum('mbje,jiab,kldc,edil,acmk', g_abab[o, v, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,i)*t2_abab(c,a,k,m)
    energy += -1.000000000000000 * einsum('mbej,ijba,kldc,deli,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,d,i,l)*t2_abab(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mbej,ijba,klcd,edil,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,i)*t2_bbbb(c,a,k,m)
    energy +=  1.000000000000000 * einsum('mbej,ijba,lkdc,deli,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,i)*t2_abab(c,a,m,k)
    energy += -1.000000000000000 * einsum('mbej,ijba,lkcd,edli,camk', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,i,l)*t2_bbbb(c,a,k,m)
    energy += -1.000000000000000 * einsum('mbej,ijba,kldc,edil,cakm', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_aaaa(d,c,i,m)
    energy +=  0.250000000000000 * einsum('mbej,ijba,kldc,eakl,dcim', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(d,c,i,m)
    energy += -0.250000000000000 * einsum('bmje,ijba,kldc,aekl,dcim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(c,d,i,m)
    energy += -0.250000000000000 * einsum('bmje,ijba,klcd,aekl,cdim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(d,c,i,m)
    energy += -0.250000000000000 * einsum('bmje,ijba,lkdc,aelk,dcim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(c,d,i,m)
    energy += -0.250000000000000 * einsum('bmje,ijba,lkcd,aelk,cdim', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,a,k,l)*t2_abab(d,c,i,m)
    energy += -0.250000000000000 * einsum('bmej,ijba,kldc,eakl,dcim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,a,k,l)*t2_abab(c,d,i,m)
    energy += -0.250000000000000 * einsum('bmej,ijba,klcd,eakl,cdim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,a,l,k)*t2_abab(d,c,i,m)
    energy += -0.250000000000000 * einsum('bmej,ijba,lkdc,ealk,dcim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,a,l,k)*t2_abab(c,d,i,m)
    energy += -0.250000000000000 * einsum('bmej,ijba,lkcd,ealk,cdim', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,k,l)*t2_aaaa(d,c,i,m)
    energy += -0.250000000000000 * einsum('mbej,ijab,kldc,eakl,dcim', g_abab[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(d,c,i,m)
    energy +=  0.250000000000000 * einsum('mbej,ijab,kldc,aekl,dcim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(c,d,i,m)
    energy +=  0.250000000000000 * einsum('mbej,ijab,klcd,aekl,cdim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(d,c,i,m)
    energy +=  0.250000000000000 * einsum('mbej,ijab,lkdc,aelk,dcim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(c,d,i,m)
    energy +=  0.250000000000000 * einsum('mbej,ijab,lkcd,aelk,cdim', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(e,a,k,l)*t2_abab(d,c,m,i)
    energy +=  0.250000000000000 * einsum('mbej,jiba,kldc,eakl,dcmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(e,a,k,l)*t2_abab(c,d,m,i)
    energy +=  0.250000000000000 * einsum('mbej,jiba,klcd,eakl,cdmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,a,l,k)*t2_abab(d,c,m,i)
    energy +=  0.250000000000000 * einsum('mbej,jiba,lkdc,ealk,dcmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(e,a,l,k)*t2_abab(c,d,m,i)
    energy +=  0.250000000000000 * einsum('mbej,jiba,lkcd,ealk,cdmi', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,k,l)*t2_bbbb(d,c,i,m)
    energy += -0.250000000000000 * einsum('bmje,jiba,kldc,eakl,dcim', g_abab[v, o, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(a,e,k,l)*t2_abab(d,c,m,i)
    energy += -0.250000000000000 * einsum('mbje,jiab,kldc,aekl,dcmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(a,e,k,l)*t2_abab(c,d,m,i)
    energy += -0.250000000000000 * einsum('mbje,jiab,klcd,aekl,cdmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(a,e,l,k)*t2_abab(d,c,m,i)
    energy += -0.250000000000000 * einsum('mbje,jiab,lkdc,aelk,dcmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(a,e,l,k)*t2_abab(c,d,m,i)
    energy += -0.250000000000000 * einsum('mbje,jiab,lkcd,aelk,cdmi', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,a,k,l)*t2_abab(d,c,m,i)
    energy += -0.250000000000000 * einsum('mbej,ijba,kldc,eakl,dcmi', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,a,k,l)*t2_abab(c,d,m,i)
    energy += -0.250000000000000 * einsum('mbej,ijba,klcd,eakl,cdmi', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,a,l,k)*t2_abab(d,c,m,i)
    energy += -0.250000000000000 * einsum('mbej,ijba,lkdc,ealk,dcmi', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,a,l,k)*t2_abab(c,d,m,i)
    energy += -0.250000000000000 * einsum('mbej,ijba,lkcd,ealk,cdmi', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,k,l)*t2_bbbb(d,c,i,m)
    energy +=  0.250000000000000 * einsum('mbej,ijba,kldc,eakl,dcim', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,l)*t2_aaaa(d,c,k,m)
    energy += -0.500000000000000 * einsum('mbej,ijba,kldc,eail,dckm', g_aaaa[o, v, v, o], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(d,c,k,m)
    energy +=  0.500000000000000 * einsum('bmje,ijba,kldc,aeil,dckm', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_abab(c,d,k,m)
    energy +=  0.500000000000000 * einsum('bmje,ijba,klcd,aeil,cdkm', g_abab[v, o, o, v], l2_aaaa, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_abab(d,c,m,k)
    energy += -0.500000000000000 * einsum('mbej,ijba,lkdc,eail,dcmk', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_abab(c,d,m,k)
    energy += -0.500000000000000 * einsum('mbej,ijba,lkcd,eail,cdmk', g_aaaa[o, v, v, o], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_bbbb(d,c,k,m)
    energy +=  0.500000000000000 * einsum('bmje,ijba,kldc,aeil,dckm', g_abab[v, o, o, v], l2_aaaa, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,a,i,l)*t2_abab(d,c,k,m)
    energy +=  0.500000000000000 * einsum('bmej,ijba,kldc,eail,dckm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,a,i,l)*t2_abab(c,d,k,m)
    energy +=  0.500000000000000 * einsum('bmej,ijba,klcd,eail,cdkm', g_abab[v, o, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||e,j>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,a,i,l)*t2_bbbb(d,c,k,m)
    energy +=  0.500000000000000 * einsum('bmej,ijba,kldc,eail,dckm', g_abab[v, o, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,a,i,l)*t2_aaaa(d,c,k,m)
    energy +=  0.500000000000000 * einsum('mbej,ijab,kldc,eail,dckm', g_abab[o, v, v, o], l2_abab, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(a,e,i,l)*t2_abab(d,c,k,m)
    energy += -0.500000000000000 * einsum('mbej,ijab,kldc,aeil,dckm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(a,e,i,l)*t2_abab(c,d,k,m)
    energy += -0.500000000000000 * einsum('mbej,ijab,klcd,aeil,cdkm', g_bbbb[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_aaaa(e,a,i,l)*t2_abab(d,c,m,k)
    energy +=  0.500000000000000 * einsum('mbej,ijab,lkdc,eail,dcmk', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_aaaa(e,a,i,l)*t2_abab(c,d,m,k)
    energy +=  0.500000000000000 * einsum('mbej,ijab,lkcd,eail,cdmk', g_abab[o, v, v, o], l2_abab, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_abab(a,e,i,l)*t2_bbbb(d,c,k,m)
    energy += -0.500000000000000 * einsum('mbej,ijab,kldc,aeil,dckm', g_bbbb[o, v, v, o], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_abab(e,a,l,i)*t2_aaaa(d,c,k,m)
    energy += -0.500000000000000 * einsum('mbej,jiba,kldc,eali,dckm', g_aaaa[o, v, v, o], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_bbbb(e,a,i,l)*t2_abab(d,c,k,m)
    energy +=  0.500000000000000 * einsum('bmje,jiba,kldc,eail,dckm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,l)*t2_abab(c,d,k,m)
    energy +=  0.500000000000000 * einsum('bmje,jiba,klcd,eail,cdkm', g_abab[v, o, o, v], l2_abab, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,a,l,i)*t2_abab(d,c,m,k)
    energy += -0.500000000000000 * einsum('mbej,jiba,lkdc,eali,dcmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(e,a,l,i)*t2_abab(c,d,m,k)
    energy += -0.500000000000000 * einsum('mbej,jiba,lkcd,eali,cdmk', g_aaaa[o, v, v, o], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,i,l)*t2_bbbb(d,c,k,m)
    energy +=  0.500000000000000 * einsum('bmje,jiba,kldc,eail,dckm', g_abab[v, o, o, v], l2_abab, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(a,e,l,i)*t2_aaaa(d,c,k,m)
    energy +=  0.500000000000000 * einsum('mbje,jiab,kldc,aeli,dckm', g_abab[o, v, o, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(a,e,l,i)*t2_abab(d,c,m,k)
    energy +=  0.500000000000000 * einsum('mbje,jiab,lkdc,aeli,dcmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(a,e,l,i)*t2_abab(c,d,m,k)
    energy +=  0.500000000000000 * einsum('mbje,jiab,lkcd,aeli,cdmk', g_abab[o, v, o, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(e,a,l,i)*t2_aaaa(d,c,k,m)
    energy +=  0.500000000000000 * einsum('mbej,ijba,kldc,eali,dckm', g_abab[o, v, v, o], l2_bbbb, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_bbbb(e,a,i,l)*t2_abab(d,c,k,m)
    energy += -0.500000000000000 * einsum('mbej,ijba,kldc,eail,dckm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,a,i,l)*t2_abab(c,d,k,m)
    energy += -0.500000000000000 * einsum('mbej,ijba,klcd,eail,cdkm', g_bbbb[o, v, v, o], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,a,l,i)*t2_abab(d,c,m,k)
    energy +=  0.500000000000000 * einsum('mbej,ijba,lkdc,eali,dcmk', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,a,l,i)*t2_abab(c,d,m,k)
    energy +=  0.500000000000000 * einsum('mbej,ijba,lkcd,eali,cdmk', g_abab[o, v, v, o], l2_bbbb, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,a,i,l)*t2_bbbb(d,c,k,m)
    energy += -0.500000000000000 * einsum('mbej,ijba,kldc,eail,dckm', g_bbbb[o, v, v, o], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,k,l)*t2_aaaa(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcef,ijba,kldc,ebkl,faij', g_aaaa[v, v, v, v], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(b,e,k,l)*t2_aaaa(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcfe,ijba,kldc,bekl,faij', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,k,l)*t2_aaaa(f,a,i,j)
    energy += -0.125000000000000 * einsum('cdfe,ijba,klcd,bekl,faij', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,l,k)*t2_aaaa(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcfe,ijba,lkdc,belk,faij', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,l,k)*t2_aaaa(f,a,i,j)
    energy += -0.125000000000000 * einsum('cdfe,ijba,lkcd,belk,faij', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_aaaa*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,k,l)*t2_abab(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcef,ijba,kldc,ebkl,faij', g_aaaa[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(b,e,k,l)*t2_abab(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcfe,ijba,kldc,bekl,faij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,k,l)*t2_abab(f,a,i,j)
    energy += -0.125000000000000 * einsum('cdfe,ijba,klcd,bekl,faij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,l,k)*t2_abab(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcfe,ijba,lkdc,belk,faij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,l,k)*t2_abab(f,a,i,j)
    energy += -0.125000000000000 * einsum('cdfe,ijba,lkcd,belk,faij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_abab(a,f,i,j)
    energy += -0.125000000000000 * einsum('dcef,ijab,kldc,ebkl,afij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,b,k,l)*t2_abab(a,f,i,j)
    energy += -0.125000000000000 * einsum('cdef,ijab,klcd,ebkl,afij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_abab(a,f,i,j)
    energy += -0.125000000000000 * einsum('dcef,ijab,lkdc,eblk,afij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(e,b,l,k)*t2_abab(a,f,i,j)
    energy += -0.125000000000000 * einsum('cdef,ijab,lkcd,eblk,afij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_abab(a,f,i,j)
    energy += -0.125000000000000 * einsum('dcef,ijab,kldc,ebkl,afij', g_bbbb[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,k,l)*t2_abab(f,a,j,i)
    energy += -0.125000000000000 * einsum('dcef,jiba,kldc,ebkl,faji', g_aaaa[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(b,e,k,l)*t2_abab(f,a,j,i)
    energy += -0.125000000000000 * einsum('dcfe,jiba,kldc,bekl,faji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,k,l)*t2_abab(f,a,j,i)
    energy += -0.125000000000000 * einsum('cdfe,jiba,klcd,bekl,faji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,l,k)*t2_abab(f,a,j,i)
    energy += -0.125000000000000 * einsum('dcfe,jiba,lkdc,belk,faji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,l,k)*t2_abab(f,a,j,i)
    energy += -0.125000000000000 * einsum('cdfe,jiba,lkcd,belk,faji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_abab(a,f,j,i)
    energy += -0.125000000000000 * einsum('dcef,jiab,kldc,ebkl,afji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(e,b,k,l)*t2_abab(a,f,j,i)
    energy += -0.125000000000000 * einsum('cdef,jiab,klcd,ebkl,afji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_abab(a,f,j,i)
    energy += -0.125000000000000 * einsum('dcef,jiab,lkdc,eblk,afji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(e,b,l,k)*t2_abab(a,f,j,i)
    energy += -0.125000000000000 * einsum('cdef,jiab,lkcd,eblk,afji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_bbbb*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_abab(a,f,j,i)
    energy += -0.125000000000000 * einsum('dcef,jiab,kldc,ebkl,afji', g_bbbb[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,b,k,l)*t2_bbbb(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcef,ijba,kldc,ebkl,faij', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,b,k,l)*t2_bbbb(f,a,i,j)
    energy += -0.125000000000000 * einsum('cdef,ijba,klcd,ebkl,faij', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,b,l,k)*t2_bbbb(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcef,ijba,lkdc,eblk,faij', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,b,l,k)*t2_bbbb(f,a,i,j)
    energy += -0.125000000000000 * einsum('cdef,ijba,lkcd,eblk,faij', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,k,l)*t2_bbbb(f,a,i,j)
    energy += -0.125000000000000 * einsum('dcef,ijba,kldc,ebkl,faij', g_bbbb[v, v, v, v], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,j,l)*t2_aaaa(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcef,ijba,kldc,ebjl,faik', g_aaaa[v, v, v, v], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(b,e,j,l)*t2_aaaa(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcfe,ijba,kldc,bejl,faik', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,j,l)*t2_aaaa(f,a,i,k)
    energy +=  0.250000000000000 * einsum('cdfe,ijba,klcd,bejl,faik', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,b,j,l)*t2_abab(a,f,i,k)
    energy +=  0.250000000000000 * einsum('dcef,ijba,lkdc,ebjl,afik', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||e,f>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,b,j,l)*t2_abab(a,f,i,k)
    energy +=  0.250000000000000 * einsum('cdef,ijba,lkcd,ebjl,afik', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_bbbb*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,j,l)*t2_abab(a,f,i,k)
    energy +=  0.250000000000000 * einsum('dcef,ijba,kldc,bejl,afik', g_bbbb[v, v, v, v], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,l,j)*t2_abab(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcfe,ijba,lkdc,belj,faik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,l,j)*t2_abab(f,a,i,k)
    energy +=  0.250000000000000 * einsum('cdfe,ijba,lkcd,belj,faik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_aaaa*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(e,b,l,j)*t2_aaaa(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcef,ijab,kldc,eblj,faik', g_aaaa[v, v, v, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||f,e>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_aaaa(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcfe,ijab,kldc,ebjl,faik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||f,e>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_bbbb(e,b,j,l)*t2_aaaa(f,a,i,k)
    energy +=  0.250000000000000 * einsum('cdfe,ijab,klcd,ebjl,faik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,l,j)*t2_abab(a,f,i,k)
    energy +=  0.250000000000000 * einsum('dcef,ijab,lkdc,eblj,afik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(e,b,l,j)*t2_abab(a,f,i,k)
    energy +=  0.250000000000000 * einsum('cdef,ijab,lkcd,eblj,afik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_abab(a,f,i,k)
    energy +=  0.250000000000000 * einsum('dcef,ijab,kldc,ebjl,afik', g_bbbb[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,j,l)*t2_abab(f,a,k,i)
    energy +=  0.250000000000000 * einsum('dcef,jiba,kldc,ebjl,faki', g_aaaa[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(b,e,j,l)*t2_abab(f,a,k,i)
    energy +=  0.250000000000000 * einsum('dcfe,jiba,kldc,bejl,faki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,j,l)*t2_abab(f,a,k,i)
    energy +=  0.250000000000000 * einsum('cdfe,jiba,klcd,bejl,faki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,b,j,l)*t2_bbbb(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcef,jiba,lkdc,ebjl,faik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||e,f>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,b,j,l)*t2_bbbb(f,a,i,k)
    energy +=  0.250000000000000 * einsum('cdef,jiba,lkcd,ebjl,faik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_bbbb*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,j,l)*t2_bbbb(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcef,jiba,kldc,bejl,faik', g_bbbb[v, v, v, v], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,j,l)*t2_abab(a,f,k,i)
    energy +=  0.250000000000000 * einsum('dcef,jiab,kldc,ebjl,afki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(e,b,j,l)*t2_abab(a,f,k,i)
    energy +=  0.250000000000000 * einsum('cdef,jiab,klcd,ebjl,afki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_aaaa*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(e,b,l,j)*t2_abab(f,a,k,i)
    energy +=  0.250000000000000 * einsum('dcef,ijba,kldc,eblj,faki', g_aaaa[v, v, v, v], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||f,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_abab(f,a,k,i)
    energy +=  0.250000000000000 * einsum('dcfe,ijba,kldc,ebjl,faki', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||f,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,b,j,l)*t2_abab(f,a,k,i)
    energy +=  0.250000000000000 * einsum('cdfe,ijba,klcd,ebjl,faki', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,b,l,j)*t2_bbbb(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcef,ijba,lkdc,eblj,faik', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <c,d||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,b,l,j)*t2_bbbb(f,a,i,k)
    energy +=  0.250000000000000 * einsum('cdef,ijba,lkcd,eblj,faik', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,c||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_bbbb(f,a,i,k)
    energy +=  0.250000000000000 * einsum('dcef,ijba,kldc,ebjl,faik', g_bbbb[v, v, v, v], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_aaaa(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,eckl,faij', g_aaaa[v, v, v, v], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_aaaa(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,eckl,faij', g_aaaa[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_aaaa(f,a,i,j)
    energy +=  0.250000000000000 * einsum('bdfe,ijba,klcd,cekl,faij', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_aaaa(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,lkdc,eclk,faij', g_aaaa[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_aaaa(f,a,i,j)
    energy +=  0.250000000000000 * einsum('bdfe,ijba,lkcd,celk,faij', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_aaaa(f,a,i,j)
    energy +=  0.250000000000000 * einsum('bdfe,ijba,kldc,eckl,faij', g_abab[v, v, v, v], l2_aaaa, l2_bbbb, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,eckl,faij', g_aaaa[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,eckl,faij', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(f,a,i,j)
    energy +=  0.250000000000000 * einsum('bdfe,ijba,klcd,cekl,faij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,lkdc,eclk,faij', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(f,a,i,j)
    energy +=  0.250000000000000 * einsum('bdfe,ijba,lkcd,celk,faij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(f,a,i,j)
    energy +=  0.250000000000000 * einsum('bdfe,ijba,kldc,eckl,faij', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(a,f,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijab,kldc,eckl,afij', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(a,f,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijab,kldc,eckl,afij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(a,f,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijab,klcd,cekl,afij', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(a,f,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijab,lkdc,eclk,afij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(a,f,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijab,lkcd,celk,afij', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(a,f,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijab,kldc,eckl,afij', g_bbbb[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(f,a,j,i)
    energy +=  0.250000000000000 * einsum('dbef,jiba,kldc,eckl,faji', g_aaaa[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(f,a,j,i)
    energy +=  0.250000000000000 * einsum('dbef,jiba,kldc,eckl,faji', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(f,a,j,i)
    energy +=  0.250000000000000 * einsum('bdfe,jiba,klcd,cekl,faji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(f,a,j,i)
    energy +=  0.250000000000000 * einsum('dbef,jiba,lkdc,eclk,faji', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(f,a,j,i)
    energy +=  0.250000000000000 * einsum('bdfe,jiba,lkcd,celk,faji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(f,a,j,i)
    energy +=  0.250000000000000 * einsum('bdfe,jiba,kldc,eckl,faji', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_abab(a,f,j,i)
    energy +=  0.250000000000000 * einsum('dbef,jiab,kldc,eckl,afji', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_abab(a,f,j,i)
    energy +=  0.250000000000000 * einsum('dbef,jiab,kldc,eckl,afji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_abab(a,f,j,i)
    energy +=  0.250000000000000 * einsum('dbef,jiab,klcd,cekl,afji', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_abab(a,f,j,i)
    energy +=  0.250000000000000 * einsum('dbef,jiab,lkdc,eclk,afji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_abab(a,f,j,i)
    energy +=  0.250000000000000 * einsum('dbef,jiab,lkcd,celk,afji', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_abab(a,f,j,i)
    energy +=  0.250000000000000 * einsum('dbef,jiab,kldc,eckl,afji', g_bbbb[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,k,l)*t2_bbbb(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,eckl,faij', g_abab[v, v, v, v], l2_bbbb, l2_aaaa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,k,l)*t2_bbbb(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,eckl,faij', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,k,l)*t2_bbbb(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,klcd,cekl,faij', g_bbbb[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,k)*t2_bbbb(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,lkdc,eclk,faij', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,k)*t2_bbbb(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,lkcd,celk,faij', g_bbbb[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,k,l)*t2_bbbb(f,a,i,j)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,eckl,faij', g_bbbb[v, v, v, v], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,j,l)*t2_aaaa(f,a,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijba,kldc,ecjl,faik', g_aaaa[v, v, v, v], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,j,l)*t2_aaaa(f,a,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijba,kldc,ecjl,faik', g_aaaa[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <b,d||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,j,l)*t2_aaaa(f,a,i,k)
    energy += -0.625000000000000 * einsum('bdfe,ijba,klcd,cejl,faik', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <b,d||e,f>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,c,j,l)*t2_abab(a,f,i,k)
    energy += -0.625000000000000 * einsum('bdef,ijba,lkcd,ecjl,afik', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <b,d||e,f>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,l)*t2_abab(a,f,i,k)
    energy += -0.625000000000000 * einsum('bdef,ijba,kldc,ecjl,afik', g_abab[v, v, v, v], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_aaaa*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,j)*t2_abab(f,a,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijba,lkdc,eclj,faik', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <b,d||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,j)*t2_abab(f,a,i,k)
    energy += -0.625000000000000 * einsum('bdfe,ijba,lkcd,celj,faik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <b,d||f,e>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_abab(f,a,i,k)
    energy += -0.625000000000000 * einsum('bdfe,ijba,kldc,ecjl,faik', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||f,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(c,e,l,j)*t2_aaaa(f,a,i,k)
    energy += -0.625000000000000 * einsum('dbfe,ijab,kldc,celj,faik', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||f,e>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_aaaa(f,a,i,k)
    energy += -0.625000000000000 * einsum('dbfe,ijab,kldc,ecjl,faik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,l,j)*t2_abab(a,f,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijab,lkdc,eclj,afik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_bbbb*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,l,j)*t2_abab(a,f,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijab,lkcd,celj,afik', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_abab(a,f,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijab,kldc,ecjl,afik', g_bbbb[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,j,l)*t2_abab(f,a,k,i)
    energy += -0.625000000000000 * einsum('dbef,jiba,kldc,ecjl,faki', g_aaaa[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_aaaa*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,j,l)*t2_abab(f,a,k,i)
    energy += -0.625000000000000 * einsum('dbef,jiba,kldc,ecjl,faki', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <b,d||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,j,l)*t2_abab(f,a,k,i)
    energy += -0.625000000000000 * einsum('bdfe,jiba,klcd,cejl,faki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <b,d||e,f>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,c,j,l)*t2_bbbb(f,a,i,k)
    energy += -0.625000000000000 * einsum('bdef,jiba,lkcd,ecjl,faik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <b,d||e,f>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,l)*t2_bbbb(f,a,i,k)
    energy += -0.625000000000000 * einsum('bdef,jiba,kldc,ecjl,faik', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,j,l)*t2_abab(a,f,k,i)
    energy += -0.625000000000000 * einsum('dbef,jiab,kldc,ecjl,afki', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,j,l)*t2_abab(a,f,k,i)
    energy += -0.625000000000000 * einsum('dbef,jiab,kldc,ecjl,afki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_bbbb*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,j,l)*t2_abab(a,f,k,i)
    energy += -0.625000000000000 * einsum('dbef,jiab,klcd,cejl,afki', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||f,e>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(c,e,l,j)*t2_abab(f,a,k,i)
    energy += -0.625000000000000 * einsum('dbfe,ijba,kldc,celj,faki', g_abab[v, v, v, v], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||f,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_abab(f,a,k,i)
    energy += -0.625000000000000 * einsum('dbfe,ijba,kldc,ecjl,faki', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,l,j)*t2_bbbb(f,a,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijba,lkdc,eclj,faik', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,l,j)*t2_bbbb(f,a,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijba,lkcd,celj,faik', g_bbbb[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.6250 <d,b||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,j,l)*t2_bbbb(f,a,i,k)
    energy += -0.625000000000000 * einsum('dbef,ijba,kldc,ecjl,faik', g_bbbb[v, v, v, v], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,c,i,j)*t2_aaaa(f,a,k,l)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,ecij,fakl', g_aaaa[v, v, v, v], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,d||e,f>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_aaaa(e,c,i,j)*t2_abab(a,f,k,l)
    energy += -0.250000000000000 * einsum('bdef,ijba,klcd,ecij,afkl', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,d||e,f>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,c,i,j)*t2_abab(a,f,l,k)
    energy += -0.250000000000000 * einsum('bdef,ijba,lkcd,ecij,aflk', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(f,a,k,l)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,ecij,fakl', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,i,j)*t2_abab(f,a,k,l)
    energy +=  0.250000000000000 * einsum('bdfe,ijba,klcd,ceij,fakl', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,i,j)*t2_abab(f,a,l,k)
    energy +=  0.250000000000000 * einsum('dbef,ijba,lkdc,ecij,falk', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,i,j)*t2_abab(f,a,l,k)
    energy +=  0.250000000000000 * einsum('bdfe,ijba,lkcd,ceij,falk', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,d||e,f>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,c,i,j)*t2_bbbb(f,a,k,l)
    energy += -0.250000000000000 * einsum('bdef,ijba,kldc,ecij,fakl', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,b||f,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(c,e,i,j)*t2_aaaa(f,a,k,l)
    energy += -0.250000000000000 * einsum('dbfe,ijab,kldc,ceij,fakl', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,i,j)*t2_abab(a,f,k,l)
    energy +=  0.250000000000000 * einsum('dbef,ijab,kldc,ecij,afkl', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,i,j)*t2_abab(a,f,k,l)
    energy +=  0.250000000000000 * einsum('dbef,ijab,klcd,ceij,afkl', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,i,j)*t2_abab(a,f,l,k)
    energy +=  0.250000000000000 * einsum('dbef,ijab,lkdc,ecij,aflk', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,i,j)*t2_abab(a,f,l,k)
    energy +=  0.250000000000000 * einsum('dbef,ijab,lkcd,ceij,aflk', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(e,c,j,i)*t2_abab(f,a,k,l)
    energy +=  0.250000000000000 * einsum('dbef,jiba,kldc,ecji,fakl', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(c,e,j,i)*t2_abab(f,a,k,l)
    energy +=  0.250000000000000 * einsum('bdfe,jiba,klcd,ceji,fakl', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(e,c,j,i)*t2_abab(f,a,l,k)
    energy +=  0.250000000000000 * einsum('dbef,jiba,lkdc,ecji,falk', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <b,d||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(c,e,j,i)*t2_abab(f,a,l,k)
    energy +=  0.250000000000000 * einsum('bdfe,jiba,lkcd,ceji,falk', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,d||e,f>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,c,j,i)*t2_bbbb(f,a,k,l)
    energy += -0.250000000000000 * einsum('bdef,jiba,kldc,ecji,fakl', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,b||f,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_abab(c,e,j,i)*t2_aaaa(f,a,k,l)
    energy += -0.250000000000000 * einsum('dbfe,jiab,kldc,ceji,fakl', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,c,j,i)*t2_abab(a,f,k,l)
    energy +=  0.250000000000000 * einsum('dbef,jiab,kldc,ecji,afkl', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(c,e,j,i)*t2_abab(a,f,k,l)
    energy +=  0.250000000000000 * einsum('dbef,jiab,klcd,ceji,afkl', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(e,c,j,i)*t2_abab(a,f,l,k)
    energy +=  0.250000000000000 * einsum('dbef,jiab,lkdc,ecji,aflk', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(c,e,j,i)*t2_abab(a,f,l,k)
    energy +=  0.250000000000000 * einsum('dbef,jiab,lkcd,ceji,aflk', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,b||f,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_bbbb(e,c,i,j)*t2_abab(f,a,k,l)
    energy += -0.250000000000000 * einsum('dbfe,ijba,kldc,ecij,fakl', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.2500 <d,b||f,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_bbbb(e,c,i,j)*t2_abab(f,a,l,k)
    energy += -0.250000000000000 * einsum('dbfe,ijba,lkdc,ecij,falk', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.2500 <d,b||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,c,i,j)*t2_bbbb(f,a,k,l)
    energy +=  0.250000000000000 * einsum('dbef,ijba,kldc,ecij,fakl', g_bbbb[v, v, v, v], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <d,a||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,j,l)*t2_aaaa(f,c,i,k)
    energy += -0.375000000000000 * einsum('daef,ijba,kldc,ebjl,fcik', g_aaaa[v, v, v, v], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <a,d||f,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,j,l)*t2_aaaa(f,c,i,k)
    energy += -0.375000000000000 * einsum('adfe,ijba,klcd,bejl,fcik', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <d,a||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,b,j,l)*t2_abab(f,c,i,k)
    energy +=  0.375000000000000 * einsum('daef,ijba,lkdc,ebjl,fcik', g_aaaa[v, v, v, v], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <a,d||e,f>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_aaaa(e,b,j,l)*t2_abab(c,f,i,k)
    energy += -0.375000000000000 * einsum('adef,ijba,lkcd,ebjl,cfik', g_abab[v, v, v, v], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <a,d||f,e>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,j,l)*t2_abab(f,c,i,k)
    energy +=  0.375000000000000 * einsum('adfe,ijba,kldc,bejl,fcik', g_abab[v, v, v, v], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <d,a||f,e>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(b,e,l,j)*t2_aaaa(f,c,i,k)
    energy +=  0.375000000000000 * einsum('dafe,ijba,kldc,belj,fcik', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <d,a||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(b,e,l,j)*t2_abab(f,c,i,k)
    energy += -0.375000000000000 * einsum('dafe,ijba,lkdc,belj,fcik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <d,a||e,f>_bbbb*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(b,e,l,j)*t2_abab(c,f,i,k)
    energy +=  0.375000000000000 * einsum('daef,ijba,lkcd,belj,cfik', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <d,a||e,f>_aaaa*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(e,b,l,j)*t2_aaaa(f,c,i,k)
    energy += -0.375000000000000 * einsum('daef,ijab,kldc,eblj,fcik', g_aaaa[v, v, v, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <a,d||f,e>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_bbbb(e,b,j,l)*t2_aaaa(f,c,i,k)
    energy += -0.375000000000000 * einsum('adfe,ijab,klcd,ebjl,fcik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <d,a||e,f>_aaaa*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(e,b,l,j)*t2_abab(f,c,i,k)
    energy +=  0.375000000000000 * einsum('daef,ijab,lkdc,eblj,fcik', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <a,d||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(e,b,l,j)*t2_abab(c,f,i,k)
    energy += -0.375000000000000 * einsum('adef,ijab,lkcd,eblj,cfik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <a,d||f,e>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_abab(f,c,i,k)
    energy +=  0.375000000000000 * einsum('adfe,ijab,kldc,ebjl,fcik', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <d,a||e,f>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,b,j,l)*t2_abab(c,f,k,i)
    energy +=  0.375000000000000 * einsum('daef,jiba,kldc,ebjl,cfki', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <d,a||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(b,e,j,l)*t2_abab(f,c,k,i)
    energy += -0.375000000000000 * einsum('dafe,jiba,kldc,bejl,fcki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <d,a||e,f>_bbbb*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(b,e,j,l)*t2_abab(c,f,k,i)
    energy +=  0.375000000000000 * einsum('daef,jiba,klcd,bejl,cfki', g_bbbb[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <d,a||e,f>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,b,j,l)*t2_bbbb(f,c,i,k)
    energy += -0.375000000000000 * einsum('daef,jiba,lkdc,ebjl,fcik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <d,a||e,f>_bbbb*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(b,e,j,l)*t2_bbbb(f,c,i,k)
    energy += -0.375000000000000 * einsum('daef,jiba,kldc,bejl,fcik', g_bbbb[v, v, v, v], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <d,a||e,f>_aaaa*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(e,b,j,l)*t2_abab(f,c,k,i)
    energy +=  0.375000000000000 * einsum('daef,jiab,kldc,ebjl,fcki', g_aaaa[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <a,d||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(e,b,j,l)*t2_abab(c,f,k,i)
    energy += -0.375000000000000 * einsum('adef,jiab,klcd,ebjl,cfki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <a,d||e,f>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,b,j,l)*t2_bbbb(f,c,i,k)
    energy +=  0.375000000000000 * einsum('adef,jiab,kldc,ebjl,fcik', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <d,a||e,f>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(e,b,l,j)*t2_abab(c,f,k,i)
    energy +=  0.375000000000000 * einsum('daef,ijba,kldc,eblj,cfki', g_abab[v, v, v, v], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <d,a||f,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_abab(f,c,k,i)
    energy += -0.375000000000000 * einsum('dafe,ijba,kldc,ebjl,fcki', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	  0.3750 <d,a||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,b,j,l)*t2_abab(c,f,k,i)
    energy +=  0.375000000000000 * einsum('daef,ijba,klcd,ebjl,cfki', g_bbbb[v, v, v, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <d,a||e,f>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(e,b,l,j)*t2_bbbb(f,c,i,k)
    energy += -0.375000000000000 * einsum('daef,ijba,lkdc,eblj,fcik', g_abab[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3750 <d,a||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,b,j,l)*t2_bbbb(f,c,i,k)
    energy += -0.375000000000000 * einsum('daef,ijba,kldc,ebjl,fcik', g_bbbb[v, v, v, v], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (1, 3), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_aaaa(f,c,i,j)
    energy += -0.125000000000000 * einsum('baef,ijba,kldc,edkl,fcij', g_aaaa[v, v, v, v], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_aaaa(f,c,i,j)
    energy +=  0.125000000000000 * einsum('baef,ijba,klcd,edkl,fcij', g_aaaa[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_aaaa(f,c,i,j)
    energy +=  0.125000000000000 * einsum('baef,ijba,lkcd,edlk,fcij', g_aaaa[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,f>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,f,i,j)
    energy +=  0.125000000000000 * einsum('baef,ijba,kldc,edkl,cfij', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,c,i,j)
    energy += -0.125000000000000 * einsum('bafe,ijba,kldc,dekl,fcij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,f>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,f,i,j)
    energy += -0.125000000000000 * einsum('baef,ijba,klcd,edkl,cfij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,c,i,j)
    energy += -0.125000000000000 * einsum('bafe,ijba,lkdc,delk,fcij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,f>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,f,i,j)
    energy += -0.125000000000000 * einsum('baef,ijba,lkcd,edlk,cfij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||f,e>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,c,i,j)
    energy +=  0.125000000000000 * einsum('bafe,ijba,kldc,edkl,fcij', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,f>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,f,i,j)
    energy +=  0.125000000000000 * einsum('abef,ijab,kldc,edkl,cfij', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||f,e>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,c,i,j)
    energy += -0.125000000000000 * einsum('abfe,ijab,kldc,dekl,fcij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,f,i,j)
    energy += -0.125000000000000 * einsum('abef,ijab,klcd,edkl,cfij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||f,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,c,i,j)
    energy += -0.125000000000000 * einsum('abfe,ijab,lkdc,delk,fcij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,f,i,j)
    energy += -0.125000000000000 * einsum('abef,ijab,lkcd,edlk,cfij', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||f,e>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,c,i,j)
    energy +=  0.125000000000000 * einsum('abfe,ijab,kldc,edkl,fcij', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,f>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,f,j,i)
    energy +=  0.125000000000000 * einsum('baef,jiba,kldc,edkl,cfji', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,c,j,i)
    energy += -0.125000000000000 * einsum('bafe,jiba,kldc,dekl,fcji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,f>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,f,j,i)
    energy += -0.125000000000000 * einsum('baef,jiba,klcd,edkl,cfji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,c,j,i)
    energy += -0.125000000000000 * einsum('bafe,jiba,lkdc,delk,fcji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,f>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,f,j,i)
    energy += -0.125000000000000 * einsum('baef,jiba,lkcd,edlk,cfji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||f,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,c,j,i)
    energy +=  0.125000000000000 * einsum('bafe,jiba,kldc,edkl,fcji', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,f>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,k,l)*t2_abab(c,f,j,i)
    energy +=  0.125000000000000 * einsum('abef,jiab,kldc,edkl,cfji', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||f,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_abab(f,c,j,i)
    energy += -0.125000000000000 * einsum('abfe,jiab,kldc,dekl,fcji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(e,d,k,l)*t2_abab(c,f,j,i)
    energy += -0.125000000000000 * einsum('abef,jiab,klcd,edkl,cfji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||f,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_abab(f,c,j,i)
    energy += -0.125000000000000 * einsum('abfe,jiab,lkdc,delk,fcji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t2_abab(e,d,l,k)*t2_abab(c,f,j,i)
    energy += -0.125000000000000 * einsum('abef,jiab,lkcd,edlk,cfji', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||f,e>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_abab(f,c,j,i)
    energy +=  0.125000000000000 * einsum('abfe,jiab,kldc,edkl,fcji', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t2_abab(d,e,k,l)*t2_bbbb(f,c,i,j)
    energy +=  0.125000000000000 * einsum('baef,ijba,kldc,dekl,fcij', g_bbbb[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,k)*t2_bbbb(f,c,i,j)
    energy +=  0.125000000000000 * einsum('baef,ijba,lkdc,delk,fcij', g_bbbb[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,k,l)*t2_bbbb(f,c,i,j)
    energy += -0.125000000000000 * einsum('baef,ijba,kldc,edkl,fcij', g_bbbb[v, v, v, v], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_aaaa(f,c,i,k)
    energy +=  0.250000000000000 * einsum('baef,ijba,kldc,edjl,fcik', g_aaaa[v, v, v, v], l2_aaaa, l2_aaaa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,a||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_aaaa(f,c,i,k)
    energy += -0.250000000000000 * einsum('baef,ijba,klcd,edjl,fcik', g_aaaa[v, v, v, v], l2_aaaa, l2_abab, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,a||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_abab(f,c,i,k)
    energy += -0.250000000000000 * einsum('baef,ijba,lkdc,edjl,fcik', g_aaaa[v, v, v, v], l2_aaaa, l2_abab, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||e,f>_aaaa*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_abab(f,c,i,k)
    energy +=  0.250000000000000 * einsum('baef,ijba,kldc,edjl,fcik', g_aaaa[v, v, v, v], l2_aaaa, l2_bbbb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,a||f,e>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,c,i,k)
    energy += -0.250000000000000 * einsum('bafe,ijba,kldc,delj,fcik', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,c,i,k)
    energy +=  0.250000000000000 * einsum('bafe,ijba,klcd,edjl,fcik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||f,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,c,i,k)
    energy +=  0.250000000000000 * einsum('bafe,ijba,lkdc,delj,fcik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||e,f>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(c,f,i,k)
    energy +=  0.250000000000000 * einsum('baef,ijba,lkcd,edlj,cfik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,a||f,e>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,c,i,k)
    energy += -0.250000000000000 * einsum('bafe,ijba,kldc,edjl,fcik', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <a,b||f,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_aaaa(f,c,i,k)
    energy += -0.250000000000000 * einsum('abfe,ijab,kldc,delj,fcik', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <a,b||f,e>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_aaaa(f,c,i,k)
    energy +=  0.250000000000000 * einsum('abfe,ijab,klcd,edjl,fcik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <a,b||f,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_abab(f,c,i,k)
    energy +=  0.250000000000000 * einsum('abfe,ijab,lkdc,delj,fcik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <a,b||e,f>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t2_abab(e,d,l,j)*t2_abab(c,f,i,k)
    energy +=  0.250000000000000 * einsum('abef,ijab,lkcd,edlj,cfik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <a,b||f,e>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_abab(f,c,i,k)
    energy += -0.250000000000000 * einsum('abfe,ijab,kldc,edjl,fcik', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,a||e,f>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_abab(c,f,k,i)
    energy += -0.250000000000000 * einsum('baef,jiba,kldc,edjl,cfki', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||f,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t2_abab(d,e,j,l)*t2_abab(f,c,k,i)
    energy +=  0.250000000000000 * einsum('bafe,jiba,kldc,dejl,fcki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||e,f>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_abab(c,f,k,i)
    energy +=  0.250000000000000 * einsum('baef,jiba,klcd,edjl,cfki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||e,f>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_bbbb(f,c,i,k)
    energy +=  0.250000000000000 * einsum('baef,jiba,lkdc,edjl,fcik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,a||e,f>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_bbbb(f,c,i,k)
    energy += -0.250000000000000 * einsum('baef,jiba,kldc,edjl,fcik', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <a,b||e,f>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t2_aaaa(e,d,j,l)*t2_abab(c,f,k,i)
    energy += -0.250000000000000 * einsum('abef,jiab,kldc,edjl,cfki', g_abab[v, v, v, v], l2_abab, l2_aaaa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <a,b||f,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t2_abab(d,e,j,l)*t2_abab(f,c,k,i)
    energy +=  0.250000000000000 * einsum('abfe,jiab,kldc,dejl,fcki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <a,b||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t2_abab(e,d,j,l)*t2_abab(c,f,k,i)
    energy +=  0.250000000000000 * einsum('abef,jiab,klcd,edjl,cfki', g_abab[v, v, v, v], l2_abab, l2_abab, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <a,b||e,f>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t2_aaaa(e,d,j,l)*t2_bbbb(f,c,i,k)
    energy +=  0.250000000000000 * einsum('abef,jiab,lkdc,edjl,fcik', g_abab[v, v, v, v], l2_abab, l2_abab, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <a,b||e,f>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t2_abab(e,d,j,l)*t2_bbbb(f,c,i,k)
    energy += -0.250000000000000 * einsum('abef,jiab,kldc,edjl,fcik', g_abab[v, v, v, v], l2_abab, l2_bbbb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t2_abab(d,e,l,j)*t2_abab(c,f,k,i)
    energy +=  0.250000000000000 * einsum('baef,ijba,kldc,delj,cfki', g_bbbb[v, v, v, v], l2_bbbb, l2_aaaa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,a||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t2_bbbb(e,d,j,l)*t2_abab(c,f,k,i)
    energy += -0.250000000000000 * einsum('baef,ijba,klcd,edjl,cfki', g_bbbb[v, v, v, v], l2_bbbb, l2_abab, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	 -0.2500 <b,a||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t2_abab(d,e,l,j)*t2_bbbb(f,c,i,k)
    energy += -0.250000000000000 * einsum('baef,ijba,lkdc,delj,fcik', g_bbbb[v, v, v, v], l2_bbbb, l2_abab, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 <b,a||e,f>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t2_bbbb(e,d,j,l)*t2_bbbb(f,c,i,k)
    energy +=  0.250000000000000 * einsum('baef,ijba,kldc,edjl,fcik', g_bbbb[v, v, v, v], l2_bbbb, l2_bbbb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||k,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(c,b,a,i,j,m)
    energy +=  0.125000000000000 * einsum('mdkl,ijba,kldc,cbaijm', g_aaaa[o, v, o, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||k,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(a,b,c,i,j,m)
    energy +=  0.125000000000000 * einsum('dmkl,ijba,kldc,abcijm', g_abab[v, o, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||k,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aaaaaa(c,b,a,i,j,m)
    energy += -0.125000000000000 * einsum('mdkl,ijba,klcd,cbaijm', g_abab[o, v, o, o], l2_aaaa, l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||l,k>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(a,b,c,i,j,m)
    energy +=  0.125000000000000 * einsum('dmlk,ijba,lkdc,abcijm', g_abab[v, o, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||l,k>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aaaaaa(c,b,a,i,j,m)
    energy += -0.125000000000000 * einsum('mdlk,ijba,lkcd,cbaijm', g_abab[o, v, o, o], l2_aaaa, l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||k,l>_bbbb*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_aabaab(a,b,c,i,j,m)
    energy += -0.125000000000000 * einsum('mdkl,ijba,kldc,abcijm', g_bbbb[o, v, o, o], l2_aaaa, l2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||k,l>_aaaa*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,b,a,i,m,j)
    energy += -0.125000000000000 * einsum('mdkl,ijba,kldc,cbaimj', g_aaaa[o, v, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||k,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(b,c,a,i,j,m)
    energy +=  0.125000000000000 * einsum('dmkl,ijba,kldc,bcaijm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||k,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(c,b,a,i,m,j)
    energy +=  0.125000000000000 * einsum('mdkl,ijba,klcd,cbaimj', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||l,k>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(b,c,a,i,j,m)
    energy +=  0.125000000000000 * einsum('dmlk,ijba,lkdc,bcaijm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||l,k>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(c,b,a,i,m,j)
    energy +=  0.125000000000000 * einsum('mdlk,ijba,lkcd,cbaimj', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||k,l>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(b,c,a,i,j,m)
    energy += -0.125000000000000 * einsum('mdkl,ijba,kldc,bcaijm', g_bbbb[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||k,l>_aaaa*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,i,m,j)
    energy += -0.125000000000000 * einsum('mdkl,ijab,kldc,cabimj', g_aaaa[o, v, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||k,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_abbabb(a,b,c,i,j,m)
    energy += -0.125000000000000 * einsum('dmkl,ijab,kldc,abcijm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||k,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,i,m,j)
    energy +=  0.125000000000000 * einsum('mdkl,ijab,klcd,cabimj', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||l,k>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,b,c,i,j,m)
    energy += -0.125000000000000 * einsum('dmlk,ijab,lkdc,abcijm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||l,k>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_aabaab(c,a,b,i,m,j)
    energy +=  0.125000000000000 * einsum('mdlk,ijab,lkcd,cabimj', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||k,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,b,c,i,j,m)
    energy +=  0.125000000000000 * einsum('mdkl,ijab,kldc,abcijm', g_bbbb[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||k,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,b,a,m,j,i)
    energy +=  0.125000000000000 * einsum('mdkl,jiba,kldc,cbamji', g_aaaa[o, v, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||k,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_abbabb(b,c,a,j,i,m)
    energy +=  0.125000000000000 * einsum('dmkl,jiba,kldc,bcajim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||k,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_aabaab(c,b,a,m,j,i)
    energy += -0.125000000000000 * einsum('mdkl,jiba,klcd,cbamji', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||l,k>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(b,c,a,j,i,m)
    energy +=  0.125000000000000 * einsum('dmlk,jiba,lkdc,bcajim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||l,k>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_aabaab(c,b,a,m,j,i)
    energy += -0.125000000000000 * einsum('mdlk,jiba,lkcd,cbamji', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||k,l>_bbbb*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(b,c,a,j,i,m)
    energy += -0.125000000000000 * einsum('mdkl,jiba,kldc,bcajim', g_bbbb[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||k,l>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,m,j,i)
    energy +=  0.125000000000000 * einsum('mdkl,jiab,kldc,cabmji', g_aaaa[o, v, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||k,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_abbabb(a,b,c,j,i,m)
    energy += -0.125000000000000 * einsum('dmkl,jiab,kldc,abcjim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||k,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,m,j,i)
    energy += -0.125000000000000 * einsum('mdkl,jiab,klcd,cabmji', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||l,k>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,b,c,j,i,m)
    energy += -0.125000000000000 * einsum('dmlk,jiab,lkdc,abcjim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||l,k>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t3_aabaab(c,a,b,m,j,i)
    energy += -0.125000000000000 * einsum('mdlk,jiab,lkcd,cabmji', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||k,l>_bbbb*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,b,c,j,i,m)
    energy +=  0.125000000000000 * einsum('mdkl,jiab,kldc,abcjim', g_bbbb[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||k,l>_aaaa*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_abbabb(c,b,a,m,j,i)
    energy += -0.125000000000000 * einsum('mdkl,ijba,kldc,cbamji', g_aaaa[o, v, o, o], l2_bbbb, l2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||k,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_bbbbbb(c,b,a,i,j,m)
    energy += -0.125000000000000 * einsum('dmkl,ijba,kldc,cbaijm', g_abab[v, o, o, o], l2_bbbb, l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||k,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,b,a,m,j,i)
    energy +=  0.125000000000000 * einsum('mdkl,ijba,klcd,cbamji', g_abab[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||l,k>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_bbbbbb(c,b,a,i,j,m)
    energy += -0.125000000000000 * einsum('dmlk,ijba,lkdc,cbaijm', g_abab[v, o, o, o], l2_bbbb, l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||l,k>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,b,a,m,j,i)
    energy +=  0.125000000000000 * einsum('mdlk,ijba,lkcd,cbamji', g_abab[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||k,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(c,b,a,i,j,m)
    energy +=  0.125000000000000 * einsum('mdkl,ijba,kldc,cbaijm', g_bbbb[o, v, o, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(c,b,a,i,k,m)
    energy += -0.500000000000000 * einsum('mdjl,ijba,kldc,cbaikm', g_aaaa[o, v, o, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(a,b,c,i,k,m)
    energy += -0.500000000000000 * einsum('dmjl,ijba,kldc,abcikm', g_abab[v, o, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aaaaaa(c,b,a,i,k,m)
    energy +=  0.500000000000000 * einsum('mdjl,ijba,klcd,cbaikm', g_abab[o, v, o, o], l2_aaaa, l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(a,b,c,i,m,k)
    energy +=  0.500000000000000 * einsum('mdjl,ijba,lkdc,abcimk', g_aaaa[o, v, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||j,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_aabaab(a,b,c,i,m,k)
    energy += -0.500000000000000 * einsum('mdjl,ijba,kldc,abcimk', g_abab[o, v, o, o], l2_aaaa, l2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,b,a,i,k,m)
    energy += -0.500000000000000 * einsum('dmlj,ijba,kldc,cbaikm', g_abab[v, o, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,l>_bbbb*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(c,b,a,i,k,m)
    energy +=  0.500000000000000 * einsum('mdjl,ijba,klcd,cbaikm', g_bbbb[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(b,c,a,i,k,m)
    energy += -0.500000000000000 * einsum('dmlj,ijba,lkdc,bcaikm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(c,b,a,i,m,k)
    energy += -0.500000000000000 * einsum('mdlj,ijba,lkcd,cbaimk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,l>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(b,c,a,i,k,m)
    energy +=  0.500000000000000 * einsum('mdjl,ijba,kldc,bcaikm', g_bbbb[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||l,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,i,k,m)
    energy += -0.500000000000000 * einsum('dmlj,ijab,kldc,cabikm', g_abab[v, o, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,i,k,m)
    energy +=  0.500000000000000 * einsum('mdjl,ijab,klcd,cabikm', g_bbbb[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,b,c,i,k,m)
    energy +=  0.500000000000000 * einsum('dmlj,ijab,lkdc,abcikm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_aabaab(c,a,b,i,m,k)
    energy += -0.500000000000000 * einsum('mdlj,ijab,lkcd,cabimk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||j,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,b,c,i,k,m)
    energy += -0.500000000000000 * einsum('mdjl,ijab,kldc,abcikm', g_bbbb[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||j,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,b,a,m,k,i)
    energy += -0.500000000000000 * einsum('mdjl,jiba,kldc,cbamki', g_aaaa[o, v, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,m||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_abbabb(b,c,a,k,i,m)
    energy += -0.500000000000000 * einsum('dmjl,jiba,kldc,bcakim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_aabaab(c,b,a,m,k,i)
    energy +=  0.500000000000000 * einsum('mdjl,jiba,klcd,cbamki', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||j,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(b,c,a,m,k,i)
    energy += -0.500000000000000 * einsum('mdjl,jiba,lkdc,bcamki', g_aaaa[o, v, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(b,c,a,m,k,i)
    energy +=  0.500000000000000 * einsum('mdjl,jiba,kldc,bcamki', g_abab[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||j,l>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,m,k,i)
    energy += -0.500000000000000 * einsum('mdjl,jiab,kldc,cabmki', g_aaaa[o, v, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_abbabb(a,b,c,k,i,m)
    energy +=  0.500000000000000 * einsum('dmjl,jiab,kldc,abckim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,m,k,i)
    energy +=  0.500000000000000 * einsum('mdjl,jiab,klcd,cabmki', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <m,d||j,l>_aaaa*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,b,c,m,k,i)
    energy +=  0.500000000000000 * einsum('mdjl,jiab,lkdc,abcmki', g_aaaa[o, v, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||j,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,b,c,m,k,i)
    energy += -0.500000000000000 * einsum('mdjl,jiab,kldc,abcmki', g_abab[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_abbabb(c,b,a,k,i,m)
    energy +=  0.500000000000000 * einsum('dmlj,ijba,kldc,cbakim', g_abab[v, o, o, o], l2_bbbb, l2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,b,a,k,i,m)
    energy += -0.500000000000000 * einsum('mdjl,ijba,klcd,cbakim', g_bbbb[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,m||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_bbbbbb(c,b,a,i,k,m)
    energy +=  0.500000000000000 * einsum('dmlj,ijba,lkdc,cbaikm', g_abab[v, o, o, o], l2_bbbb, l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,b,a,m,k,i)
    energy += -0.500000000000000 * einsum('mdlj,ijba,lkcd,cbamki', g_abab[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,d||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(c,b,a,i,k,m)
    energy += -0.500000000000000 * einsum('mdjl,ijba,kldc,cbaikm', g_bbbb[o, v, o, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(c,b,a,k,l,m)
    energy +=  0.125000000000000 * einsum('mdij,ijba,kldc,cbaklm', g_aaaa[o, v, o, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(a,b,c,k,m,l)
    energy +=  0.125000000000000 * einsum('mdij,ijba,kldc,abckml', g_aaaa[o, v, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(a,b,c,m,l,k)
    energy += -0.125000000000000 * einsum('mdij,ijba,lkdc,abcmlk', g_aaaa[o, v, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||i,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,b,a,k,l,m)
    energy += -0.125000000000000 * einsum('dmij,ijba,kldc,cbaklm', g_abab[v, o, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(b,c,a,k,l,m)
    energy +=  0.125000000000000 * einsum('dmij,ijba,kldc,bcaklm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(c,b,a,k,m,l)
    energy +=  0.125000000000000 * einsum('mdij,ijba,klcd,cbakml', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(b,c,a,l,k,m)
    energy +=  0.125000000000000 * einsum('dmij,ijba,lkdc,bcalkm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(c,b,a,m,l,k)
    energy += -0.125000000000000 * einsum('mdij,ijba,lkcd,cbamlk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||i,j>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(b,c,a,m,l,k)
    energy +=  0.125000000000000 * einsum('mdij,ijba,kldc,bcamlk', g_abab[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||i,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,k,l,m)
    energy += -0.125000000000000 * einsum('dmij,ijab,kldc,cabklm', g_abab[v, o, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_abbabb(a,b,c,k,l,m)
    energy += -0.125000000000000 * einsum('dmij,ijab,kldc,abcklm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,k,m,l)
    energy +=  0.125000000000000 * einsum('mdij,ijab,klcd,cabkml', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,b,c,l,k,m)
    energy += -0.125000000000000 * einsum('dmij,ijab,lkdc,abclkm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_aabaab(c,a,b,m,l,k)
    energy += -0.125000000000000 * einsum('mdij,ijab,lkcd,cabmlk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||i,j>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,b,c,m,l,k)
    energy += -0.125000000000000 * einsum('mdij,ijab,kldc,abcmlk', g_abab[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||j,i>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,b,a,k,l,m)
    energy += -0.125000000000000 * einsum('dmji,jiba,kldc,cbaklm', g_abab[v, o, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_abbabb(b,c,a,k,l,m)
    energy +=  0.125000000000000 * einsum('dmji,jiba,kldc,bcaklm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_aabaab(c,b,a,k,m,l)
    energy +=  0.125000000000000 * einsum('mdji,jiba,klcd,cbakml', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <d,m||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(b,c,a,l,k,m)
    energy +=  0.125000000000000 * einsum('dmji,jiba,lkdc,bcalkm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_aabaab(c,b,a,m,l,k)
    energy += -0.125000000000000 * einsum('mdji,jiba,lkcd,cbamlk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||j,i>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(b,c,a,m,l,k)
    energy +=  0.125000000000000 * einsum('mdji,jiba,kldc,bcamlk', g_abab[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||j,i>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(c,a,b,k,l,m)
    energy += -0.125000000000000 * einsum('dmji,jiab,kldc,cabklm', g_abab[v, o, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_abbabb(a,b,c,k,l,m)
    energy += -0.125000000000000 * einsum('dmji,jiab,kldc,abcklm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_aabaab(c,a,b,k,m,l)
    energy +=  0.125000000000000 * einsum('mdji,jiab,klcd,cabkml', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,m||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,b,c,l,k,m)
    energy += -0.125000000000000 * einsum('dmji,jiab,lkdc,abclkm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t3_aabaab(c,a,b,m,l,k)
    energy += -0.125000000000000 * einsum('mdji,jiab,lkcd,cabmlk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||j,i>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,b,c,m,l,k)
    energy += -0.125000000000000 * einsum('mdji,jiab,kldc,abcmlk', g_abab[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,b,a,k,l,m)
    energy += -0.125000000000000 * einsum('mdij,ijba,klcd,cbaklm', g_bbbb[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,d||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,b,a,l,k,m)
    energy += -0.125000000000000 * einsum('mdij,ijba,lkcd,cbalkm', g_bbbb[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,d||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(c,b,a,k,l,m)
    energy +=  0.125000000000000 * einsum('mdij,ijba,kldc,cbaklm', g_bbbb[o, v, o, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||k,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,a,i,j,m)
    energy +=  0.125000000000000 * einsum('mbkl,ijba,kldc,dcaijm', g_aaaa[o, v, o, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <b,m||k,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,i,j,m)
    energy +=  0.125000000000000 * einsum('bmkl,ijba,kldc,dacijm', g_abab[v, o, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||k,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,i,j,m)
    energy += -0.125000000000000 * einsum('bmkl,ijba,klcd,acdijm', g_abab[v, o, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <b,m||l,k>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,i,j,m)
    energy +=  0.125000000000000 * einsum('bmlk,ijba,lkdc,dacijm', g_abab[v, o, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||l,k>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,i,j,m)
    energy += -0.125000000000000 * einsum('bmlk,ijba,lkcd,acdijm', g_abab[v, o, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||k,l>_aaaa*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,a,i,m,j)
    energy += -0.125000000000000 * einsum('mbkl,ijba,kldc,dcaimj', g_aaaa[o, v, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||k,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,c,a,i,j,m)
    energy += -0.125000000000000 * einsum('bmkl,ijba,kldc,dcaijm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||k,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,a,i,j,m)
    energy += -0.125000000000000 * einsum('bmkl,ijba,klcd,cdaijm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||l,k>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,c,a,i,j,m)
    energy += -0.125000000000000 * einsum('bmlk,ijba,lkdc,dcaijm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||l,k>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,a,i,j,m)
    energy += -0.125000000000000 * einsum('bmlk,ijba,lkcd,cdaijm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||k,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,i,m,j)
    energy += -0.125000000000000 * einsum('mbkl,ijab,kldc,dacimj', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||k,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,i,m,j)
    energy +=  0.125000000000000 * einsum('mbkl,ijab,klcd,acdimj', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||l,k>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,i,m,j)
    energy += -0.125000000000000 * einsum('mblk,ijab,lkdc,dacimj', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||l,k>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,i,m,j)
    energy +=  0.125000000000000 * einsum('mblk,ijab,lkcd,acdimj', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||k,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,i,j,m)
    energy +=  0.125000000000000 * einsum('mbkl,ijab,kldc,acdijm', g_bbbb[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||k,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,a,m,j,i)
    energy +=  0.125000000000000 * einsum('mbkl,jiba,kldc,dcamji', g_aaaa[o, v, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||k,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,c,a,j,i,m)
    energy += -0.125000000000000 * einsum('bmkl,jiba,kldc,dcajim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||k,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,a,j,i,m)
    energy += -0.125000000000000 * einsum('bmkl,jiba,klcd,cdajim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||l,k>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,c,a,j,i,m)
    energy += -0.125000000000000 * einsum('bmlk,jiba,lkdc,dcajim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||l,k>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,a,j,i,m)
    energy += -0.125000000000000 * einsum('bmlk,jiba,lkcd,cdajim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||k,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,m,j,i)
    energy +=  0.125000000000000 * einsum('mbkl,jiab,kldc,dacmji', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||k,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,m,j,i)
    energy += -0.125000000000000 * einsum('mbkl,jiab,klcd,acdmji', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||l,k>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,m,j,i)
    energy +=  0.125000000000000 * einsum('mblk,jiab,lkdc,dacmji', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||l,k>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,m,j,i)
    energy += -0.125000000000000 * einsum('mblk,jiab,lkcd,acdmji', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||k,l>_bbbb*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,j,i,m)
    energy +=  0.125000000000000 * einsum('mbkl,jiab,kldc,acdjim', g_bbbb[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||k,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,c,a,m,j,i)
    energy += -0.125000000000000 * einsum('mbkl,ijba,kldc,dcamji', g_abab[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||k,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,a,m,j,i)
    energy += -0.125000000000000 * einsum('mbkl,ijba,klcd,cdamji', g_abab[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||l,k>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,c,a,m,j,i)
    energy += -0.125000000000000 * einsum('mblk,ijba,lkdc,dcamji', g_abab[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||l,k>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,a,m,j,i)
    energy += -0.125000000000000 * einsum('mblk,ijba,lkcd,cdamji', g_abab[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||k,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,a,i,j,m)
    energy +=  0.125000000000000 * einsum('mbkl,ijba,kldc,dcaijm', g_bbbb[o, v, o, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,a,i,k,m)
    energy += -0.500000000000000 * einsum('mbjl,ijba,kldc,dcaikm', g_aaaa[o, v, o, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,i,k,m)
    energy += -0.500000000000000 * einsum('bmjl,ijba,kldc,dacikm', g_abab[v, o, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,i,k,m)
    energy +=  0.500000000000000 * einsum('bmjl,ijba,klcd,acdikm', g_abab[v, o, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,i,m,k)
    energy +=  0.500000000000000 * einsum('mbjl,ijba,lkdc,dacimk', g_aaaa[o, v, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,i,m,k)
    energy += -0.500000000000000 * einsum('mbjl,ijba,lkcd,acdimk', g_aaaa[o, v, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||j,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,i,k,m)
    energy += -0.500000000000000 * einsum('bmjl,ijba,kldc,acdikm', g_abab[v, o, o, o], l2_aaaa, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||l,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,a,i,k,m)
    energy += -0.500000000000000 * einsum('bmlj,ijba,kldc,dcaikm', g_abab[v, o, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,c,a,i,k,m)
    energy +=  0.500000000000000 * einsum('bmlj,ijba,lkdc,dcaikm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||l,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,a,i,k,m)
    energy +=  0.500000000000000 * einsum('bmlj,ijba,lkcd,cdaikm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||l,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,a,i,k,m)
    energy += -0.500000000000000 * einsum('mblj,ijab,kldc,dcaikm', g_abab[o, v, o, o], l2_abab, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,i,k,m)
    energy += -0.500000000000000 * einsum('mbjl,ijab,kldc,dacikm', g_bbbb[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,i,k,m)
    energy +=  0.500000000000000 * einsum('mbjl,ijab,klcd,acdikm', g_bbbb[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,i,m,k)
    energy +=  0.500000000000000 * einsum('mblj,ijab,lkdc,dacimk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||l,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,i,m,k)
    energy += -0.500000000000000 * einsum('mblj,ijab,lkcd,acdimk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,i,k,m)
    energy += -0.500000000000000 * einsum('mbjl,ijab,kldc,acdikm', g_bbbb[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,a,m,k,i)
    energy += -0.500000000000000 * einsum('mbjl,jiba,kldc,dcamki', g_aaaa[o, v, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,c,a,k,i,m)
    energy +=  0.500000000000000 * einsum('bmjl,jiba,kldc,dcakim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,m||j,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,a,k,i,m)
    energy +=  0.500000000000000 * einsum('bmjl,jiba,klcd,cdakim', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,c,a,m,k,i)
    energy +=  0.500000000000000 * einsum('mbjl,jiba,lkdc,dcamki', g_aaaa[o, v, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,a,m,k,i)
    energy +=  0.500000000000000 * einsum('mbjl,jiba,lkcd,cdamki', g_aaaa[o, v, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,m||j,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,a,i,k,m)
    energy += -0.500000000000000 * einsum('bmjl,jiba,kldc,dcaikm', g_abab[v, o, o, o], l2_abab, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,m,k,i)
    energy += -0.500000000000000 * einsum('mbjl,jiab,kldc,dacmki', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,m,k,i)
    energy +=  0.500000000000000 * einsum('mbjl,jiab,klcd,acdmki', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,m,k,i)
    energy += -0.500000000000000 * einsum('mbjl,jiab,kldc,acdmki', g_abab[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||l,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,a,m,k,i)
    energy += -0.500000000000000 * einsum('mblj,ijba,kldc,dcamki', g_abab[o, v, o, o], l2_bbbb, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,c,a,k,i,m)
    energy +=  0.500000000000000 * einsum('mbjl,ijba,kldc,dcakim', g_bbbb[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,a,k,i,m)
    energy +=  0.500000000000000 * einsum('mbjl,ijba,klcd,cdakim', g_bbbb[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,c,a,m,k,i)
    energy +=  0.500000000000000 * einsum('mblj,ijba,lkdc,dcamki', g_abab[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <m,b||l,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,a,m,k,i)
    energy +=  0.500000000000000 * einsum('mblj,ijba,lkcd,cdamki', g_abab[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <m,b||j,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,a,i,k,m)
    energy += -0.500000000000000 * einsum('mbjl,ijba,kldc,dcaikm', g_bbbb[o, v, o, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,a,k,l,m)
    energy +=  0.125000000000000 * einsum('mbij,ijba,kldc,dcaklm', g_aaaa[o, v, o, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,k,m,l)
    energy +=  0.125000000000000 * einsum('mbij,ijba,kldc,dackml', g_aaaa[o, v, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,k,m,l)
    energy += -0.125000000000000 * einsum('mbij,ijba,klcd,acdkml', g_aaaa[o, v, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,m,l,k)
    energy += -0.125000000000000 * einsum('mbij,ijba,lkdc,dacmlk', g_aaaa[o, v, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,m,l,k)
    energy +=  0.125000000000000 * einsum('mbij,ijba,lkcd,acdmlk', g_aaaa[o, v, o, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_aaaa*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,m,l,k)
    energy +=  0.125000000000000 * einsum('mbij,ijba,kldc,acdmlk', g_aaaa[o, v, o, o], l2_aaaa, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||i,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,a,k,l,m)
    energy += -0.125000000000000 * einsum('bmij,ijba,kldc,dcaklm', g_abab[v, o, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,c,a,k,l,m)
    energy += -0.125000000000000 * einsum('bmij,ijba,kldc,dcaklm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,a,k,l,m)
    energy += -0.125000000000000 * einsum('bmij,ijba,klcd,cdaklm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,c,a,l,k,m)
    energy += -0.125000000000000 * einsum('bmij,ijba,lkdc,dcalkm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||i,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,a,l,k,m)
    energy += -0.125000000000000 * einsum('bmij,ijba,lkcd,cdalkm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||i,j>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,a,k,l,m)
    energy += -0.125000000000000 * einsum('bmij,ijba,kldc,dcaklm', g_abab[v, o, o, o], l2_abab, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||i,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,a,k,l,m)
    energy += -0.125000000000000 * einsum('mbij,ijab,kldc,dcaklm', g_abab[o, v, o, o], l2_abab, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,k,m,l)
    energy += -0.125000000000000 * einsum('mbij,ijab,kldc,dackml', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,k,m,l)
    energy +=  0.125000000000000 * einsum('mbij,ijab,klcd,acdkml', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,m,l,k)
    energy +=  0.125000000000000 * einsum('mbij,ijab,lkdc,dacmlk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||i,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,m,l,k)
    energy += -0.125000000000000 * einsum('mbij,ijab,lkcd,acdmlk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||i,j>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,m,l,k)
    energy += -0.125000000000000 * einsum('mbij,ijab,kldc,acdmlk', g_abab[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||j,i>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,a,k,l,m)
    energy += -0.125000000000000 * einsum('bmji,jiba,kldc,dcaklm', g_abab[v, o, o, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,c,a,k,l,m)
    energy += -0.125000000000000 * einsum('bmji,jiba,kldc,dcaklm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,a,k,l,m)
    energy += -0.125000000000000 * einsum('bmji,jiba,klcd,cdaklm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,c,a,l,k,m)
    energy += -0.125000000000000 * einsum('bmji,jiba,lkdc,dcalkm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||j,i>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,a,l,k,m)
    energy += -0.125000000000000 * einsum('bmji,jiba,lkcd,cdalkm', g_abab[v, o, o, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,m||j,i>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,a,k,l,m)
    energy += -0.125000000000000 * einsum('bmji,jiba,kldc,dcaklm', g_abab[v, o, o, o], l2_abab, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||j,i>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aaaaaa(d,c,a,k,l,m)
    energy += -0.125000000000000 * einsum('mbji,jiab,kldc,dcaklm', g_abab[o, v, o, o], l2_abab, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_aabaab(d,a,c,k,m,l)
    energy += -0.125000000000000 * einsum('mbji,jiab,kldc,dackml', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_aabaab(a,c,d,k,m,l)
    energy +=  0.125000000000000 * einsum('mbji,jiab,klcd,acdkml', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_aabaab(d,a,c,m,l,k)
    energy +=  0.125000000000000 * einsum('mbji,jiab,lkdc,dacmlk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||j,i>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t3_aabaab(a,c,d,m,l,k)
    energy += -0.125000000000000 * einsum('mbji,jiab,lkcd,acdmlk', g_abab[o, v, o, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <m,b||j,i>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,d,m,l,k)
    energy += -0.125000000000000 * einsum('mbji,jiab,kldc,acdmlk', g_abab[o, v, o, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(d,c,a,k,l,m)
    energy +=  0.125000000000000 * einsum('mbij,ijba,kldc,dcaklm', g_bbbb[o, v, o, o], l2_bbbb, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,c,a,k,l,m)
    energy +=  0.125000000000000 * einsum('mbij,ijba,kldc,dcaklm', g_bbbb[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,a,k,l,m)
    energy +=  0.125000000000000 * einsum('mbij,ijba,klcd,cdaklm', g_bbbb[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,c,a,l,k,m)
    energy +=  0.125000000000000 * einsum('mbij,ijba,lkdc,dcalkm', g_bbbb[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,a,l,k,m)
    energy +=  0.125000000000000 * einsum('mbij,ijba,lkcd,cdalkm', g_bbbb[o, v, o, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <m,b||i,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(d,c,a,k,l,m)
    energy +=  0.125000000000000 * einsum('mbij,ijba,kldc,dcaklm', g_bbbb[o, v, o, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,b,a,i,j,k)
    energy +=  0.125000000000000 * einsum('dcel,ijba,kldc,ebaijk', g_aaaa[v, v, v, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aaaaaa(e,b,a,i,j,k)
    energy +=  0.125000000000000 * einsum('dcel,ijba,kldc,ebaijk', g_abab[v, v, v, o], l2_aaaa, l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aaaaaa(e,b,a,i,j,k)
    energy +=  0.125000000000000 * einsum('cdel,ijba,klcd,ebaijk', g_abab[v, v, v, o], l2_aaaa, l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||l,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(a,b,e,i,j,k)
    energy += -0.125000000000000 * einsum('dcle,ijba,lkdc,abeijk', g_abab[v, v, o, v], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||l,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(a,b,e,i,j,k)
    energy += -0.125000000000000 * einsum('cdle,ijba,lkcd,abeijk', g_abab[v, v, o, v], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,l>_bbbb*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_aabaab(a,b,e,i,j,k)
    energy += -0.125000000000000 * einsum('dcel,ijba,kldc,abeijk', g_bbbb[v, v, v, o], l2_aaaa, l2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,l>_aaaa*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(e,b,a,i,k,j)
    energy += -0.125000000000000 * einsum('dcel,ijba,kldc,ebaikj', g_aaaa[v, v, v, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(e,b,a,i,k,j)
    energy += -0.125000000000000 * einsum('dcel,ijba,kldc,ebaikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,b,a,i,k,j)
    energy += -0.125000000000000 * einsum('cdel,ijba,klcd,ebaikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(b,e,a,i,j,k)
    energy += -0.125000000000000 * einsum('dcle,ijba,lkdc,beaijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(b,e,a,i,j,k)
    energy += -0.125000000000000 * einsum('cdle,ijba,lkcd,beaijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,l>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(b,e,a,i,j,k)
    energy += -0.125000000000000 * einsum('dcel,ijba,kldc,beaijk', g_bbbb[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,l>_aaaa*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(e,a,b,i,k,j)
    energy += -0.125000000000000 * einsum('dcel,ijab,kldc,eabikj', g_aaaa[v, v, v, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_aabaab(e,a,b,i,k,j)
    energy += -0.125000000000000 * einsum('dcel,ijab,kldc,eabikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(e,a,b,i,k,j)
    energy += -0.125000000000000 * einsum('cdel,ijab,klcd,eabikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,b,e,i,j,k)
    energy +=  0.125000000000000 * einsum('dcle,ijab,lkdc,abeijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_abbabb(a,b,e,i,j,k)
    energy +=  0.125000000000000 * einsum('cdle,ijab,lkcd,abeijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,b,e,i,j,k)
    energy +=  0.125000000000000 * einsum('dcel,ijab,kldc,abeijk', g_bbbb[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(e,b,a,k,j,i)
    energy +=  0.125000000000000 * einsum('dcel,jiba,kldc,ebakji', g_aaaa[v, v, v, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_aabaab(e,b,a,k,j,i)
    energy +=  0.125000000000000 * einsum('dcel,jiba,kldc,ebakji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,b,a,k,j,i)
    energy +=  0.125000000000000 * einsum('cdel,jiba,klcd,ebakji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(b,e,a,j,i,k)
    energy += -0.125000000000000 * einsum('dcle,jiba,lkdc,beajik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_abbabb(b,e,a,j,i,k)
    energy += -0.125000000000000 * einsum('cdle,jiba,lkcd,beajik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,l>_bbbb*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(b,e,a,j,i,k)
    energy += -0.125000000000000 * einsum('dcel,jiba,kldc,beajik', g_bbbb[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,l>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(e,a,b,k,j,i)
    energy +=  0.125000000000000 * einsum('dcel,jiab,kldc,eabkji', g_aaaa[v, v, v, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_aabaab(e,a,b,k,j,i)
    energy +=  0.125000000000000 * einsum('dcel,jiab,kldc,eabkji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_aabaab(e,a,b,k,j,i)
    energy +=  0.125000000000000 * einsum('cdel,jiab,klcd,eabkji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,b,e,j,i,k)
    energy +=  0.125000000000000 * einsum('dcle,jiab,lkdc,abejik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t3_abbabb(a,b,e,j,i,k)
    energy +=  0.125000000000000 * einsum('cdle,jiab,lkcd,abejik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,l>_bbbb*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,b,e,j,i,k)
    energy +=  0.125000000000000 * einsum('dcel,jiab,kldc,abejik', g_bbbb[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,l>_aaaa*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_abbabb(e,b,a,k,j,i)
    energy += -0.125000000000000 * einsum('dcel,ijba,kldc,ebakji', g_aaaa[v, v, v, o], l2_bbbb, l2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(e,b,a,k,j,i)
    energy += -0.125000000000000 * einsum('dcel,ijba,kldc,ebakji', g_abab[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(e,b,a,k,j,i)
    energy += -0.125000000000000 * einsum('cdel,ijba,klcd,ebakji', g_abab[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_bbbbbb(e,b,a,i,j,k)
    energy +=  0.125000000000000 * einsum('dcle,ijba,lkdc,ebaijk', g_abab[v, v, o, v], l2_bbbb, l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_bbbbbb(e,b,a,i,j,k)
    energy +=  0.125000000000000 * einsum('cdle,ijba,lkcd,ebaijk', g_abab[v, v, o, v], l2_bbbb, l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,b,a,i,j,k)
    energy +=  0.125000000000000 * einsum('dcel,ijba,kldc,ebaijk', g_bbbb[v, v, v, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,b,a,i,k,l)
    energy +=  0.125000000000000 * einsum('dcej,ijba,kldc,ebaikl', g_aaaa[v, v, v, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(a,b,e,i,k,l)
    energy +=  0.125000000000000 * einsum('dcje,ijba,kldc,abeikl', g_abab[v, v, o, v], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(a,b,e,i,k,l)
    energy +=  0.125000000000000 * einsum('cdje,ijba,klcd,abeikl', g_abab[v, v, o, v], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(a,b,e,i,l,k)
    energy +=  0.125000000000000 * einsum('dcje,ijba,lkdc,abeilk', g_abab[v, v, o, v], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(a,b,e,i,l,k)
    energy +=  0.125000000000000 * einsum('cdje,ijba,lkcd,abeilk', g_abab[v, v, o, v], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(e,b,a,i,k,l)
    energy +=  0.125000000000000 * einsum('dcej,ijba,kldc,ebaikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,b,a,i,k,l)
    energy +=  0.125000000000000 * einsum('cdej,ijba,klcd,ebaikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(e,b,a,i,l,k)
    energy +=  0.125000000000000 * einsum('dcej,ijba,lkdc,ebailk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(e,b,a,i,l,k)
    energy +=  0.125000000000000 * einsum('cdej,ijba,lkcd,ebailk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,j>_bbbb*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(b,e,a,i,k,l)
    energy += -0.125000000000000 * einsum('dcej,ijba,kldc,beaikl', g_bbbb[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_aabaab(e,a,b,i,k,l)
    energy +=  0.125000000000000 * einsum('dcej,ijab,kldc,eabikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(e,a,b,i,k,l)
    energy +=  0.125000000000000 * einsum('cdej,ijab,klcd,eabikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_aabaab(e,a,b,i,l,k)
    energy +=  0.125000000000000 * einsum('dcej,ijab,lkdc,eabilk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_aabaab(e,a,b,i,l,k)
    energy +=  0.125000000000000 * einsum('cdej,ijab,lkcd,eabilk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,b,e,i,k,l)
    energy +=  0.125000000000000 * einsum('dcej,ijab,kldc,abeikl', g_bbbb[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(e,b,a,l,k,i)
    energy +=  0.125000000000000 * einsum('dcej,jiba,kldc,ebalki', g_aaaa[v, v, v, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_abbabb(b,e,a,k,i,l)
    energy +=  0.125000000000000 * einsum('dcje,jiba,kldc,beakil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_abbabb(b,e,a,k,i,l)
    energy +=  0.125000000000000 * einsum('cdje,jiba,klcd,beakil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(b,e,a,l,k,i)
    energy += -0.125000000000000 * einsum('dcje,jiba,lkdc,bealki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_abbabb(b,e,a,l,k,i)
    energy += -0.125000000000000 * einsum('cdje,jiba,lkcd,bealki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_aaaa*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(e,a,b,l,k,i)
    energy +=  0.125000000000000 * einsum('dcej,jiab,kldc,eablki', g_aaaa[v, v, v, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_abbabb(a,b,e,k,i,l)
    energy += -0.125000000000000 * einsum('dcje,jiab,kldc,abekil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_abbabb(a,b,e,k,i,l)
    energy += -0.125000000000000 * einsum('cdje,jiab,klcd,abekil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,b,e,l,k,i)
    energy +=  0.125000000000000 * einsum('dcje,jiab,lkdc,abelki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t3_abbabb(a,b,e,l,k,i)
    energy +=  0.125000000000000 * einsum('cdje,jiab,lkcd,abelki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <d,c||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(e,b,a,k,i,l)
    energy += -0.125000000000000 * einsum('dcej,ijba,kldc,ebakil', g_abab[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1250 <c,d||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(e,b,a,k,i,l)
    energy += -0.125000000000000 * einsum('cdej,ijba,klcd,ebakil', g_abab[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(e,b,a,l,k,i)
    energy +=  0.125000000000000 * einsum('dcej,ijba,lkdc,ebalki', g_abab[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <c,d||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(e,b,a,l,k,i)
    energy +=  0.125000000000000 * einsum('cdej,ijba,lkcd,ebalki', g_abab[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1250 <d,c||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,b,a,i,k,l)
    energy +=  0.125000000000000 * einsum('dcej,ijba,kldc,ebaikl', g_bbbb[v, v, v, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,c,a,i,j,k)
    energy += -0.500000000000000 * einsum('dbel,ijba,kldc,ecaijk', g_aaaa[v, v, v, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,d||e,l>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aaaaaa(e,c,a,i,j,k)
    energy += -0.500000000000000 * einsum('bdel,ijba,klcd,ecaijk', g_abab[v, v, v, o], l2_aaaa, l2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(e,a,c,i,j,k)
    energy += -0.500000000000000 * einsum('dbel,ijba,lkdc,eacijk', g_aaaa[v, v, v, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||l,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(a,c,e,i,j,k)
    energy +=  0.500000000000000 * einsum('bdle,ijba,lkcd,aceijk', g_abab[v, v, o, v], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,d||e,l>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_aabaab(e,a,c,i,j,k)
    energy += -0.500000000000000 * einsum('bdel,ijba,kldc,eacijk', g_abab[v, v, v, o], l2_aaaa, l2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,l>_aaaa*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(e,c,a,i,k,j)
    energy +=  0.500000000000000 * einsum('dbel,ijba,kldc,ecaikj', g_aaaa[v, v, v, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,c,a,i,k,j)
    energy +=  0.500000000000000 * einsum('bdel,ijba,klcd,ecaikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,l>_aaaa*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(e,c,a,i,j,k)
    energy +=  0.500000000000000 * einsum('dbel,ijba,lkdc,ecaijk', g_aaaa[v, v, v, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,e,a,i,j,k)
    energy +=  0.500000000000000 * einsum('bdle,ijba,lkcd,ceaijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||e,l>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(e,c,a,i,j,k)
    energy +=  0.500000000000000 * einsum('bdel,ijba,kldc,ecaijk', g_abab[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||l,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(a,c,e,i,k,j)
    energy += -0.500000000000000 * einsum('dble,ijab,kldc,aceikj', g_abab[v, v, o, v], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_aabaab(e,a,c,i,k,j)
    energy +=  0.500000000000000 * einsum('dbel,ijab,kldc,eacikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(a,c,e,i,k,j)
    energy += -0.500000000000000 * einsum('dbel,ijab,klcd,aceikj', g_bbbb[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,c,e,i,j,k)
    energy += -0.500000000000000 * einsum('dble,ijab,lkdc,aceijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,l>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,e,i,j,k)
    energy += -0.500000000000000 * einsum('dbel,ijab,kldc,aceijk', g_bbbb[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(e,c,a,k,j,i)
    energy += -0.500000000000000 * einsum('dbel,jiba,kldc,ecakji', g_aaaa[v, v, v, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,d||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,c,a,k,j,i)
    energy += -0.500000000000000 * einsum('bdel,jiba,klcd,ecakji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,l>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(e,c,a,j,i,k)
    energy +=  0.500000000000000 * einsum('dbel,jiba,lkdc,ecajik', g_aaaa[v, v, v, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,e,a,j,i,k)
    energy +=  0.500000000000000 * einsum('bdle,jiba,lkcd,ceajik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||e,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(e,c,a,j,i,k)
    energy +=  0.500000000000000 * einsum('bdel,jiba,kldc,ecajik', g_abab[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||l,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(a,c,e,k,j,i)
    energy +=  0.500000000000000 * einsum('dble,jiab,kldc,acekji', g_abab[v, v, o, v], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_aabaab(e,a,c,k,j,i)
    energy += -0.500000000000000 * einsum('dbel,jiab,kldc,eackji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,l>_bbbb*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_aabaab(a,c,e,k,j,i)
    energy +=  0.500000000000000 * einsum('dbel,jiab,klcd,acekji', g_bbbb[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,c,e,j,i,k)
    energy += -0.500000000000000 * einsum('dble,jiab,lkdc,acejik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,l>_bbbb*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,e,j,i,k)
    energy += -0.500000000000000 * einsum('dbel,jiab,kldc,acejik', g_bbbb[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_abbabb(c,e,a,k,j,i)
    energy +=  0.500000000000000 * einsum('dble,ijba,kldc,ceakji', g_abab[v, v, o, v], l2_bbbb, l2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,l>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(e,c,a,k,j,i)
    energy +=  0.500000000000000 * einsum('dbel,ijba,kldc,ecakji', g_abab[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,e,a,k,j,i)
    energy +=  0.500000000000000 * einsum('dbel,ijba,klcd,ceakji', g_bbbb[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||l,e>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_bbbbbb(e,c,a,i,j,k)
    energy += -0.500000000000000 * einsum('dble,ijba,lkdc,ecaijk', g_abab[v, v, o, v], l2_bbbb, l2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,c,a,i,j,k)
    energy += -0.500000000000000 * einsum('dbel,ijba,kldc,ecaijk', g_bbbb[v, v, v, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,c,a,i,k,l)
    energy += -0.500000000000000 * einsum('dbej,ijba,kldc,ecaikl', g_aaaa[v, v, v, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(e,a,c,i,k,l)
    energy +=  0.500000000000000 * einsum('dbej,ijba,kldc,eacikl', g_aaaa[v, v, v, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(a,c,e,i,k,l)
    energy += -0.500000000000000 * einsum('bdje,ijba,klcd,aceikl', g_abab[v, v, o, v], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(e,a,c,i,l,k)
    energy +=  0.500000000000000 * einsum('dbej,ijba,lkdc,eacilk', g_aaaa[v, v, v, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(a,c,e,i,l,k)
    energy += -0.500000000000000 * einsum('bdje,ijba,lkcd,aceilk', g_abab[v, v, o, v], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||j,e>_abab*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,e,i,k,l)
    energy +=  0.500000000000000 * einsum('bdje,ijba,kldc,aceikl', g_abab[v, v, o, v], l2_aaaa, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,d||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,c,a,i,k,l)
    energy += -0.500000000000000 * einsum('bdej,ijba,klcd,ecaikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,d||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(e,c,a,i,l,k)
    energy += -0.500000000000000 * einsum('bdej,ijba,lkcd,ecailk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||e,j>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(e,c,a,i,k,l)
    energy +=  0.500000000000000 * einsum('bdej,ijba,kldc,ecaikl', g_abab[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,c,a,i,k,l)
    energy +=  0.500000000000000 * einsum('dbej,ijab,kldc,ecaikl', g_abab[v, v, v, o], l2_abab, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_aabaab(e,a,c,i,k,l)
    energy += -0.500000000000000 * einsum('dbej,ijab,kldc,eacikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(a,c,e,i,k,l)
    energy +=  0.500000000000000 * einsum('dbej,ijab,klcd,aceikl', g_bbbb[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_aabaab(e,a,c,i,l,k)
    energy += -0.500000000000000 * einsum('dbej,ijab,lkdc,eacilk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_aabaab(a,c,e,i,l,k)
    energy +=  0.500000000000000 * einsum('dbej,ijab,lkcd,aceilk', g_bbbb[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_bbbb*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(a,c,e,i,k,l)
    energy += -0.500000000000000 * einsum('dbej,ijab,kldc,aceikl', g_bbbb[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(e,c,a,l,k,i)
    energy += -0.500000000000000 * einsum('dbej,jiba,kldc,ecalki', g_aaaa[v, v, v, o], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_abbabb(e,c,a,k,i,l)
    energy += -0.500000000000000 * einsum('dbej,jiba,kldc,ecakil', g_aaaa[v, v, v, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <b,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,e,a,k,i,l)
    energy += -0.500000000000000 * einsum('bdje,jiba,klcd,ceakil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_aaaa*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(e,c,a,l,k,i)
    energy +=  0.500000000000000 * einsum('dbej,jiba,lkdc,ecalki', g_aaaa[v, v, v, o], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,e,a,l,k,i)
    energy +=  0.500000000000000 * einsum('bdje,jiba,lkcd,cealki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <b,d||j,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,c,a,i,k,l)
    energy +=  0.500000000000000 * einsum('bdje,jiba,kldc,ecaikl', g_abab[v, v, o, v], l2_abab, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||j,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(a,c,e,l,k,i)
    energy +=  0.500000000000000 * einsum('dbje,jiab,kldc,acelki', g_abab[v, v, o, v], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_abbabb(a,c,e,k,i,l)
    energy +=  0.500000000000000 * einsum('dbje,jiab,kldc,acekil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_abbabb(a,c,e,l,k,i)
    energy += -0.500000000000000 * einsum('dbje,jiab,lkdc,acelki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(e,c,a,l,k,i)
    energy +=  0.500000000000000 * einsum('dbej,ijba,kldc,ecalki', g_abab[v, v, v, o], l2_bbbb, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(e,c,a,k,i,l)
    energy +=  0.500000000000000 * einsum('dbej,ijba,kldc,ecakil', g_abab[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 <d,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,e,a,k,i,l)
    energy +=  0.500000000000000 * einsum('dbej,ijba,klcd,ceakil', g_bbbb[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_abab*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(e,c,a,l,k,i)
    energy += -0.500000000000000 * einsum('dbej,ijba,lkdc,ecalki', g_abab[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,e,a,l,k,i)
    energy += -0.500000000000000 * einsum('dbej,ijba,lkcd,cealki', g_bbbb[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 <d,b||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,c,a,i,k,l)
    energy += -0.500000000000000 * einsum('dbej,ijba,kldc,ecaikl', g_bbbb[v, v, v, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,d,c,i,j,k)
    energy +=  0.125000000000000 * einsum('bael,ijba,kldc,edcijk', g_aaaa[v, v, v, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(e,d,c,i,j,k)
    energy += -0.125000000000000 * einsum('bael,ijba,lkdc,edcijk', g_aaaa[v, v, v, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,l>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(e,c,d,i,j,k)
    energy += -0.125000000000000 * einsum('bael,ijba,lkcd,ecdijk', g_aaaa[v, v, v, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||l,e>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,i,k,j)
    energy += -0.125000000000000 * einsum('bale,ijba,kldc,cdeikj', g_abab[v, v, o, v], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,i,k,j)
    energy += -0.125000000000000 * einsum('bael,ijba,kldc,edcikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,l>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,i,k,j)
    energy += -0.125000000000000 * einsum('bael,ijba,klcd,ecdikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,i,j,k)
    energy += -0.125000000000000 * einsum('bale,ijba,lkdc,decijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||l,e>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,i,j,k)
    energy +=  0.125000000000000 * einsum('bale,ijba,lkcd,cdeijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,l>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,i,j,k)
    energy +=  0.125000000000000 * einsum('bael,ijba,kldc,edcijk', g_abab[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||l,e>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,i,k,j)
    energy += -0.125000000000000 * einsum('able,ijab,kldc,cdeikj', g_abab[v, v, o, v], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,i,k,j)
    energy += -0.125000000000000 * einsum('abel,ijab,kldc,edcikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||e,l>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,i,k,j)
    energy += -0.125000000000000 * einsum('abel,ijab,klcd,ecdikj', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,i,j,k)
    energy += -0.125000000000000 * einsum('able,ijab,lkdc,decijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||l,e>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,i,j,k)
    energy +=  0.125000000000000 * einsum('able,ijab,lkcd,cdeijk', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,l>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,i,j,k)
    energy +=  0.125000000000000 * einsum('abel,ijab,kldc,edcijk', g_abab[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||l,e>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,k,j,i)
    energy +=  0.125000000000000 * einsum('bale,jiba,kldc,cdekji', g_abab[v, v, o, v], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,k,j,i)
    energy +=  0.125000000000000 * einsum('bael,jiba,kldc,edckji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,l>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,k,j,i)
    energy +=  0.125000000000000 * einsum('bael,jiba,klcd,ecdkji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,j,i,k)
    energy += -0.125000000000000 * einsum('bale,jiba,lkdc,decjik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||l,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,j,i,k)
    energy +=  0.125000000000000 * einsum('bale,jiba,lkcd,cdejik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,l>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,j,i,k)
    energy +=  0.125000000000000 * einsum('bael,jiba,kldc,edcjik', g_abab[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||l,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,k,j,i)
    energy +=  0.125000000000000 * einsum('able,jiab,kldc,cdekji', g_abab[v, v, o, v], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,k,j,i)
    energy +=  0.125000000000000 * einsum('abel,jiab,kldc,edckji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,l>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,k,j,i)
    energy +=  0.125000000000000 * einsum('abel,jiab,klcd,ecdkji', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,j,i,k)
    energy += -0.125000000000000 * einsum('able,jiab,lkdc,decjik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||l,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,j,i,k)
    energy +=  0.125000000000000 * einsum('able,jiab,lkcd,cdejik', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,l>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,j,i,k)
    energy +=  0.125000000000000 * einsum('abel,jiab,kldc,edcjik', g_abab[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,e,c,k,j,i)
    energy +=  0.125000000000000 * einsum('bael,ijba,kldc,deckji', g_bbbb[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,e,k,j,i)
    energy += -0.125000000000000 * einsum('bael,ijba,klcd,cdekji', g_bbbb[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,l>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,d,c,i,j,k)
    energy +=  0.125000000000000 * einsum('bael,ijba,kldc,edcijk', g_bbbb[v, v, v, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,kldc,edcikl', g_aaaa[v, v, v, o], l2_aaaa, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,kldc,edcikl', g_aaaa[v, v, v, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,i,k,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,klcd,ecdikl', g_aaaa[v, v, v, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(e,d,c,i,l,k)
    energy +=  0.125000000000000 * einsum('baej,ijba,lkdc,edcilk', g_aaaa[v, v, v, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(e,c,d,i,l,k)
    energy +=  0.125000000000000 * einsum('baej,ijba,lkcd,ecdilk', g_aaaa[v, v, v, o], l2_aaaa, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_aaaa*l2_aaaa(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,kldc,edcikl', g_aaaa[v, v, v, o], l2_aaaa, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_abab*l2_abab(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,kldc,edcikl', g_abab[v, v, v, o], l2_abab, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,kldc,edcikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,i,k,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,klcd,ecdikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,d,c)*t3_aabaab(e,d,c,i,l,k)
    energy +=  0.125000000000000 * einsum('baej,ijba,lkdc,edcilk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_abab*l2_abab(i,j,b,a)*l2_abab(l,k,c,d)*t3_aabaab(e,c,d,i,l,k)
    energy +=  0.125000000000000 * einsum('baej,ijba,lkcd,ecdilk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_abab*l2_abab(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,kldc,edcikl', g_abab[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,j>_abab*l2_abab(i,j,a,b)*l2_aaaa(k,l,d,c)*t3_aaaaaa(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('abej,ijab,kldc,edcikl', g_abab[v, v, v, o], l2_abab, l2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,d,c)*t3_aabaab(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('abej,ijab,kldc,edcikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(k,l,c,d)*t3_aabaab(e,c,d,i,k,l)
    energy +=  0.125000000000000 * einsum('abej,ijab,klcd,ecdikl', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,d,c)*t3_aabaab(e,d,c,i,l,k)
    energy +=  0.125000000000000 * einsum('abej,ijab,lkdc,edcilk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,j>_abab*l2_abab(i,j,a,b)*l2_abab(l,k,c,d)*t3_aabaab(e,c,d,i,l,k)
    energy +=  0.125000000000000 * einsum('abej,ijab,lkcd,ecdilk', g_abab[v, v, v, o], l2_abab, l2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||e,j>_abab*l2_abab(i,j,a,b)*l2_bbbb(k,l,d,c)*t3_abbabb(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('abej,ijab,kldc,edcikl', g_abab[v, v, v, o], l2_abab, l2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||j,e>_abab*l2_abab(j,i,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,l,k,i)
    energy +=  0.125000000000000 * einsum('baje,jiba,kldc,cdelki', g_abab[v, v, o, v], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,e,c,k,i,l)
    energy +=  0.125000000000000 * einsum('baje,jiba,kldc,deckil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,e,k,i,l)
    energy += -0.125000000000000 * einsum('baje,jiba,klcd,cdekil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,l,k,i)
    energy += -0.125000000000000 * einsum('baje,jiba,lkdc,declki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||j,e>_abab*l2_abab(j,i,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,l,k,i)
    energy +=  0.125000000000000 * einsum('baje,jiba,lkcd,cdelki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||j,e>_abab*l2_abab(j,i,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('baje,jiba,kldc,edcikl', g_abab[v, v, o, v], l2_abab, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||j,e>_abab*l2_abab(j,i,a,b)*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,l,k,i)
    energy +=  0.125000000000000 * einsum('abje,jiab,kldc,cdelki', g_abab[v, v, o, v], l2_abab, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,d,c)*t3_abbabb(d,e,c,k,i,l)
    energy +=  0.125000000000000 * einsum('abje,jiab,kldc,deckil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(k,l,c,d)*t3_abbabb(c,d,e,k,i,l)
    energy += -0.125000000000000 * einsum('abje,jiab,klcd,cdekil', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <a,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,l,k,i)
    energy += -0.125000000000000 * einsum('abje,jiab,lkdc,declki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||j,e>_abab*l2_abab(j,i,a,b)*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,l,k,i)
    energy +=  0.125000000000000 * einsum('abje,jiab,lkcd,cdelki', g_abab[v, v, o, v], l2_abab, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <a,b||j,e>_abab*l2_abab(j,i,a,b)*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('abje,jiab,kldc,edcikl', g_abab[v, v, o, v], l2_abab, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_aaaa(k,l,d,c)*t3_aabaab(c,d,e,l,k,i)
    energy +=  0.125000000000000 * einsum('baej,ijba,kldc,cdelki', g_bbbb[v, v, v, o], l2_bbbb, l2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,d,c)*t3_abbabb(d,e,c,k,i,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,kldc,deckil', g_bbbb[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(k,l,c,d)*t3_abbabb(c,d,e,k,i,l)
    energy += -0.125000000000000 * einsum('baej,ijba,klcd,cdekil', g_bbbb[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1250 <b,a||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,d,c)*t3_abbabb(d,e,c,l,k,i)
    energy += -0.125000000000000 * einsum('baej,ijba,lkdc,declki', g_bbbb[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_abab(l,k,c,d)*t3_abbabb(c,d,e,l,k,i)
    energy +=  0.125000000000000 * einsum('baej,ijba,lkcd,cdelki', g_bbbb[v, v, v, o], l2_bbbb, l2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||e,j>_bbbb*l2_bbbb(i,j,b,a)*l2_bbbb(k,l,d,c)*t3_bbbbbb(e,d,c,i,k,l)
    energy +=  0.125000000000000 * einsum('baej,ijba,kldc,edcikl', g_bbbb[v, v, v, o], l2_bbbb, l2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
  

    return energy

 
