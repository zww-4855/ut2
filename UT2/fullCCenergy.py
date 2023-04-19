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

    o=oa
    v=va

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

        t1_aa=t1_bb=t4_aaaaaaaa=t4_bbbbbbbb=t4_aaabaaab=t4_aabbaabb=t4_abbbabbb=None

        t4_aaaa=ccsdtq_t4_aaaaaaaa_residual(t1_aa, t1_bb,
                                t2_aaaa, t2_bbbb, t2_abab,
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb,
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb,
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        t4_bbbb=ccsdtq_t4_bbbbbbbb_residual(t1_aa, t1_bb,
                                t2_aaaa, t2_bbbb, t2_abab,
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb,
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb,
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        t4_aaab=ccsdtq_t4_aaabaaab_residual(t1_aa, t1_bb,
                                t2_aaaa, t2_bbbb, t2_abab,
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb,
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb,
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        t4_aabb=ccsdtq_t4_aabbaabb_residual(t1_aa, t1_bb,
                                t2_aaaa, t2_bbbb, t2_abab,
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb,
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb,
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        t4_abbb=ccsdtq_t4_abbbabbb_residual(t1_aa, t1_bb,
                                t2_aaaa, t2_bbbb, t2_abab,
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb,
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb,
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)



        t2_dag_aa=g_aaaa[v,v,o,o]*ccd_kernel.denom["D2aa"]
        t2_dagger_aa=t2_dag_aa.transpose(2,3,0,1)

        t2_dag_ab=g_abab[v,v,o,o]*ccd_kernel.denom["D2ab"]
        t2_dagger_ab=t2_dag_ab.transpose(2,3,0,1)
 
        t2_dag_bb=g_bbbb[v,v,o,o]*ccd_kernel.denom["D2bb"]
        t2_dagger_bb=t2_dag_bb.transpose(2,3,0,1)


        qf_corr=energy_pertQf(t2_dagger_aa,t2_dagger_ab,t2_dagger_bb,t4_aaaa,t4_aaab,t4_aabb,t4_abbb,t4_bbbb,l2_aaaa,l2_bbbb,l2_abab,o,v)

        return qf_corr*0.50000000000000000
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



def energy_pertQf(g_aaaa,g_abab,g_bbbb,t4_aaaaaaaa,t4_aaabaaab,t4_aabbaabb,t4_abbbabbb,t4_bbbbbbbb,l2_aaaa,l2_bbbb,l2_abab,o,v):

    #	  0.0625 <l,k||c,d>_aaaa*l2_aaaa(i,j,b,a)*t4_aaaaaaaa(c,d,b,a,i,j,l,k)
    energy =  0.062500000000000 * einsum('lkcd,ijba,cdbaijlk', g_aaaa, l2_aaaa, t4_aaaaaaaa[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||c,d>_abab*l2_aaaa(i,j,b,a)*t4_aaabaaab(c,a,b,d,i,j,l,k)
    energy += -0.062500000000000 * einsum('lkcd,ijba,cabdijlk', g_abab, l2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <k,l||c,d>_abab*l2_aaaa(i,j,b,a)*t4_aaabaaab(c,a,b,d,i,j,k,l)
    energy += -0.062500000000000 * einsum('klcd,ijba,cabdijkl', g_abab, l2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||d,c>_abab*l2_aaaa(i,j,b,a)*t4_aaabaaab(a,d,b,c,i,j,l,k)
    energy +=  0.062500000000000 * einsum('lkdc,ijba,adbcijlk', g_abab, l2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <k,l||d,c>_abab*l2_aaaa(i,j,b,a)*t4_aaabaaab(a,d,b,c,i,j,k,l)
    energy +=  0.062500000000000 * einsum('kldc,ijba,adbcijkl', g_abab, l2_aaaa, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||c,d>_bbbb*l2_aaaa(i,j,b,a)*t4_aabbaabb(b,a,c,d,i,j,l,k)
    energy +=  0.062500000000000 * einsum('lkcd,ijba,bacdijlk', g_bbbb, l2_aaaa, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||c,d>_aaaa*l2_abab(i,j,b,a)*t4_aaabaaab(c,d,b,a,i,k,l,j)
    energy += -0.062500000000000 * einsum('lkcd,ijba,cdbaiklj', g_aaaa, l2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||c,d>_abab*l2_abab(i,j,b,a)*t4_aabbaabb(c,b,d,a,i,l,j,k)
    energy +=  0.062500000000000 * einsum('lkcd,ijba,cbdailjk', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <k,l||c,d>_abab*l2_abab(i,j,b,a)*t4_aabbaabb(c,b,d,a,i,k,l,j)
    energy += -0.062500000000000 * einsum('klcd,ijba,cbdaiklj', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||d,c>_abab*l2_abab(i,j,b,a)*t4_aabbaabb(b,d,c,a,i,l,j,k)
    energy += -0.062500000000000 * einsum('lkdc,ijba,bdcailjk', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <k,l||d,c>_abab*l2_abab(i,j,b,a)*t4_aabbaabb(b,d,c,a,i,k,l,j)
    energy +=  0.062500000000000 * einsum('kldc,ijba,bdcaiklj', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||c,d>_bbbb*l2_abab(i,j,b,a)*t4_abbbabbb(b,d,c,a,i,j,l,k)
    energy += -0.062500000000000 * einsum('lkcd,ijba,bdcaijlk', g_bbbb, l2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||c,d>_aaaa*l2_abab(i,j,a,b)*t4_aaabaaab(c,d,a,b,i,k,l,j)
    energy += -0.062500000000000 * einsum('lkcd,ijab,cdabiklj', g_aaaa, l2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||c,d>_abab*l2_abab(i,j,a,b)*t4_aabbaabb(c,a,b,d,i,l,j,k)
    energy += -0.062500000000000 * einsum('lkcd,ijab,cabdiljk', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <k,l||c,d>_abab*l2_abab(i,j,a,b)*t4_aabbaabb(c,a,b,d,i,k,l,j)
    energy +=  0.062500000000000 * einsum('klcd,ijab,cabdiklj', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||d,c>_abab*l2_abab(i,j,a,b)*t4_aabbaabb(a,d,b,c,i,l,j,k)
    energy +=  0.062500000000000 * einsum('lkdc,ijab,adbciljk', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <k,l||d,c>_abab*l2_abab(i,j,a,b)*t4_aabbaabb(a,d,b,c,i,k,l,j)
    energy += -0.062500000000000 * einsum('kldc,ijab,adbciklj', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||c,d>_bbbb*l2_abab(i,j,a,b)*t4_abbbabbb(a,d,b,c,i,j,l,k)
    energy +=  0.062500000000000 * einsum('lkcd,ijab,adbcijlk', g_bbbb, l2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||c,d>_aaaa*l2_abab(j,i,b,a)*t4_aaabaaab(c,d,b,a,k,j,l,i)
    energy +=  0.062500000000000 * einsum('lkcd,jiba,cdbakjli', g_aaaa, l2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||c,d>_abab*l2_abab(j,i,b,a)*t4_aabbaabb(c,b,d,a,l,j,i,k)
    energy += -0.062500000000000 * einsum('lkcd,jiba,cbdaljik', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <k,l||c,d>_abab*l2_abab(j,i,b,a)*t4_aabbaabb(c,b,d,a,k,j,l,i)
    energy +=  0.062500000000000 * einsum('klcd,jiba,cbdakjli', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||d,c>_abab*l2_abab(j,i,b,a)*t4_aabbaabb(b,d,c,a,l,j,i,k)
    energy +=  0.062500000000000 * einsum('lkdc,jiba,bdcaljik', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <k,l||d,c>_abab*l2_abab(j,i,b,a)*t4_aabbaabb(b,d,c,a,k,j,l,i)
    energy += -0.062500000000000 * einsum('kldc,jiba,bdcakjli', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||c,d>_bbbb*l2_abab(j,i,b,a)*t4_abbbabbb(b,d,c,a,j,i,l,k)
    energy += -0.062500000000000 * einsum('lkcd,jiba,bdcajilk', g_bbbb, l2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||c,d>_aaaa*l2_abab(j,i,a,b)*t4_aaabaaab(c,d,a,b,k,j,l,i)
    energy +=  0.062500000000000 * einsum('lkcd,jiab,cdabkjli', g_aaaa, l2_abab, t4_aaabaaab[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||c,d>_abab*l2_abab(j,i,a,b)*t4_aabbaabb(c,a,b,d,l,j,i,k)
    energy +=  0.062500000000000 * einsum('lkcd,jiab,cabdljik', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <k,l||c,d>_abab*l2_abab(j,i,a,b)*t4_aabbaabb(c,a,b,d,k,j,l,i)
    energy += -0.062500000000000 * einsum('klcd,jiab,cabdkjli', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||d,c>_abab*l2_abab(j,i,a,b)*t4_aabbaabb(a,d,b,c,l,j,i,k)
    energy += -0.062500000000000 * einsum('lkdc,jiab,adbcljik', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <k,l||d,c>_abab*l2_abab(j,i,a,b)*t4_aabbaabb(a,d,b,c,k,j,l,i)
    energy +=  0.062500000000000 * einsum('kldc,jiab,adbckjli', g_abab, l2_abab, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||c,d>_bbbb*l2_abab(j,i,a,b)*t4_abbbabbb(a,d,b,c,j,i,l,k)
    energy +=  0.062500000000000 * einsum('lkcd,jiab,adbcjilk', g_bbbb, l2_abab, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||c,d>_aaaa*l2_bbbb(i,j,b,a)*t4_aabbaabb(c,d,b,a,l,k,i,j)
    energy +=  0.062500000000000 * einsum('lkcd,ijba,cdbalkij', g_aaaa, l2_bbbb, t4_aabbaabb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||c,d>_abab*l2_bbbb(i,j,b,a)*t4_abbbabbb(c,d,b,a,l,j,i,k)
    energy += -0.062500000000000 * einsum('lkcd,ijba,cdbaljik', g_abab, l2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <k,l||c,d>_abab*l2_bbbb(i,j,b,a)*t4_abbbabbb(c,d,b,a,k,j,l,i)
    energy +=  0.062500000000000 * einsum('klcd,ijba,cdbakjli', g_abab, l2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.0625 <l,k||d,c>_abab*l2_bbbb(i,j,b,a)*t4_abbbabbb(d,c,b,a,l,j,i,k)
    energy += -0.062500000000000 * einsum('lkdc,ijba,dcbaljik', g_abab, l2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <k,l||d,c>_abab*l2_bbbb(i,j,b,a)*t4_abbbabbb(d,c,b,a,k,j,l,i)
    energy +=  0.062500000000000 * einsum('kldc,ijba,dcbakjli', g_abab, l2_bbbb, t4_abbbabbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.0625 <l,k||c,d>_bbbb*l2_bbbb(i,j,b,a)*t4_bbbbbbbb(c,d,b,a,i,j,l,k)
    energy +=  0.062500000000000 * einsum('lkcd,ijba,cdbaijlk', g_bbbb, l2_bbbb, t4_bbbbbbbb[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
   


    return energy 

def ccsdtq_t4_aaaaaaaa_residual(t1_aa, t1_bb, 
                                t2_aaaa, t2_bbbb, t2_abab, 
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||k,l>_aaaa*t2_aaaa(a,b,j,m)*t2_aaaa(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,abjm,cdin->abcdijkl', g_aaaa[o, o, o, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
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
    
    #	  1.0000 P(j,k)*P(a,b)<m,a||k,l>_aaaa*t3_aaaaaa(b,c,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('makl,bcdijm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||i,l>_aaaa*t3_aaaaaa(b,c,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mail,bcdjkm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,a||j,k>_aaaa*t3_aaaaaa(b,c,d,i,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('majk,bcdilm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)<m,c||k,l>_aaaa*t3_aaaaaa(a,b,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mckl,abdijm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||i,l>_aaaa*t3_aaaaaa(a,b,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcil,abdjkm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)<m,c||j,k>_aaaa*t3_aaaaaa(a,b,d,i,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcjk,abdilm->abcdijkl', g_aaaa[o, v, o, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<a,b||e,l>_aaaa*t3_aaaaaa(e,c,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('abel,ecdijk->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<a,b||e,j>_aaaa*t3_aaaaaa(e,c,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('abej,ecdikl->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<a,d||e,l>_aaaa*t3_aaaaaa(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('adel,ebcijk->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<a,d||e,j>_aaaa*t3_aaaaaa(e,b,c,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('adej,ebcikl->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,d)<b,c||e,l>_aaaa*t3_aaaaaa(e,a,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('bcel,eadijk->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,d)<b,c||e,j>_aaaa*t3_aaaaaa(e,a,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('bcej,eadikl->abcdijkl', g_aaaa[v, v, v, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    return quadruples_res


def ccsdtq_t4_aaabaaab_residual(t1_aa, t1_bb, 
                                t2_aaaa, t2_bbbb, t2_abab, 
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	  1.0000 P(i,j)*P(b,c)<m,n||k,l>_abab*t2_aaaa(a,b,j,m)*t2_abab(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('mnkl,abjm,cdin->abcdijkl', g_abab[o, o, o, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
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
    
    #	 -1.0000 P(j,k)*P(a,b)<a,m||k,l>_abab*t3_aabaab(b,c,d,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('amkl,bcdijm->abcdijkl', g_abab[v, o, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<a,m||i,l>_abab*t3_aabaab(b,c,d,j,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('amil,bcdjkm->abcdijkl', g_abab[v, o, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,a||i,k>_aaaa*t3_aabaab(b,c,d,j,m,l)
    contracted_intermediate =  1.000000000000000 * einsum('maik,bcdjml->abcdijkl', g_aaaa[o, v, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,a||j,k>_aaaa*t3_aabaab(b,c,d,i,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('majk,bcdiml->abcdijkl', g_aaaa[o, v, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<c,m||k,l>_abab*t3_aabaab(a,b,d,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('cmkl,abdijm->abcdijkl', g_abab[v, o, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,d||k,l>_abab*t3_aaaaaa(a,b,c,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdkl,abcijm->abcdijkl', g_abab[o, v, o, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 <c,m||i,l>_abab*t3_aabaab(a,b,d,j,k,m)
    quadruples_res += -1.000000000000000 * einsum('cmil,abdjkm->abcdijkl', g_abab[v, o, o, o], t3_aabaab)
    
    #	 -1.0000 <m,d||i,l>_abab*t3_aaaaaa(a,b,c,j,k,m)
    quadruples_res += -1.000000000000000 * einsum('mdil,abcjkm->abcdijkl', g_abab[o, v, o, o], t3_aaaaaa)
    
    #	  1.0000 <m,c||i,k>_aaaa*t3_aabaab(a,b,d,j,m,l)
    quadruples_res +=  1.000000000000000 * einsum('mcik,abdjml->abcdijkl', g_aaaa[o, v, o, o], t3_aabaab)
    
    #	 -1.0000 P(i,k)<m,c||j,k>_aaaa*t3_aabaab(a,b,d,i,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('mcjk,abdiml->abcdijkl', g_aaaa[o, v, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<a,b||e,k>_aaaa*t3_aabaab(e,c,d,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('abek,ecdijl->abcdijkl', g_aaaa[v, v, v, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<a,b||e,j>_aaaa*t3_aabaab(e,c,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('abej,ecdikl->abcdijkl', g_aaaa[v, v, v, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<a,d||e,l>_abab*t3_aaaaaa(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('adel,ebcijk->abcdijkl', g_abab[v, v, v, o], t3_aaaaaa)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<a,d||k,e>_abab*t3_aabaab(c,b,e,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('adke,cbeijl->abcdijkl', g_abab[v, v, o, v], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<a,d||j,e>_abab*t3_aabaab(c,b,e,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('adje,cbeikl->abcdijkl', g_abab[v, v, o, v], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 <c,d||e,l>_abab*t3_aaaaaa(e,a,b,i,j,k)
    quadruples_res +=  1.000000000000000 * einsum('cdel,eabijk->abcdijkl', g_abab[v, v, v, o], t3_aaaaaa)
    
    #	 -1.0000 <b,c||e,k>_aaaa*t3_aabaab(e,a,d,i,j,l)
    quadruples_res += -1.000000000000000 * einsum('bcek,eadijl->abcdijkl', g_aaaa[v, v, v, o], t3_aabaab)
    
    #	 -1.0000 <c,d||k,e>_abab*t3_aabaab(b,a,e,i,j,l)
    quadruples_res += -1.000000000000000 * einsum('cdke,baeijl->abcdijkl', g_abab[v, v, o, v], t3_aabaab)
    
    #	  1.0000 P(i,j)<b,c||e,j>_aaaa*t3_aabaab(e,a,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('bcej,eadikl->abcdijkl', g_aaaa[v, v, v, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<c,d||j,e>_abab*t3_aabaab(b,a,e,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('cdje,baeikl->abcdijkl', g_abab[v, v, o, v], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    return quadruples_res


def ccsdtq_t4_aabbaabb_residual(t1_aa, t1_bb, 
                                t2_aaaa, t2_bbbb, t2_abab, 
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	  1.0000 P(i,j)<n,m||k,l>_bbbb*t2_abab(a,c,j,m)*t2_abab(b,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmkl,acjm,bdin->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
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
    
    #	  1.0000 P(a,b)<a,m||j,l>_abab*t3_abbabb(b,c,d,i,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('amjl,bcdikm->abcdijkl', g_abab[v, o, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<a,m||i,l>_abab*t3_abbabb(b,c,d,j,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('amil,bcdjkm->abcdijkl', g_abab[v, o, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<a,m||j,k>_abab*t3_abbabb(b,c,d,i,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('amjk,bcdilm->abcdijkl', g_abab[v, o, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,a||j,i>_aaaa*t3_abbabb(b,c,d,m,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('maji,bcdmlk->abcdijkl', g_aaaa[o, v, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<m,c||k,l>_bbbb*t3_aabaab(a,b,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mckl,abdijm->abcdijkl', g_bbbb[o, v, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<m,c||j,l>_abab*t3_aabaab(a,b,d,i,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mcjl,abdimk->abcdijkl', g_abab[o, v, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||i,l>_abab*t3_aabaab(a,b,d,j,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mcil,abdjmk->abcdijkl', g_abab[o, v, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||j,k>_abab*t3_aabaab(a,b,d,i,m,l)
    contracted_intermediate = -1.000000000000000 * einsum('mcjk,abdiml->abcdijkl', g_abab[o, v, o, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<a,c||e,l>_abab*t3_aabaab(e,b,d,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('acel,ebdijk->abcdijkl', g_abab[v, v, v, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<a,b||e,j>_aaaa*t3_abbabb(e,c,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('abej,ecdikl->abcdijkl', g_aaaa[v, v, v, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<a,c||j,e>_abab*t3_abbabb(b,e,d,i,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('acje,bedikl->abcdijkl', g_abab[v, v, o, v], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<a,d||e,l>_abab*t3_aabaab(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('adel,ebcijk->abcdijkl', g_abab[v, v, v, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<a,d||j,e>_abab*t3_abbabb(b,e,c,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('adje,becikl->abcdijkl', g_abab[v, v, o, v], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<b,c||e,l>_abab*t3_aabaab(e,a,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('bcel,eadijk->abcdijkl', g_abab[v, v, v, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<d,c||e,l>_bbbb*t3_aabaab(b,a,e,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('dcel,baeijk->abcdijkl', g_bbbb[v, v, v, o], t3_aabaab)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<b,c||j,e>_abab*t3_abbabb(a,e,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('bcje,aedikl->abcdijkl', g_abab[v, v, o, v], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    return quadruples_res


def ccsdtq_t4_abbbabbb_residual(t1_aa, t1_bb, 
                                t2_aaaa, t2_bbbb, t2_abab, 
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	  1.0000 P(b,c)<n,m||k,l>_bbbb*t2_abab(a,b,i,m)*t2_bbbb(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmkl,abim,cdjn->abcdijkl', g_bbbb[o, o, o, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
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
    
    #	 -1.0000 P(j,k)<m,b||k,l>_bbbb*t3_abbabb(a,c,d,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbkl,acdijm->abcdijkl', g_bbbb[o, v, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<a,m||i,l>_abab*t3_bbbbbb(b,c,d,j,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('amil,bcdjkm->abcdijkl', g_abab[v, o, o, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<m,b||i,l>_abab*t3_abbabb(a,c,d,m,k,j)
    contracted_intermediate =  1.000000000000000 * einsum('mbil,acdmkj->abcdijkl', g_abab[o, v, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 <m,b||j,k>_bbbb*t3_abbabb(a,c,d,i,l,m)
    quadruples_res += -1.000000000000000 * einsum('mbjk,acdilm->abcdijkl', g_bbbb[o, v, o, o], t3_abbabb)
    
    #	 -1.0000 <a,m||i,j>_abab*t3_bbbbbb(b,c,d,k,l,m)
    quadruples_res += -1.000000000000000 * einsum('amij,bcdklm->abcdijkl', g_abab[v, o, o, o], t3_bbbbbb)
    
    #	  1.0000 <m,b||i,j>_abab*t3_abbabb(a,c,d,m,l,k)
    quadruples_res +=  1.000000000000000 * einsum('mbij,acdmlk->abcdijkl', g_abab[o, v, o, o], t3_abbabb)
    
    #	  1.0000 P(j,k)*P(c,d)<m,c||k,l>_bbbb*t3_abbabb(a,b,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mckl,abdijm->abcdijkl', g_bbbb[o, v, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||i,l>_abab*t3_abbabb(a,b,d,m,k,j)
    contracted_intermediate = -1.000000000000000 * einsum('mcil,abdmkj->abcdijkl', g_abab[o, v, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<m,c||j,k>_bbbb*t3_abbabb(a,b,d,i,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcjk,abdilm->abcdijkl', g_bbbb[o, v, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||i,j>_abab*t3_abbabb(a,b,d,m,l,k)
    contracted_intermediate = -1.000000000000000 * einsum('mcij,abdmlk->abcdijkl', g_abab[o, v, o, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<a,b||e,l>_abab*t3_abbabb(e,c,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('abel,ecdijk->abcdijkl', g_abab[v, v, v, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<a,b||e,j>_abab*t3_abbabb(e,c,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('abej,ecdikl->abcdijkl', g_abab[v, v, v, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<a,b||i,e>_abab*t3_bbbbbb(e,c,d,j,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('abie,ecdjkl->abcdijkl', g_abab[v, v, o, v], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<a,d||e,l>_abab*t3_abbabb(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('adel,ebcijk->abcdijkl', g_abab[v, v, v, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<b,d||e,l>_bbbb*t3_abbabb(a,e,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('bdel,aecijk->abcdijkl', g_bbbb[v, v, v, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 <a,d||e,j>_abab*t3_abbabb(e,b,c,i,k,l)
    quadruples_res +=  1.000000000000000 * einsum('adej,ebcikl->abcdijkl', g_abab[v, v, v, o], t3_abbabb)
    
    #	  1.0000 <b,d||e,j>_bbbb*t3_abbabb(a,e,c,i,k,l)
    quadruples_res +=  1.000000000000000 * einsum('bdej,aecikl->abcdijkl', g_bbbb[v, v, v, o], t3_abbabb)
    
    #	  1.0000 <a,d||i,e>_abab*t3_bbbbbb(e,b,c,j,k,l)
    quadruples_res +=  1.000000000000000 * einsum('adie,ebcjkl->abcdijkl', g_abab[v, v, o, v], t3_bbbbbb)
    
    #	 -1.0000 P(k,l)*P(b,d)<b,c||e,l>_bbbb*t3_abbabb(a,e,d,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('bcel,aedijk->abcdijkl', g_bbbb[v, v, v, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(b,d)<b,c||e,j>_bbbb*t3_abbabb(a,e,d,i,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('bcej,aedikl->abcdijkl', g_bbbb[v, v, v, o], t3_abbabb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    return quadruples_res


def ccsdtq_t4_bbbbbbbb_residual(t1_aa, t1_bb, 
                                t2_aaaa, t2_bbbb, t2_abab, 
                                t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, 
                                t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, 
                                f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||k,l>_bbbb*t2_bbbb(a,b,j,m)*t2_bbbb(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,abjm,cdin->abcdijkl', g_bbbb[o, o, o, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
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
    
    #	  1.0000 P(j,k)*P(a,b)<m,a||k,l>_bbbb*t3_bbbbbb(b,c,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('makl,bcdijm->abcdijkl', g_bbbb[o, v, o, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||i,l>_bbbb*t3_bbbbbb(b,c,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mail,bcdjkm->abcdijkl', g_bbbb[o, v, o, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,a||j,k>_bbbb*t3_bbbbbb(b,c,d,i,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('majk,bcdilm->abcdijkl', g_bbbb[o, v, o, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)<m,c||k,l>_bbbb*t3_bbbbbb(a,b,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mckl,abdijm->abcdijkl', g_bbbb[o, v, o, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||i,l>_bbbb*t3_bbbbbb(a,b,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcil,abdjkm->abcdijkl', g_bbbb[o, v, o, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)<m,c||j,k>_bbbb*t3_bbbbbb(a,b,d,i,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcjk,abdilm->abcdijkl', g_bbbb[o, v, o, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<a,b||e,l>_bbbb*t3_bbbbbb(e,c,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('abel,ecdijk->abcdijkl', g_bbbb[v, v, v, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<a,b||e,j>_bbbb*t3_bbbbbb(e,c,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('abej,ecdikl->abcdijkl', g_bbbb[v, v, v, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<a,d||e,l>_bbbb*t3_bbbbbb(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('adel,ebcijk->abcdijkl', g_bbbb[v, v, v, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<a,d||e,j>_bbbb*t3_bbbbbb(e,b,c,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('adej,ebcikl->abcdijkl', g_bbbb[v, v, v, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,d)<b,c||e,l>_bbbb*t3_bbbbbb(e,a,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('bcel,eadijk->abcdijkl', g_bbbb[v, v, v, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,d)<b,c||e,j>_bbbb*t3_bbbbbb(e,a,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('bcej,eadikl->abcdijkl', g_bbbb[v, v, v, o], t3_bbbbbb)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    return quadruples_res

