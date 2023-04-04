import numpy as np
from numpy import einsum


def ccd_energyMain(ccd_kernel):
    sliceInfo=ccd_kernel.sliceInfo
    oa=sliceInfo["occ_aa"]
    ob=sliceInfo["occ_bb"]
    va=sliceInfo["virt_aa"]
    vb=sliceInfo["virt_bb"]

    t2_aaaa=ccd_kernel.tamps["t2aa"]
    t2_bbbb=ccd_kernel.tamps["t2bb"]
    t2_abab=ccd_kernel.tamps["t2ab"]

    fock=ccd_kernel.ints["oei"]
    tei=ccd_kernel.ints["tei"]

    f_aa=fock["faa"]
    f_bb=fock["fbb"]


    g_aaaa=tei["g_aaaa"]
    g_bbbb=tei["g_bbbb"]
    g_abab=tei["g_abab"]

    return ccd_energy_with_spin(t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)

def ccd_energy_with_spin(t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | e(-T) H e(T) | 0> :

    o = oa
    v = va
    energy =  1.000000000000000 * einsum('ii', f_aa[o, o])

    #     1.0000 f_bb(i,i)
    energy +=  1.000000000000000 * einsum('ii', f_bb[o, o])

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

    return energy


