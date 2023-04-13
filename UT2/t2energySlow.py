import numpy as np
from numpy import einsum

import UT2.modify_T2energy_pertQfSlow as pertQf

def ccd_energyMain(ccd_kernel,get_perturbCorr=False):
    sliceInfo=ccd_kernel.sliceInfo
    oa=sliceInfo["occ_aa"]
    va=sliceInfo["virt_aa"]

    t2_aaaa=ccd_kernel.tamps["t2aa"]

    fock=ccd_kernel.ints["oei"]
    tei=ccd_kernel.ints["tei"]



    if get_perturbCorr==True:
        l2dic=ccd_kernel.get_l2amps()
        qf_corr=pertQf.energy_pertQf(tei,l2dic["l2aa"],t2_aaaa,oa,va)
        return qf_corr
    else:    
        return ccdEnery(t2_aaaa,fock,tei,oa,va) 

def ccdEnery(t2,f,g,o,v):
  
    #         1.0000 f(i,i)
    energy =  1.000000000000000 * einsum('ii', f[o, o])
    #        -0.5000 <j,i||j,i>
    energy += -0.500000000000000 * einsum('jiji', g[o, o, o, o])
    #         0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy +=  0.250000000000000 * einsum('jiab,abji', g[o, o, v, v], t2)
    
    return energy

