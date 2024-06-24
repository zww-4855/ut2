import numpy as np

def ccenergy_driver(driveCCobj,cc_info):
    o=driveCCobj.occSliceInfo["occ_aa"]
    v=driveCCobj.occSliceInfo["virt_aa"]
    
    W=driveCCobj.integralInfo["tei"]
    Fock=driveCCobj.integralInfo["oei"]
    if "slowSOcalc" in cc_info: #spin-orb energy
        t2=driveCCobj.tamps["t2aa"]
        if "t1aa" in driveCCobj.tamps:
            t1=driveCCobj.tamps["t1aa"]
        else:
            t1=None

        ccsd_energy=spinorbitalCCSDE(t2,W,Fock,o,v)
        print('ccsd e',ccsd_energy) 
        return ccsd_energy




def spinorbitalCCSDE(T2,W,F,o,v,T1=None):
    #         1.0000 f(i,i)
    energy =  1.000000000000000 * np.einsum('ii', F[o, o])
    #        -0.5000 <j,i||j,i>
    energy += -0.500000000000000 * np.einsum('jiji', W[o, o, o, o])
    energy += 0.250000000 * np.einsum("ijab,abij->",T2,W[v,v,o,o],optimize="optimal")

    if T1:
        energy += 1.000000000 * np.einsum("ai,ia->",F[v,o],T1,optimize="optimal")
        energy += 0.500000000 * np.einsum("ia,jb,abij->",T1,T1,W[v,v,o,o],optimize="optimal")


    return energy
