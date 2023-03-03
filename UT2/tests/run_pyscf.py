import pyscf
from pyscf import ao2mo


def run_RHFpyscf(bas,molecule,method):
    mol = pyscf.M(
        atom=molecule,
        verbose=5,
        basis=bas)



    mf = mol.RHF()
    mf.conv_tol_grad=1E-10
    mf.run()


    orb=mf.mo_coeff
    cc_runtype=method   #{"ccdType":"CCDQf-1"}

    correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)
    value.append(correlatedEnergy)
    print('Final energy:', correlatedEnergy)
    if count%2 == 1:
        diff=abs(abs(value[0])-abs(value[1]*2))
        assert diff <= 10**-10
        value=[]
    count+=1
    print('Final energy:', correlatedEnergy)

    return correlatedEnergy
