import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 

def test_Size_consistencyRHFT2():
    # run pyscf for some reason
    basis = ['ccpvdz']#,'6-31G','ccpvtz']#,'augccpvdz','augccdpvtz']
    atomString = ['Ne 0 0 0; Ne 0 0 50', 'Ne 0 0 0']
    testMethods=[{"ccdType":"CCD"},{"ccdType":"CCDQf-1"},{"ccdType":"CCDQf-2"},{"ccdType":"CCD(Qf)"},{"ccdType":"pCCD"}]
    value=[]
    count=0
    for bas, molecule,method in itertools.product(basis, atomString,testMethods):
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

#    diff=value[0]-value[1]*2
#    assert diff <= 10**-10

def main():
    basis = ['ccpvdz']#,'ccpvtz','augccpvdz','augccdpvtz']
    atomString = ['Ne 0 0 0; Ne 0 0 50', 'Ne 0 0 0']
    testMethods=[{"ccdType":"CCD"}]

test_Size_consistencyRHFT2()
