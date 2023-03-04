import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 

def test_H2energy():
    # run pyscf for some reason
    basis = ['ccpvdz']#,'6-31G','ccpvtz']#,'augccpvdz','augccdpvtz']
    atomString = ['H 0 0 0; H 0 0 0.98']
    testMethods=[{"ccdType":"CCD"},{"ccdType":"CCDQf-1"},{"ccdType":"CCDQf-2"},{"ccdType":"CCD(Qf)"}]
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
        if count == 0:
            value.append(correlatedEnergy)
        else:
            diff=abs(abs(value[0])-abs(correlatedEnergy))
            assert diff <= 10**-10
        count+=1
        print('Final energy:', correlatedEnergy)

test_H2energy()
