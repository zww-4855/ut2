"""
Unit and regression test for the UT2 package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo

from UT2.run_ccd import * 

def test_UT2_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "UT2" in sys.modules
    assert "pyscf" in sys.modules

def test_T2module_size_consistency():
    # run pyscf for some reason
    basis = ['ccpvdz']#,'ccpvtz','augccpvdz','augccdpvtz']
    atomString = ['Ne 0 0 0; Ne 0 0 50', 'Ne 0 0 0']
    testMethods=[{"ccdType":"CCD"}]
    value=[]
    for bas, molecule,method in itertools.product(basis, atomString,testMethods):
        mol = pyscf.M(
            atom=molecule,
            verbose=5,
            basis=bas)
    
    
    
        mf = mol.RHF()
        mf.conv_tol_grad=1E-10
        mf.run()
    
    
    
        cc_runtype=method   #{"ccdType":"CCDQf-1"}
    
        correlatedEnergy=run_ccd.ccd_main(mf,mol,orb,cc_runtype)
        value.append(correlatedEnergy)

        print('Final energy:', correlatedEnergy)

    diff=value[0]-value[1]*2
    assert diff <= 10**-10



