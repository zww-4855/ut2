import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 

''' 
Test to verify that T2 methods are numerically stable, meaning give the same result after being run several times in a row.  
'''

@pytest.mark.parametrize("Basis,Method",[('6-31G',{"ccdType":"CCD"}),
    ('6-31G',{"ccdType":"CCDQf-1"}),
    ('6-31G',{"ccdType":"CCDQf-2"}),
    ('6-31G',{"ccdType":"CCD(Qf)"}),])

def test_NumericalStability(Basis,Method):
    atomString = 'H 0 0 0; F 0 0 0.917'
    value=[]
    mol = pyscf.M(
        atom=atomString,
        verbose=5,
        basis=Basis)


    for i in range(5):
        mf = mol.UHF()
        mf.conv_tol_grad=1E-10
        mf.run()
    
    
        orb=mf.mo_coeff
        cc_runtype=Method   
    
        correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)
        if i==0:
            tmpE=correlatedEnergy
        else:
            currentE=correlatedEnergy
            diff=abs(abs(tmpE)-abs(currentE))
            assert diff <= 10**-10


#Basis='6-31G'
#Method={"ccdType":"CCD"}
#test_NumericalStability(Basis,Method)

