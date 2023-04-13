import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 

''' 
Test to verify that T2 methods yield consistent energies. Intends to verify that methods give a consistent set of energies, irrespective of any changes or modifications to the existing code. Standardizes the resulting set of energies for the HF molecule in the 6-31G basis set. 
'''


@pytest.mark.parametrize("Basis,Method,Answer",[('6-31G',{"ccdType":"CCD"},-100.114058950498),
    ('6-31G',{"ccdType":"CCDQf-1"},-100.114512316484),
    ('6-31G',{"ccdType":"CCDQf-2"},-100.114494929544),
    ('6-31G',{"ccdType":"CCD(Qf)"},-100.114950279379),])

def test_ConsistentEnergy(Basis,Method,Answer):

    atomString = 'H 0 0 0; F 0 0 0.917'
    mol = pyscf.M(
        atom=atomString,
        verbose=5,
        basis=Basis)



    #mf = mol.UHF()
    mf = mol.RHF()
    mf.conv_tol_grad=1E-10
    mf.run()
    
    
    orb=mf.mo_coeff
    cc_runtype=Method   
    correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)

    diff=abs(abs(Answer)-abs(correlatedEnergy))
    print('diff:',diff)
    assert diff <= 10**-10


#Basis='6-31G'
#Method={"ccdTypeSlow":"CCDQf-2"}
#Answer=-100.114494929544 #-100.114058950498
#test_ConsistentEnergy(Basis,Method,Answer)

