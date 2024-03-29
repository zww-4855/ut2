import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 


@pytest.mark.parametrize("Basis,Method,Answer",[('cc-pvdz',{"fullCCType":"CCSDT(Qf)"}, -100.230561268464),
                        ('cc-pvdz',{"fullCCType":"CCSDTQf-1"},-100.230558317210),
                        ('cc-pvdz',{"fullCCType":"CCSDT"},-100.230179771926)])

def test_fullCCTypes(Basis,Method,Answer):
    '''
    Test to verify CCSDT and CCSDT(Qf) give ACES2 quality results. Also verifies CCSDTQf-1 is within .002 mHa of CCSDT(Qf), as per Monika and Stan's paper. 
    '''
    atomString = 'H 0 0 0; F 0 0 1.73287873154'
    value=[]
    mol = pyscf.M(
        atom=atomString,
        verbose=5,
        unit='B',
        basis=Basis)


    mf = mol.RHF()
    mf.conv_tol_grad=1E-10
    mf.run()
    
    
    orb=mf.mo_coeff
    cc_runtype=Method   
    
    correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)

    diff=abs(abs(Answer)-abs(correlatedEnergy))
    assert diff <= 10**-8

#Basis='cc-pvdz'
#Method={"fullCCType":"CCSDT(Qf*)"}
#test_fullCCTypes(Basis,Method,-100.230561268464)

