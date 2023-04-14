import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 



@pytest.mark.parametrize("Basis,Method1,Method2",[('6-31G',{"ccdType":"CCD"},{"ccdTypeSlow":"CCD"}),
                            ('6-31G', {"ccdType":"CCD(Qf)"},{"ccdTypeSlow":"CCD(Qf)"}),
                            ('6-31G', {"ccdType":"CCDQf-1"},{"ccdTypeSlow":"CCDQf-1"}),
                            ('6-31G', {"ccdType":"CCDQf-2"},{"ccdTypeSlow":"CCDQf-2"}),])

def test_ccdTypesCheck(Basis,Method1,Method2):
    '''
    Test to verify that existing spin-integrated T2 methods yield equivalent answers to their spin-orbital-based analogs.
    '''

    value=[]    
    for i in range(2):
        atomString = 'H 0 0 0; F 0 0 0.917'
        mol = pyscf.M(
            atom=atomString,
            verbose=5,
            basis=Basis)
    
    
    
        mf = mol.RHF()
        mf.conv_tol_grad=1E-10
        mf.run()
        
        
        orb=mf.mo_coeff
        if i==0:
            cc_runtype=Method1
        else:
            cc_runtype=Method2
   
        correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)
        value.append(correlatedEnergy)

    diff=abs(abs(value[0])-abs(value[1]))
    print('diff:',diff)
    assert diff <= 10**-10


#Basis='6-31G'
#Method={"ccdTypeSlow":"CCD"}
#Answer=-100.114058950498
#test_ConsistentEnergy(Basis,Method,Answer)

