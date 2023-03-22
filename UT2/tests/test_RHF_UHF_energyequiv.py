import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 

''' 
Test to verify that RHF T2 methods give the same energy as UHF T2 methods before the bifurcation point of F2 
'''

@pytest.mark.parametrize("Basis,Method",[('ccpvdz',{"ccdType":"CCD"}),
    ('ccpvtz',{"ccdType":"CCD"}),
    ('ccpvdz',{"ccdType":"CCDQf-1"}),
    ('ccpvtz',{"ccdType":"CCDQf-1"}),
    ('ccpvdz',{"ccdType":"CCDQf-2"}),
    ('ccpvtz',{"ccdType":"CCDQf-2"}),
    ('ccpvdz',{"ccdType":"CCD(Qf)"}),
    ('ccpvtz',{"ccdType":"CCD(Qf)"}),])
#    ('ccpvdz',{"ccdType":"pCCD"}),
#    ('ccpvtz',{"ccdType":"pCCD"}),])

def test_RHFequalUHF(Basis,Method):
    atomString = 'F 0 0 0; F 0 0 1.1'
    value=[]
    mol = pyscf.M(
        atom=atomString,
        verbose=5,
        basis=Basis)


    for i in range(2):
        if i==0:
            mf = mol.RHF()
        else:
            mf = mol.UHF()

        mf.conv_tol_grad=1E-10
        mf.run()
    
    
        orb=mf.mo_coeff
        cc_runtype=Method   
    
        correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)
        value.append(correlatedEnergy)
        print('Final energy:', correlatedEnergy)
        if i%2 == 1:
            diff=abs(abs(value[0])-abs(value[1]))
            assert diff <= 10**-10
            print('Final energy:', correlatedEnergy)


#Basis='ccpvdz'
#Method={"ccdType":"pCCD"}
#test_RHFequalUHF(Basis,Method)

