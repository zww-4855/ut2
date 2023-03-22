import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 
'''
   Test each T2 method's size-consistency property for dimeric Neon
REMINDER: pCCD is *NOT* size-consistent by design
'''
@pytest.mark.parametrize("Basis,Method",[('ccpvdz',{"ccdType":"CCD"}),
    ('ccpvtz',{"ccdType":"CCD"}),
    ('ccpvdz',{"ccdType":"CCDQf-1"}),
    ('ccpvtz',{"ccdType":"CCDQf-1"}),
    ('ccpvdz',{"ccdType":"CCDQf-2"}),
    ('ccpvtz',{"ccdType":"CCDQf-2"}),
    ('ccpvdz',{"ccdType":"CCD(Qf)"}),
    ('ccpvtz',{"ccdType":"CCD(Qf)"}),])


def test_Size_consistencyRHFT2(Basis,Method):
    atomString = ['Ne 0 0 0; Ne 0 0 50', 'Ne 0 0 0']
    value=[]
    for i in range(2):
        mol = pyscf.M(
            atom=atomString[i],
            verbose=5,
            basis=Basis)

        mf = mol.RHF()
        mf.conv_tol_grad=1E-10
        mf.run()

        orb=mf.mo_coeff
        cc_runtype=Method

        correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)
        value.append(correlatedEnergy)
        print('Final energy:', correlatedEnergy)
        if i%2 == 1:
            diff=abs(abs(value[0])-abs(value[1]*2))
            assert diff <= 10**-10
            value=[]
        print('Final energy:', correlatedEnergy)


#Basis='ccpvdz'
#Method={"ccdType":"CCD(Qf)"}
#test_Size_consistencyRHFT2(Basis,Method)
