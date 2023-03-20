import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 

''' Test that ensures all higher order T2 methods beyond CCD reduces to CCD whenever    terms associated with T4, T6, etc are 0 by construction. In this case, the test     involves H2, where we know effects associated with T4, etc are zero because 
    there are only 2 electrons. CCD reference energy taken from ACES2 result. '''

@pytest.mark.parametrize("Basis,Method",[('ccpvdz',{"ccdType":"CCD"}), 
    ('ccpvtz',{"ccdType":"CCD"}),
    ('augccpvtz',{"ccdType":"CCD"}),
    ('ccpvdz',{"ccdType":"CCDQf-1"}),
    ('ccpvtz',{"ccdType":"CCDQf-1"}),
    ('augccpvtz',{"ccdType":"CCDQf-1"}),
    ('ccpvdz',{"ccdType":"CCDQf-2"}),
    ('ccpvtz',{"ccdType":"CCDQf-2"}),
    ('augccpvtz',{"ccdType":"CCDQf-2"}),
    ('ccpvdz',{"ccdType":"CCD(Qf)"}),
    ('ccpvtz',{"ccdType":"CCD(Qf)"}),
    ('augccpvtz',{"ccdType":"CCD(Qf)"}),])

def test_H2energy(Basis,Method):
    if Basis=="ccpvdz":
        aces2CCD=-1.142679147402
    elif Basis=="ccpvtz":
        aces2CCD=-1.148497898757
    elif Basis=="augccpvtz":
        aces2CCD=-1.149022864393
    atomString = 'H 0 0 0; H 0 0 1.851932'
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
    diff=abs(abs(aces2CCD)-abs(correlatedEnergy))
    assert diff <= 10**-10
    print('Final energy:', correlatedEnergy)

#Basis='ccpvdz'
#Method={"ccdType":"CCDQf-1"}
#test_H2energy(Basis,Method)
