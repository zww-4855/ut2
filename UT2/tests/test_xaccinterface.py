import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 



@pytest.mark.parametrize("Basis,Method1,Method2",[('6-31G',{"fullCCType":"CCD"},{"ccdTypeSlow":"CCD"}),])
def test_xaccinputfile(Basis,Method1,Method2):
    '''
    Placeholder test that reads relevant 2e integral files, etc, that are written 
    to file from Xacc, then performs T3-like energy corrections based on PT.
    '''
    atomString='C 0.0 0.0 0.0; C 0.0 0.0 1.2'#'H 0.0 0.0 0.0; H 1.0 0.0 0.0; H 2.0 0.0 0.0; H 3.0 0.0 0.0' #'C 0.0 0.0 0.0; C 0.0 0.0 1.2' #f'C 0.0 0.0 0.0; C {ref_bondDist+x} 0.0 0.0'


    value=[]
    basis='STO-6G'
    mol = pyscf.M(
        atom=atomString,
        verbose=5,
    #    symmetry =True,
        basis=basis)
    mf = mol.RHF(mol)
    
    mf.conv_tol_grad=1E-7
    mf.run()
    
    orb=mf.mo_coeff

    infiles={"T2infile":None, "T1infile":None}
    xaccfiles={'fock':'xacc_infiles/water_sto6g_fock.dat','tamps':'xacc_infiles/water_sto6g_abij.out','ints':'xacc_infiles/water_sto6g_h2.dat'}
    cc_runtype={"pertOrderSoftware":'xacc',"xaccfiles":xaccfiles,"pertCorr":infiles,"pertCorrOrders":"wicked_parT"}
    correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)


#test_ccdTypesCheck('Basis','Method',{"fullCCType":"CCD"})

