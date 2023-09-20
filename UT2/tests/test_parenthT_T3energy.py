import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 



@pytest.mark.parametrize("Basis,Method1,Method2",[('6-31G',{"fullCCType":"CCD"},{"ccdTypeSlow":"CCD"}),])
def test_squareBrackT(Basis,Method1,Method2):
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

    cc_runtype={"ccdTypeSlow":"CCD","stopping_eps":10**-9,"diis_size":3,"diis_start_cycle":4,"dump_tamps":True}
    correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)
    print('final CCD total energy is:', correlatedEnergy)
    
    
    infiles={"T2infile":'tamps.pickle', "T1infile":None}
    cc_runtype={"pertOrderSoftware":None,"pertCorr":infiles,"pertCorrOrders":"pdagq_parT"}
    correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)
   
   # Now run pyscf-based CCD to extract (T) correction
    mycc = cc.CCSD(mf)
    old_update_amps = mycc.update_amps
    def update_amps(t1, t2, eris):
        t1, t2 = old_update_amps(t1, t2, eris)
        return t1*0, t2
    mycc.update_amps = update_amps
    mycc.kernel()
    
    
    
    et=mycc.ccsd_t()
    print('triples correction:',et)
   
    assert abs(abs(et)-abs(corrCo)) <10E-7

test_squareBrackT('Basis','Method',{"fullCCType":"CCD"})

