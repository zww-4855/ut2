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
    Runs the pyscf CCD, and adds a (T) correction on top of the converged T2 amplitudes ====> resulting in CCD[T]. Checks the resulting correction against those generated inside UT2 using pdaggerq and wicked diagrams. 
    '''
    atomString='C 0.0 0.0 0.0; C 0.0 0.0 1.2'

    value=[]
    basis='STO-6G'
    mol = pyscf.M(
        atom=atomString,
        verbose=5,
        basis=basis)
    mf = mol.RHF(mol)
    
    mf.conv_tol_grad=1E-7
    mf.run()
    
    orb=mf.mo_coeff

    # Initialize the T2 amplitude file by running UT2
    cc_runtype={"ccdTypeSlow":"CCD","stopping_eps":10**-9,"diis_size":3,"diis_start_cycle":4,"dump_tamps":True}
    correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)
    print('final CCD total energy is:', correlatedEnergy)
  
    # Start running [T] correction, by first checking pdaggerq module
    
    infiles={"T2infile":'tamps.pickle', "T1infile":None}
    cc_runtype={"pertOrderSoftware":None,"pertCorr":infiles,"pertCorrOrders":"pdagq_parT"}
    correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)

   # Now check the wicked-generated [T] correction
    infiles={"T2infile":'tamps.pickle', "T1infile":None}
    cc_runtype={"pertOrderSoftware":None,"pertCorr":infiles,"pertCorrOrders":"pdagq_parT"}
    correlatedEnergy,wicked_squareBrackT3=ccd_main(mf,mol,orb,cc_runtype)

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

    print(et,wicked_squareBrackT3)
    # Verify all [T]-based energy corrections agree
    assert abs(abs(et)-abs(corrCo)) <10E-7
    assert abs(abs(et)-abs(wicked_squareBrackT3)) < 10E-7

test_squareBrackT('Basis','Method',{"fullCCType":"CCD"})

