import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo
import itertools
from UT2.run_ccd import * 




def test_SCFroutines():
    '''
    Tests the validity of the auxillary subroutines that convert the 1/2 electron integrals by verifying the accuracy of the final SCF energy. 
    '''

    value=[]    
    for i in range(1):
        atomString = 'H 0 0 0; F 0 0 0.917'
        mol = pyscf.M(
            atom=atomString,
            verbose=5,
            basis='6-31G')
    
    
    
        if i==0:
            mf = mol.UHF()
            mf.conv_tol_grad=1E-10
            mf.run()
    
    
            orb=mf.mo_coeff

            cc_runtype={"ccdType":"CCD", "hf_energy":mf.e_tot, "nuclear_energy":mf.energy_nuc()}
            storedInfo=StoredInfo()
            storedInfo = convertSCFinfo(mf,mol,orb,cc_runtype,storedInfo)
            ints = storedInfo.integralInfo
            oei = ints["oei"]
            faa=oei["faa"]
            fbb=oei["fbb"]

            h1e = np.array((mf.get_hcore(), mf.get_hcore()))
            h1aa = orb[0].T @ h1e[0] @ orb[0]
            h1bb = orb[1].T @ h1e[1] @ orb[1] 
            occNums=storedInfo.occInfo
            na=occNums["nocc_aa"]

            nb=na
            e1 = 0.5 * np.einsum("ii", h1aa[:na, :na]) + 0.5 * np.einsum("ii", h1bb[:nb, :nb])
            e2 = 0.5 * np.einsum("ii", faa[:na, :na]) + 0.5 * np.einsum("ii", fbb[:nb, :nb])
            totSCFenergy = e1 + e2 + mf.energy_nuc()


            diff=abs(abs(totSCFenergy)-abs(mf.e_tot))
            assert diff <= 10**-10
        else:
            cc_runtype={"ccdTypeSlow":"CCD", "hf_energy":mf.e_tot, "nuclear_energy":mf.energy_nuc()}

 




