Getting Started
===============

All UT2 routines rely upon a prior PySCF self-consistent field (SCF) calculation. Unless specifically mentioned otherwise, UT2 can be used in conjunction with RHF or UHF orbitals. After installation of PySCF, this can be accomplished via the following generic script:


>>> import pyscf
>>> atomString = 'H 0 0 0; F 0 0 0.917'
>>> mol = pyscf.M(
>>>     atom=atomString,
>>>     verbose=5,
>>>     basis='cc-pvdz')
>>> mf = mol.RHF()
>>> mf.conv_tol_grad=1E-10
>>> mf.run()

After the UT2 module has been loaded, the converged set of SCF orbitals have to be extracted from the mean-field object 'mf', then type of UT2 calculation has to be specified in dictionary format (more on this in the following subsection). These details, in addition to the 'mol' and 'mf' PySCF objects, are then passed into the run_ccd driver that handles iteration of the pertinent CC equations.

>>> from UT2.run_ccd import *
>>> orb=mf.mo_coeff
>>> cc_runtype= {"ccdType":"CCD"} 
>>> correlatedEnergy,corrCo=ccd_main(mf,mol,orb,cc_runtype)

The return values are the final, correlated energy and the correlation correction to the SCF energy, respectively.

