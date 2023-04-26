Calculation Setup
=================

At the current time, all post-SCF method calculation details are controlled by parameters contained within the 'cc_runtype' dictionary. The following parameter - expressed as a key in dict having a corresponding value - specifications are supported:
  
* "stopping_eps" - Sets the convergence tolerance of the CC residual eqns in terms of a float. Default is set to abnormally high tolerance of 10**-10. 
* "max_iter"     - Sets the maximum number of iterations for iterating the CC residual eqns. Default is set to 75 iterations. 
* "dum_tamps"    - Boolean that, when set to True, dumps the set of T amplitudes to pickled output file. Default is False. 
* "diis_size"    - Sets the maximum dimension of space that stores the DIIS vectors for extrapolation. When applicable, default is set to 10 vectors. 
* "diis_start_cycle" - Specifies which CC iteration the DIIS algorithm should begin. When applicable, default is set to the first CC iteration. 
* "hf_energy"    - Sets the electronic energy found at the SCF level. Default is specified by virtue of the prior PySCF calculation
* "nuclear_energy" - Sets the nuclear repulsion energy, found from PySCF SCF. 
* "ccdTypeSlow"        - String used to specify running any of the available spin-orbital based codes. Note that these methods are inherently slow. A list of supported values corresponding to this key are as follows:
  #. "pCCD" : Simplified version of CCD having curiously good performance at capturing static correlation effects.
  #. "CCD"  : Standard implementation of CCD
  #. "CCD(Qf)" : Augments the base CCD theory with a perturbative energy correction, attributable to T4-like effects that are correct thru 3rd order in T4. Guarantees energy correct through fifth order in perturbation theory (PT). 
  #. "CCD(Qf*)": Augments the base CCD theory with a perturbative energy correction, attributable to T4-like effects. Includes effects from T4 correct thru both 3rd and 4th order. Guarantees energy correct through sixth order in PT.    
  #. "CCD(Qf*Hf)": Augments the base CCD theory with a perturbative energy correction, attributable to the highest order T4, and lowest order (5th) T6 amplitudes. Guarantees energy correct thru eigth order in PT. 
* "fullCCtype" - Partially complete, spin-integrated CC code. Note that these methods are inherently faster to some extent than those in "ccdTypeSlow", but method support is more limited. A list of supported values correspond to this key are as follows:
  #. "pCCD"
  #. "CCD"
  #. "CCSDT"     : Full CCSDT method that includes the N^8 triples
  #. "CCSDT(Qf)" : Augments CCSDT results by perturbative energy attributable to lowest order Wn(T2^2/2 + T3) terms in the T4 residual equations. In priciple, scales no worse than CCSDT
  #. "CCSDT(Qf*)" : Augments CCSDT results by bolstering the base (Qf) method by including the higher order T2^3 term in the T4 residual equations.
  #. "CCSDTQf-1"  : Augments CCSDT results by contracting T2^\dagger with the Wn(T2^2/2 + T3) terms recovered in the T4 residual equations, to bolster the T2 residual equations. 
  

 
