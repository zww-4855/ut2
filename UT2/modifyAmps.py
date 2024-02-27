import numpy as np
import UT2.XCCDbasebuilder as XCCDbasebuilder
import UT2.kernel as kernel

class BuildBaseAmps(kernel.UltT2CC):
    """
    Constructs the base amplitudes for a given method (ie XCCD(n), UCCD(n), or perturbatively corrected CC theory

    :param UltT2CC.t_base: A dictionary containing the base amplitudes
    :return: Sets the  t_base parameters in parent UltT2CC class
    """
    def __init__(self):
        self.t_base={}
        self.contractInfo=super().contractInfo
        self.t_amps=None

    def buildXCCDbase(self, order=5):
        """
    Constructs the base T2 amplitudes for XCCD(5-9) and sets them to the UltT2CC class parameter for later use

    :param order: XCCD order
        """
        t2 = super().tamps["t2aa"]
        self.t_base = XCCDbasebuilder.build_XCCDbase(t2,order,self.contractInfo)
        super().t_base({order:t_base})
        return self.t_base


    def buildUCCDbase(self,order=5):
        pass




class ContractAdjointAmps(object):
    def __init__(self):
        self.result={}
        self.contractInfo=super().contractInfo
        self.sliceInfo=super().sliceInfo

    def buildXCCD_T2residEqns(self,order=5):
        """
    Returns the XCCD(order) modification to the T2 residual equations

    :param order: order of XCCD
 
    :return: Modification to the T2 residual eqns are the given order
        """
        t2_resid = super().t_base[order]
        return t2_resid

    def buildXCCD_T2energy(self,order=5, factorization=False):
        """
    Returns XCCD-like corrections to the energy. Note, that this can be in the style of CCSDT(Qf) where we cap with T2^\dag and one W-2, or it can be in style of XCCD where we simply cap with all T2^\dag

    :param order: order of XCCD
    :factorization: Boolean variable that determines if we cap with a final W-2 (True) or a T2^\dag (False). Default is False.

    :return: XCCD-like energy correction at some order
        """
        t2=super().tamps["t2aa"]
        if factorization == False: # Do XCCD-like correction
            t2_dag=t2.transpose(2,3,0,1)
        else:
            g=super().ints["tei"]
            D2=super().denom["D2aa"]
            oa=self.sliceInfo["occ_aa"]
            va=self.sliceInfo["virt_aa"]
            t2_dag = g[va,va,oa,oa] / D2

        self.t2_order=XCCDbasebuilder.build_XCCDbase(t2,order,self.contractInfo)
        t2energy_mod = (1.0/4.0)*einsum("ijab,abij",t2_dag,t2_order)

        self.result.update({order:t2energy_mod})
        return t2energy_mod
        




