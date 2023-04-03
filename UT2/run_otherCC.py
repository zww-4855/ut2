import numpy as np
import pyscf
from pyscf import ao2mo
from pyscf import cc
import UT2.t2energy as t2energy
import UT2.t2residEqns as t2residEqns
import sys

import UT2.modify_T2resid_T4Qf1 as qf1
import UT2.modify_T2resid_T4Qf2 as qf2
import UT2.modify_T2energy_pertQf as pertQf
from numpy import linalg
import UT2.run_ccd as hlp
''' Function that drives other coupled-cluster programs outside of the T2 context, such as CCSDT, CCSDTQ, and CCSDTQf'''
def otherCC_main(mf, mol, orb, cc_runtype):
    (
        na,
        nb,
        nvirta,
        nvirtb,
        occ_aa,
        occ_bb,
        virt_aa,
        virt_bb,
        faa,
        fbb,
        gaaaa,
        gbbbb,
        gabab,
        eabij_aa,
        eabij_bb,
        eabij_ab,
        otherCCDenomInfo
    ) = hlp.convertSCFinfo(mf, mol, orb)

def otherCC_kernel(
    na,
    nb,
    nvirta,
    nvirtb,
    occaa,
    virtaa,
    occbb,
    virtbb,
    faa,
    fbb,
    gaaaa,
    gbbbb,
    gabab,
    eabij_aa,
    eabij_bb,
    eabij_ab,
    hf_energy,
    nucE,
    diis_size=None,
    diis_start_cycle=4,
    cc_runtype=None,otherCCDenomInfo):

