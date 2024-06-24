import numpy as np
import UT2.tamps as tamps

def cceqns_driver(driveCCobj,cc_info):
    o=driveCCobj.occSliceInfo["occ_aa"]
    v=driveCCobj.occSliceInfo["virt_aa"]

    nocc=driveCCobj.occInfo["nocc_aa"]
    nvirt=driveCCobj.occInfo["nvirt_aa"]

    W=driveCCobj.integralInfo["tei"]
    Fock=driveCCobj.integralInfo["oei"]
    if "slowSOcalc" in cc_info: # spin-orb eqns
        if "CCSD" in cc_info["slowSOcalc"]: #Generate CCSD resid eqns
            T1=driveCCobj.tamps["t1aa"]
            T2=driveCCobj.tamps["t2aa"]
            D1=driveCCobj.denomInfo["D1aa"]
            D2=driveCCobj.denomInfo["D2aa"]
            resid_t1=ccsd_t1resid(Fock,W,T1,T2,o,v)
            resid_t2=ccsd_t2resid(Fock,W,T1,T2,o,v,nocc,nvirt)
            resid_t1+=np.reciprocal(driveCCobj.denomInfo["D1aa"])
            resid_t2+=np.reciprocal(driveCCobj.denomInfo["D2aa"])
            driveCCobj.tamps.update({"t1aa":resid_t1*D1,"t2aa":resid_t2*D2})

        elif "D" in cc_info["slowSOcalc"]: #Generate CCD base resids
            T2=driveCCobj.tamps["t2aa"]
            D2=driveCCobj.denomInfo["D2aa"]
            resid_t2=ccsd_t2resid(Fock,W,np.zeros((nocc,nvirt)),T2,o,v,nocc,nvirt)
            resid_t2+=np.reciprocal(driveCCobj.denomInfo["D2aa"])
            driveCCobj.tamps.update({"t2aa":resid_t2*D2})

def ccsd_t1resid(F,W,T1,T2,o,v):
    # contributions to the residual
    rov = -1.000000000 * np.einsum("ij,ja->ia",F[o,o],T1,optimize="optimal")
    rov += -1.000000000 * np.einsum("bj,ja,ib->ia",F[v,o],T1,T1,optimize="optimal")
    rov += 1.000000000 * np.einsum("bj,ijab->ia",F[v,o],T2,optimize="optimal")
    rov += 1.000000000 * np.einsum("ia->ia",F[o,v],optimize="optimal")
    rov += 1.000000000 * np.einsum("ba,ib->ia",F[v,v],T1,optimize="optimal")
    rov += -1.000000000 * np.einsum("ja,ib,kc,bcjk->ia",T1,T1,T1,W[v,v,o,o],optimize="optimal")
    rov += -1.000000000 * np.einsum("ja,kb,ibjk->ia",T1,T1,W[o,v,o,o],optimize="optimal")
    rov += -0.500000000 * np.einsum("ja,ikbc,bcjk->ia",T1,T2,W[v,v,o,o],optimize="optimal")
    rov += -1.000000000 * np.einsum("ib,jc,bcja->ia",T1,T1,W[v,v,o,v],optimize="optimal")
    rov += -0.500000000 * np.einsum("ib,jkac,bcjk->ia",T1,T2,W[v,v,o,o],optimize="optimal")
    rov += 1.000000000 * np.einsum("jb,ikac,bcjk->ia",T1,T2,W[v,v,o,o],optimize="optimal")
    rov += -1.000000000 * np.einsum("jb,ibja->ia",T1,W[o,v,o,v],optimize="optimal")
    rov += -0.500000000 * np.einsum("jkab,ibjk->ia",T2,W[o,v,o,o],optimize="optimal")
    rov += -0.500000000 * np.einsum("ijbc,bcja->ia",T2,W[v,v,o,v],optimize="optimal")
    return rov


def ccsd_t2resid(F,W,T1,T2,o,v,nocc,nvir):
    # contributions to the residual
    roovv = 0.500000000 * np.einsum("ik,jkab->ijab",F[o,o],T2,optimize="optimal")
    roovv += 0.500000000 * np.einsum("ck,ka,ijbc->ijab",F[v,o],T1,T2,optimize="optimal")
    roovv += 0.500000000 * np.einsum("ck,ic,jkab->ijab",F[v,o],T1,T2,optimize="optimal")
    roovv += -0.500000000 * np.einsum("ca,ijbc->ijab",F[v,v],T2,optimize="optimal")
    roovv += 0.250000000 * np.einsum("ka,lb,ic,jd,cdkl->ijab",T1,T1,T1,T1,W[v,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ka,lb,ic,jckl->ijab",T1,T1,T1,W[o,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ka,lb,ijcd,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ka,lb,ijkl->ijab",T1,T1,W[o,o,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ka,ic,jd,cdkb->ijab",T1,T1,T1,W[v,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ka,ic,jlbd,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ka,ic,jckb->ijab",T1,T1,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ka,lc,ijbd,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ka,ilbc,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ka,ijcd,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ka,ijkb->ijab",T1,W[o,o,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ic,jd,klab,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ic,jd,cdab->ijab",T1,T1,W[v,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ic,kd,jlab,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ic,klab,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ic,jkad,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ic,jcab->ijab",T1,W[o,v,v,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("kc,ilab,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("kc,ijad,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikab,jlcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klab,ijcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klab,ijkl->ijab",T2,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,klbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jckb->ijab",T2,W[o,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ijcd,cdab->ijab",T2,W[v,v,v,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijab->ijab",W[o,o,v,v],optimize="optimal")

    roovv=tamps.antisym_T2(roovv,nocc,nvir)

    return roovv


