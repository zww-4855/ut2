import numpy as np
import UT2.tamps as tamps


def ccdq_t2eqns(F,W,T2,T4,o,v):
    roovv = 0.500000000 * np.einsum("ik,jkab->ijab",F[o,o],T2,optimize="optimal")
    roovv += -0.500000000 * np.einsum("ca,ijbc->ijab",F[v,v],T2,optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikab,jlcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klab,ijcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klab,ijkl->ijab",T2,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,klbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jckb->ijab",T2,W[o,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ijcd,cdab->ijab",T2,W[v,v,v,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("ijklabcd,cdkl->ijab",T4,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijab->ijab",W[o,o,v,v],optimize="optimal")
    nocc=nvir=None
    roovv=tamps.antisym_T2(roovv,nocc,nvir)
    return roovv


def ccdq_t4eqns(F,W,T2,T4,o,v):
    # contributions to the residual
    roooovvvv = 0.006944444 * np.einsum("im,jklmabcd->ijklabcd",F[o,o],T4,optimize="optimal")
    roooovvvv += -0.006944444 * np.einsum("ea,ijklbcde->ijklabcd",F[v,v],T4,optimize="optimal")
    roooovvvv += -0.031250000 * np.einsum("imab,jncd,klef,efmn->ijklabcd",T2,T2,T2,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.062500000 * np.einsum("imab,jncd,klmn->ijklabcd",T2,T2,W[o,o,o,o],optimize="optimal")
    roooovvvv += 0.250000000 * np.einsum("imab,jkce,lndf,efmn->ijklabcd",T2,T2,T2,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.250000000 * np.einsum("imab,jkce,lemd->ijklabcd",T2,T2,W[o,v,o,v],optimize="optimal")
    roooovvvv += -0.020833333 * np.einsum("imab,jklncdef,efmn->ijklabcd",T2,T4,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.031250000 * np.einsum("mnab,ijce,kldf,efmn->ijklabcd",T2,T2,T2,W[v,v,o,o],optimize="optimal")
    roooovvvv += 0.002604167 * np.einsum("mnab,ijklcdef,efmn->ijklabcd",T2,T4,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.062500000 * np.einsum("ijae,klbf,efcd->ijklabcd",T2,T2,W[v,v,v,v],optimize="optimal")
    roooovvvv += -0.020833333 * np.einsum("ijae,klmnbcdf,efmn->ijklabcd",T2,T4,W[v,v,o,o],optimize="optimal")
    roooovvvv += 0.027777778 * np.einsum("imae,jklnbcdf,efmn->ijklabcd",T2,T4,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.003472222 * np.einsum("mnae,ijklbcdf,efmn->ijklabcd",T2,T4,W[v,v,o,o],optimize="optimal")
    roooovvvv += 0.002604167 * np.einsum("ijef,klmnabcd,efmn->ijklabcd",T2,T4,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.003472222 * np.einsum("imef,jklnabcd,efmn->ijklabcd",T2,T4,W[v,v,o,o],optimize="optimal")
    roooovvvv += 0.005208333 * np.einsum("ijmnabcd,klmn->ijklabcd",T4,W[o,o,o,o],optimize="optimal")
    roooovvvv += -0.027777778 * np.einsum("ijkmabce,lemd->ijklabcd",T4,W[o,v,o,v],optimize="optimal")
    roooovvvv += 0.005208333 * np.einsum("ijklabef,efcd->ijklabcd",T4,W[v,v,v,v],optimize="optimal")
    nocc=nvir=None
    roooovvvv = tamps.antisym_T4(roooovvvv,nocc,nvir)
    return roooovvvv
