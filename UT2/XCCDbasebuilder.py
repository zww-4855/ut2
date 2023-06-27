import numpy as np
import UT2.antisym_t4resids as antisym_t4_residual

def build_XCCDbase(t2_aa,order,contractInfo={})
    """
    Builds the appropriate T2 base, which depends on contractions with T2^\dag and products of HnT2

    :param t2_aa: UltT2 module's set of T2 amps, in (v,v,o,o) ordering
    :param order: order in XCCD
    :param contractInfo: dictionary containing the UltT2CC information needed to construct the diagrams, such as 2e- integrals, and T2 amps, etc

    :return: Returns the pertinent T2 base
    """
    nocc=contractInfo["nocc"]
    nvir=contractInfo["nvir"]
    t=t2_aa.transpose(2,3,0,1)
    t2_dag=t2_aa #contractInfo["tamps"]

    g=contractInfo["ints"]
    o=contractInfo["oa"]
    v=contractInfo["va"]

    Roooovvvv = np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
    v_oo=g[o,o,o,o]
    v_vo=g[o,v,o,v]
    v_vv=g[v,v,v,v]
    v_m2=g[v,v,o,o]

    if order == 5:
        # Do base-line (Qf) style of constructing T4
        Roooovvvv += -0.062500000 * np.einsum("imab,jncd,klmn->ijklabcd",t,t,v_oo,optimize="optimal")
        Roooovvvv += -0.250000000 * np.einsum("imab,jkce,lemd->ijklabcd",t,t,v_vo,optimize="optimal")
        Roooovvvv += -0.062500000 * np.einsum("ijae,klbf,efcd->ijklabcd",t,t,v_vv,optimize="optimal")

        t4=antisym_t4_residual(Roooovvvv,nocc,nvir)
        t4=t4.transpose(4,5,6,7,0,1,2,3)
        resid_aaaa = (1.0/8.0)*einsum('klcd,abcdijkl->abij',t2_dag,t4)

    elif order == 6:
        # construct T2^\dag [Wn_2 T2^3/3!]c
        Roooovvvv = -0.031250000 * np.einsum("imab,jncd,klef,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")
        Roooovvvv += 0.250000000 * np.einsum("imab,jkce,lndf,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")
        Roooovvvv += -0.031250000 * np.einsum("mnab,ijce,kldf,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")

        t4=antisym_t4_residual(Roooovvvv,nocc,nvir)
        t4=t4.transpose(4,5,6,7,0,1,2,3)
        resid_aaaa = (1.0/8.0)*einsum('klcd,abcdijkl->abij',t2_dag,t4)

    elif order == 7:
        # construct [T2^\dag T2^\dag [W0T2^3/3!]]c



    elif order == 8:




    elif order == 9:


    else:
        print("The specified order for XCCD is not implemented! Please chhoose from orders 5-9.")
        sys.exit()

    return resid_aaaa

