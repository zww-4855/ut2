import numpy as np
import UT2.antisym_t4resids as antisym
import pickle
from numpy import einsum

def build_XCCDbase(t2_aa,order,contractInfo={}):
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
    t2_dag=t #contractInfo["tamps"]

    g=contractInfo["ints"]
    o=contractInfo["oa"]
    v=contractInfo["va"]
    print('nocc,nvirt orbs', nocc,nvir)
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

        t4=antisym.antisym_t4_residual(Roooovvvv,nocc,nvir)
        with open('roooovvvv_t4_third.pickle', 'wb') as handle:
            pickle.dump(t4, handle)
        t4T=t4.transpose(4,5,6,7,0,1,2,3)
        print(t4.shape,Roooovvvv.shape,t4T.shape,t4.transpose(4,5,6,7,0,1,2,3).shape)
        print(t2_dag.shape)
        resid_aaaa = (1.0/8.0)*einsum('klcd,abcdijkl->abij',t2_dag,t4T)

    elif order == 6:
        # construct T2^\dag [Wn_2 T2^3/3!]c
        Roooovvvv = -0.031250000 * np.einsum("imab,jncd,klef,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")
        Roooovvvv += 0.250000000 * np.einsum("imab,jkce,lndf,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")
        Roooovvvv += -0.031250000 * np.einsum("mnab,ijce,kldf,efmn->ijklabcd",t,t,t,v_m2,optimize="optimal")

        t4=antisym.antisym_t4_residual(Roooovvvv,nocc,nvir)
        t4=t4.transpose(4,5,6,7,0,1,2,3)
        resid_aaaa = (1.0/8.0)*einsum('klcd,abcdijkl->abij',t2_dag,t4)

    elif order == 7:
        # construct [T2^\dag T2^\dag [W0T2^3/3!]]c
        t4=np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
        with open('roooovvvv_t4_third.pickle', 'rb') as handle:
            t4=pickle.load(handle)

        Roooovvvv = get_xcc7residual(nocc,nvir,t4,t2,t2_dag)
        t4=antisym.antisym_t4_residual(Roooovvvv,nocc,nvir)
        t4=t4.transpose(4,5,6,7,0,1,2,3)
        resid_aaaa = (1.0/8.0)*einsum('klcd,abcdijkl->abij',t2_dag,t4)

    elif order == 8:
        pass



    elif order == 9:
        t4=np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
        with open('roooovvvv_t4_third.pickle', 'rb') as handle:
            t4=pickle.load(handle)

        Roooovvvv = get_xcc9residual(nocc,nvir,t4,t2,t2_dag)
        t4=antisym.antisym_t4_residual(Roooovvvv,nocc,nvir)
        t4=t4.transpose(4,5,6,7,0,1,2,3)
        resid_aaaa = (1.0/8.0)*einsum('klcd,abcdijkl->abij',t2_dag,t4)


    else:
        print("The specified order for XCCD is not implemented! Please chhoose from orders 5-9.")
        sys.exit()

    return resid_aaaa


def get_xcc7residual(nocc,nvir,t4,t2,t_dag):
    """
    Constructs the XCCD(7) diagram, in a form looking like a net T4

    :param nocc: Number of occupied orbitals
    :param nvir: Number of virtual orbitals
    :param t4: Third order net-T4 constructed at the XCCD(5) level
    :param t2: T2 amplitude
    :param t_dag: Adjoint of the T2 amplitude

    :return: XCCD(7) net-T4 like amplitude for contraction against T2^t
    """
    roooovvvv = np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
    roooovvvv += -0.020833333 * np.einsum("imab,jklncdef,efmn->ijklabcd",t2,t4,t_dag,optimize="optimal")
    roooovvvv += 0.002604167 * np.einsum("mnab,ijklcdef,efmn->ijklabcd",t2,t4,t_dag,optimize="optimal")
    roooovvvv += -0.020833333 * np.einsum("ijae,klmnbcdf,efmn->ijklabcd",t2,t4,t_dag,optimize="optimal")
    roooovvvv += 0.027777778 * np.einsum("imae,jklnbcdf,efmn->ijklabcd",t2,t4,t_dag,optimize="optimal")
    roooovvvv += -0.003472222 * np.einsum("mnae,ijklbcdf,efmn->ijklabcd",t2,t4,t_dag,optimize="optimal")
    roooovvvv += 0.002604167 * np.einsum("ijef,klmnabcd,efmn->ijklabcd",t2,t4,t_dag,optimize="optimal")
    roooovvvv += -0.003472222 * np.einsum("imef,jklnabcd,efmn->ijklabcd",t2,t4,t_dag,optimize="optimal")
    return roooovvvv



def get_xcc9residual(nocc,nvir,t4,t2,t_dag):
    """
    Constructs the XCCD(9) diagram, in a form looking like a net T4
  
    :param nocc: Number of occupied orbitals
    :param nvir: Number of virtual orbitals
    :param t4: Third order net-T4 constructed at the XCCD(5) level
    :param t2: T2 amplitude
    :param t_dag: Adjoint of the T2 amplitude

    :return: XCCD(9) net-T4 like amplitude for contraction against T2^t
    """
    roooovvvv = np.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir))
    roooovvvv += -0.007812500 * np.einsum("imab,jncd,klefghop,opef,ghmn->ijklabcd",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.010416667 * np.einsum("imab,ncde,jklfghop,ghmn,opcf->ijklabde",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("imab,jkce,lndfghop,opdf,ehmn->ijklabcg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.125000000 * np.einsum("imab,jnce,kldfghop,opnf,ehmd->ijklabcg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("imab,jnce,kldfghop,epdf,homn->ijklabcg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("imab,jnce,kldfghop,opdf,ehmn->ijklabcg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("imab,ncde,jklfghop,homn,epcf->ijklabdg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("imab,ncde,jklfghop,opnc,ehmf->ijklabdg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("imab,ncde,jklfghop,opcf,ehmn->ijklabdg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("imab,jkef,lncdghop,fpcd,eomn->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.015625000 * np.einsum("imab,jkef,lncdghop,opcd,efmn->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.031250000 * np.einsum("imab,jnef,klcdghop,efmc,opnd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.125000000 * np.einsum("imab,jnef,klcdghop,eomc,fpnd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.015625000 * np.einsum("imab,jnef,klcdghop,efcd,opmn->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("imab,jnef,klcdghop,fpcd,eomn->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("imab,ncef,jkldghop,eomn,fpcd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.010416667 * np.einsum("imab,ncef,jkldghop,opmn,efcd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("imab,ncef,jkldghop,fpnc,eomd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.005208333 * np.einsum("imab,ncef,jkldghop,opnc,efmd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.001302083 * np.einsum("mnab,cdef,ijklghop,ghmc,opnd->ijklabef",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("mnab,ijce,kldfghop,opnf,ehmd->ijklabcg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.015625000 * np.einsum("mnab,ijce,kldfghop,opdf,ehmn->ijklabcg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("mnab,icde,jklfghop,ehmn,opcf->ijklabdg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.083333333 * np.einsum("mnab,icde,jklfghop,homc,epnf->ijklabdg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.041666667 * np.einsum("mnab,icde,jklfghop,opnf,ehmc->ijklabdg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("mnab,cdef,ijklghop,opnd,fhmc->ijklabeg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.002604167 * np.einsum("mnab,cdef,ijklghop,opcd,fhmn->ijklabeg",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.015625000 * np.einsum("mnab,ijef,klcdghop,eomn,fpcd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.015625000 * np.einsum("mnab,ijef,klcdghop,efmc,opnd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("mnab,ijef,klcdghop,eomc,fpnd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("mnab,icef,jkldghop,eomn,fpcd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.083333333 * np.einsum("mnab,icef,jkldghop,eomc,fpnd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("mnab,icef,jkldghop,opmc,efnd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.002604167 * np.einsum("mnab,cdef,ijklghop,eomn,fpcd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.010416667 * np.einsum("mnab,cdef,ijklghop,eomc,fpnd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.002604167 * np.einsum("mnab,cdef,ijklghop,opmc,efnd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.000868056 * np.einsum("ijkmabcd,lnefghop,opef,ghmn->ijklabcd",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.000651042 * np.einsum("ijmnabcd,klefghop,ghmn,opef->ijklabcd",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.002604167 * np.einsum("ijmnabcd,klefghop,ghme,opnf->ijklabcd",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.001736111 * np.einsum("imnabcde,jklfghop,ghmn,opaf->ijklbcde",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.000108507 * np.einsum("mnabcdef,ijklghop,ghmn,opab->ijklcdef",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.000868056 * np.einsum("ijklabce,mndfghop,opdf,ehmn->ijklabcg",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.013888889 * np.einsum("ijkmabce,lndfghop,ehnd,opmf->ijklabcg",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.006944444 * np.einsum("ijkmabce,lndfghop,opdf,ehmn->ijklabcg",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.010416667 * np.einsum("ijmnabce,kldfghop,opmn,ehdf->ijklabcg",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("ijmnabce,kldfghop,homd,epnf->ijklabcg",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.005208333 * np.einsum("ijmnabce,kldfghop,opdf,ehmn->ijklabcg",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.013888889 * np.einsum("imnabcde,jklfghop,ehmn,opaf->ijklbcdg",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.013888889 * np.einsum("imnabcde,jklfghop,homn,epaf->ijklbcdg",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.001736111 * np.einsum("mnabcdef,ijklghop,opab,fhmn->ijklcdeg",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.000651042 * np.einsum("ijklabef,mncdghop,efmn,opcd->ijklabgh",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.002604167 * np.einsum("ijklabef,mncdghop,eomn,fpcd->ijklabgh",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.010416667 * np.einsum("ijkmabef,lncdghop,efnc,opmd->ijklabgh",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("ijkmabef,lncdghop,eonc,fpmd->ijklabgh",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.005208333 * np.einsum("ijkmabef,lncdghop,opcd,efmn->ijklabgh",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.015625000 * np.einsum("ijmnabef,klcdghop,efmc,opnd->ijklabgh",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.031250000 * np.einsum("ijmnabef,klcdghop,eomc,fpnd->ijklabgh",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.003906250 * np.einsum("ijmnabef,klcdghop,efcd,opmn->ijklabgh",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.015625000 * np.einsum("ijmnabef,klcdghop,eocd,fpmn->ijklabgh",t4,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.007812500 * np.einsum("ijae,klbf,mncdghop,opcd,efmn->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("ijae,kmbf,lncdghop,efnc,opmd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.125000000 * np.einsum("ijae,kmbf,lncdghop,fpcd,eomn->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("ijae,kmbf,lncdghop,opcd,efmn->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.031250000 * np.einsum("ijae,mnbf,klcdghop,eomn,fpcd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.015625000 * np.einsum("ijae,mnbf,klcdghop,opmn,efcd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.125000000 * np.einsum("ijae,mnbf,klcdghop,eomc,fpnd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("ijae,mnbf,klcdghop,opnd,efmc->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.010416667 * np.einsum("ijae,klfb,mncdghop,efmn,bpcd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("ijae,kmfb,lncdghop,efnc,bpmd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("ijae,kmfb,lncdghop,fbcd,epmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("ijae,kmfb,lncdghop,bpcd,efmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("ijae,mnfb,klcdghop,efmc,bpnd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("ijae,mnfb,klcdghop,fbnd,epmc->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.010416667 * np.einsum("ijae,mnfb,klcdghop,efcd,bpmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.005208333 * np.einsum("ijae,mnfb,klcdghop,fbcd,epmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.125000000 * np.einsum("imae,jnbf,klcdghop,opmd,efnc->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.250000000 * np.einsum("imae,jnbf,klcdghop,eonc,fpmd->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.062500000 * np.einsum("imae,jnbf,klcdghop,efcd,opmn->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.125000000 * np.einsum("imae,jnbf,klcdghop,fpcd,eomn->ijklabgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.083333333 * np.einsum("imae,nbcf,jkldghop,eomn,fpbd->ijklacgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.166666667 * np.einsum("imae,nbcf,jkldghop,fomn,epbd->ijklacgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.083333333 * np.einsum("imae,nbcf,jkldghop,opmn,efbd->ijklacgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.083333333 * np.einsum("imae,nbcf,jkldghop,eonb,fpmd->ijklacgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("imae,nbcf,jkldghop,opnb,efmd->ijklacgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.083333333 * np.einsum("imae,jkfb,lncdghop,ebcd,fpmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("imae,jkfb,lncdghop,epcd,fbmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("imae,jkfb,lncdghop,bpcd,efmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.083333333 * np.einsum("imae,jnfb,klcdghop,efmc,bpnd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.083333333 * np.einsum("imae,jnfb,klcdghop,fbmc,epnd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.166666667 * np.einsum("imae,jnfb,klcdghop,efnc,bpmd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.083333333 * np.einsum("imae,jnfb,klcdghop,ebcd,fpmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("imae,jnfb,klcdghop,fbcd,epmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.055555556 * np.einsum("imae,nbfc,jkldghop,fpmn,ecbd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.013888889 * np.einsum("imae,nbfc,jkldghop,fcmb,epnd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.013888889 * np.einsum("imae,nbfc,jkldghop,ecnb,fpmd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.013888889 * np.einsum("imae,nbfc,jkldghop,epnb,fcmd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.013888889 * np.einsum("imae,nbfc,jkldghop,cpnb,efmd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.013888889 * np.einsum("imae,nbfc,jkldghop,fcbd,epmn->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.020833333 * np.einsum("mnae,bcdf,ijklghop,eomb,fpnc->ijkladgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.005208333 * np.einsum("mnae,bcdf,ijklghop,opmb,efnc->ijkladgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.005208333 * np.einsum("mnae,bcdf,ijklghop,eobc,fpmn->ijkladgh",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("mnae,ijfb,klcdghop,fpmn,ebcd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.083333333 * np.einsum("mnae,ijfb,klcdghop,efmc,bpnd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("mnae,ijfb,klcdghop,fbmc,epnd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.006944444 * np.einsum("mnae,ibfc,jkldghop,fcmn,epbd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.027777778 * np.einsum("mnae,ibfc,jkldghop,fpmn,ecbd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.055555556 * np.einsum("mnae,ibfc,jkldghop,fpmb,ecnd->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.027777778 * np.einsum("mnae,ibfc,jkldghop,fpmd,ecnb->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.027777778 * np.einsum("mnae,ibfc,jkldghop,fcnd,epmb->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.000868056 * np.einsum("mnae,bcfd,ijklghop,fdmn,epbc->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.001736111 * np.einsum("mnae,bcfd,ijklghop,fpmn,edbc->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.006944444 * np.einsum("mnae,bcfd,ijklghop,fpmb,ednc->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.003472222 * np.einsum("mnae,bcfd,ijklghop,fdnc,epmb->ijklagho",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.001302083 * np.einsum("ijef,klab,mncdghop,eamn,fbcd->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("ijef,kmab,lncdghop,fbcd,eamn->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.002604167 * np.einsum("ijef,kmab,lncdghop,abcd,efmn->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.002604167 * np.einsum("ijef,mnab,klcdghop,efmc,abnd->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.010416667 * np.einsum("ijef,mnab,klcdghop,eamc,fbnd->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.002604167 * np.einsum("ijef,mnab,klcdghop,eacd,fbmn->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.005208333 * np.einsum("imef,jnab,klcdghop,efnc,abmd->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.020833333 * np.einsum("imef,jnab,klcdghop,eanc,fbmd->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.005208333 * np.einsum("imef,jnab,klcdghop,eacd,fbmn->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.000868056 * np.einsum("imef,nabc,jkldghop,efna,bcmd->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.003472222 * np.einsum("imef,nabc,jkldghop,fcna,ebmd->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += 0.001736111 * np.einsum("imef,nabc,jkldghop,efnd,bcma->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    roooovvvv += -0.006944444 * np.einsum("imef,nabc,jkldghop,ebnd,fcma->ijklghop",t2,t2,t4,t_dag,t_dag,optimize="optimal")
    return roooovvvv


