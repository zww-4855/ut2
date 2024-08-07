import numpy as np
import UT2.tamps as tamps



def wnT2T3_toT4(W,T2,T3,o,v):
    roooovvvv = -0.125000000 * np.einsum("imab,jkncde,lemn->ijklabcd",T2,T3,W[o,v,o,o],optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("imab,jklcef,efmd->ijklabcd",T2,T3,W[v,v,o,v],optimize="optimal")
    roooovvvv += -0.020833333 * np.einsum("mnab,ijkcde,lemn->ijklabcd",T2,T3,W[o,v,o,o],optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("ijae,kmnbcd,lemn->ijklabcd",T2,T3,W[o,v,o,o],optimize="optimal")
    roooovvvv += -0.125000000 * np.einsum("ijae,klmbcf,efmd->ijklabcd",T2,T3,W[v,v,o,v],optimize="optimal")
    roooovvvv += 0.083333333 * np.einsum("imae,jknbcd,lemn->ijklabcd",T2,T3,W[o,v,o,o],optimize="optimal")
    roooovvvv += 0.083333333 * np.einsum("imae,jklbcf,efmd->ijklabcd",T2,T3,W[v,v,o,v],optimize="optimal")
    roooovvvv += -0.020833333 * np.einsum("ijef,klmabc,efmd->ijklabcd",T2,T3,W[v,v,o,v],optimize="optimal")

    roooovvvv=tamps.antisym_T4(roooovvvv,None,None)
    return roooovvvv


def wnT2cubed_toT4(W,T2,o,v):
    roooovvvv = -0.031250000 * np.einsum("imab,jncd,klef,efmn->ijklabcd",T2,T2,T2,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.062500000 * np.einsum("imab,jncd,klmn->ijklabcd",T2,T2,W[o,o,o,o],optimize="optimal")
    roooovvvv += 0.250000000 * np.einsum("imab,jkce,lndf,efmn->ijklabcd",T2,T2,T2,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.250000000 * np.einsum("imab,jkce,lemd->ijklabcd",T2,T2,W[o,v,o,v],optimize="optimal")
    roooovvvv += -0.031250000 * np.einsum("mnab,ijce,kldf,efmn->ijklabcd",T2,T2,T2,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.062500000 * np.einsum("ijae,klbf,efcd->ijklabcd",T2,T2,W[v,v,v,v],optimize="optimal")

    roooovvvv=tamps.antisym_T4(roooovvvv,None,None)
    return roooovvvv


def wnT2cubed_toT2(W,T2,o,v):
    T2dag=T2.transpose(2,3,0,1)
    roovv = 0.500000000 * np.einsum("ikab,jlcd,mnef,celm,dfkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikab,jlcd,mnef,eflm,cdkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikab,jlcd,mnef,cemn,dfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("ikab,lmcd,jnef,eflm,cdkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klab,ijcd,mnef,cekm,dfln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klab,ijcd,mnef,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klab,ijcd,mnef,cemn,dfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klab,imcd,jnef,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klab,imcd,jnef,cekn,dflm->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klab,imcd,jnef,cemn,dfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("ijac,klbd,mnef,efkl,cdmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ijac,klbd,mnef,dekm,cfln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijac,klbd,mnef,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijac,klbd,mnef,demn,cfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ikac,jlbd,mnef,efkl,cdmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jlbd,mnef,dekm,cfln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,mnef,eflm,cdkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,mnef,demn,cfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ikac,lmbd,jnef,dekl,cfmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,lmbd,jnef,efkl,cdmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,lmbd,jnef,dekn,cflm->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,lmbd,jnef,eflm,cdkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,lmbd,jnef,deln,cfkm->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klac,imbd,jnef,dekl,cfmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("klac,imbd,jnef,dekn,cflm->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klac,mnbd,ijef,dekl,cfmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klac,mnbd,ijef,dekm,cfln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ijcd,klae,mnbf,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikcd,jlae,mnbf,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klcd,imae,jnbf,efkl,cdmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")


    roovv=tamps.antisym_T2(roovv,None,None)
    return roovv


def wnT2T3_toT2(W,T2,T3,o,v):
    T2dag=T2.transpose(2,3,0,1)
    roovv = 0.250000000 * np.einsum("ikab,cdlm,jmncde,lekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikab,cdlm,lmncde,jekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikab,cdlm,jlmdef,efkc->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klab,cdkm,ijncde,meln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klab,cdkm,imncde,jeln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klab,cdkm,ijmdef,eflc->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klab,cdmn,ijncde,mekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klab,cdmn,imncde,jekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijac,dekl,lmnbde,kcmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijac,dekl,klmbef,cfmd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijac,dekl,klmdef,cfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikac,dekl,jmnbde,lcmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,dekl,lmnbde,jcmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,dekl,jlmbef,cfmd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,dekl,jlmdef,cfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,cdlm,jmnbde,lekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,cdlm,lmnbde,jekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikac,cdlm,jlmbef,efkd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,cdlm,jlmdef,efkb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikac,delm,jmnbde,lckn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,delm,lmnbde,jckn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikac,delm,jlmbef,cfkd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,delm,jlmdef,cfkb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klac,dekl,imnbde,jcmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,dekl,ijmbef,cfmd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klac,dekl,ijmdef,cfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klac,cdkm,ijnbde,meln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("klac,cdkm,imnbde,jeln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klac,cdkm,ijmbef,efld->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,cdkm,ijmdef,eflb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,dekm,ijnbde,mcln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klac,dekm,imnbde,jcln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klac,dekm,ijmbef,cfld->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,dekm,ijmdef,cflb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klac,cdmn,ijnbde,mekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,cdmn,imnbde,jekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijcd,cekl,lmnabe,kdmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijcd,cekl,klmabf,dfme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijcd,cekl,klmaef,dfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("ijcd,efkl,klmabf,cdme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.062500000 * np.einsum("ijcd,efkl,klmaef,cdmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikcd,cekl,jmnabe,ldmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,cekl,lmnabe,jdmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikcd,cekl,jlmabf,dfme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikcd,cekl,jlmaef,dfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikcd,efkl,jlmabf,cdme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,efkl,jlmaef,cdmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,cdlm,jmnabe,lekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikcd,cdlm,lmnabe,jekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ikcd,cdlm,jlmaef,efkb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikcd,celm,jmnabe,ldkn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,celm,lmnabe,jdkn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,celm,jlmabf,dfke->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikcd,celm,jlmaef,dfkb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klcd,cekl,imnabe,jdmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klcd,cekl,ijmabf,dfme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klcd,cekl,ijmaef,dfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klcd,efkl,ijmabf,cdme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klcd,efkl,ijmaef,cdmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klcd,cdkm,ijnabe,meln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klcd,cdkm,imnabe,jeln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klcd,cdkm,ijmaef,eflb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klcd,cekm,ijnabe,mdln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klcd,cekm,imnabe,jdln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klcd,cekm,ijmabf,dfle->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klcd,cekm,ijmaef,dflb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klcd,cdmn,ijnabe,mekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klcd,cdmn,imnabe,jekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")

    roovv=tamps.antisym_T2(roovv,None,None)
    return roovv

