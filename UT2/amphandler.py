import pickle
import UT2.pdagq_T3corr as pdagq_T3corr
import numpy as np
import UT2.wicked_T3corr as wicked_T3corr
class AmpHandler():
    def __init__(self,o,v,storedInfo,T2infile=None,T1infile=None,xaccObj=None):
        self.t1=None
        self.t2=None
        self.t3={}
        self.T2infile=T2infile
        self.T1infile=T1infile
        self.cc_runtype=storedInfo.cc_runtype

        self.o=o
        self.v=v
        self.nocc=storedInfo.occInfo["nocc_aa"]
        self.nvirt=storedInfo.occInfo["nvirt_aa"]
        if T2infile or T1infile:
            self.getAmps()
        elif xaccObj:
            self.t1=xaccObj.t1amps
            self.t2=xaccObj.t2amps
        else:
            print('T1infile,T2infile, and xaccObj are all None; try again')
            sys.exit()
            
        self.g=storedInfo.integralInfo["tei"]
        self.fock=storedInfo.integralInfo["oei"]
        self.denoms=storedInfo.denomInfo
        #print(self.denoms.keys())



    def getAmps(self):
        # Assume the storing convention is virt, virt, occ, occ
        occ=self.nocc
        virt=self.nvirt
        self.t2=np.zeros((virt,virt,occ,occ))
        self.t1=np.zeros((virt,occ))

        with open(self.T2infile,'rb') as t2handle:
            t2_tmp=pickle.load(t2handle)

        print('shape of T2:',np.shape(self.t2))
        if self.T1infile:
            with open(self.T1infile,'rb') as t1handle:
                self.t1=pickle.load(t1handle)
        else: # assumes t1ordering (virt,occ)
            t1_tmp=np.zeros((self.nvirt,self.nocc))

        try:
            self.t1=t1_tmp
            self.t2=t2_tmp
        except:
            self.t1=t1_tmp.transpose(1,0)
            self.t2=t2_tmp.transpose(2,3,0,1)


    def run_UCCDcheck(self):
        t2_dag=self.t2
        self.t2=self.t2.transpose(2,3,0,1)
        # build diagram A
        energy=0.0
        t2=self.t2
        g=self.g
        o=self.o
        v=self.v
        fock=self.fock
        print('o,v',o,v)
        energyA= 0.250000000 * np.einsum("ijab,abij->",t2,g[v,v,o,o],optimize="optimal")
        energyB=0.250000000 * np.einsum("abij,ijab->",t2_dag,g[o,o,v,v],optimize="optimal")

        energyC= 0.125000000 * np.einsum("ijab,cdij,abcd->",t2,t2_dag,g[v,v,v,v],optimize="optimal")
        energyC += -1.000000000 * np.einsum("ijab,acik,kbjc->",t2,t2_dag,g[o,v,o,v],optimize="optimal")
        energyC += 0.125000000 * np.einsum("ijab,abkl,klij->",t2,t2_dag,g[o,o,o,o],optimize="optimal")

        print('energies A,B,C',energyA,energyB,energyC)
        fockpart = -0.500000000 * np.einsum("ji,ikab,abjk->",fock[o,o],t2,t2_dag,optimize="optimal")
        fockpart += 0.500000000 * np.einsum("ba,ijbc,acij->",fock[v,v],t2,t2_dag,optimize="optimal")

        print('fockpart:',fockpart)
        energy=energyA+energyB+energyC
        print('total corr E',energy,energy+-1.11632556448611)



        # checking T2 residual route
        roovv = 0.125000000 * np.einsum("klab,ijkl->ijab",t2,g[o,o,o,o],optimize="optimal")
        roovv += -1.000000000 * np.einsum("ikac,jckb->ijab",t2,g[o,v,o,v],optimize="optimal")
        roovv += 0.125000000 * np.einsum("ijcd,cdab->ijab",t2,g[v,v,v,v],optimize="optimal")
        roovv = wicked_T3corr.antisym_T2(roovv, self.nocc, self.nvirt)
        # now contract <0|T2^ (WnT2)|0>
        newE=0.250000000 * np.einsum("ijab,abij->",roovv,t2_dag,optimize="optimal")
        print('new diagram C energy:',newE)
        sys.exit()

    def run_pdagq(self):
        # need t2 shape (virt,virt,occ,occ), so if not in this shape
        # transpose the t2 tensor
        if np.shape(self.t2)[0] != self.nvirt:
            print('inside first if caution')
            l1=self.t1
            l2=self.t2
            self.t2=self.t2.transpose(2,3,0,1)
            self.t1=self.t1.transpose(1,0)
        else:
            print('inside second if: true')
            print(np.shape(self.t1))
            l1=self.t1.transpose(1,0)
            l2=self.t2.transpose(2,3,0,1)
        t3resid=pdagq_T3corr.build_T3(self.g,self.o,self.v,self.t2)
        t3=t3resid*self.denoms["D3aa"]
        #self.t3.update["T3_secondOrder"]=t3
        #l1=self.t1 #.transpose(1,0)
        #l2=self.t2 #.transpose(2,3,0,1)
        energy=pdagq_T3corr.pdagq_T3energy(self.g,self.o,self.v,l1,l2,t3)
        print('(T)-based triples correction to CCSD is:', energy)
        ccsd_energy=pdagq_T3corr.ccsd_energy(self.g,self.fock,self.o,self.v,self.t1,self.t2)
        return ccsd_energy,energy

    def run_wickedTest(self):#run [T] test **ONLY**
        # need tensors in occ,occ,virt,virt ordering
#        if "xaccfiles" in self.cc_runtype:
#            self.t1=self.t1.transpose(1,0)
#            self.t2=self.t2.transpose(2,3,0,1)

        t1_dag=self.t1
        self.t1=self.t1.transpose(1,0)
        t2_dag=self.t2
        self.t2=self.t2.transpose(2,3,0,1)
        ##### Correct for pCCD and CCSpD
        def determine_pCCD_correction(T2op,nocc,nvirt,g,v,o):
            fullE = 0.250000000 * np.einsum("abij,ijab->",g[v,v,o,o],T2op,optimize="optimal")
            subtractE=0.0
            zerod_T2=T2op
            for i in range(nocc-1):
                for a in range(nvirt-1):
                    subtractE+=0.25*T2op[i,i+1,a,a+1]*g[a,a+1,i,i+1]
                    zerod_T2[i,i+1,a,a+1]=0.0
            print('fullE:',fullE, 'diagonal portion of E:',subtractE)
            print('test of off-diagon:',0.250000000 * np.einsum("abij,ijab->",g[v,v,o,o],zerod_T2,optimize="optimal"))
            offDiagE=fullE-subtractE
            print('final pCCD related E',offDiagE)
            return offDiagE

        # Do D2T2=Wn
        #t2_firstOrder_pCCD=0.250000000 * np.einsum("ijab->ijab",self.g[self.o,self.o,self.v,self.v],optimize="optimal")
        #t2_firstOrder_pCCD=wicked_T3corr.antisym_T2(t2_firstOrder_pCCD,self.nocc,self.nvirt)
        #t2_firstOrder_pCCD=t2_firstOrder_pCCD.transpose(2,3,0,1)

        t2_firstOrder_pCCD=self.g[self.v,self.v,self.o,self.o]*self.denoms["D2aa"]
        t2_firstOrder_pCCD=t2_firstOrder_pCCD.transpose(2,3,0,1)
        offDiag_MP2_E=determine_pCCD_correction(t2_firstOrder_pCCD,self.nocc,self.nvirt,self.g,self.v,self.o)
        print('Off diag pCCD MP2 energy:',offDiag_MP2_E)

        # NEED TO ANTISYMMETRIZE THIS

        # Do D2T2= WnT2
        
        t2_secondOrder_pCCD = 0.125000000 * np.einsum("klab,ijkl->ijab",t2_firstOrder_pCCD,self.g[self.o,self.o,self.o,self.o],optimize="optimal")
        t2_secondOrder_pCCD += -1.000000000 * np.einsum("ikac,jckb->ijab",t2_firstOrder_pCCD,self.g[self.o,self.v,self.o,self.v],optimize="optimal")
        t2_secondOrder_pCCD += 0.125000000 * np.einsum("ijcd,cdab->ijab",t2_firstOrder_pCCD,self.g[self.v,self.v,self.v,self.v],optimize="optimal")

        t2_secondOrder_pCCD=wicked_T3corr.antisym_T2(t2_secondOrder_pCCD,self.nocc,self.nvirt)
        t2_secondOrder_pCCD=t2_secondOrder_pCCD.transpose(2,3,0,1)
        t2_secondOrder_pCCD=t2_secondOrder_pCCD*self.denoms["D2aa"]
        t2_secondOrder_pCCD=t2_secondOrder_pCCD.transpose(2,3,0,1)
        offDiag_MP3_E=determine_pCCD_correction(-1.0*t2_secondOrder_pCCD,self.nocc,self.nvirt,self.g,self.v,self.o)
        print('Off diag pCCD MP3 energy:',offDiag_MP3_E)


        # DO D2T2=WnT1
        t2_fromT1_pCCD = -0.500000000 * np.einsum("ka,ijkb->ijab",self.t1,self.g[self.o,self.o,self.o,self.v],optimize="optimal")
        t2_fromT1_pCCD += -0.500000000 * np.einsum("ic,jcab->ijab",self.t1,self.g[self.o,self.v,self.v,self.v],optimize="optimal")

        t2_fromT1_pCCD=wicked_T3corr.antisym_T2(t2_fromT1_pCCD,self.nocc,self.nvirt)
        t2_fromT1_pCCD_Orig=t2_fromT1_pCCD # ijab index
        t2_fromT1_pCCD=t2_fromT1_pCCD.transpose(2,3,0,1)
        t2_fromT1_pCCD=t2_fromT1_pCCD*self.denoms["D2aa"] # abij index

        print('listing T1 amps:')
        for i in range(self.nocc):
            for a in range(self.nvirt):
                print('t1amp:',i,a+self.nocc,self.t1[i,a])
        fullE = 0.250000000 * np.einsum("abij,ijab->",t2_fromT1_pCCD,t2_fromT1_pCCD_Orig,optimize="optimal")
        subtractE=0.0
        print('full singles portion of CCSpD T1 corr:',fullE)
        for i in range(self.nocc-1):
            for a in range(self.nvirt-1):
                subtractE+=0.25*t2_fromT1_pCCD_Orig[i,i+1,a,a+1]*t2_fromT1_pCCD[a,a+1,i,i+1]
        print('fullE:',fullE, 'diagonal portion of E:',subtractE)
        offDiagE=fullE-subtractE
        print('final pCCD related E',offDiagE)
        print('UCCSpD off diagonal singles correction:', offDiagE)
        print('total UCCSpD off diagonal singles correction:',offDiagE+offDiag_MP2_E+offDiag_MP3_E)

        #After I antisymmetrize, divide these operators by D2 denominator to form the amplitude eqn
        #THEN call each amplitude with the determine_pCCD_correction() function to determine the 
        # off-diagonal contribution

        # USING PCCD AMPS TO GENERATE LCCD EQNS
        def build_MP2_T2(W,self):
            roovv = 0.250000000 * np.einsum("ijab->ijab",W,optimize="optimal")
            roovv = wicked_T3corr.antisym_T2(roovv,self.nocc,self.nvirt)
            roovv = roovv.transpose(2,3,0,1)
            roovv = roovv*self.denoms["D2aa"]
            roovv = roovv.transpose(2,3,0,1)
            return roovv



        def build_LCCD_T2(T2,W,o,v,self):
            roovv = 0.125000000 * np.einsum("klab,ijkl->ijab",T2,W[o,o,o,o],optimize="optimal")
            roovv += -1.000000000 * np.einsum("ikac,jckb->ijab",T2,W[o,v,o,v],optimize="optimal")
            roovv += 0.125000000 * np.einsum("ijcd,cdab->ijab",T2,W[v,v,v,v],optimize="optimal")

            roovv = wicked_T3corr.antisym_T2(roovv,self.nocc,self.nvirt)
            roovv = roovv.transpose(2,3,0,1)
            roovv = roovv*self.denoms["D2aa"]
            roovv = roovv.transpose(2,3,0,1)
            return roovv


        def get_WnT2_energy(T2,g):
            energy = 0.250000000 * np.einsum("ijab,abij->",T2,g,optimize="optimal")
            return energy


        def kill_Diag_T2(roovv,nocc,nvirt):
            for i in range(nocc-1):
                for a in range(nvirt-1):
                    roovv[i,i+1,a,a+1]=0.0
                    roovv[i+1,i,a,a+1]=0.0
                    roovv[i,i+1,a+1,a]=0.0
                    roovv[i+1,i,a+1,a]=0.0

            return roovv


        def build_WnT1_T2(T1,self):
            # DO D2T2=WnT1
            t2_fromT1_pCCD = -0.500000000 * np.einsum("ka,ijkb->ijab",T1,self.g[self.o,self.o,self.o,self.v],optimize="optimal")
            t2_fromT1_pCCD += -0.500000000 * np.einsum("ic,jcab->ijab",T1,self.g[self.o,self.v,self.v,self.v],optimize="optimal")

            roovv = wicked_T3corr.antisym_T2(t2_fromT1_pCCD,self.nocc,self.nvirt)
            roovv = roovv.transpose(2,3,0,1)
            roovv = roovv*self.denoms["D2aa"]
            roovv = roovv.transpose(2,3,0,1)
            return roovv


        def build_CCD_T2_sqr(T2,W,self):
            roovv = -0.500000000 * np.einsum("ikab,jlcd,cdkl->ijab",T2,T2,W,optimize="optimal")
            roovv += 0.125000000 * np.einsum("klab,ijcd,cdkl->ijab",T2,T2,W,optimize="optimal")
            roovv += -0.500000000 * np.einsum("ijac,klbd,cdkl->ijab",T2,T2,W,optimize="optimal")
            roovv += 1.000000000 * np.einsum("ikac,jlbd,cdkl->ijab",T2,T2,W,optimize="optimal")
            roovv = 0.5*roovv

            roovv = wicked_T3corr.antisym_T2(roovv,self.nocc,self.nvirt)
            roovv = roovv.transpose(2,3,0,1)
            roovv = roovv*self.denoms["D2aa"]
            roovv = roovv.transpose(2,3,0,1)
            return roovv


        firstOrder_T2 = build_MP2_T2(self.g[self.o,self.o,self.v,self.v],self)
        firstOrderT2_killOD=np.copy(firstOrder_T2)

        firstOrderT2_killOD = kill_Diag_T2(firstOrderT2_killOD,self.nocc,self.nvirt)

        MP2fullE=get_WnT2_energy(firstOrder_T2,self.g[self.v,self.v,self.o,self.o])
        MP2offDiagE = get_WnT2_energy(firstOrderT2_killOD,self.g[self.v,self.v,self.o,self.o])

        print('full MP2 energy:',MP2fullE)
        print('off Diag MP2 energy:',MP2offDiagE)

        secondOrder_T2=build_LCCD_T2(firstOrder_T2,self.g,self.o,self.v,self)
        MP3fullE=get_WnT2_energy(secondOrder_T2,self.g[self.v,self.v,self.o,self.o])
        print('full MP3 energy:', -MP3fullE)

        secondOrder_T2_killOD = build_LCCD_T2(firstOrderT2_killOD,self.g,self.o,self.v,self)
        MP3offDiagE = get_WnT2_energy(secondOrder_T2_killOD,self.g[self.v,self.v,self.o,self.o])
        print('off diag MP3 energy:',-MP3offDiagE)

        


        pLCCD_T2 = build_LCCD_T2(self.t2,self.g,self.o,self.v,self)
        pLCCD_T2KOD = np.copy(pLCCD_T2)
        pLCCD_T2KOD=kill_Diag_T2(pLCCD_T2KOD,self.nocc,self.nvirt)
        testE=get_WnT2_energy(pLCCD_T2,self.g[self.v,self.v,self.o,self.o])
        pLCCD_e=get_WnT2_energy(pLCCD_T2KOD,self.g[self.v,self.v,self.o,self.o])
        print('test LCCD energy from pCCD:',-testE)
        print('off diag LCCD energy from pCCD amps:',-pLCCD_e)

        CCSpD_t2_from_T1 = build_WnT1_T2(self.t1,self)
        CCSpD_t2_from_T1KOD = np.copy(CCSpD_t2_from_T1)
        CCSpD_t2_from_T1KOD = kill_Diag_T2(CCSpD_t2_from_T1KOD,self.nocc,self.nvirt)
        t1_energy = get_WnT2_energy(CCSpD_t2_from_T1,self.g[self.v,self.v,self.o,self.o])
        t1_energyOD = get_WnT2_energy(CCSpD_t2_from_T1KOD,self.g[self.v,self.v,self.o,self.o])

        print('UCCSpD full correction from T1:',t1_energy)
        print('UCCSpD off diag correction from T1:',t1_energyOD)


        t2_sqrCCD=build_CCD_T2_sqr(firstOrderT2_killOD,self.g[self.v,self.v,self.o,self.o],self)
        t2_sqrCCD_E=get_WnT2_energy(t2_sqrCCD,self.g[self.v,self.v,self.o,self.o])
        print('Wnt2^2 from MP2 off-diagonal amps:',t2_sqrCCD_E)
        sys.exit()
        #########################################################################
        # end
        #########################################################################
        roovv_LCCD = 0.125000000 * np.einsum("klab,ijkl->ijab",self.t2,self.g[self.o,self.o,self.o,self.o],optimize="optimal")
        roovv_LCCD += -1.000000000 * np.einsum("ikac,jckb->ijab",self.t2,self.g[self.o,self.v,self.o,self.v],optimize="optimal")
        roovv_LCCD += 0.125000000 * np.einsum("ijcd,cdab->ijab",self.t2,self.g[self.v,self.v,self.v,self.v],optimize="optimal")


        roovv_LCCD = wicked_T3corr.antisym_T2(roovv_LCCD,self.nocc,self.nvirt)
        roovv_LCCD=roovv_LCCD.transpose(2,3,0,1)
        roovv_LCCD=roovv_LCCD*self.denoms["D2aa"] # abij index
        roovv_LCCD=roovv_LCCD.transpose(2,3,0,1)
        # Now zero out diagonal of LCCD T2 tensor
        for i in range(self.nocc-1):
            for a in range(self.nvirt-1):
                roovv_LCCD[i,i+1,a,a+1]=0.0
        # now build <0|WnT2|0>
        roovv_LCCD_energy=0.250000000 * np.einsum("ijab,abij->",roovv_LCCD,self.g[self.v,self.v,self.o,self.o],optimize="optimal")
        print('LCCD energy correction to pCCD:',roovv_LCCD_energy)

        def partB_UCC4E(T2,T2dag,W):
            r = 1.000000000 * np.einsum("ijab,acik,bdjl,klcd->",T2["oovv"],T2dag["vvoo"],T2dag["vvoo"],W["oovv"],optimize="optimal")
            r += 0.500000000 * np.einsum("ijab,cdjk,abil,klcd->",T2["oovv"],T2dag["vvoo"],T2dag["vvoo"],W["oovv"],optimize="optimal")
            r += 0.125000000 * np.einsum("ijab,abkl,cdij,klcd->",T2["oovv"],T2dag["vvoo"],T2dag["vvoo"],W["oovv"],optimize="optimal")
            r += 0.500000000 * np.einsum("ijab,bckl,adij,klcd->",T2["oovv"],T2dag["vvoo"],T2dag["vvoo"],W["oovv"],optimize="optimal")
            r += 0.125000000 * np.einsum("ijab,cdkl,abij,klcd->",T2["oovv"],T2dag["vvoo"],T2dag["vvoo"],W["oovv"],optimize="optimal")
            return r

        # END OF PCCD CORRECTION SECTION
        #############################
        # START OF 1/2(1/2WNT2^2) SECTION

        def antisym_UCC4_T2(self,roovv_LCCD):
            roovv_LCCD = wicked_T3corr.antisym_T2(roovv_LCCD,self.nocc,self.nvirt)
            roovv_LCCD=roovv_LCCD.transpose(2,3,0,1)
            roovv_LCCD=roovv_LCCD*self.denoms["D2aa"] # abij index
            roovv_LCCD=roovv_LCCD.transpose(2,3,0,1)
            return roovv_LCCD
 
        def zero_diagT2(self,roovv_LCCD):
            for i in range(self.nocc-1):
                for a in range(self.nvirt-1):
                    roovv_LCCD[i,i+1,a,a+1]=0.0
            return roovv_LCCD


        def Wn_into_T2(W,self):
            roovv = 0.250000000 * np.einsum("ijab->ijab",W,optimize="optimal")
            roovv = antisym_UCC4_T2(self,roovv)
        #    roovv = zero_diagT2(self,roovv)
            return roovv

        def CCD_T2_sqr(T2,W,self):
            roovv = -0.500000000 * np.einsum("ikab,jlcd,cdkl->ijab",T2,T2,W,optimize="optimal")
            roovv += 0.125000000 * np.einsum("klab,ijcd,cdkl->ijab",T2,T2,W,optimize="optimal")
            roovv += -0.500000000 * np.einsum("ijac,klbd,cdkl->ijab",T2,T2,W,optimize="optimal")
            roovv += 1.000000000 * np.einsum("ikac,jlbd,cdkl->ijab",T2,T2,W,optimize="optimal")
            roovv = 0.5*roovv
            roovv = antisym_UCC4_T2(self,roovv)
            roovv = zero_diagT2(self,roovv)
            return roovv

        def get_WnT2_energy(roovv_LCCD,self):
            return 0.250000000 * np.einsum("ijab,abij->",roovv_LCCD,self.g[self.v,self.v,self.o,self.o],optimize="optimal")


        wn_into_T2=Wn_into_T2(self.g[self.o,self.o,self.v,self.v],self)
        wn_into_T2_energy=get_WnT2_energy(wn_into_T2,self)
        print('check full MP2 energy:',wn_into_T2_energy)
        t2sqr_UCC4=CCD_T2_sqr(roovv_LCCD,self.g[self.v,self.v,self.o,self.o],self)
        t2sqr_UCC4_energy=get_WnT2_energy(t2sqr_UCC4,self)


        checkMP3_base=Wn_into_T2(self.g[self.o,self.o,self.v,self.v],self)
        checkMP3_base=checkMP3_base.transpose(2,3,0,1)
        checkMP3_base=checkMP3_base*self.denoms["D2aa"] # abij index
        checkMP3_base=checkMP3_base.transpose(2,3,0,1)
        print('check full MP3 energy:',0.250000000 * np.einsum("abij,ijab->",self.g[self.v,self.v,self.o,self.o],checkMP3_base,optimize="optimal"))
        sys.exit()

        print('pCCD off diagonal E from W_n:',wn_into_T2_energy)
        print('pCCD off diagonal E from WnT2:',roovv_LCCD_energy)
        print('pCCD off diagonal 0.25WnT2^2 going into <0|WnT2|0>',t2sqr_UCC4_energy)
        print('total pCCD off diagonal E:',wn_into_T2_energy+roovv_LCCD_energy+t2sqr_UCC4_energy)



        ###################################################
        # END OF PCCD CORRECTION
        ###################################################

        t3resid=wicked_T3corr.build_T3_secondO(self.g,self.o,self.v,self.t2)
        t3resid=wicked_T3corr.antisym_T3(t3resid,self.nocc,self.nvirt)
        t3residOrigContract=t3resid
        t3resid=t3resid.transpose(3,4,5,0,1,2)
        t3=t3resid*self.denoms["D3aa"]
        t3Contract=t3
        t3=t3.transpose(3,4,5,0,1,2)
        
        energy=wicked_T3corr.getE_squareBrackT(self.g,self.o,self.v,t3,t2_dag)
        energy_t1=wicked_T3corr.getE_parenT_t1(self.g,self.o,self.v,t3,t1_dag)
        print('[T]-based triples energy correction to CC is:',energy)
        print('(T)-based triples w/ T1_dag:',energy+energy_t1)
        newwayE=0.111111111 * np.einsum("ijkabc,abcijk->",t3residOrigContract,t3Contract,optimize="optimal")

        print('redone energy:',newwayE)
        square_brackTenergy=wicked_T3corr.wicked_main(self.g,self.o,self.v,self.t1,t1_dag,self.t2,t2_dag,self.denoms["D3aa"],self.denoms["D2aa"],self.nocc,self.nvirt)

        netT2=wicked_T3corr.build_netT2(self.g,self.o,self.v,t3)
        netT2=wicked_T3corr.antisym_T2(netT2,self.nocc,self.nvirt)
        t2=netT2.transpose(2,3,0,1)
        t2=t2#*self.denoms["D2aa"]
        t2=t2.transpose(2,3,0,1)
        t2_likeE=0.250000000 * np.einsum("abij,ijab->",t2_dag,t2,optimize="optimal")
        print('netT2-like energy:',t2_likeE)
        ## ADDED 12/18/2023 to accomodate (T*) correction ##

        #correct approximated T2:
        corr_T2=netT2.transpose(2,3,0,1)
        corr_T2=corr_T2*self.denoms["D2aa"]
        corr_T2=corr_T2.transpose(2,3,0,1)

        o=self.o
        v=self.v
        # handle T1 residuals
        rov = -0.500000000 * np.einsum("jkab,ibjk->ia",corr_T2,self.g[o,v,o,o],optimize="optimal")
        rov += -0.500000000 * np.einsum("ijbc,bcja->ia",corr_T2,self.g[v,v,o,v],optimize="optimal")
        # Now take <0|T1^(D1T1)|0>
        t1_diagramE=np.einsum("ia,ai->",rov,t1_dag,optimize="optimal")
        print('Fifth order (T*) diagram:',t1_diagramE)
        print('Total (T*) contrib including [T]:',t1_diagramE+energy)


        r = -0.250000000 * np.einsum("ijkabc,adij,bckd->",t3,t2_dag,self.g[v,v,o,v],optimize="optimal")
        r += -0.250000000 * np.einsum("ijkabc,abil,lcjk->",t3,t2_dag,self.g[o,v,o,o],optimize="optimal")
        print('try again:',r)

        #Build T1 correction-if applicable
        netT1=wicked_T3corr.get_netT1_fromT2(self.g,self.o,self.v,self.t2)
        t1_bar=netT1.transpose(1,0)*self.denoms["D1aa"]
        print('shapes',np.shape(netT1),np.shape(t1_bar))
        print('check on [S]',np.einsum("ia,ai->",t1_bar,netT1))
        t1_bar=t1_bar.transpose(1,0)
        t2_bar_resid=wicked_T3corr.get_T2resid_fromT1bar(self.g,self.o,self.v,t1_bar)
        t2_bar_resid=wicked_T3corr.antisym_T2(t2_bar_resid,self.nocc,self.nvirt)
        singles_correction=0.250000000 * np.einsum("abij,ijab->",t2_dag,t2_bar_resid,optimize="optimal")
        print('[S] singles correction to CCD:',singles_correction)

        #BUILD NEXT ORDER T1 CORRECTION:
        thirdO_t1=wicked_T3corr.get_netT1_thirdO(self.g,self.o,self.v,self.t2) 
        fifthO_singles=np.einsum("ia,ai->",thirdO_t1,t1_bar.transpose(1,0))
        print('fifth order (S) correction to CCD:', fifthO_singles)
        print('second diagram (S) correction to CCD:',np.einsum("ia,ai->",t1_bar,thirdO_t1.transpose(1,0)))
        fifthO=fifthO_singles+np.einsum("ia,ai->",t1_bar,thirdO_t1.transpose(1,0))
        thirdO_t1_bar=thirdO_t1.transpose(1,0)*self.denoms["D1aa"]
        print('sixth order (S) correction to CCD:',np.einsum("ia,ai->",thirdO_t1,thirdO_t1_bar))
        finalParSenergy=singles_correction+fifthO+np.einsum("ia,ai->",thirdO_t1,thirdO_t1_bar)
        print('Full sixth order (S) correction:',finalParSenergy)
        return None,square_brackTenergy
