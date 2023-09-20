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
        t2=t2*self.denoms["D2aa"]
        t2=t2.transpose(2,3,0,1)
        t2_likeE=0.250000000 * np.einsum("abij,ijab->",t2_dag,t2,optimize="optimal")
        print('netT2-like energy:',t2_likeE)

        o=self.o
        v=self.v
        r = -0.250000000 * np.einsum("ijkabc,adij,bckd->",t3,t2_dag,self.g[v,v,o,v],optimize="optimal")
        r += -0.250000000 * np.einsum("ijkabc,abil,lcjk->",t3,t2_dag,self.g[o,v,o,o],optimize="optimal")
        print('try again:',r)
        return None,square_brackTenergy
