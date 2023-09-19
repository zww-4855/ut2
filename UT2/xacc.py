import numpy as np
from math import floor


class run_xacc():
    def __init__(self,bkgrd_infile,tamp_infile=None,tei_infile=None):
        self.nocc=None
        self.nvirt=None
        self.mo_energies=None
        self.read_bkgrd(bkgrd_infile)

        self.o=slice(None,self.nocc)
        self.v=slice(self.nocc,None)
        self.denoms=self.set_denoms()

        self.t2amps=np.zeros((self.nvirt,self.nvirt,self.nocc,self.nocc))
        self.t1amps=np.zeros((self.nvirt,self.nocc))
        self.read_tamps(tamp_infile)


        nbas=self.nocc+self.nvirt
        self.tei=np.zeros((nbas,nbas,nbas,nbas))
        self.read_tei(tei_infile)
        #self.ccd_energyTest()
        #self.mp2_energy()

    def set_denoms(self):
        eps_a = np.asarray(self.mo_energies)
        eps_b = np.asarray(self.mo_energies)
        eps = np.append(eps_a, eps_b)
        eps=np.sort(eps)
        n=np.newaxis
        v=self.v
        o=self.o
        self.e_ai = 1 / (-eps[v,n] + eps[n,o])
        self.e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
        self.e_abcijk = 1/(-eps[v,n,n,n,n,n]-eps[n,v,n,n,n,n]-eps[n,n,v,n,n,n]+
                   eps[n,n,n,o,n,n]+eps[n,n,n,n,o,n]+eps[n,n,n,n,n,o])
        
    def ccd_energyTest(self):
        n=np.newaxis
        o=slice(None,self.nocc)
        v=slice(self.nocc,None)
        ccd_test=0.250000000000000 * np.einsum('jiab,abji',self.tei[o, o, v, v],self.t2amps)
        print('ccd test E:',ccd_test)

    def mp2_energy(self):
        n=np.newaxis
        o=slice(None,self.nocc)
        v=slice(self.nocc,None)
        print(self.mo_energies,type(self.mo_energies[0]),self.nocc,o,v)
        eps_a = np.asarray(self.mo_energies)
        eps_b = np.asarray(self.mo_energies)
        eps = np.append(eps_a, eps_b)
        eps=np.sort(eps)
        e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    
        t2=e_abij*self.tei[v,v,o,o]
    
        mp2E=0.250000000000000 * np.einsum('jiab,abji',self.tei[o, o, v, v], t2)

        ccd_test=0.250000000000000 * np.einsum('jiab,abji',self.tei[o, o, v, v],self.t2amps)
        print('ccd test E:',ccd_test)
        print('i',self.nocc,'a',self.nvirt)
        print('mp2E:', mp2E)
        e=0.0
        print(np.shape(e_abij))
        for i in range(8):
            for j in range(i+1,8):
                for a in range(8,12):
                    for b in range(a+1,12):
                        denom=self.mo_energies[floor(i/2)]+self.mo_energies[floor(j/2)]-self.mo_energies[floor(a/2)]-self.mo_energies[floor(b/2)]
                        if abs(self.t2amps[a-self.nocc,b-self.nocc,i,j])>10E-6:
                            e+=(self.tei[a,b,i,j])**2/denom
                            print(a,b,i,j,self.t2amps[a-self.nocc,b-self.nocc,i,j])#self.tei[a,b,i,j],denom)
        print('looped e:',e)
        print(self.tei[8,9,0,1],self.tei[8,9,1,0])
        #sys.exit()
    def read_tei(self,tei_infile):
        tei={}
        with open(tei_infile,'r') as f:
            for line in f:
                tei_key=float(line.split()[-1])
                index_list=line.split()
                operator_list=[]
                for operator in range(4): # max T2, min T1
                    if index_list[operator] == '|':
                        break
                    operator_list.append(int(index_list[operator].strip('^')))
                print('op list/key:',operator_list,tei_key,operator_list[0])
                idx=str(operator_list[0])+','+str(operator_list[1])+','+str(operator_list[2])+','+str(operator_list[3])
                a=operator_list[0]
                b=operator_list[1]
                c=operator_list[2]
                d=operator_list[3]
                self.tei[a,b,c,d]=4.0*tei_key

                tei.update({tei_key:operator_list})
        #self.tei=expand_tei(tei,self.nocc,self.nvirt)
        #print('tei',self.tei)

    def read_tamps(self,tamp_infile):
        t2amp={}
        t1amp={}
        read_amps=False
        with open(tamp_infile,'r') as f:
            for line in f:
                if read_amps:
                    amp_key=float(line.split()[-1])
                    index_list=line.split()
                    operator_list=[]
                    for operator in range(4): # max T2, min T1
                        if index_list[operator] == '|':
                            break
                        operator_list.append(int(index_list[operator].strip('^')))
                    print('op list:',operator_list,'amp key:',amp_key)
                    if len(operator_list)==4: #dealing with t2amp
#                        t2amp.update({amp_key:operator_list})
                        a=operator_list[0]-self.nocc
                        b=operator_list[1]-self.nocc
                        i=operator_list[2]
                        j=operator_list[3]
                        self.t2amps[a,b,i,j]=amp_key
                        self.t2amps[b,a,i,j]= -1.0* amp_key
                        self.t2amps[a,b,j,i]= -1.0*amp_key
                        self.t2amps[b,a,j,i]=amp_key
                        print(self.t2amps[a,b,i,j]) 
                    else: # dealing with t1amp
                        a=operator_list[0]-self.nocc
                        i=operator_list[1]
                        self.t1amps[a,i]=amp_key

                if line[:5]=="+++++":#parse the file until this str is read
                    read_amps=True

        print('t1:',self.t1amps)
        print('t2:',self.t2amps)
        self.t2amps=self.t2amps*0.25

    def read_bkgrd(self,bkgrd_infile):
        '''
        Reads important bkgrnd info, like # of occupied/virtual orbs
        '''
        with open(bkgrd_infile,'r') as f:
            lines=f.readlines()
        self.nocc=2*int(lines[1].strip().split()[-1])
        self.nvirt=2*int(lines[2].strip().split()[-1])
        tmp_energies=lines[3].strip().split()[-1]
        mo_energies=[]
        for element in tmp_energies.split(','):
            mo_energies.append(float(element.strip('[').strip(']')))
        self.mo_energies=mo_energies



