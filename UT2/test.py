




class SetupCC():
    def __init__(self,pyscf_mf,cc_info):#Set the defaults up for CC calculation
        # Load mean-field information from PySCF object
        self.hf_e=pyscf_mf.e_tot
        self.nuc_e=pyscf_mf.energy_nuc()
        
        # Initialize basics of CC calculation
        self.max_iter=cc_info.get("max_iter",75)
        self.dump_tamps=cc_info.get('dump_tamps',False)
        self.dropcore=cc_info.get('dropcore',0)
        self.stopping_eps=cc_info.get("stopping_eps","10**-8")
        self.diis_size=cc_info.get("diis_size")
        self.diis_start_cycle=cc_info.get("diis_start_cycle")

        # Initialize data dictionaries
        self.occInfo=None
        self.occSliceInfo=None
        self.denomInfo=None
        self.integralInfo=None

        if "slowSOcalc" in cc_info:
            self.cc_calcs=cc_info.get("slowSOcalc",'CCD')
            self.get_occInfo(pyscf_mf)



        print(cc_info)



    def get_occInfo(self,pyscf_mf):
        dropcore=self.dropcore
        print('dropcore:',dropcore)
        if 'RHF' in str(type(mf)): # running RHF calculation
            if dropcore>0: 
                print('dropcore not implemented for RHF')
                sys.exit()
            occ = pyscf_mf.mo_occ
            nele = int(sum(occ))
            nocc = nele // 2
            norbs = pyscf_mf.get_fock().shape[0] #oei.shape[0]
            nsvirt = 2 * (norbs - nocc)
            nsocc = 2 * nocc
            self.occInfo={"nocc_aa":nsocc,"nvirt_aa":nsvirt}
    
        elif 'UHF' in str(type(pyscf_mf)): # running UHF calculation
            norbs=pyscf_mf.get_fock().shape[0] + pyscf_mf.get_fock().shape[1]
            na,nb=pyscf_mf.nelec
            nele=na+nb-2*dropcore
            nsvirt = 2 * (pyscf_mf.get_fock()[0].shape[0] - na)#(norbs - nocc)
            self.occInfo={"nocc_aa":nele,"nvirt_aa":nsvirt}
    
    
        n = np.newaxis
        o = slice(None, nele)
        v = slice(nele, None)
        self.occSliceInfo={"occ_aa":o,"virt_aa":v}



    def print(self):
        print(self.cc_calcs)



class DriveCC(SetupCC):
    def __init__(self,pyscf_mf,cc_info):
        SetupCC.__init__(self,pyscf_mf,cc_info)
        print(self.cc_calcs)


