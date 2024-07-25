

def D1denomSlow(epsaa,occ_aa,virt_aa,n):
    D1=1.0/(-epsaa[virt_aa,n]+epsaa[n,occ_aa])
    D1=D1.transpose(1,0)
    return D1

def D2denomSlow(epsaa,occ_aa,virt_aa,n):
    D2=1.0 / (
        -epsaa[virt_aa, n, n, n]
        - epsaa[n, virt_aa, n, n]
        + epsaa[n, n, occ_aa, n]
        + epsaa[n, n, n, occ_aa]
    )
    D2=D2.transpose(2,3,0,1)
    return D2

def D3denomSlow(epsaa,occ_aa,virt_aa,n):
    D3 = 1.0/(
            -epsaa[virt_aa,n,n,n,n,n]
            -epsaa[n,virt_aa,n,n,n,n]
            -epsaa[n,n,virt_aa,n,n,n]
            +epsaa[n,n,n,occ_aa,n,n]
            +epsaa[n,n,n,n,occ_aa,n]
            +epsaa[n,n,n,n,n,occ_aa] )
    D3=D3.transpose(3,4,5,0,1,2)
    return D3


def D4denomSlow(epsaa,occ_aa,virt_aa,n):
    D4=1.0/(
            -epsaa[virt_aa, n, n, n, n, n, n, n]
           -epsaa[n,      virt_aa, n, n, n, n, n, n]
           -epsaa[n, n,           virt_aa, n, n, n, n, n]
           -epsaa[n, n, n,                virt_aa, n, n, n, n]
           +epsaa[n, n, n, n, occ_aa, n, n, n]
           +epsaa[n, n, n, n, n,       occ_aa, n, n]
           +epsaa[n, n, n, n, n, n,            occ_aa, n]
           +epsaa[n, n, n, n, n, n, n,                 occ_aa] )
    D4=D4.transpose(4,5,6,7,0,1,2,3)
    return D4
