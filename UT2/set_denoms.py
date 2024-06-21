

def D1denomSlow(epsaa,occ_aa,virt_aa,n):
    D1=1.0/(-epsaa[virt_aa,n]+epsaa[n,occ_aa])
    return D1

def D2denomSlow(epsaa,occ_aa,virt_aa,n):
    D2=1.0 / (
        -epsaa[virt_aa, n, n, n]
        - epsaa[n, virt_aa, n, n]
        + epsaa[n, n, occ_aa, n]
        + epsaa[n, n, n, occ_aa]
    )
    return D2

def D3denomSlow(epsaa,occ_aa,virt_aa,n):
    D3 = 1.0/(
            -epsaa[virt_aa,n,n,n,n,n]
            -epsaa[n,virt_aa,n,n,n,n]
            -epsaa[n,n,virt_aa,n,n,n]
            +epsaa[n,n,n,occ_aa,n,n]
            +epsaa[n,n,n,n,occ_aa,n]
            +epsaa[n,n,n,n,n,occ_aa] )
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
    return D4
