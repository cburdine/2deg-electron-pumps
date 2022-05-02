import numpy as np
import scipy.linalg as la

def Bound(vpot, x, meff=0.067): 
    """
    This script is a translation of the 'Bound.m' MATLAB
    script written by Craig Lent. It should be inpirted into
    a Python3 script. This script also requires the numpy package to
    be installed.

    Usage: Solves a 1-D Schodinger bound-state problem using an FEM
           method

    args:
        vpot: Potential at x nodes (numpy array of shape (numnp,))
        x:    x nodes (numpy array of shape (numnp,))
        meff: effective mass coefficient

    returns:
        eigs, vecs
        
        eigs: sorted eigenvalues and eigenvectors
              (numpy array of shape (numnp-2,)
              with units in eV)

        vecs: sorted corresponding unnormalized
              eigenvectors (numpy arrray of shape
              (numnp,numnp-2) )
    """   
    # declare natural constants:
    hbarc = 1973 # hbar times speed of light
    c = 2.998e18 # the speed of light (Angstroms/sec) 
    mc2=511000   # electron rest mass (eV) 
    
    xpref=hbarc*hbarc/(2*meff*mc2)
    
    # allocate arrays
    numnp = x.shape[0]
    VA = np.zeros(shape=(numnp,numnp))
    TA = np.zeros(shape=(numnp,numnp))
    MA = np.zeros(shape=(numnp,numnp))

    # compute entries of FEM arrays:
    for j in range(numnp-1):
        h=x[j+1]-x[j]
        VA[j,j] += h*(vpot[j]/4.0 + vpot[j+1]/12.0)
        VA[j,j+1] += h*(vpot[j]/12.0 + vpot[j+1]/12.0)
        VA[j+1,j] += h*(vpot[j]/12.0 + vpot[j+1]/12.0)
        VA[j+1,j+1] += h*(vpot[j]/12.0 + vpot[j+1]/4.0)
        
        TA[j,j] += 1/h
        TA[j+1,j+1] += 1/h
        TA[j,j+1] -= 1/h
        TA[j+1,j] -= 1/h
        
        MA[j,j] += h/3
        MA[j+1,j+1] += h/3
        MA[j+1,j] += h/6
        MA[j,j+1] += h/6
    
    TA *= xpref    
    # form Hamiltonian matirx:
    HA = TA+VA
    
    # trim first and last cols/rows:
    H = HA[1:-1,1:-1]
    M = MA[1:-1,1:-1]

    # find eigenvectors and eigenvalues & sort them:
    d, V = la.eig(H,M)
    idx = np.argsort(d)
    eigs = np.real(d[idx])
    vecs = np.real(V[:,idx])
    vecs = np.vstack((np.zeros(numnp-2),vecs,np.zeros(numnp-2)))
    
    for iv in range(numnp-2):
        psi = vecs[:,iv]
        norm = np.trapz(np.abs(psi*psi.conj()),x)
        vecs[:,iv] = psi/np.sqrt(norm)

    return vecs, eigs
