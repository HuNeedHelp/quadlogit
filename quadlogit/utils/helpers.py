#----------------------------------------------------------
# Function for constructing quadruples and computing SEs  
#----------------------------------------------------------
#
# Note:
# --- Below functions are based on the matlab file provided
#     in the supplemental material of Joachmans (2019), and
#     has translated to Python, and has been largely 
#     accerelated using tensor computations.

import itertools as it
import numpy as np
import scipy as sp
import pkg_resources 
from scipy import io

# Constructing quadruples
def rearragement_fast(G,u,rearragement_index,N):
    rho = int(((N*(N-1))/2)*((N-2)*(N-2-1))/2)
    G1diff = G[rearragement_index[:,0],rearragement_index[:,1]] - G[rearragement_index[:,0],rearragement_index[:,3]]
    G2diff = G[rearragement_index[:,2],rearragement_index[:,1]] - G[rearragement_index[:,2],rearragement_index[:,3]]
    ss = ((G1diff>0) * (G2diff<0)) + ((G1diff<0) * (G2diff>0))
    zz = (G1diff-G2diff)
    rr = (u[rearragement_index[:,0],rearragement_index[:,1]] - u[rearragement_index[:,0],rearragement_index[:,3]])-( u[rearragement_index[:,2],rearragement_index[:,1]] - u[rearragement_index[:,2],rearragement_index[:,3]])
    return zz/2, rr, ss
# Computing SEs -- fast version.
def standarderror_fast(beta_QL,G,u,m_star,permutations,N):
    rho = int(((N*(N-1))/2)*((N-2)*(N-2-1))/2); pn = m_star/rho
    r = np.zeros((N,N,N,N),dtype='int8')
    a = np.zeros((N,N,N,N),dtype='int8')
    c = np.zeros((N,N,N,N),dtype='int8')
    G1diff = G[permutations[:,0],permutations[:,1]] - G[permutations[:,0],permutations[:,3]]
    G2diff = G[permutations[:,2],permutations[:,1]] - G[permutations[:,2],permutations[:,3]]
    c[permutations[:,0],permutations[:,1],permutations[:,2],permutations[:,3]] = (((G1diff>0) * (G2diff<0)) + ((G1diff<0) * (G2diff>0)))
    a[permutations[:,0],permutations[:,1],permutations[:,2],permutations[:,3]] = ((G1diff>0) * (G2diff<0))
    r[permutations[:,0],permutations[:,1],permutations[:,2],permutations[:,3]] = (u[permutations[:,0],permutations[:,1]] - u[permutations[:,0],permutations[:,3]])-(u[permutations[:,2],permutations[:,1]] - u[permutations[:,2],permutations[:,3]]) 

    # standard GMM - conditional likelihood
    eee = np.exp(r*beta_QL)
    FFF = eee/(1+eee) 
    rc=r*c
    xi = 4*np.sum(np.sum(((a- FFF)*rc),axis=3),axis=2)/((N-2)*(N-3)); V = np.sum(xi**2)/(N*(N-1))
    Q  =   (-r*(FFF/(1+eee))*rc).sum()/(N*(N-1)*(N-2)*(N-3))

    Q = Q/pn; V = V/pn;

    W = (1/(Q.T*Q).T)*(Q.T*V*Q)*(1/(Q.T*Q)); se1 = np.sqrt( W /(N*(N-1)*pn) )
    W = (1/(Q))        *V*      (1/(Q));     se2 = np.sqrt( W /(N*(N-1)*pn) )
    return np.array([se1]), np.array([se2])


# Read data files of the pre-created indice for QL, for each setting of N
def generate_quad_indices(N):
    path = pkg_resources.resource_filename('quadlogit', 'utils/')
    rearragement_index=sp.io.loadmat(path+'N100_rearragement_index.mat')
    permutations=sp.io.loadmat(path+'N100_permutations.mat')
    # Adjust the index: starting from 0 instead of 1
    rearragement_index=rearragement_index.get('rearragement_index')-1
    permutations=permutations.get('permutations')-1
    # Adjust the index for the current working dataset 
    if N<100:
        # While we currently load indices for N=100, it is possible to adjust them for N<100.
        rearragement_index = rearragement_index[rearragement_index[:,0]<N]
        rearragement_index = rearragement_index[rearragement_index[:,1]<N]
        rearragement_index = rearragement_index[rearragement_index[:,2]<N]
        rearragement_index = rearragement_index[rearragement_index[:,3]<N]

        permutations = permutations[permutations[:,0]<N]
        permutations = permutations[permutations[:,1]<N]
        permutations = permutations[permutations[:,2]<N]
        permutations = permutations[permutations[:,3]<N]

    rearragement_index=rearragement_index.astype(np.intp)
    permutations=permutations.astype(np.intp)

    return rearragement_index, permutations



