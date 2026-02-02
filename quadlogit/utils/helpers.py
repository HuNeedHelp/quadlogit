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
# def rearrangement_fast(G,u,rearragement_index,N):
#     rho = int(((N*(N-1))/2)*((N-2)*(N-2-1))/2)
#     G1diff = G[rearragement_index[:,0],rearragement_index[:,1]] - G[rearragement_index[:,0],rearragement_index[:,3]]
#     G2diff = G[rearragement_index[:,2],rearragement_index[:,1]] - G[rearragement_index[:,2],rearragement_index[:,3]]
#     ss = ((G1diff>0) * (G2diff<0)) + ((G1diff<0) * (G2diff>0))
#     zz = (G1diff-G2diff)
#     rr = (u[rearragement_index[:,0],rearragement_index[:,1]] - u[rearragement_index[:,0],rearragement_index[:,3]])-( u[rearragement_index[:,2],rearragement_index[:,1]] - u[rearragement_index[:,2],rearragement_index[:,3]])
#     return zz/2, rr, ss
def rearrangement_fast(G, u, N):
    G_cols = G.T.reshape(1, N, N, 1)    # (1, col_j1, row_i2, 1)
    G_rows = G.reshape(N, 1, 1, N)      # (row_i1, 1, 1, col_j2)
    G_self = G.reshape(1, 1, N, N)      # (1, 1, row_i2, col_j2)
    G_cross = G.reshape(N, N, 1, 1)     # (row_i1, col_j1, 1, 1)

    if len(u.shape) == 3:
        # u is (N, N, features)
        u = u.transpose(2, 0, 1)   # (features, N, N)
    else:
        u = u.reshape(1, N, N)     # (1, N, N)
    u_cols = u.transpose(0, 2, 1).reshape(-1, 1, N, N, 1) # (features, 1, col_j1, row_i2, 1)
    u_rows = u.reshape(-1, N, 1, 1, N) # (features, row_i1, 1, 1, col_j2)
    u_self = u.reshape(-1, 1, 1, N, N) # (features, 1, 1, row_i2, col_j2)
    u_cross = u.reshape(-1, N, N, 1, 1) # (features, row_i1, col_j1, 1, 1)
    
    G1_diff = G_cross - G_rows
    G2_diff = G_self - G_cols
    ss = ((G1_diff > 0) & (G2_diff < 0)) | ((G1_diff < 0) & (G2_diff > 0))
    zz = G1_diff - G2_diff  # shape (N,N,N,N), - (G_cols + G_rows) + G_cross + G_self
    rr = u_cross - u_rows - u_cols + u_self  # shape (features, N,N,N,N)
    
    # Apply mask to ensure i1 < i2 and j1 != j2
    row_mask = np.triu(np.ones((N, N), dtype=bool), k=1) # i1 < i2
    col_mask = ~np.eye(N, dtype=bool) # j1 != j2
    total_mask = row_mask.reshape(N, 1, N, 1) & col_mask.reshape(1, N, 1, N)
    ss = ss[total_mask] # shape (num_quadruples,)
    zz = zz[total_mask] # shape (num_quadruples,)
    rr = rr[:, total_mask].reshape(-1) if u.shape[0] == 1 else rr[:, total_mask].T # shape (num_quadruples,) or (num_quadruples, features)
    # Drop non-informative quadruples
    zz = zz[ss] / 2
    rr = rr[ss] if len(rr.shape) == 1 else rr[ss, :]
    return (zz + 1)/2, rr, ss


# def rearrangement_fast_optimized(G, u, N):
#     # ---- u reshape ----
#     if u.ndim == 3:
#         u = u.transpose(2, 0, 1)   # (features, N, N)
#     else:
#         u = u.reshape(1, N, N)

#     F = u.shape[0]

#     zz_list = []
#     rr_list = []
#     ss_list = []

#     for i1 in range(N):
#         for i2 in range(i1 + 1, N):
#             # i1 < i2
#             G1_diff = G[i1, :, None] - G[i1, None, :]   # (i1, [j1, 1]) - (i1, [1, j2]) => (N, N)
#             G2_diff = G[i2, :, None] - G[i2, None, :]   # (i2, [j1, 1]) - (i2, [1, j2]) => (N, N)

#             # j1 != j2
#             col_mask = ~np.eye(N, dtype=bool)

#             ss = ((G1_diff > 0) & (G2_diff < 0)) | ((G1_diff < 0) & (G2_diff > 0))
#             ss &= col_mask

#             if not ss.any():
#                 continue

#             zz = (G1_diff - G2_diff)[ss] / 2   # (num_quad,)

#             # rr
#             u = u[None, :, :] if len(u.shape) < 3 else u
#             rr = (
#                 u[:, i1, :, None] - u[:, i1, None, :]
#                 - u[:, i2, :, None] + u[:, i2, None, :]
#             )  # (F, N, N)

#             if F == 1:
#                 rr = rr[0][ss]
#             else:
#                 rr = rr[:, ss].T  # (num_quad, F)

#             zz_list.append(zz)
#             rr_list.append(rr)
#             ss_list.append(ss[ss])

#     zz = np.concatenate(zz_list)
#     rr = np.concatenate(rr_list, axis=0)
#     ss = np.concatenate(ss_list)

#     return zz, rr, ss

# Computing SEs -- fast version.
def standarderror_fast(beta_QL,G,u,m_star,N):
    G_cols = G.T.reshape(1, N, N, 1)    # (1, col_j1, row_i2, 1)
    G_rows = G.reshape(N, 1, 1, N)      # (row_i1, 1, 1, col_j2)
    G_self = G.reshape(1, 1, N, N)      # (1, 1, row_i2, col_j2)
    G_cross = G.reshape(N, N, 1, 1)     # (row_i1, col_j1, 1, 1)
    G1diff = G_cross - G_rows
    G2diff = G_self - G_cols

    if len(u.shape) == 3:
        # u is (N, N, features)
        u = u.transpose(2, 0, 1)   # (features, N, N)
    else:
        u = u.reshape(1, N, N)     # (1, N, N)
    u_cols = u.transpose(0, 2, 1).reshape(-1, 1, N, N, 1) # (features, 1, col_j1, row_i2, 1)
    u_rows = u.reshape(-1, N, 1, 1, N) # (features, row_i1, 1, 1, col_j2)
    u_self = u.reshape(-1, 1, 1, N, N) # (features, 1, 1, row_i2, col_j2)
    u_cross = u.reshape(-1, N, N, 1, 1) # (features, row_i1, col_j1, 1, 1)

    c = (((G1diff>0) * (G2diff<0)) + ((G1diff<0) * (G2diff>0)))
    a = ((G1diff>0) * (G2diff<0))
    r = u_cross - u_rows - u_cols + u_self

    # Apply mask to ensure i1, i2, j1, j2 are all distinct
    mask = np.eye(N, dtype=bool)
    # reshape成(N, N, N, N)的形式，然后应用mask
    mask_4d = mask.reshape(N, 1, N, 1) | mask.reshape(1, N, 1, N) | mask.reshape(N, 1, 1, N) | mask.reshape(1, N, N, 1)
    mask_4d = ~mask_4d
    c = c * mask_4d
    a = a * mask_4d
    r = r * mask_4d   # shape (num_quadruples,) or (num_quadruples, features)
    rho = len(c[mask_4d])  # number of valid quadruples
    pn = m_star/rho

    # rho = int(((N*(N-1))/2)*((N-2)*(N-2-1))/2); pn = m_star/rho
    # r = np.zeros((N,N,N,N),dtype='int8')
    # a = np.zeros((N,N,N,N),dtype='int8')
    # c = np.zeros((N,N,N,N),dtype='int8')
    # G1diff = G[permutations[:,0],permutations[:,1]] - G[permutations[:,0],permutations[:,3]]
    # G2diff = G[permutations[:,2],permutations[:,1]] - G[permutations[:,2],permutations[:,3]]
    # c[permutations[:,0],permutations[:,1],permutations[:,2],permutations[:,3]] = (((G1diff>0) * (G2diff<0)) + ((G1diff<0) * (G2diff>0)))
    # a[permutations[:,0],permutations[:,1],permutations[:,2],permutations[:,3]] = ((G1diff>0) * (G2diff<0))
    # r[permutations[:,0],permutations[:,1],permutations[:,2],permutations[:,3]] = (u[permutations[:,0],permutations[:,1]] - u[permutations[:,0],permutations[:,3]])-(u[permutations[:,2],permutations[:,1]] - u[permutations[:,2],permutations[:,3]]) 

    # standard GMM - conditional likelihood
    eee = np.exp(r*beta_QL) * mask_4d
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



