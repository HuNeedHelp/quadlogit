import time
import numpy as np
import quadlogit


# ======================================================================
# 优化尝试rearragement_fast

def rearragement_fast(G, u):
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
    rr = rr[:, total_mask].reshape(-1) if u.shape[0] == 1 else rr[:, total_mask] # shape (num_quadruples,) or (features, num_quadruples)
    
    return zz, rr, ss

N = 45 # 45
G,Xmat,Zmat,density,degree,transitivity,separation,trueParameter=quadlogit.demo.GenData(N, directed=True, mutual=False,specification="A1",seed=111)
# print("Xmat shape: ", Xmat.shape)

time_start = time.time()
zz, rr, ss = rearragement_fast(G, Xmat)
print("Number of elements is: ", len(zz))
print("Time taken: ", time.time() - time_start)

# 45
# Number of elements is:  1960200
# Time taken:  0.06407690048217773

# N: 100
# Number of elements is:  49005000
# Time taken:  2.385293960571289

# N: 150
# Number of elements is:  249761250
# Time taken:  58.128456592559814

# N: 200
# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 11.9 GiB for an array with shape (1, 200, 200, 200, 200) and data type float64