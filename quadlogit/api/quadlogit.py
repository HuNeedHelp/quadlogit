"""
Created on Thu Sep 25 2025

Last modified on Fri Oct 10 2025

Authors: Shiran Hu, Muyang Guo, Xinran Cheng, Xuan Zhou
Supervised by: Zizhong Yan
"""
#----------------------------------------------------------
# Load library dependencies
#----------------------------------------------------------
import sys
import numpy as np
import scipy as sp
import pandas as pd
import time
from scipy import stats
from ..lib.ipt import logit
from ..utils.helpers import rearrangement_fast
from ..utils.helpers import standarderror_fast
from ..utils.helpers import generate_quad_indices
#----------------------------------------------------------
# Class of various network regression
#----------------------------------------------------------
class fit:
    """
        Quadruple Logit Regression for Directed Network Formation Models

        A class for estimating parameters of directed network formation models using 
        the quadruple logit methodology. This approach constructs informative quadruples 
        from network adjacency data and estimates the model via conditional logit regression.

        Parameters
        ----------
        G : ndarray
            Adjacency matrix of shape (N, N) representing the directed network.
            Must be a 2D binary array where G[i,j] = 1 indicates a link from i to j.
            G[i,j] = 0 indicates no link.
            
        X : ndarray, optional
            Covariates for network formation. Can be:
            - 2D array of shape (N, N) for a single covariate
            - 3D array of shape (N, N, K) for K covariates
            Currently only the first covariate (X[:,:,0]) is used in estimation.
            Default is None.
            
        X_names : list of str, optional
            Names of the covariates for display in estimation results.
            If None and X is provided, automatic names "X1", "X2", etc. are generated.
            Default is None.
            
        silent : bool, optional
            If True, suppresses the printing of estimation results summary.
            If False, prints detailed regression output.
            Default is False.
            
        indices : tuple, optional
            Precomputed quadruple indices (rearranges, permutations) to avoid 
            recomputation. Currently not actively used in the implementation.
            Default is None.

        Attributes
        ----------
        N : int
            Number of agents (network size).
            
        Nchoose4 : int
            Number of informative quadruples used in estimation.
            
        paras : ndarray
            Estimated coefficients from quadruple logit regression.
            
        se : ndarray
            Standard errors of the estimated coefficients.
            
        success : int
            Indicator of estimation success (1 if successful, 0 otherwise).

        Notes
        -----
        - The adjacency matrix G must be binary (containing only 0s and 1s).
        - The estimation is based on Graham's ipt package (netrics).
        - Currently only the first covariate from 3D X arrays is utilized.
        - The method constructs all possible quadruples (i1, j1, i2, j2) where 
        i1 < i2 and j1 != j2, filters for informative quadruples (where 
        the network links have opposite patterns), and estimates parameters 
        via conditional logit.

        Examples
        --------
        >>> import numpy as np
        >>> G = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        >>> X = np.random.randn(3, 3)
        >>> model = fit(G, X=X, X_names=['Distance'])
    """
    def __init__(self, G, X=None, X_names=None, silent=False, indices=None):
        #----------------------------------------------------------
        # Preparations
        #----------------------------------------------------------
        # [> Check compatibility of input variables <]
        self.N = np.shape(G)[0]
        # Check adjacency matrix
        if G.shape[0]!=G.shape[1]: 
            sys.exit("Error: adjacency matrix G is not a square matrix.")
        if G.ndim!=2: 
            sys.exit("Error: adjacency matrix G is not a 2d NumPy array. Please see the helpfile: help(fit)")
        if np.all(np.unique(G)==np.array([0,1]))!=True:
            sys.exit("Error: adjacency matrix G is not correctly defined, or is not binary.")
        # Check covariates
        if X is not None: 
            # For a three dim X (multiple covariates), we currently only consider the first covariate
            if X.ndim==2: X=X[:,:,None] 
        # [> Change all input variables to float 64bit <]
        if G.dtype != 'float64': G = G.astype('float64')
        if X is not None:  
            if X.dtype != 'float64': X = X.astype('float64')
        #----------------------------------------------------------
        # Pre-Estimation 1 - Read indices
        #----------------------------------------------------------
        start_time = time.time()
        # if indices is None:
        #     if silent is False: 
        #         print("Rearrangement/permutation options not specified (quadruple-logit indices were not preloaded).")
        #         print("For N <= 100, indices can be loaded automatically.")
        #     rearranges, permutations = generate_quad_indices(self.N) 
        # if indices is not None:
        #     rearranges, permutations = indices
        #----------------------------------------------------------
        # Pre-Estimation 2 - Construct quadruples
        #----------------------------------------------------------
        # Construct quadruples
        # zz,rr,ss=rearrangement_fast(G,X[:,:,0],rearranges,self.N) # For a three dim X (multiple covariates), we currently only consider the first covariate
        zz, rr, ss = rearrangement_fast(G, X[:,:,0], self.N)
        # Drop non-informative quadruples
        zzz = zz[ss==1].reshape(-1,1); zzz = (zzz+1)/2;
        rrr = rr[ss==1].reshape(-1,1);
        # LHS (dependent) variable in quadruple logit
        lhs =  pd.Series(zzz.reshape(-1), name='lhs')
        # RHS (independent) variable in quadruple logit
        rhs =  pd.DataFrame()
        rhs['rhs'] = rrr.reshape(-1)
        self.Nchoose4 = rhs.shape[0]
        #----------------------------------------------------------
        # Estimation
        #----------------------------------------------------------
        # Estimation is based on Graham's ipt package (https://github.com/bryangraham/netrics)
        self.success = 1
        try:
            self.paras, _, _, _, _, self.success= logit(lhs,rhs, nocons=True , s_wgt=None, silent=True, full=False)
            _, self.se = standarderror_fast(self.paras.reshape(-1,1),G,X[:,:,0],np.sum(ss),self.N) # For a three dim X (multiple covariates), we currently only consider the first covariate
        except:
            self.success = 0
            sys.exit("Estimation failed.")
        end_time = time.time() - start_time
        #----------------------------------------------------------
        # Broadcasting
        #----------------------------------------------------------
        if silent is False: 
            print("",)
            print("--------------------------------------------------------------------------------")            
            print("---- ESTIMATION RESULTS --------------------------------------------------------")             
            print("        DIRECTED NETWORK FORMATION MODEL -- QUADRUPLE LOGIT REGRESSION")
            print("--------------------------------------------------------------------------------")            
            print("Number of agents: %3s                       Number of quadruples: %3s" % (self.N,rhs.shape[0]))
            print("                                            Time spent (seconds): %5.3f" % end_time)
            print("--------------------------------------------------------------------------------")            
            print("Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]")
            print("--------------------------------------------------------------------------------")            
            if X is not None:
                if X_names is None:
                    X_names = []
                    for kk in range(0,np.shape(X)[2]):
                        X_names.append("X" + str(kk+1))
            for kk in range(0,np.shape(X)[2]):
                print("%20s%15s%14s%8.3f%12s%11s" % (X_names[kk][:15],
                                            str(self.paras[kk])[:11],
                                            str(self.se[kk])[:10],
                                            2*sp.stats.norm.sf(abs(self.paras[kk]/self.se[kk])),
                                            str(self.paras[kk]-1.9599*self.se[kk])[:8]  ,
                                            str(self.paras[kk]+1.9599*self.se[kk])[:8]  ))
            print("--------------------------------------------------------------------------------")     
            print("")
#not oOo