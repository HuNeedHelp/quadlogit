"""
Created on Thu Sep 25 2025

Last modified on Fri Oct 10 2025

Authors: Zizhong Yan
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
from ..utils.helpers import rearragement_fast
from ..utils.helpers import standarderror_fast
from ..utils.helpers import generate_quad_indices
#----------------------------------------------------------
# Class of various network regression
#----------------------------------------------------------
class fit:
    """
    ---to be add a helpfile---
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
        if indices is None:
            if silent is False: 
                print("Rearrangement/permutation options not specified (quadruple-logit indices were not preloaded).")
                print("For N <= 100, indices can be loaded automatically.")
            rearranges, permutations = generate_quad_indices(self.N) 
        if indices is not None:
            rearranges, permutations = indices
        #----------------------------------------------------------
        # Pre-Estimation 2 - Construct quadruples
        #----------------------------------------------------------
        # Construct quadruples
        zz,rr,ss=rearragement_fast(G,X[:,:,0],rearranges,self.N) # For a three dim X (multiple covariates), we currently only consider the first covariate
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
            _, self.se = standarderror_fast(self.paras.reshape(-1,1),G,X[:,:,0],np.sum(ss),permutations,self.N) # For a three dim X (multiple covariates), we currently only consider the first covariate
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