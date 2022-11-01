import numpy as  np 
import random 
from LinearPnP import LinearPnP

def PnPRANSAC(X,x,K,iterations=100,threshold=0.01):
    R_best= []
    C_best= []
    n= 0 
    N = X.shape[0] 
    I = np.identity(3)
    X = np.hstack((X, np.ones((X.shape[0], 1)))).T 
    for i in iterations:
        rdm_idx_12 = random.sample(range(N),12) 
        X12 = X[rdm_idx_12]
        x12 = x[rdm_idx_12]
        R,C = LinearPnP(X12,x12,K)
        P = np.dot(K, np.dot(R, np.hstack((I, -C))))
        Errors = (P[0].dot(X)/P[2].dot(X)-x[:,0].T)**2 + (P[1].dot(X)/P[2].dot(X)-x[:,1])**2
        Errors[Errors<=threshold] = 1
        Errors[Errors>threshold] = 0 
        count= np.sum(Errors)
        if count> n :
            n = count 
            R_best = R
            C_best = C
    return R_best , C_best 




















