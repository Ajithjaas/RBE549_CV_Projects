import numpy as  np 
import random 
from LinearPnP import LinearPnP

def PnPRANSAC(X,x,K,iterations=1000,threshold=0.01):
    N = X.shape[0] 
    rdm_idx_12 = np.transpose(random.sample(range(N),12)) 
    X12 = X[rdm_idx_12]
    x12 = x[rdm_idx_12]
    R_best,C_best = LinearPnP(X12,x12,K) #initializing some R, C 
    n= 0 
    I = np.identity(3)
    X = np.hstack((X, np.ones((X.shape[0], 1)))) 
    XT = X.T
    for i in range(iterations):
        rdm_idx_12 = np.transpose(random.sample(range(N),12)) 
        X12 = X[rdm_idx_12][:,:-1]
        x12 = x[rdm_idx_12]
        R,C = LinearPnP(X12,x12,K)
        P = np.dot(K, np.dot(R, np.hstack((I, -C))))
        Errors = (P[0].dot(XT)/P[2].dot(XT)-x[:,0].T)**2 + (P[1].dot(XT)/P[2].dot(XT)-x[:,1])**2
        # print(Errors)
        # input("Close HERE")
        Errors = Errors/np.linalg.norm(Errors) #normalize the error 
        # print(Errors)
        # input()

        err1 = Errors<=threshold
        err0 = Errors>threshold
        Errors[err1] = 1
        Errors[err0] = 0 
        inliers= np.sum(Errors)
        if inliers> n :
            n = inliers 
            R_best = R
            C_best = C
    return R_best , C_best 

if __name__ == "__main__":
    X= np.random.randint(1000,size= (1000,3))
    x = np.random.randint(1000,size= (1000,2))
    K = np.loadtxt("P3Data/calibration.txt")
    print(PnPRANSAC(X,x,K))




















