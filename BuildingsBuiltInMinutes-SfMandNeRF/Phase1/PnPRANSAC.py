import numpy as  np 
import random 
from LinearPnP import LinearPnP

def PnPRANSAC(X,x,K,iterations=1000,threshold=1):
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
        Errors[Errors<=threshold] = 1
        Errors[Errors>threshold] = 0 
        count= np.sum(Errors)
        if count> n :
            n = count 
            R_best = R
            C_best = C
    return R_best , C_best 

if __name__ == "__main__":
    X= np.random.randint(1000,size= (1000,3))
    x = np.random.randint(1000,size= (1000,2))
    K = np.loadtxt("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/BuildingsBuiltInMinutes-SfMandNeRF/P3Data/calibration.txt")
    print(PnPRANSAC(X,x,K))




















