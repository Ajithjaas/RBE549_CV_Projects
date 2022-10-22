from EstimateFundamentalMatrix import EstimateFundamentalMatrix
import numpy as np
import cv2
import random 

def GetInliersRANSAC(M1,M2,iterations=100,threshold=0.01):
    rows = M1.shape[0] 
    x1 = np.concatenate([M1, np.ones((M1.shape[0],1),dtype=M1.dtype)], axis=1)
    x2 = np.transpose(np.concatenate([M2, np.ones((M2.shape[0],1),dtype=M2.dtype)], axis=1))
    F = [] 
    n= 0 
    for i in range(iterations):
        #get random choices from matches 
        eight_random_idx = random.sample(range(rows),4) 
        m1_eight = M1[eight_random_idx,:]
        m2_eight = M2[eight_random_idx,:]
        f = EstimateFundamentalMatrix(m1_eight, m2_eight)
        Errors = np.abs(x2.dot(f.dot(x1)))
        Errors[Errors<=threshold] = 1
        Errors[Errors>threshold] = 0 
        inliers= np.sum(Errors)
        if inliers> n :
            n = inliers 
            F = f
    return F

