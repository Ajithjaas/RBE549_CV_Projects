import numpy as np

def DisambiguateCameraPose(Rs,Cs,Xs):
    best_n =0 
    best_idx =0 
    for i in range(len(Rs)):
        R,C = R[i],C[i]
        r3 = R[2] #get third row of R
        n=0 
        for X in Xs:
            if r3.dot(X-C)>0:
                n+=1
        if n > best_n:
            best_n = n 
            best_idx = i 
    return Rs[best_idx],Cs[best_idx] , Xs[best_idx] 



