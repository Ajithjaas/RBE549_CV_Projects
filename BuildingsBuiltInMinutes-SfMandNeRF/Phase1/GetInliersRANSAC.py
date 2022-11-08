from EstimateFundamentalMatrix import EstimateFundamentalMatrix
import numpy as np
import random 

def GetInliersRANSAC(x1,x2,iterations=1000,threshold=0.2):
    rows = x1.shape[0] 
    x1 = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x2 = np.hstack((x2, np.ones((x2.shape[0], 1))))
    n= 0 
    eight_random_idx = random.sample(range(rows),8) 
    x1_inliers = x1[eight_random_idx,:][:,:2] #initializing random inliers 
    x2_inliers = x2[eight_random_idx,:][:,:2] #initializing random inliers 
    F = EstimateFundamentalMatrix(x1_inliers, x2_inliers)  #initializing random F 
    for i in range(iterations):
        #get random choices from matches 
        eight_random_idx = random.sample(range(rows),8) 
        x18 = x1[eight_random_idx,:][:,:2]
        x28 = x2[eight_random_idx,:][:,:2]
        f = EstimateFundamentalMatrix(x18, x28)
        Errors = np.abs(x2.dot(f.dot(x1.T)))  # it should be x2T.F.x1 , but that way dimensions are not matching 
        Errors = np.mean(Errors, axis=1)
        Errors = np.reshape(Errors,(1,Errors.shape[0]))
        err1 = Errors<threshold
        err0 = Errors>threshold
        Errors[err1] = 1
        Errors[err0] = 0 
        inliers= np.sum(Errors)
        if inliers> n :
            n = inliers 
            F = f
            indices = Errors==1
            x1_inliers = x1[indices[0],:][:,:2]
            x2_inliers = x2[indices[0],:][:,:2]
    return x1_inliers, x2_inliers  # We don't need  inliers anymore , but returning just in case 

if __name__ == "__main__":
    x1 = np.random.randint(100,size=(20,2))
    x2 = np.random.randint(100,size=(20,2))
    print(GetInliersRANSAC(x1,x2))
