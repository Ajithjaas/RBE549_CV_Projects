import numpy as np 
from scipy.spatial.transform import Rotation 
from scipy.optimize import least_squares

def NonlinearPnP(X,x,R,C,K):
    q0 = Rotation.from_matrix(R).as_quat() # Converting rotation matrix to quaternions 
    # We need to minimize loss function w.r.t C and Quaternion 
    C = C.T
    Cq0 = np.hstack((C[0],q0))
    optimized = least_squares(fun = Error,x0=Cq0,args=[X, x,K])
    Cq = optimized.x
    C = Cq[:3]
    q = Cq[3:]
    R = Rotation.from_quat(q).as_matrix() 
    return R, C.T

def Error(Cq0,X,x,K):
    C0 = Cq0[:3]
    C0 = np.expand_dims(C0, axis=1) #for some reason np.transpose isn't working on this so doing this way 
    q0 = Cq0[3:]
    R0 = Rotation.from_quat(q0).as_matrix() 
    I = np.identity(3)
    P = np.dot(K, np.dot(R0, np.hstack((I, -C0))))
    X = np.hstack((X, np.ones((X.shape[0], 1)))).T
    E = (P[0].dot(X)/P[2].dot(X)-x[:,0].T)**2 + (P[1].dot(X)/P[2].dot(X)-x[:,1])**2
    return np.sum(E) 

if __name__ == "__main__":
    from PnPRANSAC import PnPRANSAC
    X= np.random.randint(1000,size= (1000,3))
    x = np.random.randint(1000,size= (1000,2))
    K = np.loadtxt("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/BuildingsBuiltInMinutes-SfMandNeRF/P3Data/calibration.txt")
    R,C = PnPRANSAC(X,x,K)
    print(R,C)
    print(NonlinearPnP(X,x,R,C,K))
