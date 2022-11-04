import numpy as np 
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation 

def BundleAdjustment(X,x,Rs,Cs,K,V):
    """X- all the global coordinates captured by all the cameras
    Rs - all the rotation matrices
    Cs - all the camera centers 
    K - camera caliberation matrix
    V - Visibility matrix  - assuming I already have this 
    x - contains the image coordinate of each camera . so this is a 3D array 
    """
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    J = X.shape[0] # Total number of global points 
    I = Rs.shape[0] # number of rotaions . this is same as number of Cs or same as number of cameras 
    XRCs0 = X.flatten()
    # print(XRCs0)
    for R in Rs:
        XRCs0 = np.hstack((XRCs0,Rotation.from_matrix(R).as_quat()))
    for C in Cs:
        XRCs0 = np.hstack((XRCs0,C.flatten()))
    # Here XRCs is the initial parameter array 
    optimized = least_squares(fun = Error,x0=XRCs0,args=[x,K,V,I,J])
#     print(optimized.x)
    XRCs = optimized.x

    X = [XRCs[i] for i in range(J*4)]
    X = np.reshape(X,(J,4)) # contains all the X values 
    XRCs = XRCs[J*4:] #everything remaining are Rs and Cs
    Rs = [Rotation.from_quat(XRCs[i:i+4]).as_matrix() for i in range(0,I*4,4) ] 
    XRCs = XRCs[I*4:] #eveything remains is Cs 
    Cs = [XRCs[i:i+3] for i in range(0,I*3,3)] 

    return Rs, Cs , X

def Error(x0,x,K,V,I,J):

    # X,Rs and Cs has to be extracted back from the initial values 
    X = [x0[i] for i in range(J*4)]
    X = np.reshape(X,(J,4)) # contains all the X values 
    x0 = x0[J*4:] #everything remaining are Rs and Cs
    Rs = [Rotation.from_quat(x0[i:i+4]).as_matrix() for i in range(0,I*4,4) ] 
    x0 = x0[I*4:] #eveything remains is Cs 
    Cs = [x0[i:i+3] for i in range(0,I*3,3)] 
    E =0 
    I3 = np.identity(3) # Identity matrix
    XT = X.T
    for i,(R,C) in enumerate(zip(Rs,Cs)):
        P = np.dot(K, np.dot(R, np.hstack((I3, -C.reshape((3,1))))))
        e = (P[0].dot(XT)/P[2].dot(XT)-x[i][:,0])**2 + (P[1].dot(XT)/P[2].dot(XT)-x[i][:,1])**2
        E+= V[i].dot(e)
    return E 

if __name__ == "__main__":

    Rs = np.random.randint(10,size=(4,3,3))
    Cs = np.random.randint(100,size=(4,3,1))
    K = np.random.randint(10,size=(3,3))
    X = np.random.randint(100,size= (100,3))
    x = np.random.randint(100,size= (Rs.shape[0],X.shape[0],2))

    V = np.random.randint(2,size=(Rs.shape[0],X.shape[0]))
    print(Rs,Cs)
    print("______________")
    print(X)
    print(BundleAdjustment(X,x,Rs,Cs,K,V))
