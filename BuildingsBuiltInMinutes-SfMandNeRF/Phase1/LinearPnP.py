import numpy as np

def LinearPnP(X, x, K):
    A =[]
    for i,(Xi,xi) in enumerate(zip(X,x)):
        Xx,Xy,Xz = Xi 
        xx,xy = xi 
        A.append([Xx,Xy,Xz,1,0,0,0,0,-xx*Xx,-xx*Xy,-xx*Xz,-xx])
        A.append([0,0,0,0,Xx,Xy,Xz,1,-xy*Xx,-xy*Xy,-xy*Xz,-xy])
    
    # To get the values of P , now do the SVD 
    u, s, vh = np.linalg.svd(A)
    p = vh[np.argmin(s),:]
    p = np.reshape(p,(len(p), 1)) # this p vector has to be arranges such a way that we get original P 
    K_inv = np.linalg.inv(K) 
    P = np.reshape(p,(3,4))
    U,D,VT = np.linalg.svd(K_inv.dot(P[0:3,0:3]))
    R = U.dot(VT)
    T = K_inv.dot(P[:,3:])/D[0]
    if np.linalg.det(R) < 0:
        R = -R
        T = -T
    return R,T 

if __name__ == "__main__":
    X= np.random.randint(1000,size= (12,3))
    x = np.random.randint(1000,size= (12,2))
    K = np.loadtxt("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/BuildingsBuiltInMinutes-SfMandNeRF/P3Data/calibration.txt")
    print(LinearPnP(X,x,K))