import numpy as np

def LinearPnP(X, x, K):
    A =[]
    for i,Xi,xi in enumerate(zip(X,x)):
        Xx,Xy,Xz = Xi 
        xx,xy,xz = xi 
        A.append([Xx,Xy,Xz,1,0,0,0,0,-xx*Xx,-xx*Xy,-xx*Xz,-xx])
        A.append([0,0,0,0,Xx,Xy,Xz,1,-xy*Xx,-xy*Xy,-xy*Xz,-xy])
    
    # To get the values of P , now do the SVD 
    u, s, vh = np.linalg.svd(A)
    p = vh.T[:, -1]
    p = np.reshape(p,(len(p), 1)) # this p vector has to be arranges such a way that we get original P 
    K_inv = np.linalg.inv(K) 
    P = np.reshape(p,(3,4))
    U,D,VT = np.linalg.svd(K_inv.dot(P[:,0:3]))
    R = U.dot(VT)
    T = K_inv.dot(P[:,3:])/D[0,0]





