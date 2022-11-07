import numpy as np
from scipy.optimize import least_squares

def NonLinearTriangulation(K,R1,C1,R2,C2,x1, x2, X):
    I = np.identity(3)
    C1 = C1.reshape((3,1))
    C2 = C2.reshape((3,1))
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1)))) 
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))
    X_new = [] 
    for i,(Xi,x1i,x2i)  in enumerate(zip(X,x1,x2)):
        optimized= least_squares(fun=Error, x0=Xi, args=[P1, P2,x1i,x2i])
        X_new.append(optimized.x)
    X_new = np.array(X_new)
    h= X_new[:,3]
    h = h.reshape((h.shape[0],1))  # Homogenized coordinates 
    return X_new/h

def Error(X,P1,P2,x1,x2):
    # X = np.hstack((X,1)).T
    E1 = (P1[0].dot(X)/P1[2].dot(X)-x1[0])**2 + (P1[1].dot(X)/P1[2].dot(X)-x1[1])**2
    E2 = (P2[0].dot(X)/P2[2].dot(X)-x2[0])**2 + (P2[1].dot(X)/P2[2].dot(X)-x2[1])**2
    return E1+E2 


    
