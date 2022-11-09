import numpy as np

def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    x1 = np.hstack((x1, np.ones((x1.shape[0], 1)))) #Appending one to x1 and x2 
    x2 = np.hstack((x2, np.ones((x2.shape[0], 1))))
    C1 = C1.reshape((3,1))
    C2 = C2.reshape((3,1))
    I = np.identity(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))  # the P is written this way but not as P = K[R T] because the C and R here means the rotation and translation between cameras 
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))
    # Now we need to find the skew symmetric matrix of these coordinates 
    skew = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    X = []
    for i,(x1i, x2i) in enumerate(zip(x1,x2)):
        # Now find  the dot product of this with P1 and P2 and stack them 
        s1 = skew(x1i).dot(P1) 
        s2 = skew(x2i).dot(P2)
        A = np.vstack((s1,s2))
        # find svd of A 
        u, s, vh  = np.linalg.svd(A)
        x = vh[np.argmin(s),:] 
        x = x/x[3]  
        X.append(x)
    X = np.vstack(X) 
    return X

if __name__=="__main__":
    C1 = np.zeros((1,3))
    R1 = np.eye(3)

