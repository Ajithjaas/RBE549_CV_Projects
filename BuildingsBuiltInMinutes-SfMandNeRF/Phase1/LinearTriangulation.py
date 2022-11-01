import numpy as np

def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))

    I = np.identity(3)

    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))  # the P is written this way but not as P = K[R T] because the C and R here means the rotation and translation between cameras 
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    x1 = np.hstack((x1, np.ones((x1.shape[0], 1)))) #Appending one to x1 and x2 
    x2 = np.hstack((x2, np.ones((x1.shape[0], 1))))

    # Now we need to find the skew symmetric matrix of these coordinates 
    s = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])
    X = np.zeros((x1.shape[0], 3))
    for i,(x1_i,x2_i) in enumerate(zip(x1,x2)):
        s1 = s(x1_i)
        s2 = s(x2_i)
        # Now we need to stack the dot product of this with P1 and P2 and stack them 
        A = np.vstack((np.dot(s1, P1), np.dot(s2, P2)))
        # find svd of A 
        u, s, vh = np.linalg.svd(A)
        x = vh.T[:, -1]
        x = np.reshape(x,(len(x), 1))
        X[i, :] = x

    return X 
