import numpy as np

def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))
    P1 = K.dot(np.hstack((R1, C1)))
    P2 = K.dot(np.hstack((R2, C2)))


