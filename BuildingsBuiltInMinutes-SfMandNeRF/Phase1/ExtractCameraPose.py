import numpy as np

def ExtractCameraPose(E):
    U,D,VT = np.linalg.svd(E) 
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = [U.dot(W.dot(VT)),
         U.dot(W.dot(VT)),
         U.dot(np.transpose(W).dot(VT)),
         U.dot(np.transpose(W).dot(VT))] 
    C=[U[:, 2],-U[:, 2],U[:, 2],-U[:, 2]]
    
    R = [-R[i] if(np.linalg.det(R[i]) < 0) else R[i] for i in range(4)]
    C = [-C[i] if(np.linalg.det(R[i]) < 0) else C[i] for i in range(4)]
    return R, C
