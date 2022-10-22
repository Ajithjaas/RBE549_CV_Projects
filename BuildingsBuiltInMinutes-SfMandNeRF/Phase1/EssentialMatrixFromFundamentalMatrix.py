import numpy as np
def EssentialMatrixFromFundamentalMatrix(K, F):
    E = np.transpose(K).dot(F.dot(K))
    u, s, vh = np.linalg.svd(E)
    s = np.diag([1,1,0])
    return u.dot(s.dot(np.transpose(vh)))

    