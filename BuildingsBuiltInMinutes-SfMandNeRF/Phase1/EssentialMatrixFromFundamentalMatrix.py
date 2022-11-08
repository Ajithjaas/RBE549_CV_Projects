import numpy as np
# from scipy.io import  loadmat 
def EssentialMatrixFromFundamentalMatrix(K, F):
    # Should the coordinates be normalized ? 
    E = np.transpose(K).dot(F.dot(K))
    u, s, vh = np.linalg.svd(E)
    s = np.diag([1,1,0])
    return u.dot(s.dot(vh))


if __name__ == "__main__":
    K = np.loadtxt("P3Data/calibration.txt")
    F = np.random.randint(10, size = (3,3)) 
    print(EssentialMatrixFromFundamentalMatrix(K,F))

