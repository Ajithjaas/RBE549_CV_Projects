import numpy as np
# from scipy.io import  loadmat 
def EssentialMatrixFromFundamentalMatrix(K, F):
    E = np.transpose(K).dot(F.dot(K))
    u, s, vh = np.linalg.svd(E)
    s = np.diag([1,1,0])
    return u.dot(s.dot(np.transpose(vh)))

if __name__ == "__main__":
    K = np.loadtxt("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/BuildingsBuiltInMinutes-SfMandNeRF/P3Data/calibration.txt")
    F = np.random.randint(10, size = (3,3)) 
    print(EssentialMatrixFromFundamentalMatrix(K,F))

