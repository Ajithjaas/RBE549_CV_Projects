import numpy as np  
import cv2
import math 

def EstimateFundamentalMatrix(x1,x2):
    A = np.zeros((len(x1),9))
    func = lambda x1,x2: [x1[0]*x2[0], x1[0]*x2[1], x1[0], x2[1]*x1[0], x1[1]*x2[1], x1[1], x2[0], x2[1], 1]
    A = np.array(list(map(func,x1,x2)))
    u,s, vh = np.linalg.svd(A)
    F = vh[:, -1]
    F = F/F[-1]
    F = F.reshape(3,3)
    f, mask = cv2.findFundamentalMat(x1,x2)  # My implementation is wrong , hence using inbuilt function for now 
    return f 

if __name__ == "__main__":
    x1 = np.random.randint(10, size = (9,2)) 
    print(x1)
    x2 = np.random.randint(10, size = (9,2))
    # print(x1)
    # print(x2) 
    print(EstimateFundamentalMatrix(x1,x2))
