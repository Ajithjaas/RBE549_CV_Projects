import numpy as np  

def EstimateFundamentalMatrix(x1,x2):
    A = np.zeros((len(x1),9))
    func = lambda x1,x2: [x1[0]*x2[0], x2[0]*x1[1], x2[0], x2[1]*x1[0], x2[1]*x1[0], x2[1], x1[0], x1[1], 1]
    A = np.array(list(map(func,x1,x2)))
    u,s, vh = np.linalg.svd(A)
    F = vh.T[:, -1]
    F = F.reshape(3,3)
    return F 

if __name__ == "__main__":
    x1 = np.random.randint(10, size = (9,2)) 
    x2 = np.random.randint(10, size = (9,2)) 
    print(EstimateFundamentalMatrix(x1,x2))
