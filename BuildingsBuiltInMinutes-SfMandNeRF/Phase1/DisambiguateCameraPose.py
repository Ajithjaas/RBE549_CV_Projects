import numpy as np 

def DisambiguateCameraPose(Rs,Cs,Xs):
    best_n =0 
    best_idx =0 
    for i,R in enumerate(Rs):
        C = np.transpose(Cs[i]).flatten()
        r3 = R[2] #get third row of R
        n=0 
        for X in Xs:
            if r3.dot(X-C)>0:
                n+=1
        if n > best_n:
            best_n = n 
            best_idx = i 
    return Rs[best_idx],Cs[best_idx]

if __name__ == "__main__":

    Rs = np.random.randint(10,size=(4,3,3))
    Cs = np.random.randint(100,size=(4,3,1))
    Xs = np.random.randint(100,size=(1000,3))
    print(DisambiguateCameraPose(Rs,Cs,Xs))
