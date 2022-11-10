import numpy as np 

def FilterCorrespondences(x_sos,X,x_sos_new,x_new_sos):

    """x_sos- the image coordinates from one of the images at start of the service . As the new images are being added, this can be from any image 
    X- global coordinates  at the start of the service 
    x_sos_new - correspondence between one image from start of the service and the new image for example x1 read from macthing13.txt file 
    x_new_sos - x3 read from matching13.txt file 
    """
    ind =[]
    indx = []
    for i,x in enumerate(x_sos_new):
        try:
            ind.append(np.argwhere(np.isin(x_sos, x).all(axis=1)).flatten()[0])
            indx.append(i)
        except:
            pass
    return X[ind][:,:3],x_sos_new[indx],x_new_sos[indx]



if __name__ == "__main__":
    # from PnPRANSAC import PnPRANSAC

    # match1 = np.loadtxt('P3Data/matches/matches12.txt')
    # x12 = match1[:,:2]
    # x21 = match1[:,2:4]

    # match1 = np.loadtxt('P3Data/matches/matches13.txt')
    # x13 = match1[:,:2]
    # x31 = match1[:,2:4]

    # X= np.random.randint(100,size=(x12.shape[0],3))

    # X,x1,x2 = FilterCorrespondences(x12,X,x13,x31)
    # K = np.loadtxt('P3Data/calibration.txt')

    # print(PnPRANSAC(X,x2,K))

    x12 = np.random.randint(10,size=(10,2))
    X = np.random.randint(10,size=(10,3))
    X = np.hstack((X,np.ones((X.shape[0],1))))

    ind = np.random.randint(x12.shape[0],size=(5))

    x13 = x12[ind]
    x31 = np.random.randint(10,size=(x13.shape[0],2))
    # print("x12",x12)
    # print("X", X)
    # print("x13",x13)
    # print("x31",x31)
    print(FilterCorrespondences(x12,X,x13,x31))





