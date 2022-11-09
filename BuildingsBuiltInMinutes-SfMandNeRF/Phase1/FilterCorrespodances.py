import numpy as np 

def FilterCorrespondences(x_sos,X,x_sos_new,x_new_sos):

    """x_sos- the image coordinates from one of the images at start of the service . As the new images are being added, this can be from any image 
    X- global coordinates  at the start of the service 
    x_sos_new - correspondence between one image from start of the service and the new image for example x1 read from macthing13.txt file 
    x_new_sos - x3 read from matching13.txt file 
    """
    ind_x_sos_new = []
    ind_x_sos =[]
    for i in x_sos_new:

        ind = np.argwhere(i==x_sos)

        if len(ind)>0:
            ind_x_sos.append(ind[0][0])
            ind_x_sos_new.append(np.argwhere(i==x_sos_new)[0][0])

    # finally return the global coordinates that are already computed at SoS but correspond to new image , and the filtered correspondences  

    return X[ind_x_sos], x_new_sos[ind_x_sos_new]  # this data cna be passed as it is to PnP RANSAC .........Hopefully! 



if __name__ == "__main__":

    match1 = np.loadtxt('P3Data/matches/matches12.txt')
    x12 = match1[:,:2]
    x21 = match1[:,2:4]

    match1 = np.loadtxt('P3Data/matches/matches13.txt')
    x13 = match1[:,:2]
    x31 = match1[:,2:4]

    X= np.random.randint(100,size=(x12.shape[0],3))

    print(FilterCorrespondences(x12,X,x13,x31))