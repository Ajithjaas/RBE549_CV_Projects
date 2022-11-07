from mimetypes import init
import os
import cv2
# import pry
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import argparse
import scipy.io as sio
from GetInliersRANSAC import GetInliersRANSAC
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from NonlinearTriangulation import NonLinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from Plot3D import Plot3D
import glob 


def createMatchFiles(path = 'P3Data/'):
        """
            Function used to create image pair match files
            Input:
                path (Default - 'Phase1/P3Data/matching1.txt')
            Output:
                Saving image pair match file in location
        """
        save_path = "P3Data/matches/"
        for i in range(1,6):
            for j in range(i+1,6):
                with open(save_path+'matches'+str(i)+str(j)+'.txt', 'w') as fp:
                    pass

        for i in range(1,5):
            file_name   = "matching"+str(i)+".txt"
            file        = open(path+file_name,'r')
            content     = file.readlines()
            for line in content[1:]:
                val         = line.split()
                matches     = val[6:]
                for j,match in enumerate(matches):
                    if(j%3==0):
                        file_save   = open(save_path +"matches"+str(i)+str(match)+ ".txt", 'a')
                        points      = val[4] + " " + val[5] + " " + matches[j+1] + " " + matches[j+2] + " " + val[1] + " " + val[2] + " " + val[3] + "\n"
                        file_save.write(points)
                        file_save.close()

def main():
    X = []
    #******* MATCHING FILE PREPARATION *********
    createMatchFiles()
    match1 = np.loadtxt('P3Data/matches/matches12.txt')
    x1 = match1[:,:2]
    x2 = match1[:,2:4]

    #******* CAMERA PARAMETERS *******
    K = np.loadtxt('P3Data/calibration.txt')
    x1_in,x2_in = GetInliersRANSAC(x1,x2,threshold=1)  # Doing only for the first two images 
    print("Inliers size ", x1_in.shape[0])

    F = EstimateFundamentalMatrix(x1_in,x2_in)
    print("Fundamental Matrix is", F)
    # E = EssentialMatrixFromFundamentalMatrix(K,F)
    E, mask = cv2.findEssentialMat(x1_in,x2_in, K, cv2.RANSAC, prob=0.999, threshold=1.0)
    print("Essential Matrix is", E)
    Rs,Cs = ExtractCameraPose(E)
    C1 = np.zeros((1,3)) #first camera's position
    R1 = np.eye(3) # first camera's pose 
    # print(Rs,Cs)
    for i in range(4):
        X.append(LinearTriangulation(K,C1,R1,Cs[i],Rs[i],x1_in,x2_in))
    R,C,X = DisambiguateCameraPose(Rs,Cs,X)
    X = np.array(X)
    print(X)
    fig = plt.figure()
    # ax = fig.add_subplot(projection='2d')
    plt.scatter(X[:, 0], X[:, 2],c="g",s=1,label="Linear")
    
    # plt.show()
    X = NonLinearTriangulation(K,R1,C1,R,C,x1_in, x2_in, X) #Nonlinear Triangulation 
    print("_______")
    print(X)
    plt.scatter(X[:, 0], X[:, 2],c="r",s=1,label="Non Linear")
    plt.legend()
    plt.show()

    Cset =[]
    Rset =[] 
    Cset.append(C)
    Rset.append(R)

    
if __name__ == '__main__':
    main()