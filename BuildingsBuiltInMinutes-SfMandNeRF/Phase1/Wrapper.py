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
from Plot3D import Plot3D, DrawCameras
import glob 
from FilterCorrespodances import FilterCorrespondences
from PnPRANSAC import PnPRANSAC

from NonlinearPnP import NonlinearPnP 


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
    creatematches =True 
    if creatematches:
        createMatchFiles()
    match1 = np.loadtxt('P3Data/matches/matches12.txt')
    x1 = match1[:,:2]
    x2 = match1[:,2:4]

    #******* CAMERA PARAMETERS *******
    K = np.loadtxt('P3Data/calibration.txt')
    x1_in,x2_in = GetInliersRANSAC(x1,x2)  # Doing only for the first two images 
    print("Inliers size ", x1_in.shape[0])
    F = EstimateFundamentalMatrix(x1_in,x2_in)
    print("Fundamental Matrix is", F)
    E = EssentialMatrixFromFundamentalMatrix(K,F)
    print("Essential Matrix is", E)
    Rs,Cs = ExtractCameraPose(E)
    C1 = np.zeros((1,3)) #first camera's position
    R1 = np.eye(3) # first camera's pose 
    # print(Rs,Cs)
    for i in range(4):
        X.append(LinearTriangulation(K,C1,R1,Cs[i],Rs[i],x1_in,x2_in))
    R,C,X = DisambiguateCameraPose(Rs,Cs,X)
    X = np.array(X)
    # print(X)
    fig = plt.figure()
    ax = fig.add_subplot()#projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2],c="g",s=1,label="Linear")

    plt.scatter(X[:, 0], X[:, 2],c="g",s=1,label="Linear")
    
    X = NonLinearTriangulation(K,R1,C1,R,C,x1_in, x2_in, X) #Nonlinear Triangulation 
    # print("_______")
    # print(X)
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2],c="r",s=1,label="Non Linear")
    plt.scatter(X[:, 0], X[:, 2],c="r",s=1,label="Non Linear")
    DrawCameras(R1,C1[0],plt,ax,"1") #Draw 1st camera 
    DrawCameras(R,C,plt,ax,"2")  # Draw 2nd camera

    pairs_of_interest = ['P3Data/matches/matches13.txt',
                         'P3Data/matches/matches14.txt',
                         'P3Data/matches/matches15.txt'
                         ]

    Xglobal =np.unique(np.copy(X),axis=0)

    i=3

    for pair in pairs_of_interest:
        match1 = np.loadtxt(pair)
        x1f = match1[:,:2]
        x2f = match1[:,2:4]
        # x1f,x2f = GetInliersRANSAC(x1f,x2f) 
        Xf,x1f,x2f = FilterCorrespondences(x1_in,X,x1f,x2f)
        R,C = PnPRANSAC(Xf,x2f,K)
        R,C = NonlinearPnP(Xf,x2f,R,C,K)
        DrawCameras(R,C,plt,ax,str(i))  # Draw 2nd camera
        Xnew = LinearTriangulation(K,C1,R1,C,R,x1_in,x2f)
        Xnew = NonLinearTriangulation(K,R1,C1,R,C,x1_in, x2f, Xnew)
        plt.scatter(Xnew[:, 0], Xnew[:, 2],c="b",s=1,label="Non")
        i+=1

    # pairs_of_interest = ['P3Data/matches/matches23.txt',
    #                      'P3Data/matches/matches34.txt',
    #                      'P3Data/matches/matches45.txt'
    #                      ]

    # Xglobal =np.unique(np.copy(X),axis=0)

    # i=3

    # for pair in pairs_of_interest:
    #     match = np.loadtxt(pair)
    #     x1f = match[:,:2]
    #     x2f = match[:,2:4]
    #     X,x2_in,x2f = FilterCorrespondences(x2_in,X,x1f,x2f)
    #     print(X)
    #     R,C = PnPRANSAC(X,x2f,K)
    #     R,C = NonlinearPnP(X,x2f,R,C,K)
    #     DrawCameras(R,C,plt,ax,str(i))  # Draw 2nd camera
    #     X = LinearTriangulation(K,C1,R1,C,R,x2_in,x2f)
    #     X = NonLinearTriangulation(K,R1,C1,R,C,x2_in, x2f, X)
    #     plt.scatter(X[:, 0], X[:, 2],c="b",s=1,label="Non")
    #     x2_in = np.copy(x2f)
    #     i+=1

    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()