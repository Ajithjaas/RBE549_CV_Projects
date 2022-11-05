from mimetypes import init
import os
import cv2
import pry
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import argparse
import scipy.io as sio
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation

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
    F = EstimateFundamentalMatrix(x1,x2)
    E = EssentialMatrixFromFundamentalMatrix(K,F)
    R,C = ExtractCameraPose(E)
    print(R,C)
    for i in range(3):
        X.append(LinearTriangulation(K,C[i],R[i],C[i+1],R[i+1],x1,x2))
    print(len(X))
    
if __name__ == '__main__':
    main()