#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Ajith Kumar Jayamoorthy (ajayamoorthy@wpi.edu)
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2

# Add any python libraries here
import os
import zipfile
import argparse
import matplotlib.pyplot as plt


def patch_creator(img_gray, patch_size, rho):
    x_min = rho                                         # min left-top x-coordinate for patch considering the -rho pertubation 
    x_max = img_gray.shape[1] - patch_size - rho - 1    # max left-top x-coordinate for patch considering the patch size as well as +rho pertubation
    y_min = rho                                         # min left-top y-coordinate for patch considering the -rho pertubation 
    y_max = img_gray.shape[0] - patch_size - rho - 1    # max left-top y-coordinate for patch considering the patch size as well as +rho pertubation

    # Selecting Random top-left anchor point in the Original Image to extract patch from.
    PatchA_xMin = np.random.randint(x_min,x_max)
    PatchA_yMin = np.random.randint(y_min,y_max)

    # Clockwise corner co-ordinates of Patch A
    PatchA_corners =   [[PatchA_xMin             , PatchA_yMin              ],  # Left-Top
                        [PatchA_xMin             , PatchA_yMin + patch_size ],  # Right-Top
                        [PatchA_xMin + patch_size, PatchA_yMin + patch_size ],  # Right-Bottom
                        [PatchA_xMin + patch_size, PatchA_yMin              ]]  # Left-Botton
    
    # Generating the PatchB corners by adding random pertubations to the PatchA corners
    PatchB_corners = []
    for c in PatchA_corners:
        PatchB_corners.append([c[0] + np.random.randint(-rho,rho), c[1] + np.random.randint(-rho,rho)])

    # Converting the corner from list to numpy array of type float
    PatchA_corners = np.float32(PatchA_corners)
    PatchB_corners = np.float32(PatchB_corners)

    # Calculating the Homography matrix between PatchA_corners and PatchB_corners
    H_AtoB      = cv2.getPerspectiveTransform(PatchA_corners,PatchB_corners)                    # Give the homography transformation from A to B (i.e. Mutiply with H_AtoB converts PatchA to PatchB)
    H_BtoA      = np.linalg.inv(H_AtoB)                                                         # Taking the inverse of the Homography matrix (i.e. Mutiplying with H_BtoA converts PatchB to PatchA)
    img_warped  = cv2.warpPerspective(img_gray, H_BtoA, (img_gray.shape[1],img_gray.shape[0]))  # Produces Images that has been warped (i.e. PatchA_corners co-ordinates in Warped image corresponds to the PatchB_corner co-ordinates in the original image)
    
    PatchA  =   img_gray[PatchA_yMin:PatchA_yMin+patch_size, PatchA_xMin:PatchA_xMin+patch_size]    # PatchA extracted from the original Image
    PatchB  = img_warped[PatchA_yMin:PatchA_yMin+patch_size, PatchA_xMin:PatchA_xMin+patch_size]    # PatchB extracted from the warped Image
    H4Pt    = PatchB_corners - PatchA_corners                                                       # Calculating the difference in the values of corner co-ordinates of PatchA and PatchB

    ## Plot check for verification of the tranformation Matrices
    # plt.imshow(PatchA)
    # plt.show()
    # plt.imshow(PatchB)
    # plt.show()
    # plt.imshow(img_gray)
    # plt.show()
    # plt.imshow(img_warped)
    # plt.show()
    # C = np.hstack((PatchA_corners,np.array([1,1,1,1]).reshape(-1,1)))
    # C = C.T
    # B = np.dot(H_AtoB,C)
    # B_norm = B / B[-1,:]
    # D = np.hstack((PatchB_corners,np.array([1,1,1,1]).reshape(-1,1)))
    # D = D.T
    # A = np.dot(H_BtoA,C)
    # A_norm = A / A[-1,:]
    # print('A :',A)
    # print('Output:',A_norm)
    # print('PatchB :',PatchA_corners)
    # print('B :',B)
    # print('Output:',B_norm)
    # print('PatchB :',PatchB_corners)
    return img_warped, PatchA_corners, PatchB_corners, PatchA, PatchB, H4Pt


def patchGenerator(ZipFileString, NumFeatures, originalImgPath, patchImgPath, ZipPath, LabelsPath):
    # Number of Files evaluated based on the input
    if ZipFileString == 'Train':
        numberOfFiles = 5000
    elif ZipFileString == 'Val':
        numberOfFiles = 1000
    else:
        numberOfFiles = 1000
    
    # Checking for the folder to store Original Images extracted from the zip files. In case the folder doesn't exist, then it is created.
    if not os.path.exists(originalImgPath):
        os.makedirs(originalImgPath)
        print("The new directory for Original Images is created!")

    # Loop to extract the images from the zip folder and store it in the Original Image folder.
    for i in range(1,numberOfFiles+1):
        with zipfile.ZipFile(ZipPath, 'r') as zfile:
            data        = zfile.read(ZipFileString+'/'+str(i)+'.jpg')
        img         = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        img_gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(originalImgPath+str(i)+'.jpg', img_gray)
    
    # Checking for the folder to store Patches extracted from the Original image files. In case the folder doesn't exist, then it is created.
    if not os.path.exists(patchImgPath):
        os.makedirs(patchImgPath)
        print("The new directory for Patch Images is created!")

    # Patch_Creation Function
    TrainLabels = open(LabelsPath, "w") # Creating a file in the Txt folder to store the Labels we generated from the patch_creation function
    rho         = 16                     # Maximum pertubation value in pixels
    patch_size  = NumFeatures           # Dimension of the patch (The final patch dimensions = patch_size x patch_size)

    for i in range(1,numberOfFiles+1):
        _, PatchA_corners, PatchB_corners, PatchA, PatchB, H4Pt = patch_creator(cv2.imread(originalImgPath+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE), patch_size, rho) # Generating Patches
        
        training_patch          = np.dstack((PatchA,PatchB)) # Stacking Patch-A and Patch-B together to create one sample of data
        training_patch_corners  = np.dstack((PatchA_corners,PatchB_corners)) # Stacking the Patch-A and Patch-B corners together
        np.save(patchImgPath+str(i)+'.npy',training_patch)  # Saving the stacked patches as numpy array to be used for training
        np.save(patchImgPath+str(i)+'_corners.npy',training_patch_corners)

        # Storing the respective labels (i.e. H4Pt) values in a textfile to be used for training
        string_name = str(i)+","
        for idx,val in enumerate(H4Pt.astype(int)):
            string_name = string_name + str(val[0])+","
            if idx == len(H4Pt)-1:
                string_name = string_name + str(val[1])
            else:
                string_name = string_name + str(val[1])+","
        TrainLabels.write(string_name+"\n")

    return

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', type=int, default=64, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--Data', default='Train', help='The data that need to be prepared \'Train\' or \'Test\', Default:\'Train\'')
    Args        = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    data_type   = Args.Data

    if data_type == 'Train':
        trainZipFileString      = 'Train'
        originalTrainImgPath    = '../Data/TrainOrgImgs/'
        patchTrainImgPath       = '../Data/Train/'
        trainZipPath            = '../Data/Train.zip'
        trainLabelsPath         = '../Code/TxtFiles/LabelsTrain.txt'
        patchGenerator(trainZipFileString, NumFeatures, originalTrainImgPath, patchTrainImgPath, trainZipPath, trainLabelsPath)

        valZipFileString      = 'Val'
        originalValImgPath    = '../Data/ValOrgImgs/'
        patchValImgPath       = '../Data/Val/'
        valZipPath            = '../Data/Val.zip'
        valLabelsPath         = '../Code/TxtFiles/LabelsVal.txt'
        patchGenerator(valZipFileString, NumFeatures, originalValImgPath, patchValImgPath, valZipPath, valLabelsPath)

    else:
        testZipFileString   = 'P1TestSet/Phase2'
        originalTestImgPath = '../Data/TestOrgImgs/'
        patchTestImgPath    = '../Data/Test/'
        testZipPath         = '../Data/P1TestSet.zip'
        testLabelsPath      = '../Code/TxtFiles/LabelsTest.txt'
        patchGenerator(testZipFileString, NumFeatures, originalTestImgPath, patchTestImgPath, testZipPath, testLabelsPath)

if __name__ == "__main__":
    main()
