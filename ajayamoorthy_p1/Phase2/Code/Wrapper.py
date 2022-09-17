#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
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

def patch_generator(img,PatchSize,rho):
    x_min = rho                             # min left-top x-coordinate for patch considering the -rho pertubation  
    x_max = img.shape[1]-PatchSize-rho-1    # max left-top x-coordinate for patch considering the patch size as well as +rho pertubation
    y_min = rho                             # min left-top y-coordinate for patch considering the -rho pertubation 
    y_max = img.shape[0]-PatchSize-rho-1    # max left-top y-coordinate for patch considering the patch size as well as +rho pertubation

    x_patchA = np.random.randint(x_min,x_max) # Considering a random left-top x-coordinate for Patch-A within limits - (x_min,x_max)
    y_patchA = np.random.randint(y_min,y_max) # Considering a random left-top y-coordinate for Patch-A within limits - (y_min,y_max)

    # Clockwise corner co-ordinates of Patch A
    PatchA_corners = [[x_patchA             , y_patchA],            # Left-Top
                    [x_patchA+PatchSize   , y_patchA],            # Right-Top
                    [x_patchA+PatchSize   , y_patchA+PatchSize],  # Right-Bottom
                    [x_patchA             , y_patchA+PatchSize]]  # Left-Bottom
                    
    PatchB_corners = []
    for c in PatchA_corners:
        PatchB_corners.append([c[0] + np.random.randint(-rho,rho), c[1] + np.random.randint(-rho,rho)])

    PatchA_corners = np.float32(PatchA_corners)
    PatchB_corners = np.float32(PatchB_corners)
    
    # Calgulating the Transformation Matrix from A to B and then from B to A
    H_AB = cv2.getPerspectiveTransform(PatchA_corners,PatchB_corners)
    H_BA = np.linalg.inv(H_AB)
    img_warped = cv2.warpPerspective(img, H_BA, (img.shape[1],img.shape[0]))
    # print(img.shape)
    # print(img_warped.shape)

    # print(PatchA_corners)
    PatchA  =        img[y_patchA:(y_patchA+PatchSize),x_patchA:(x_patchA+PatchSize)]
    PatchB  = img_warped[y_patchA:(y_patchA+PatchSize),x_patchA:(x_patchA+PatchSize)]
    H4pt    = PatchB_corners - PatchA_corners
    return PatchA_corners, PatchB_corners, PatchA, PatchB, H4pt

def patch_creator(data_type, path, zip_path, data_path, labels_path, NumFeatures):
    if data_type == 'Train':
        n = 5001
    else:
        n = 1001

    if not os.path.exists(path):
        # Create a new directory because it does not exist 
        os.makedirs(path)
        print("The new directory is created!")

    #train_img = []
    for i in range(1,n):
        with zipfile.ZipFile(zip_path, 'r') as zfile:
            data = zfile.read(data_type+'/'+str(i)+'.jpg')
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path+str(i)+'.jpg', img_gray)
        # train_img.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))  


    if not os.path.exists(data_path):
        # Create a new directory because it does not exist 
        os.makedirs(data_path)
        print("The new directory is created!")


    TrainLabels = open(labels_path, "w")
    for i in range(1,n):
        rho = 8  # In pixels
        patch_size = int(NumFeatures) # Dimension of the patch (The final patch dimensions = patch_size x patch_size)
        _, _, PatchA, PatchB, H4Pt = patch_generator(cv2.imread(path+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE),patch_size,rho)
        training_imgs = np.dstack((PatchA,PatchB))
        # print(PatchA.shape)
        # print(PatchB.shape)
        # if i == 1:
        #     training_imgs = np.dstack((PatchA,PatchB))
        #     training_label = H4Pt
        #     # print(training_imgs.shape)
        # else:
        #     training_imgs = np.dstack((training_imgs,PatchA))
        #     training_imgs = np.dstack((training_imgs,PatchB))
        #     training_label = np.dstack((training_label,H4Pt))
        #     # print(training_imgs.shape)
        # cv2.imwrite(data_path+str(i)+'a.jpg',PatchA)
        # cv2.imwrite(data_path+str(i)+'b.jpg',PatchB)
        # np.save('training_label.npy', H4Pt)
        np.save(data_path+str(i)+'.npy',training_imgs)
        string_name = str(i)+","
        for idx,val in enumerate(H4Pt.astype(int)):
            string_name = string_name + str(val[0])+","
            if idx == len(H4Pt)-1:
                string_name = string_name + str(val[1])
            else:
                string_name = string_name + str(val[1])+","

        TrainLabels.write(string_name+"\n")

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=64, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--Data', default='Train', help='The data that need to be prepared \'Train\' or \'Test\', Default:\'Train\'')
    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    data_type = Args.Data

    """
    Read a set of images for Panorama stitching
    """
    if data_type=='Train':
        path = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/ajayamoorthy_p1/Phase2/Data/Train_Images/'
        data_path = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/ajayamoorthy_p1/Phase2/Data/Train/'
        trainzip_path = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/ajayamoorthy_p1/Phase2/Data/Train.zip'
        labels_path = "/home/ajith/Documents/git_repos/RBE549_CV_Projects/ajayamoorthy_p1/Phase2/Code/TxtFiles/LabelsTrain.txt"
        patch_creator('Train', path, trainzip_path, data_path, labels_path, NumFeatures)
    else:
        path = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/ajayamoorthy_p1/Phase2/Data/Test_Images/'
        data_path = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/ajayamoorthy_p1/Phase2/Data/Test/'
        testzip_path = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/ajayamoorthy_p1/Phase2/Data/P1TestSet.zip'
        labels_path = "/home/ajith/Documents/git_repos/RBE549_CV_Projects/ajayamoorthy_p1/Phase2/Code/TxtFiles/LabelsTest.txt"
        patch_creator('P1TestSet/Phase2', path, testzip_path, data_path, labels_path, NumFeatures)
 
if __name__ == "__main__":
    main()
