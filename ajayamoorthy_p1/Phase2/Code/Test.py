#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import argparse
import glob
import math as m
import os
import random
import shutil
import string
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
from Wrapper import * 
from skimage import data, exposure, img_as_float
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToTensor
from tqdm import tqdm
from Network.Network import SupHomographyModel, UnSupHomographyModel
from Train import SupGenerateBatch

if torch.cuda.is_available():
  device = torch.device("cuda")
  torch.cuda.empty_cache()
else:
  device = torch.device("cpu")

# Don't generate pyc codes
sys.dont_write_bytecode = True


########################################################
######            SETUPALL FUNCTION              #######
########################################################
def ReadDirNames(ReadPath):
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames

def ReadLabels(LabelsPath):
    if not (os.path.isfile(LabelsPath)):
        print("ERROR: Test Labels do not exist in " + LabelsPath)
        sys.exit()
    else:
        TestLabels = open(LabelsPath, "r")
        TestLabels = TestLabels.read()
        TestLabels = np.array(re.split(",|\n",TestLabels)[:-1],dtype=int)
        TestLabels = TestLabels.reshape(-1,9)
        TestLabels = dict(zip(TestLabels[:,0],TestLabels[:,1:]))
    return TestLabels

def SetupAll(LabelsPath, NumFeature, PathDirNamesTest):
    # Setup DirNames
    DirNamesTest    = ReadDirNames(PathDirNamesTest)
    TestLabels      = ReadLabels(LabelsPath)
    # Image Input Shape
    ImageSize       = [NumFeature, NumFeature, 2]
    NumTestSamples  = len(DirNamesTest)
    # Number of classes
    NumClasses      = 8
    return TestLabels, DirNamesTest, ImageSize, NumTestSamples, NumClasses




def TestOperation(ImgSize, BasePath, DirNamesTest, TestLabels, ModelPath, LabelsPathPred, NumTestSamples, NumClasses):
    """
    Inputs:
    NumFeatures is the side dimension of the Image.
    ImageSize       - size of the image
    ModelPath       - Path to load trained model from
    BasePath        - The test dataset
    TestLabels      - Test Labels
    LabelsPathPred  - Path to save predictions
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model       = SupHomographyModel(ImgSize, NumClasses)
    CheckPoint  = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    model.to(device)
    
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )
    
    # Shuffiling the Order of the training images to avoid bias 
    DirNamesTest_ = random.sample(list(zip(DirNamesTest, list(TestLabels))), len(DirNamesTest))
    try:
        PredLabels = open(LabelsPathPred, "x")
    except:
        PredLabels = open(LabelsPathPred, "w")
    
    Output = []
    Minibatchsize = 100
    Iterations = int(NumTestSamples/Minibatchsize)
    for idx in tqdm(range(Iterations)):
        batch   = SupGenerateBatch(idx, BasePath, DirNamesTest_, TestLabels, ImgSize, Minibatchsize)
        result  = model.validation_step(batch)
        delta   = model.test_step(batch)
        delta   = delta.cpu().detach().numpy()
        Output.append(result["val_loss"].item())
        # Storing the respective labels (i.e. H4Pt) values in a textfile to be used for training
        BatchDirNamesTest = DirNamesTest_[idx*Minibatchsize: idx*Minibatchsize + Minibatchsize]
        for i in range(delta.shape[0]):
            string_name = BatchDirNamesTest[i][0] + ","
            string_name = string_name + str(delta[i,:])+","
            PredLabels.write(string_name+"\n")

    print("Average Test Loss = ", np.mean(np.array(Output)))


def TestSample(BasePath, zip_path, path, LabelsPath, NumFeatures, PathDirNamesTest, ImgSize, ModelPath):
    model       = SupHomographyModel(ImgSize, OutputSize=8)
    CheckPoint  = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    model.to(device)

  
    if not os.path.exists(path):
        # Create a new directory because it does not exist 
        os.makedirs(path)
        print("The new directory is created!")

    for i in range(1,5):
        with zipfile.ZipFile(zip_path, 'r') as zfile:
            data = zfile.read('P1TestSet/Phase2/'+str(i)+'.jpg')
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        cv2.imwrite(path+str(i)+'.jpg', img)

    rho = 16  # In pixels
    patch_size = int(NumFeatures) # Dimension of the patch (The final patch dimensions = patch_size x patch_size)
    TestLabels, DirNamesTest, ImgSize, NumTestSamples, NumClasses = SetupAll(LabelsPath, NumFeatures, PathDirNamesTest)
    
    DirNamesTest_ = random.sample(list(zip(DirNamesTest, list(TestLabels))), len(DirNamesTest))

    for i in range(1,5):
        img = cv2.imread(path+str(i)+'.jpg')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_gray = cv2.imread(path+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
        _,PatchA_corners, PatchB_corners,_,_,_ = patch_creator(img_gray,patch_size,rho)
        batch   = SupGenerateBatch(i, BasePath, DirNamesTest_, TestLabels, ImgSize, 1)
        NN_H4PT = model.test_step(batch)
        NN_H4PT = NN_H4PT.cpu().detach().numpy().reshape(4,2)
        NN_Corners = PatchA_corners + NN_H4PT
        A_corn = np.vstack((PatchA_corners, PatchA_corners[0]))
        B_corn = np.vstack((PatchB_corners, PatchB_corners[0]))
        NN_corn = np.vstack((NN_Corners, NN_Corners[0]))
        xA, yA = zip(*A_corn)
        xB, yB = zip(*B_corn)
        xC, yC = zip(*NN_corn)

        plt.axis('off')
        plt.imshow(img)
        plt.scatter(PatchA_corners[:,0], PatchA_corners[:,1], c='b')
        plt.scatter(PatchB_corners[:,0], PatchB_corners[:,1], c='r')
        plt.scatter(NN_Corners[:,0], NN_Corners[:,1], c='y')
        plt.plot(xA, yA, c='b')
        plt.plot(xB, yB, c='r')
        plt.plot(xC, yC, c='y')
        plt.show()

def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="../Checkpoints/4model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="../Data",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )
    Parser.add_argument(
        "--NumFeatures",
        type = int,
        default=100,
        help="Dimension of the Patch, Default:100",
    )
    Parser.add_argument(
        "--ModelType",
        default='Sup',
        help="Model type is either Supervised ('Sup') or Unsupervised ('Unsup'), Default:'Sup'",
    )

    Args = Parser.parse_args()
    ModelPath   = Args.ModelPath
    BasePath    = Args.BasePath
    LabelsPath  = Args.LabelsPath
    NumFeatures = Args.NumFeatures
    ModelType   = Args.ModelType

    PathDirNamesTest = "../Code/TxtFiles/DirNamesTest.txt"
    TestLabels, DirNamesTest, ImgSize, NumTestSamples, NumClasses = SetupAll(LabelsPath, NumFeatures, PathDirNamesTest)

    # Define PlaceHolder variables for Input and Predicted output
    LabelsPathPred = "../Code/TxtFiles/PredOut.txt"  # Path to save predicted labels
    TestOperation(ImgSize, BasePath, DirNamesTest, TestLabels, ModelPath, LabelsPathPred, NumTestSamples, NumClasses)

    zip_path = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/ajayamoorthy_p1/Phase2/Data/P1TestSet.zip'
    path = '../Data/TrainSamples/'
    TestSample(BasePath, zip_path, path, LabelsPath, NumFeatures, PathDirNamesTest, ImgSize, ModelPath)

if __name__ == "__main__":
    main()
