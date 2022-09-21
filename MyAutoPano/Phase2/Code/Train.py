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
# termcolor, do (pip install termcolor)

from ctypes.wintypes import HACCEL
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import UnSupHomographyModel
from Network.Network import SupHomographyModel
from Network.Network import UnSupLossFn
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

if torch.cuda.is_available():
  device = torch.device("cuda")
  torch.cuda.empty_cache()
else:
  device = torch.device("cpu")
print(torch.cuda.get_device_name())


###  GENERATION OF BATCH SUPERVISED
def SupGenerateBatch(PerEpochCounter, BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath            - Path to COCO folder without "/" at the end
    DirNamesTrain       - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates    - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize           - Size of the Image
    MiniBatchSize       - Size of the MiniBatch

    Outputs:
    I1Batch             - Batch of images
    CoordinatesBatch    - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []
    BatchDirNamesTrain = DirNamesTrain[PerEpochCounter*MiniBatchSize: PerEpochCounter*MiniBatchSize + MiniBatchSize]
 
    ImageNum = 0
    # print("\n Img Location :", BatchDirNamesTrain[ImageNum][0])
    # print("\n Cordinates   :", TrainCoordinates[BatchDirNamesTrain[ImageNum][1]])

    while ImageNum < MiniBatchSize:
        # Generate random image
        RandImageName   = BasePath + os.sep + BatchDirNamesTrain[ImageNum][0] + ".npy"
        patch           = np.float32(np.load(RandImageName))
        To_Tensor       = ToTensor()
        I1              = To_Tensor(patch)
        Coordinates     = TrainCoordinates[BatchDirNamesTrain[ImageNum][1]]
        # Append All Images and Mask
        I1Batch.append(I1)
        CoordinatesBatch.append(torch.tensor(Coordinates, dtype=torch.float32))
        ImageNum += 1

    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device)


###  GENERATION OF BATCH SUPERVISED
def UnSupGenerateBatch(PerEpochCounter, BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath            - Path to COCO folder without "/" at the end
    DirNamesTrain       - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates    - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize           - Size of the Image
    MiniBatchSize       - Size of the MiniBatch

    Outputs:
    I1Batch             - Batch of images
    CoordinatesBatch    - Batch of coordinates
    """
    I1Batch = []
    CornersBatch = []
    CoordinatesBatch = []
    BatchDirNamesTrain = DirNamesTrain[PerEpochCounter*MiniBatchSize: PerEpochCounter*MiniBatchSize + MiniBatchSize]
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandImageName   = BasePath + os.sep + BatchDirNamesTrain[ImageNum][0] + ".npy"
        PatchCornName   = BasePath + os.sep + BatchDirNamesTrain[ImageNum][0] + "_corners.npy"
        patch           = np.float32(np.load(RandImageName))
        patch_corner    = np.float32(np.load(PatchCornName))
        To_Tensor       = ToTensor()
        I1              = To_Tensor(patch)
        Coordinates     = TrainCoordinates[BatchDirNamesTrain[ImageNum][1]]
        # Append All Images and Mask
        I1Batch.append(I1)
        CornersBatch.append(patch_corner)
        CoordinatesBatch.append(torch.tensor(Coordinates, dtype=torch.float32))
        ImageNum += 1

    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device), CornersBatch


#### PRINTING THE TRAINING PARAMETERS
def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


#### TRAINING OPERATION
def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    NumClasses,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain       - Variable with Subfolder paths to train files
    TrainCoordinates    - Coordinates corresponding to Train/Test
    NumTrainSamples     - length(Train)
    ImageSize           - Size of the image
    NumEpochs           - Number of passes through the Train data
    MiniBatchSize       - Size of the MiniBatch
    SaveCheckPoint      - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath      - Path to save checkpoints/model
    DivTrain            - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile          - Latest checkpointfile to continue training
    BasePath            - Path to COCO folder without "/" at the end
    LogsPath            - Path to save Tensorboard Logs
    NumClasses          - Number of output values
    ModelType           - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    if ModelType == 'Sup':
        # Predict output with forward pass
        model = SupHomographyModel(ImageSize,NumClasses)
        model.to(device)
    else:
        model = UnSupHomographyModel()
        model.to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    # Optimizer = torch.optim.SGD(model.parameters(),lr = 0.00001)
    Optimizer = torch.optim.AdamW(model.parameters(),lr = 0.0001)
    print("Optimizer Information: \n", Optimizer.state_dict)
    
    # Tensorboard
    # writer.add_graph(net,images)  # used to visualize the network
    from torchsummary import summary
    summary(model,input_size=(64,64,2))
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")


    # Shuffiling the Order of the training images to avoid bias 
    DirNamesTrain_ = random.sample(list(zip(DirNamesTrain, list(TrainCoordinates))), len(DirNamesTrain))

    # Starting the training process
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):

            if ModelType == 'Sup':
                I1Batch = SupGenerateBatch(PerEpochCounter, BasePath, DirNamesTrain_, TrainCoordinates, ImageSize, MiniBatchSize)
            else:
                I1Batch = UnSupGenerateBatch(PerEpochCounter, BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize)

            # Predict output with forward pass
            if ModelType == 'Sup':
                LossThisBatch = model.training_step(I1Batch)
            else:
                LossThisBatch = model.training_step(I1Batch)
                # delta_arr     = LossThisBatch["delta_arr"]
                # H_AB          = model.DLT(delta_arr,ConerBatch)
                LossThisBatch = UnSupLossFn(PredicatedCoordinatesBatch, CoordinatesBatch)

            Optimizer.zero_grad()
            LossThisBatch["loss"].backward()
            Optimizer.step()

            # # Save checkpoint every some SaveCheckPoint's iterations
            # if PerEpochCounter % SaveCheckPoint == 0:
            #     # Save the Model learnt in this epoch
            #     SaveName = (
            #         CheckPointPath
            #         + str(Epochs)
            #         + "a"
            #         + str(PerEpochCounter)
            #         + "model.ckpt"
            #     )

            #     torch.save(
            #         {
            #             "epoch": Epochs,
            #             "model_state_dict": model.state_dict(),
            #             "optimizer_state_dict": Optimizer.state_dict(),
            #             "loss": LossThisBatch,
            #         },
            #         SaveName,
            #     )
            #     print("\n" + SaveName + " Model Saved...")

            result = model.validation_step(I1Batch)
            # Tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                result["val_loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n Validation Loss :",result["val_loss"].item())
        print("\n" + SaveName + " Model Saved...")



def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:../Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=10,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=100,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    Parser.add_argument(
        "--NumFeatures",
        type=int,
        default=64,
        help="Path to save Logs for Tensorboard, Default=64",
    )

    Args            = Parser.parse_args()
    NumEpochs       = Args.NumEpochs
    BasePath        = Args.BasePath
    DivTrain        = float(Args.DivTrain)
    MiniBatchSize   = Args.MiniBatchSize
    LoadCheckPoint  = Args.LoadCheckPoint
    CheckPointPath  = Args.CheckPointPath
    LogsPath        = Args.LogsPath
    ModelType       = Args.ModelType
    NumFeatures     = Args.NumFeatures

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath, NumFeatures)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        NumClasses,
        ModelType,
    )


if __name__ == "__main__":
    main()
