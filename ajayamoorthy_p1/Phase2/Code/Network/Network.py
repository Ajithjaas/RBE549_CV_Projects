"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(delta, corners):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    Loss = nn.MSELoss()
    loss = Loss(delta,corners)
    # print("\n",loss)
    return loss


class HomographyModel(pl.LightningModule):
    def __init__(self, InputSize,OutputSize):
        super(HomographyModel, self).__init__()                                                                          
        self.model = Net(InputSize,OutputSize)

    def forward(self, b):
        return self.model(b)

    def training_step(self, batch):
        patch, corners = batch
        delta = self.model(patch)
        loss = LossFn(delta, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch):
        patch, corners = batch
        delta = self.model(patch)
        loss = LossFn(delta, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(nn.Module):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        ''' CNN model'''
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc1 = nn.Sequential(
            nn.Linear(int(InputSize[0]/4)*int(InputSize[1]/4)*128, 4096),
            nn.ReLU()) 
        self.fc2= nn.Sequential(
            nn.Linear(4096, OutputSize))

    def forward(self, xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        out = self.layer1(xb)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
