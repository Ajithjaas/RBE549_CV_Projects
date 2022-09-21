"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Ajith Kumar Jayamoorthy (ajayamoorthy@wpi.edu)
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

#########################################################################################
###                                 SUPERVISED NETWORK                                ###
#########################################################################################

def SupLossFn(delta, corners):
    mse     = nn.MSELoss()
    loss    = mse(delta,corners) 
    return loss

class SupHomographyModel(pl.LightningModule):
    def __init__(self, InputSize, OutputSize):
        super(SupHomographyModel, self).__init__()
        self.model = SupNet(InputSize,OutputSize)

    def forward(self, a):
        return self.model(a)

    def training_step(self, batch):
        patch, H4Pt = batch
        delta       = self.model(patch)
        loss        = SupLossFn(delta, H4Pt)
        logs        = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch):
        patch, H4Pt = batch
        delta       = self.model(patch)
        loss        = SupLossFn(delta, H4Pt)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss    = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs        = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def test_step(self, batch):
        patch,_ = batch
        delta   = self.model(patch)
        return delta


class SupNet(nn.Module):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        ''' CNN model'''
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=0.3))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=0.3))
        self.fc1 = nn.Sequential(
            nn.Linear(int(InputSize[0]/4)*int(InputSize[1]/4)*128, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)) 
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5)) 
        self.fc3= nn.Sequential(
            nn.Linear(2048, OutputSize))



    def forward(self, xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        out = self.layer1(xb)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out




#########################################################################################
###                                UNSUPERVISED NETWORK                               ###
#########################################################################################


def UnSupLossFn(patch_b, patch_b_pred):
    loss = np.sum(abs(patch_b-patch_b_pred))
    return loss

class UnSupHomographyModel(pl.LightningModule):
    def __init__(self, InputSize, OutputSize):
        super(UnSupHomographyModel, self).__init__()
        self.model = UnSupNet(InputSize,OutputSize)

    def training_step(self, batch):
        patch, H4Pt, Corner = batch
        patchB_pred         = self.model(patch,Corner)
        PhotoLoss           = UnSupLossFn(patch, patchB_pred)
        # delta_arr           = delta.cpu().detach().numpy()
        # H_AB                = self.model.DLT(delta_arr,ConerBatch)
        logs = {"loss": PhotoLoss}
        return {"loss": PhotoLoss, "log": logs}#, "delta_arr": delta_arr}

    def validation_step(self, batch):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = UnSupLossFn(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class UnSupNet(nn.Module):
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
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=0.3))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p=0.3))
        self.fc1 = nn.Sequential(
            nn.Linear(int(InputSize[0]/4)*int(InputSize[1]/4)*128, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)) 
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5)) 
        self.fc3= nn.Sequential(
            nn.Linear(2048, OutputSize))

        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def DLT(self,delta,Corners):
        H4Pt = delta.cpu().detach().numpy()
        H_AB = np.empty((3,3,3))
        for i in range(H4Pt.shape[0]):
            label = np.array(H4Pt[i].reshape(-1,2))
            pred_label = np.array(H4Pt[i].reshape(-1,2)) 
            M = np.zeros((8,8))
            N = np.zeros((8,1))
            j=0
            for real_corner, pred_corner in zip(label,pred_label):
                ru,rv    = real_corner
                pu,pv    = pred_corner
                M[j,:]   = np.array([0,0,0,-ru,-rv,-1,pu*ru,pv*rv])
                M[j+1,:] = np.array([ru,rv,1,0,0,0,-pu*ru,-pv*rv])
                N[j]     = pv
                N[j+1]   = pu 
                j+=2
            H = np.reshape(np.append(np.linalg.solve(M,N),[1]),(3,3))
            H_AB[i] = H 
        return H_AB

    def stn(self, x):
        "Spatial transformer network forward function"
        xs      = self.localization(x)
        xs      = xs.view(-1, 10 * 3 * 3)
        theta   = self.fc_loc(xs)
        theta   = theta.view(-1, 2, 3)

        grid    = F.affine_grid(theta, x.size())
        x       = F.grid_sample(x, grid)
        return x

    def forward(self, xb, corner):
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
        out = self.fc3(out)
        H_AB = self.DLT(out,corner)
        PB_pred = self.stn(H_AB,xb)
        return PB_pred
