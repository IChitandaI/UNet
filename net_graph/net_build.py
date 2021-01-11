import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels=n_channels
        self.n_classes=n_classes
        
        self.input=DoubleConv(n_channels, 64)
        self.down1=DoubleConv(x1, 64, 128)
        self.down2=DoubleConv(x1, 128, 256)
        self.down3=DoubleConv(x1, 256, 512)
        self.down4=DoubleConv(x1, 512, 1024)

        self.up1=DoubleConv(x1, 1024, 512)
        self.up2=DoubleConv(x1, 512, 256) 
        self.up3=DoubleConv(x1, 256, 128)
        self.up4=DoubleConv(x1, 128, 64)

        self.output=Output(64, n_classes)
    def forward(self, x):
        x1=self.input(x)

        x2=MaxPool(x1)
        x2=self.down1(x2)

        x3=MaxPool(x2)
        x3=self.down2(x3)

        x4=MaxPool(x3)
        x4=self.down3(x4)

        x5=MaxPool(x4)
        x5=self.down4(x5)

        
        x=UpAndCombine(x5, x4)
        x=self.up1(x)
        x=UpAndCombine(x, x3)
        x=self.up2(x) 
        x=UpAndCombine(x, x2)
        x=self.up3(x)
        x=UpAndCombine(x, x1)
        x=self.up4(x)
        
        return self.output(x)