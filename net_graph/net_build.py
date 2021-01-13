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
        self.down1=DoubleConv(64, 128)
        self.down2=DoubleConv(128, 256)
        self.down3=DoubleConv(256, 512)
        self.down4=DoubleConv(512, 1024)

        self.up1=DoubleConv(1024+512, 512)
        self.up2=DoubleConv(512+256, 256) 
        self.up3=DoubleConv(256+128, 128)
        self.up4=DoubleConv(128+64, 64)

        self.Maxpool=MaxPool()

        self.upandcombine=UpAndCombine()
        self.output=Output(64, n_classes)
    def forward(self, x):
        x1=self.input(x)

        x2=self.Maxpool(x1)
        x2=self.down1(x2)

        x3=self.Maxpool(x2)
        x3=self.down2(x3)

        x4=self.Maxpool(x3)
        x4=self.down3(x4)

        x5=self.Maxpool(x4)
        x5=self.down4(x5)
        x=self.upandcombine(x5, x4)
        x=self.up1(x)
        x=self.upandcombine(x, x3)
        x=self.up2(x) 
        x=self.upandcombine(x, x2)
        x=self.up3(x)
        x=self.upandcombine(x, x1)
        x=self.up4(x)
        return self.output(x)