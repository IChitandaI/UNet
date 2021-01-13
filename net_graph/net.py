import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.doubleconv=nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.doubleconv(x)

class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool=nn.MaxPool2d(2)
    def forward(self, x):
        return self.maxpool(x)

class UpAndCombine(nn.Module):
    def __init__(self):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x_new, x_last):
        x_new=self.up(x_new)
        diffX=x_last.size()[3]-x_new.size()[3]
        diffY=x_last.size()[2]-x_new.size()[2]
        x_new=F.pad(x_new, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return torch.cat([x_last,x_new], dim=1)


class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.outconv=nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.outconv(x)
