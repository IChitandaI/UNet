import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super.__init__()
        self.doubleconv=nn.sequential(
            nn.conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
            nn.BatchNorm2d(mid_channels)
            nn.ReLU(inplace=True)
            nn.conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
            
        )