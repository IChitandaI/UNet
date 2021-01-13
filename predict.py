from net_graph.net_build import UNet
from predata import Data_set
from data_vis import plot_img_and_mask

import logging
import os
import sys
from glob import glob

import torch.nn as nn
from torch import optim
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

net_path='checkpoint/checkpoint_epoch1.eth'
predict_pic_path='predict_pic/img'

def pre_dict(net,
             full_img,
             device,
             scale_factor=1
             out_threshold=0.5):
    net.eval()
    img=torch.from_numpy(Data_set.resize(full_img, scale_factor)).type(torch.FloatTensor)
    img=img.unsqueeze(0)
    img=img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output=net(img)
        probs=F.softmax(output, dim=1)
        probs=probs.squeeze(0)
        tf=transforms.compose(
            transforms.ToPILImage()
            transforms.Resize(full_img.size[1])
            transforms.ToTensor()
        )
        probs=tf(probs.cpu())#Try to remove ".cpu". It is still working?
        full_mask=probs.cpu().numpy()
    return full_mask>out_threshold

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__=="__main__":
    net = UNet(n_channels=3, n_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(net_path, map_location=device))


    img = Image.open(predict_pic_path)
    mask=pre_dict(img=img, scale_factor=0.5)
    plot_img_and_mask(img, mask)

