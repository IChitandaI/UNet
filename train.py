from net_graph.net_build import UNet
from predata import Data_set

import loggint
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
from torch.utils.tensorboard import SummaryWriter

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'
data_num=5088

def pre_pic(img):


def train(net,
          device,
          epochs=5,
          batch_size=1,
          lr=0.001,
          val_percent=0.1,
          save_cp=True,
          img_scale=0.5):
    
    #writer = SummaryWriter(comment=f'lr_{lr}_batch_size_{batch_size}_img_scale_{img_scale}')
    train_data=Data_set(dir_img, dir_mask)

    optimizer=optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.5)
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    

    for epoch in range(epochs):
        net.train()
        epoch_loss=0
        
        for batch in train_loader:
            imgs=batch['image']
            mask=batch['mask']

            imgs=imgs.to(device=device, dtype=torch.float32)
            mask=mask.to(device=device, dtype=torch.float32)

            pre_mask=net(imgs)
            loss=nn.BCEWithLogitsLoss(pre_mask, mask)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

    if save_cp:
        try:
            os.mkdir(dir_checkpoint)
            logging.info("Creating checkpoint directory")
        except OSError:
            pass
        torch.save(net.state_dict(), dir_checkpoint+f'checkpoint_epoch{epoch + 1}.pth')
        loggint.info(f'Checkpoint_epoch{epoch + 1} has saved!')
    
    #writer.close()


if __name__=="__main__":
    net=UNet(n_channels=3, n_classes=1)
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu' )
    net=UNet(n_channels=3, n_classes=1)
    net.to(device=device)
    train(net=net, device=device, epoch=data_num)

