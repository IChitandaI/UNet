import logging
from os.path import splitext
from os import listdir

import numpy as np
import torch
from torch.utils.data import Dataset

from glob import glob
from PIL import Image



class Data_set(Dataset):
    def __init__(self, dir_img, dir_mask, scale=1):
        self.dir_img = dir_img
        self.dir_mask = dir_mask
        self.scale=scale
        self.name = [splitext(file)[0] for file in listdir(
            dir_img) if not file.startswith('.')]

    def __len__(self):
        return len(self.name)
    @classmethod
    def resize(cls, img, scale):
        W, H = img.size
        new_W = int(W*scale)
        new_H = int(H*scale)
        new_img = img.resize((new_W, new_H))
        a = np.array(new_img)
        if len(a.shape) == 2:
            a = np.expand_dims(a, axis=2)
        a = a.transpose((2, 0, 1))
        if a.max() > 1:
            a = a/255
        return a

    def __getitem__(self, i):
        x = self.name[i]
        file_img = glob(self.dir_img + x + '.*')
        file_mask = glob(self.dir_mask + x + '_mask.*')
        try:
            img = Image.open(file_img[0])
            mask = Image.open(file_mask[0])
        except:
            print(file_img)
            print(file_mask)
        img = self.resize(img, self.scale)
        mask = self.resize(mask, self.scale)

        return {
            'img': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
        }  # return a dict
