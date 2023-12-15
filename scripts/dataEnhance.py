import os
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from random import sample
import random


def imageEnhanceRandom(src,idx):
    N = 12
    #src.save('./mvtec_result' + str(idx) + "SRC001.png")
    dst = src.copy()
    img_array = np.array(src)
    H, W, C = img_array.shape
    scale = int(H / N)
    pointMax = H - 1
    number = 30

    method_flag = np.random.randint(low=0, high=3)
    if method_flag == 0:
        method = 'blur'
    elif method_flag == 1:
        method = 'white'
    else:
        method = 'mask'
    
    
    for nnn in range(number):
        negative_point = np.random.randint(low = 0, high = H - 1, size = 2)
        point_TL_X = negative_point[0]
        point_TL_Y = negative_point[1]
        if (point_TL_X + scale) >= pointMax:
            point_BR_X = pointMax + 1
        else:
            point_BR_X = point_TL_X + scale
        if (point_TL_Y + scale) >= pointMax:
            point_BR_Y = pointMax + 1
        else:
            point_BR_Y = point_TL_Y + scale
        box = (point_TL_X, point_TL_Y, point_BR_X, point_BR_Y)
        if nnn == 0 :
            mask = src.crop((0, 0, point_BR_X - point_TL_X,point_BR_Y - point_TL_Y))

        if method == 'blur':
            mask = src.crop((point_TL_X, point_TL_Y, point_BR_X, point_BR_Y))
            mask = mask.filter(ImageFilter.GaussianBlur(radius = 9))
        elif method == 'cutpaste':
            if mask.width >= (point_BR_X - point_TL_X) or mask.height >= (point_BR_Y - point_TL_Y):
                if mask.width < (point_BR_X - point_TL_X):
                    mask = mask.crop((0, 0, mask.width, point_BR_Y - point_TL_Y))
                elif mask.height < (point_BR_Y - point_TL_Y):
                    mask = mask.crop((0, 0, point_BR_X - point_TL_X, mask.height))
                else:
                    mask = mask.crop((0, 0, point_BR_X - point_TL_X, point_BR_Y - point_TL_Y))
                box2 = (point_TL_X, point_TL_Y, point_TL_X + mask.width, point_TL_Y + mask.height)
                dst.paste(mask, box2)
            else:
                box2 = (point_TL_X , point_TL_Y, point_TL_X + mask.width, point_TL_Y + mask.height)
                dst.paste(mask, box2)
            mask = src.crop((point_TL_X, point_TL_Y, point_BR_X, point_BR_Y))
        elif method == 'mask':
            mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (0, 0, 0))
        elif method == 'white':
            mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (255, 255, 255))
        elif method == 'RGB':
            negative_flag = np.random.randint(low=0, high=8)
            if negative_flag == 0:
                mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (0, 0, 0))
            elif negative_flag == 1:
                mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (255, 0, 0))
            elif negative_flag == 2:
                mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (0, 255, 0))
            elif negative_flag == 3:
                mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (0, 0, 255))
            elif negative_flag == 4:
                mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (255, 255, 0))
            elif negative_flag == 5:
                mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (255, 0, 255))
            elif negative_flag == 6:
                mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (0, 255, 255))
            else:
                mask = Image.new('RGB', (point_BR_X - point_TL_X, point_BR_Y - point_TL_Y), (255, 255, 255))
        if method != 'cutpaste':
            dst.paste(mask, box)
    #dst.save('./' + method + '_' +  str(idx) + "DST001.png")

    return dst

