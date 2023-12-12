# official code: https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py
# reference from: https://github.com/rximg/EfficientAD/blob/main/data_loader.py
import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

data_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
             'pcb3', 'pcb4', 'pipe_fryum']

class VisaDataset(Dataset):

    def __init__(self, root, transform, gt_transform, phase, category=None,split_ratio=0.8):
        self.phase = phase
        self.root = root
        self.category = category
        self.transform = transform
        self.gt_transform = gt_transform
        self.split_ratio = split_ratio
        self.split_file = root + "/split_csv/1cls.csv"
        assert os.path.isfile(self.split_file), 'Error VsiA dataset'
        assert os.path.isdir(os.path.join(self.root,category)), 'Error VsiA dataset category:{}'.format(category)
            
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1


    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        with open(self.split_file,'r') as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
                category, split, label, image_path, mask_path = row
                image_name = image_path.split("/")[-1]
                mask_name = mask_path.split("/")[-1]
                if split=='train' and self.phase=='eval':
                    split='eval'
                if self.phase == split and self.category == category :
                    img_src_path = os.path.join(self.root,image_path)
                    if label == "normal":
                        gt_src_path = 0
                        index = 0
                        types = "good"
                    else:
                        index = 1
                        types = "bad"
                        gt_src_path = os.path.join(self.root,mask_path)
                    
                    img_tot_paths.append(img_src_path)
                    gt_tot_paths.append(gt_src_path)
                    tot_labels.append(index)
                    tot_types.append(types)
        train_len = int(len(img_tot_paths)*self.split_ratio)
        img_tot_paths, gt_tot_paths, tot_labels, tot_types = syn_shuffle(img_tot_paths, gt_tot_paths, tot_labels, tot_types)
        if self.phase == "train":
            img_tot_paths = img_tot_paths[:train_len]
            gt_tot_paths = gt_tot_paths[:train_len]
            tot_labels = tot_labels[:train_len]
            tot_types = tot_types[:train_len]
        elif self.phase == 'eval':
            img_tot_paths = img_tot_paths[train_len:]
            gt_tot_paths = gt_tot_paths[train_len:]
            tot_labels = tot_labels[train_len:]
            tot_types = tot_types[train_len:]

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        origin = img
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        # return img, gt, label, os.path.basename(img_path[:-4]), img_type
        return {
            'origin':np.array(origin),
            'image': img,
            'gt': gt,
            'label': label,
            'name': os.path.basename(img_path[:-4]),
            'type': img_type
        }
