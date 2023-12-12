import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MVTecLOCODataset(Dataset):

    def __init__(self, root, transform, gt_transform, phase,category, split_ratio=None):
        self.phase==phase
        if phase=='train':
            self.img_path = os.path.join(root,category, 'train')
        if phase=='eval':
            self.img_path = os.path.join(root,category, 'validation')
            # self.gt_path = os.path.join(root,category, 'ground_truth')
        else:
            self.img_path = os.path.join(root,category, 'test')
            self.gt_path = os.path.join(root,category, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        assert os.path.isdir(os.path.join(root,category)), 'Error MVTecLOCODataset category:{}'.format(category)
        # load dataset

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1


    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*")
                gt_paths = [g for g in gt_paths if os.path.isdir(g)]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                if len(gt_paths)==0:
                    gt_paths = [0]*len(img_paths)
                
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

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
            names = os.listdir(gt)
            ims = [cv2.imread(os.path.join(gt, name), cv2.IMREAD_GRAYSCALE) for name in names]
            ims = [im for im in ims if isinstance(im, np.ndarray)]
            imzeros = np.zeros_like(ims[0])
            for im in ims:
                imzeros[im==255] = 255
            gt = Image.fromarray(imzeros)
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
    