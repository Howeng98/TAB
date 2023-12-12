import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'

CLASS_NAMES = ['zipper', 'screw', 'hazelnut', 'capsule', 'carpet', 'grid',
               'cable', 'leather', 'metal_nut', 'pill', 'bottle',
               'tile', 'toothbrush', 'transistor', 'wood']
# CLASS_NAMES = ['01', '02', '03']
# CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
#              'pcb3', 'pcb4', 'pipe_fryum']


class MVTecDataset(Dataset):
    def __init__(self, dataset_path='C:\dataset\mvtec_anomaly_detection', class_name='01', is_train=True,
                 resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.LANCZOS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        

        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            if "01" in img_dir or "03" in img_dir:
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                        for f in os.listdir(img_type_dir)
                                        if f.endswith('.bmp')])
            else:
                # MVTecAD
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                        for f in os.listdir(img_type_dir)
                                        if f.endswith('.png')])
                
                # VisA
                # img_fpath_list = sorted([os.path.join(img_type_dir, f)
                #                         for f in os.listdir(img_type_dir)
                #                         if f.endswith('.JPG')])

            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                if  "03" in img_dir:
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.bmp')
                                    for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)
                elif "01" in img_dir or "02" in img_dir:
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                    for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)

                # #MVTecAD
                else:
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                    for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)

                # #VisA
                # else:
                #     gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                #                     for img_fname in img_fname_list]
                #     mask.extend(gt_fpath_list)
                    
                                 
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
