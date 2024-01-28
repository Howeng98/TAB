import os
import cv2
import torch
import numpy as np
import imgaug.augmenters as iaa
import sys
sys.path.append("./scripts")
from PIL import ImageFile, Image
from glob import glob
from torchvision import transforms
from utils import select_another_image_from_same_class, set_class_label
from self_sup_tasks import patch_ex
from perlin import rand_perlin_2d_np
from gen_mask import gen_mask
from cutpaste import CutPasteUnion

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Manufacturing_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, dataset_path, input_size=224, is_labels=True, s=1.0, preprocess=False, self_sup_args={}, load_memory=False, is_train=True):
        self.is_labels = is_labels
        self.self_sup_args = self_sup_args
        self.input_size = input_size
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.CLASS_NAMES = list()
        self.counter = 0
        self.load_memory = load_memory
        self.is_train = is_train                
                        
        # MVTecAD, BTAD, KSDD2  /  Industrial_dataset
        if self.dataset_name == 'MvTecAD' or self.dataset_name == 'BTAD' or self.dataset_name == 'KSDD2':
            self.image_files = glob(os.path.join(self.dataset_path, '*', 'train/good', '*.*'))
        elif self.dataset_name == 'Industrial_dataset':
            self.image_files = glob(os.path.join(self.dataset_path, '*', '*.*'))
                                                       
        self.anomaly_source_paths = sorted(glob('../dtd/images/'+"/*/*.jpg"))
        
        class_names = os.listdir(self.dataset_path)
        self.CLASS_NAMES = [class_name for class_name in class_names]
                                
        self.simple_transform = transforms.Compose([
            transforms.Resize([256, 256], Image.LANCZOS),        
            # transforms.RandomResizedCrop([self.input_size, self.input_size]),
            transforms.CenterCrop(size=((self.input_size, self.input_size))),
            # transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.8),
            # transforms.RandomRotation(30),
            # transforms.RandomGrayscale(0.5),
            # transforms.RandomHorizontalFlip(p=0.8),
            # transforms.RandomVerticalFlip(p=0.8),         
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])                        
        
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
        ]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.mask_generator = gen_mask([16], 52, self.input_size)
        self.cutpaste = CutPasteUnion(input_size=self.input_size)
    
    def __getitem__(self, idx):
        
        if self.is_train:
            img_path = self.image_files[idx]
            img_path, img_path1 = select_another_image_from_same_class(img_path)
            class_name = self.__get_label__(img_path)
            class_label = self.CLASS_NAMES.index(class_name) # (e.g. class_name=screw, class_label=2, ...)
            
            idx2 = np.random.randint(len(self.image_files))
            img_path2 = self.image_files[idx2]
            
            # ori
            ori_img = self.load_image(img_path)
            image = self.simple_transform(ori_img)
                        
            # another image as synthetic patch source        
            ano_image = self.load_image(img_path1)
            ano_image = self.simple_transform(ano_image)
            
            # mask and label
            mask = torch.zeros([1, self.input_size, self.input_size]).float()
            label = torch.tensor([0], dtype=torch.float32)            
            
            # NSA synthetic image 
            # if torch.rand(1).numpy()[0] > 0.5:
            augmented_image, augmented_mask, ori_source_image = patch_ex(ima_dest=np.asarray(image), ima_src=np.asarray(ano_image), **self.self_sup_args)            
            augmented_image = self.norm_transform(augmented_image)
            augmented_mask[augmented_mask!=0]=1
            augmented_mask = self.to_tensor(augmented_mask).float()       
            # return augmented_image, augmented_mask, class_name, class_label
                               
            # PERLIN synthetic image
            # anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()                        
            # aug_PERLIN_ori, anomaly_mask_PERLIN_ori = self.augment_image(image.copy(), self.anomaly_source_paths[anomaly_source_idx])
            # aug_PERLIN = np.transpose(aug_PERLIN_ori, (2, 0, 1)).astype(np.float32) / 255.0
            # anomaly_mask_PERLIN = np.transpose(anomaly_mask_PERLIN_ori, (2, 0, 1))
            # augmented_image, augmented_mask = aug_PERLIN, anomaly_mask_PERLIN
            
            # MASK synthetic image
            # aug_MASK_ori = image.copy()
            # aug_MASK_ori = np.array(aug_MASK_ori)
            # anomaly_mask_MASK_ori = np.zeros((self.input_size, self.input_size, 1)) 
            # masks = next(self.mask_generator)
            # i = np.random.randint(0, len(masks))
            # mask = masks[i]            
            # aug_MASK_ori[mask == 0] = 0
            # anomaly_mask_MASK_ori[mask == 0] = 1
            # augmented_image = np.transpose(aug_MASK_ori, (2, 0, 1)).astype(np.float32) / 255.0
            # augmented_mask = np.transpose(anomaly_mask_MASK_ori, (2, 0, 1))                    
            
            # CutPaste synthetic image
            # augmented_image, augmented_mask = self.cutpaste(image)
            # augmented_image = self.norm_transform(augmented_image)
            # augmented_mask[augmented_mask!=0]=1
            # augmented_mask = np.transpose(augmented_mask, (2, 0, 1))
        
            #
            label = torch.tensor([1], dtype=torch.float32)
            image = self.norm_transform(image)
            image2 = self.norm_transform(ano_image)            
            
            return image, augmented_image, class_name, self.CLASS_NAMES.index(class_name), augmented_mask
        
    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 5, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]],
                              self.augmenters[aug_ind[3]],
                              self.augmenters[aug_ind[4]],
                              ]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6 #  6-->8 
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path, cv2.IMREAD_COLOR)
        anomaly_source_img = cv2.cvtColor(anomaly_source_img,cv2.COLOR_BGR2RGB) # 20230405 for pretrain is RGB bu cv2.imread is BGR
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
            self.input_size, self.input_size))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                              perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                              perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np(
            (self.input_size, self.input_size), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.65  # 0.75 cannot
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(
            perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(
            np.float32) * perlin_thr / 255.0 

        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
        
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly = 0.0
        return augmented_image, msk

    def __len__(self):
        return len(self.image_files)
    
    def __get_num_classes__(self):
        return len(self.CLASS_NAMES)
    
    def __get_labels_list__(self):
        return self.CLASS_NAMES
        
    def __get_cls_names__(self):
        return self.CLASS_NAMES
    
    def __get_label__(self, image_file):
        if self.dataset_name == 'MvTecAD' or self.dataset_name == 'BTAD' or self.dataset_name == 'KSDD2':
            image_class = os.path.dirname(image_file).split("/")[-3]
            if image_class != '..':                
                return image_class
        elif self.dataset_name == 'Industrial_dataset':            
            image_class = os.path.dirname(image_file).split("/")[-1]        
            return image_class
        return -1
    
    def configure_self_sup(self, on=True, self_sup_args={}):
        self.self_sup = on
        self.self_sup_args.update(self_sup_args)
        
    def load_image(self, path):
        return Image.open(path).convert('RGB')        
    