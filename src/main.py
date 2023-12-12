import os
import torch
import torch.backends.cudnn as cudnn
import warnings
import random
import numpy as np


from train import train
from config import get_args
from dataloader import Manufacturing_dataset
from utils import DataLoaderX, Repeat, setup_seed
from constant import SETTINGS
warnings.filterwarnings('ignore')

def main(args):        
    
    args.device = torch.device('cuda:0')            
    setting = SETTINGS.get(args.settings)
    
    
    if args.dataset == "MvTecAD":
        args.data_path = '../MvTecAD'
    elif args.dataset == "Industrial_dataset":
        args.data_path = '../Manufacturing_Dataset'
    
    train_dataset = Manufacturing_dataset(    
        dataset_name = args.dataset,    
        dataset_path = args.data_path, 
        input_size = args.input_size,
        is_labels = False, 
        is_train = True,
        load_memory = False
    )
    
    train_dataset.configure_self_sup(self_sup_args=setting.get('self_sup_args'))    
        
    print("Dataset Num Classes: {}".format(train_dataset.__get_num_classes__()))
    print("Dataset Num Samples: {}".format(train_dataset.__len__()))

    args.num_classes = train_dataset.__get_num_classes__()    
    
    train_loader = DataLoaderX(
        Repeat(train_dataset, 3),
        pin_memory=True, 
        num_workers=args.workers,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
    )    
    with torch.cuda.device(args.gpu_index):
        train(train_dataset, train_loader, args)


if __name__ == "__main__":
    # setup_seed(1024)
    args = get_args()
    main(args)