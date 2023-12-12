import argparse
from torchvision import models


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_args():
    
    parser = argparse.ArgumentParser(description='Anomaly Backbone')
    
    # Path setting
    parser.add_argument('--dataset', default='MvTecAD', choices=['MvTecAD', 'Industrial_dataset'])

    # Hyperparameters
    parser.add_argument('--backbone-name', default='wide_resnet50_2', choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2'])
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', type=float, default=0.001) #0.0001
    parser.add_argument('--batch-size', type=int, default=64) #128
    parser.add_argument('--input-size', default=224)
    parser.add_argument('--gpu-index', default=0, type=int)
    
    #
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--output_dim', default=128, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--data-parallel', default=False)
    parser.add_argument('--resume', default=False, help='Resume training from ckpt')
    parser.add_argument('--optm', default='adam')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--settings', type=str, default="Shift-Intensity-923874273") #Shift-Intensity-923874273
    parser.add_argument('--note', type=str, default="This is note")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12], help="features used")

    # https://github.com/facebookresearch/ov-seg/blob/main/open_clip_training/README.md
    return parser.parse_args()