import torch
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Build basic config.")
    parser.add_argument("-device", required=False, default=0, type=int, help="cuda device")
    parser.add_argument("-conf", required=True, type=str, help="config file to choose model")
    args, unknown = parser.parse_known_args()
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    args.arch = args.conf.split("_")[0]
    args.dataset = args.conf.split("_")[1]
    return args


args = get_args()
device = args.device


tiny_imagenet_pth = "./datasets"
imagenet_pth = "./datasets/data/ImageNet"
cifar10_path = "./datasets/data/CIFAR10"


cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
cifar10_std = np.array([0.2471, 0.2435, 0.2616])

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])