"""
static
"""
import argparse
from multiprocessing import dummy
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models.quantization as models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_dataset.imagenet import MyImageNet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--onnx", dest="onnx")

def main():
    args = parser.parse_args()

    save_dir = './saved_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    dummy_input = torch.randn(16, 3, 224, 224)
    model = models.__dict__["resnet18"](pretrained=True, quantize=False)
    if args.onnx:
        orig_model = model
        torch.onnx.export(model, dummy_input, "./saved_results/origin_model.onnx")

    traindir = r"C:\Users\Local_Admin\Work\Dataset\dataset"
    valdir = r"C:\Users\Local_Admin\Work\Dataset\dataset"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = MyImageNet(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)


    val_dataset = MyImageNet(valdir, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    from neural_compressor.experimental import Quantization, common
    model.eval()
    model.fuse_model()
    quantizer = Quantization("./conf.yaml")
    # quantizer.train_dataloader = train_loader
    quantizer.calib_dataloader = val_loader
    quantizer.eval_dataloader = val_loader
    quantizer.model = common.Model(model)
    q_model = quantizer.fit()
    q_model.save(save_dir)

    if args.onnx:
        from neural_compressor.utils.pytorch import load
        orig_model = load(os.path.join(save_dir, 'best_model.pt'), orig_model)
        torch.onnx.export(orig_model, dummy_input, os.path.join(save_dir, "best_model.onnx"))
        print("The onnx model have been transfered and saved!")
    return


if __name__ == "__main__":
    main()