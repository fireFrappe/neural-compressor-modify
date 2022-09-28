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


def main():
    save_dir = './saved_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    dummy_input = torch.randn(16, 3, 224, 224)
    model = models.__dict__["resnet18"](pretrained=True, quantize=False)
    orig_model = model
    torch.onnx.export(model, dummy_input, "./saved_results/origin_model.onnx")

    from neural_compressor.experimental import Quantization, common
    model.eval()
    quantizer = Quantization("./conf.yaml")
    quantizer.model = common.Model(model)
    q_model = quantizer.fit()
    q_model.save(save_dir)

    from neural_compressor.utils.pytorch import load
    orig_model = load(os.path.join(save_dir, 'best_model.pt'), orig_model)
    torch.onnx.export(orig_model, dummy_input, os.path.join(save_dir, "best_model.onnx"))
    return


if __name__ == "__main__":
    main()