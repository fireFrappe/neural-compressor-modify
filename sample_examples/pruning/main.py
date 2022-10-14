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

def main(args):

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_results')
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

    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    from neural_compressor.experimental import Pruning, common
    prune = Pruning(args.config)

    def train_func(model):
        # To add a learning rate scheduler, here redefine a training func

        global best_acc1
        # prune.on_train_begin()
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args)

            prune.on_epoch_begin(epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, prune)
            prune.on_epoch_end()

            # evaluate on validation set
            acc1, acc5 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.distributed or (args.distributed 
                and hvd.rank() == 0):
                print("=> save checkpoint")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'topology': args.topology,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)
        return model

    prune.model = common.Model(model)
    prune.train_dataloader = train_loader
    prune.eval_dataloader = val_loader
    prune.train_func = train_func
    model = prune.fit()
    model.save(args.output_model)
    return

    if args.onnx:
        from neural_compressor.utils.pytorch import load
        orig_model = load(os.path.join(save_dir, 'best_model.pt'), orig_model)
        torch.onnx.export(orig_model, dummy_input, os.path.join(save_dir, "best_model.onnx"))
        print("The onnx model have been transfered and saved!")
    return

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)