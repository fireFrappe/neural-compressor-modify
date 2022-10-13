import glob
from PIL import Image
import numpy as np
import os
import  torch

class MyImageNet(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        assert os.path.exists(self.root), "Datapath doesn't exist!"

        self.transform = transform

        self.image_list = []
        with open(os.path.join(self.root, "val_1000.txt"), 'r') as label_file:
            # self.image_list = {img: int(label) for img, label in label_file.readlines().strip().split(' ')}
            for line in label_file.readlines():
                img, label = line.strip().split(" ")

                with Image.open(os.path.join(self.root, "img_1000", img)) as image:
                    if len(image.split()) == 3:
                        self.image_list.append((image, label))

    def __getitem__(self, index):
        sample = self.image_list[index]
        label = int(sample[1])

        if self.transform is not None:
            image = self.transform(sample[0])
        return image, label
        
    def __len__(self):
        return len(self.image_list)


