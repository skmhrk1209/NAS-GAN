import torch
from torch import utils
from PIL import Image
import os


class ImageNet(utils.data.Dataset):

    def __init__(self, root, meta, transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.metas = []
        with open(meta) as file:
            for line in file.readlines():
                path, target = line.rstrip().split()
                self.metas.append((path, int(target)))

    def __len__(self):

        return len(self.metas)

    def __getitem__(self, index):

        path, target = self.metas[index]

        with open(os.path.join(self.root, path), 'rb') as file:
            image = Image.open(file).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
