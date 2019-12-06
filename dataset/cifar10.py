from torchvision import datasets, transforms
import torch
import numpy as np
from PIL import Image
import PIL
from config import options


class CIFAR10:
    def __init__(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True)
            self.images = train_dataset.data
            self.labels = np.array(train_dataset.targets)
        elif mode == 'test':
            test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True)
            self.images = test_dataset.data
            self.labels = np.array(test_dataset.targets)

    def __getitem__(self, index):
        img = self.images[index]
        target = torch.tensor(self.labels[index]).long()

        # augmentation and normalization
        if self.mode == 'train':
            img = Image.fromarray(img, mode='RGB')
            img = transforms.RandomResizedCrop(options.img_w, scale=(0.8, 1.))(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomRotation(degrees=90, resample=PIL.Image.BICUBIC)(img)

        img = transforms.ToTensor()(img).float()

        return img, target

    def __len__(self):
        return len(self.labels)
