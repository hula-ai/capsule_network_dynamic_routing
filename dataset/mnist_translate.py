from torchvision import datasets, transforms
import h5py
import torch


class MNIST:
    def __init__(self, mode='train'):

        if mode == 'train':
            h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/capsule_network_pytorch/dataset/MNIST_translate.h5', 'r')
            self.images = h5f['X_train'][:]
            self.labels = h5f['y_train'][:]
            h5f.close()
        elif mode == 'test':
            h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/capsule_network_pytorch/dataset/MNIST_translate.h5', 'r')
            self.images = h5f['X_test'][:]
            self.labels = h5f['y_test'][:]
            h5f.close()

    def __getitem__(self, index):
        img = torch.tensor(self.images[index]).unsqueeze(0).div(255.).float()
        target = torch.tensor(self.labels[index])

        # normalization & augmentation
        # img = transforms.Normalize((0.1307,), (0.3081,))(img)

        return img, target

    def __len__(self):
        return len(self.labels)
