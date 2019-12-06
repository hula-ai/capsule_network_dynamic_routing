from torchvision import datasets, transforms


class MNIST:
    def __init__(self, mode='train'):

        if mode == 'train':
            train_dataset = datasets.MNIST('./dataset', train=True, download=True)
            self.images = train_dataset.data.float()
            self.labels =train_dataset.targets
        elif mode == 'test':
            test_dataset = datasets.MNIST('./dataset', train=False, download=True)
            self.images = test_dataset.data.float()
            self.labels = test_dataset.targets

    def __getitem__(self, index):
        img = self.images[index].unsqueeze(0).div(255.)
        target = self.labels[index]

        # normalization & augmentation
        img = transforms.Normalize((0.1307,), (0.3081,))(img)

        return img, target

    def __len__(self):
        return len(self.labels)
