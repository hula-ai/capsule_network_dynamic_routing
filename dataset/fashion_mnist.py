from torchvision import datasets, transforms


class FashionMNIST:
    def __init__(self, mode='train'):
        # dataset_transform = transforms.Compose([transforms.ToTensor(),
        #                                         transforms.Normalize((0.1307,), (0.3081,))
        #                                         ])
        if mode == 'train':
            train_dataset = datasets.FashionMNIST('./dataset', train=True, download=True)
            self.images = train_dataset.data.float()
            self.labels =train_dataset.targets
        elif mode == 'test':
            test_dataset = datasets.FashionMNIST('./dataset', train=False, download=True)
            self.images = test_dataset.data.float()
            self.labels =test_dataset.targets

    def __getitem__(self, index):
        img = self.images[index].unsqueeze(0).div(255.)
        target = self.labels[index]

        # normalization & augmentation
        # img = transforms.Normalize((0.1307,), (0.3081,))(img)

        return img, target

    def __len__(self):
        return len(self.labels)
