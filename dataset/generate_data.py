import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import h5py


def translate(batch, width=60, height=60):
    """Inserts MNIST digits at random locations in larger blank background."""

    n, width_img, height_img, c_img = batch.shape

    data = np.zeros((n, width, height, c_img))  # blank background for each image

    for k in range(n):
        # sample location
        x_pos = np.random.randint(0, width - width_img)
        y_pos = np.random.randint(0, height - height_img)

        # insert in blank image
        data[k, x_pos:x_pos + width_img, y_pos:y_pos + height_img, :] += batch[k]

    return data


def clutter(batch, train_data, width=60, height=60, n_patches=6):
    """Inserts MNIST digits at random locations in larger blank background and
    adds 8 by 8 subpatches from other random MNIST digits."""

    # get dimensions
    n, width_img, height_img, c_img = batch.shape
    width_sub, height_sub = 8, 8  # subpatch

    assert n > 4, 'There must be more than 4 images in the batch (there are {})'.format(n)

    data = np.zeros((n, width, height, c_img))  # blank background for each image

    for k in range(n):

        # sample location
        x_pos = np.random.randint(0, width - width_img)
        y_pos = np.random.randint(0, height - height_img)

        # insert in blank image
        data[k, x_pos:x_pos + width_img, y_pos:y_pos + height_img, :] += batch[k]

        # add 8 x 8 subpatches from random other digits
        for i in range(n_patches):
            digit = train_data[np.random.randint(0, train_data.shape[0] - 1)]
            c1, c2 = np.random.randint(0, width_img - width_sub, size=2)
            i1, i2 = np.random.randint(0, width - width_sub, size=2)
            data[k, i1:i1 + width_sub, i2:i2 + height_sub, :] += digit[c1:c1 + width_sub, c2:c2 + height_sub, :]

    data = np.clip(data, 0., 1.)

    return data


if __name__ == '__main__':

    W, H = 60, 60  # background size
    train_dataset = datasets.MNIST('.', train=True, download=True)
    train_data = train_dataset.data.reshape(-1, 28, 28, 1).numpy()
    train_label = train_dataset.targets.numpy()
    # train_data_translated = torch.tensor(np.squeeze(translate(train_data, W, H)))

    test_dataset = datasets.MNIST('.', train=False, download=True)
    test_data = test_dataset.data.reshape(-1, 28, 28, 1).numpy()
    test_label = test_dataset.targets.numpy()
    # test_data_translated = torch.tensor(np.squeeze(translate(test_data, W, H)))

    # h5f = h5py.File('MNIST_translate.h5', 'w')
    # h5f.create_dataset('X_train', data=train_data_translated)
    # h5f.create_dataset('y_train', data=train_label)
    # h5f.create_dataset('X_test', data=test_data_translated)
    # h5f.create_dataset('y_test', data=test_label)
    # h5f.close()

    train_data_cluttered = torch.tensor(np.squeeze(clutter(train_data, train_data, W, H)))
    test_data_cluttered = torch.tensor(np.squeeze(clutter(test_data, test_data, W, H)))

    h5f = h5py.File('MNIST_clutter.h5', 'w')
    h5f.create_dataset('X_train', data=train_data_cluttered)
    h5f.create_dataset('y_train', data=train_label)
    h5f.create_dataset('X_test', data=test_data_cluttered)
    h5f.create_dataset('y_test', data=test_label)
    h5f.close()

    print()






