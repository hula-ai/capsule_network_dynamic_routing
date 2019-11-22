import os
import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader

from mnist_attention_old import CapsuleNet, CapsuleLoss

from torch.optim import Adam
import numpy as np
from config import options
import torch
import torch.nn.functional as F
from utils.eval_utils import compute_accuracy
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms


os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2 ,3'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def visualize_attention(img, att_maps, title='', img_save_path=None):

    img = np.minimum(np.maximum(img, 0), 1)

    avg_att_map = np.mean(att_maps, axis=0)
    max_ = avg_att_map.max()
    min_ = avg_att_map.min()
    avg_att_map = (avg_att_map - min_) / (max_ - min_)
    heatmap = cv2.applyColorMap(np.uint8(255 * np.squeeze(avg_att_map)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img, cmap='gray', alpha=0.5)
    ax.imshow(heatmap, alpha=0.3)
    ax.axis('off')
    fig.suptitle(title)
    plt.savefig(img_save_path + '.png')
    plt.close()

    fig, axes = plt.subplots(nrows=4, ncols=8)
    for i, att_map in enumerate(att_maps):
        ax = axes.ravel()[i]

        att_map = np.minimum(np.maximum(att_map, 0), 1)
        heatmap = cv2.applyColorMap(np.uint8(255 * np.squeeze(att_map)), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        ax.imshow(img, cmap='gray', alpha=0.5)
        ax.imshow(heatmap, alpha=0.3)
        ax.axis('off')
    fig.suptitle(title)
    plt.savefig(img_save_path + '_2.png')
    plt.close()


def visualize(x, y, y_pred, att_maps, batch_id):
    batch_size, num_maps, H, W, num_cls = att_maps.size()
    attention_map = att_maps[torch.arange(batch_size), :, :, :, y_pred]
    attention_map = attention_map.view(batch_size, num_maps, -1)  # (B, K, H * W)
    attention_map_max, _ = attention_map.max(dim=2, keepdim=True)  # (B, K, 1)
    attention_map_min, _ = attention_map.min(dim=2, keepdim=True)  # (B, K, 1)
    attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min)  # (B, K, H * W)
    attention_map = attention_map.view(batch_size, num_maps, H, W)  # (B, K, H, W)
    attention_map = F.upsample_bilinear(attention_map, size=(x.size(2), x.size(3)))
    for batch_index in range(batch_size):
        title = 'True: {}, Pred: {}'.format(y[batch_index], y_pred[batch_index])
        img_num = str(batch_id * batch_size + batch_index)
        img_save_path = os.path.join(viz_dir, img_num)
        visualize_attention(x[batch_index].squeeze().cpu().numpy()*255., attention_map[batch_index].cpu().numpy(), title, img_save_path)


@torch.no_grad()
def evaluate():

    capsule_net.eval()
    test_loss = 0
    targets, predictions = [], []

    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        target_ohe = F.one_hot(target, options.num_classes)
        y_pred, x_reconst, v_length, c_maps, _ = capsule_net(data, target_ohe)
        loss = capsule_loss(data, target_ohe, v_length, x_reconst)

        # visualize(data, target, y_pred.argmax(dim=1), c_maps, batch_id)

        targets += [target_ohe]
        predictions += [y_pred]
        test_loss += loss

    test_loss /= (len(test_loader) * options.batch_size)
    test_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))

    # display
    log_string("test_loss: {0:.7f}, test_accuracy: {1:.04%}"
               .format(test_loss, test_acc))


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    iter_num = options.load_model_path.split('/')[-1].split('.')[0]

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))
    img_dir = os.path.join(save_dir, 'imgs')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    viz_dir = os.path.join(img_dir, iter_num)
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    # bkp of inference
    os.system('cp {}/inference.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    capsule_net = CapsuleNet(options)
    log_string('Model Generated.')
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in capsule_net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    capsule_net.cuda()
    capsule_net = nn.DataParallel(capsule_net)

    ##################################
    # Load the trained model
    ##################################
    ckpt = options.load_model_path
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    capsule_net.load_state_dict(state_dict)
    log_string('Model successfully loaded from {}'.format(ckpt))

    ##################################
    # Loss and Optimizer
    ##################################

    capsule_loss = CapsuleLoss(options)
    optimizer = Adam(capsule_net.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################
    if options.data_name == 'mnist':
        from dataset.mnist import MNIST as data
    elif options.data_name == 'fashion_mnist':
        from dataset.fashion_mnist import FashionMNIST as data
    elif options.data_name == 't_mnist':
        from dataset.mnist_translate import MNIST as data
    elif options.data_name == 'c_mnist':
        from dataset.mnist_clutter import MNIST as data
    elif options.data_name == 'cub':
        from dataset.dataset_CUB import CUB as data

    train_dataset = data(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')

    evaluate()
