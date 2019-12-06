import torch
import torch.nn as nn
import torch.nn.functional as F
from config import options


def squash(input_tensor, dim=-1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector


class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, cap_dim, num_cap_map):
        super(PrimaryCapsLayer, self).__init__()

        self.capsule_dim = cap_dim
        self.num_cap_map = num_cap_map
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)

    def forward(self, x):
        batch_size = x.size(0)
        outputs = self.conv_out(x)
        map_dim = outputs.size(-1)
        outputs = outputs.view(batch_size, self.capsule_dim, self.num_cap_map, map_dim, map_dim)    # [bs, 8, 32, 6, 6]
        outputs = outputs.view(batch_size, self.capsule_dim, -1).transpose(-1, -2)                  # [bs, 1152, 8]
        outputs = squash(outputs)
        return outputs


class DigitCapsLayer(nn.Module):
    def __init__(self, num_digit_cap, num_prim_cap, in_cap_dim, out_cap_dim, num_iterations):
        super(DigitCapsLayer, self).__init__()
        self.num_prim_cap = num_prim_cap
        self.num_digit_cap = num_digit_cap
        self.num_iterations = num_iterations
        self.W = nn.Parameter(0.01 * torch.randn(1, num_digit_cap, num_prim_cap, out_cap_dim, in_cap_dim))
        # [1, 10, 1152, 16, 8]

    def forward(self, x):
        batch_size = x.size(0)              # [bs,1152, 16]
        u = x.unsqueeze(1).unsqueeze(4)     # [bs, 1, 1152, 16, 1]
        u_hat = torch.matmul(self.W, u)     # [bs, 10, 1152, 16, 1]
        u_hat = u_hat.squeeze(-1)           # [bs, 10, 1152, 16]

        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.num_digit_cap, self.num_prim_cap, 1).cuda()   # [bs, 10, 1152, 1]
        for i in range(self.num_iterations - 1):
            c = F.softmax(b, dim=1)
            s = (c * temp_u_hat).sum(dim=2)     # [bs, 10, 16]
            v = squash(s)
            # [bs, 10, 1152, 16] . [batch_size, 10, 16, 1]
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))  # [batch_size, 10, 1152, 1]
            b += uv

        c = F.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)
        v = squash(s)
        return v
        # [bs, 10, 16]


class CapsuleNet(nn.Module):
    def __init__(self, args):
        super(CapsuleNet, self).__init__()
        self.args = args

        # convolution layer
        self.conv1 = nn.Conv2d(in_channels=args.img_c, out_channels=args.f_conv1, kernel_size=args.k_conv1, stride=1)

        # primary capsule layer
        assert args.f_prim % args.primary_cap_dim == 0
        self.num_primary_cap_map = int(args.f_prim / args.primary_cap_dim)
        self.primary_capsules = PrimaryCapsLayer(in_channels=args.f_conv1, out_channels=args.f_prim,
                                                 kernel_size=args.k_prim, stride=options.s_prim,
                                                 cap_dim=args.primary_cap_dim,
                                                 num_cap_map=self.num_primary_cap_map)

        num_prim_cap = int((args.img_h - 2*(args.k_prim-1)) * (args.img_h - 2*(args.k_prim-1)) / (args.s_prim*args.s_prim))

        self.digit_capsules = DigitCapsLayer(num_digit_cap=args.num_classes,
                                             num_prim_cap=self.num_primary_cap_map * num_prim_cap,
                                             in_cap_dim=args.primary_cap_dim,
                                             out_cap_dim=args.digit_cap_dim,
                                             num_iterations=args.num_iterations)

        if args.add_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(16 * args.num_classes, args.h1),
                nn.ReLU(inplace=True),
                nn.Linear(args.h1, args.h2),
                nn.ReLU(inplace=True),
                nn.Linear(args.h2, args.img_h * args.img_w),
                nn.Sigmoid()
            )

    def forward(self, imgs, y=None):
        x = F.relu(self.conv1(imgs), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)          # [bs, 10, 16]

        v_length = torch.norm(x, dim=-1)    # [bs, 10]

        _, y_pred = v_length.max(dim=1)     # [bs]
        y_pred_ohe = F.one_hot(y_pred, self.args.num_classes)   # [bs, 10]

        if y is None:
            y = y_pred_ohe

        img_reconst = torch.zeros_like(imgs)
        if self.args.add_decoder:
            img_reconst = self.decoder((x * y[:, :, None].float()).view(x.size(0), -1))

        return y_pred_ohe, img_reconst, v_length


class CapsuleLoss(nn.Module):
    def __init__(self, args):
        super(CapsuleLoss, self).__init__()
        self.args = args

    def forward(self, images, labels, v_c, reconstructions):
        present_error = F.relu(self.args.m_plus - v_c, inplace=True) ** 2  # max(0, m_plus-||v_c||)^2
        absent_error = F.relu(v_c - self.args.m_minus, inplace=True) ** 2  # max(0, ||v_c||-m_minus)^2

        l_c = labels.float() * present_error + self.args.lambda_val * (1. - labels.float()) * absent_error
        margin_loss = l_c.sum(dim=1).mean()

        reconstruction_loss = 0
        if self.args.add_decoder:
            assert torch.numel(images) == torch.numel(reconstructions)
            images = images.view(reconstructions.size()[0], -1)
            reconstruction_loss = torch.mean((reconstructions - images) ** 2)

        return margin_loss + self.args.alpha * reconstruction_loss
