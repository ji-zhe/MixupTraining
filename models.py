import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchvision

class GanGenerator(nn.Module): #z_dim=100
    def __init__(self, z_dim=100, dim=64, y_dim=None):
        super(GanGenerator, self).__init__()

        def dconv_bn_relu(z_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(z_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(z_dim+y_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, z, cond):
        # cond = torch.nn.functional.one_hot(cond, self.y_dim)
        main_input = torch.cat((z,cond), dim=1)
        y = self.l1(main_input)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class GanDiscriminator(nn.Module):
    def __init__(self, channel_num=3, dim=64, y_dim=None):
        super(GanDiscriminator, self).__init__()
        self.y_dim = y_dim

        def conv_ln_lrelu(z_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(z_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(channel_num+y_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x, cond):
        # cond = torch.nn.functional.one_hot(cond, self.y_dim)
        cond = cond.reshape(*cond.shape,1,1)
        cond = cond.expand(*cond.shape[:-2],*x.shape[2:])
        main_input = torch.cat((x,cond), dim=1)
        y = self.ls(main_input)
        y = y.view(-1)
        return y



