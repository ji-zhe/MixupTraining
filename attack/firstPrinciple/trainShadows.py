import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import grad

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import urllib3
from models import *
import json

gender_idx = 20
target_idx=gender_idx
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gradient_penalty(x, y, c):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = discriminator(z, c)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp

if __name__ == "__main__":
    f = open('./jsonfile/idx.json','r')
    datasetIdx = json.load(f)
    f.close()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--n_iters", type=int, default=100000, help="number of iterations of training")
    parser.add_argument("--lr", type=float, default=4e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--n_class", type=int, default=2, help="number of classes")
    opt = parser.parse_args()
    print(opt)
    bs = 100
    lambda_gp = 10
    latent_dim = opt.latent_dim
    n_class = opt.n_class

    trans_crop = transforms.CenterCrop(128)
    trnas_resize = transforms.Resize(64)
    trans_tensor = transforms.ToTensor()
    trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])

    datasetT = torchvision.datasets.CelebA('../dataset/', split= 'test', target_type= 'attr', transform = trans, target_transform = None, download = False)

    for shadowID in range(512):
        dataset = torch.utils.data.Subset(datasetT, datasetIdx[shadowID])
        dataloader = torch.utils.data.DataLoader(dataset, bs, shuffle=True)

        generator = GanGenerator(z_dim=latent_dim, y_dim=n_class)
        generator = nn.DataParallel(generator).cuda()
        discriminator = GanDiscriminator(y_dim=n_class)
        discriminator = nn.DataParallel(discriminator).cuda()

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        result_dir = f'./models/base/{shadowID}/images/'
        os.makedirs(result_dir,exist_ok=True)
        store_param = f'./models/base/{shadowID}/params/'
        os.makedirs(store_param,exist_ok=True)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        for epoch in range(opt.n_epochs):
            for i, (imgs, cond,_) in enumerate(dataloader):
                real_imgs = imgs.cuda()
                cond = cond[:,target_idx].cuda()
                real_label = torch.ones(real_imgs.shape[0]).cuda()
                fake_label = torch.zeros(real_imgs.shape[0]).cuda()
                # cond = cond[:,target_idx].cuda()
                cond = torch.nn.functional.one_hot(cond, n_class)
                # Sample noise as generator input
                z = torch.randn((imgs.shape[0], latent_dim), requires_grad=True).cuda()

                # Generate a batch of images
                fake_imgs = generator(z, cond)

                # Real images
                real_validity = discriminator(real_imgs, cond)
                fake_validity = discriminator(fake_imgs, cond)
                gp = gradient_penalty(real_imgs.data, fake_imgs.data, cond.data)

                # Adversarial loss
                # d_loss = loss_fn(real_validity, real_label) + loss_fn(fake_validity, fake_label)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()
                if i % 5 == 0:
                    fake_imgs = generator(z, cond)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = discriminator(fake_imgs, cond)

                    # z1, z2 = torch.randn(bs, latent_dim).cuda(), torch.randn(bs, latent_dim).cuda()
                    # loss_div = -torch.mean(torch.norm(Enc(generator(z1)) - Enc(generator(z2))) / torch.norm(z1 - z2))
                    # g_loss = loss_fn(fake_validity, real_label)  # + 0.01 * loss_div
                    g_loss = -torch.mean(fake_validity) #+ 0.01 * loss_div

                    g_loss.backward()
                    optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [loss_div: %s]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), "deprecated")
                        # loss_div.item()
                    )

            if epoch % 10 == 0:
                torch.save(generator.module.state_dict(), store_param + 'G_{}.pkl'.format(epoch))
                torch.save(discriminator.module.state_dict(), store_param + 'D_{}.pkl'.format(epoch))
                save_image(fake_imgs.detach().data, result_dir + '/img_{}.jpg'.format(epoch))

        torch.save(generator.module.state_dict(), store_param + 'G_{}.pkl'.format(epoch))
        torch.save(discriminator.module.state_dict(), store_param + 'D_{}.pkl'.format(epoch))
        save_image(fake_imgs.detach().data, result_dir + '/img_{}.jpg'.format(epoch))
