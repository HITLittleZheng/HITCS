import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm

workspace_dir = '.'

def data_processing(str):
    if str == 'mnist':
        transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        channels = 1
    elif str == 'cifar':
        transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        channels = 3
    else:
        return None, None
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    return dataloader, channels

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):

    def __init__(self, channels, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, channels, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):

    def __init__(self, channels, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        self.ls = nn.Sequential(
            nn.Conv2d(channels, dim, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

def plot_loss(g_loss, d_loss):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(g_loss)), g_loss, label="Generator Loss", alpha=0.7)
    plt.plot(np.arange(len(d_loss)), d_loss, label="Discriminator Loss", alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator and Discriminator Loss Over Training")
    plt.grid(True)
    plt.show()

def training(dataloader, channels, z_dim, n_epoch, n_critic, learning_rate=1e-4):
    z_sample = Variable(torch.randn(100, z_dim)).cuda()

    log_dir = os.path.join(workspace_dir, 'logs_final')
    ckpt_dir = os.path.join(workspace_dir, 'checkpoints_final')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    G = Generator(channels, z_dim).cuda()
    D = Discriminator(channels).cuda()
    G.train()
    D.train()

    criterion = nn.BCELoss()
    opt_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    steps = 0
    g_loss, d_loss = [], []
    for e, epoch in enumerate(range(n_epoch)):
        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar, 0):
            imgs = data[0]
            imgs = imgs.cuda()
            bs = imgs.size(0)

            # 训练判别器
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2
            D.zero_grad()
            loss_D.backward()
            opt_D.step()
            d_loss.append(loss_D.detach().cpu().numpy())
            # 训练生成器
            if steps % n_critic == 0:
                z = Variable(torch.randn(bs, z_dim)).cuda()
                f_imgs = G(z)
                f_logit = D(f_imgs)
                loss_G = criterion(f_logit, r_label)
                G.zero_grad()
                loss_G.backward()
                opt_G.step()
                g_loss.append(loss_G.detach().cpu().numpy())
            steps += 1
            progress_bar.set_description(f'Epoch {e + 1}/{n_epoch}')
            progress_bar.set_postfix(loss_D=round(loss_D.item(), 4), loss_G=round(loss_G.item(), 4))

        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
        G.train()

        if (e + 1) % 5 == 0 or e == 0:
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))
    plot_loss(g_loss, d_loss)


if __name__ == '__main__':
    dataloader, channels = data_processing('mnist')
    z_dim = 100
    n_epoch = 10
    n_critic = 1
    training(dataloader, channels, z_dim, n_epoch, n_critic)