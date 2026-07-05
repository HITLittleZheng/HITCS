import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ======================= 全局默认参数 =======================
DEFAULT_CONFIG = {
    'batch_size': 128,
    'epochs': 20,               # 每个对比实验的训练轮数（减少以节省时间）
    'lr': 1e-3,
    'latent_dim': 20,
    'use_dropout': False,
    'dropout_rate': 0.2,
    'data_fraction': 1.0,
    'activation': 'relu',       # 可选: 'relu', 'leaky_relu', 'elu'
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ======================= 实验配置列表（控制变量法） =======================
# 每个实验只改变一个因素，其他使用默认值
EXPERIMENTS = [
    # 1. 不同数据量
    {'name': 'data_10%', 'data_fraction': 0.1},
    {'name': 'data_30%', 'data_fraction': 0.3},
    {'name': 'data_100%', 'data_fraction': 1.0},
    
    # 2. Dropout 对比
    {'name': 'dropout_False', 'use_dropout': False},
    {'name': 'dropout_True', 'use_dropout': True},
    
    # 3. 隐变量维度对比
    {'name': 'latent_10', 'latent_dim': 10},
    {'name': 'latent_20', 'latent_dim': 20},
    {'name': 'latent_50', 'latent_dim': 50},
    
    # 4. 激活函数对比
    {'name': 'act_relu', 'activation': 'relu'},
    {'name': 'act_leaky_relu', 'activation': 'leaky_relu'},
    {'name': 'act_elu', 'activation': 'elu'},
]

# ======================= 模型组件 =======================
def get_activation_fn(name):
    if name == 'relu':
        return F.relu
    elif name == 'leaky_relu':
        return lambda x: F.leaky_relu(x, negative_slope=0.2)
    elif name == 'elu':
        return F.elu
    else:
        raise ValueError(f"Unsupported activation: {name}")

class Encoder(nn.Module):
    def __init__(self, latent_dim, use_dropout, dropout_rate, activation_fn):
        super(Encoder, self).__init__()
        self.act = activation_fn
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.act(self.conv1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.act(self.conv2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.act(self.conv3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, use_dropout, dropout_rate, activation_fn):
        super(Decoder, self).__init__()
        self.act = activation_fn
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.act(self.deconv1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.act(self.deconv2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.sigmoid(self.deconv3(x))
        x = x[:, :, 2:30, 2:30]   # 32 -> 28
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim, use_dropout, dropout_rate, activation_fn):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, use_dropout, dropout_rate, activation_fn)
        self.decoder = Decoder(latent_dim, use_dropout, dropout_rate, activation_fn)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.reshape(-1, 784), x.reshape(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ======================= 数据加载辅助 =======================
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

def get_train_loader(data_fraction, batch_size):
    if data_fraction < 1.0:
        num = int(len(full_train_dataset) * data_fraction)
        indices = torch.randperm(len(full_train_dataset))[:num]
        subset = Subset(full_train_dataset, indices)
    else:
        subset = full_train_dataset
    return DataLoader(subset, batch_size=batch_size, shuffle=True)

# ======================= 训练与记录 =======================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader.dataset)

def run_experiment(cfg, exp_name, results_dir='experiment_results'):
    """运行单个实验配置，保存结果和损失曲线"""
    exp_dir = os.path.join(results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    device = cfg['device']
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    lr = cfg['lr']
    latent_dim = cfg['latent_dim']
    use_dropout = cfg['use_dropout']
    dropout_rate = cfg['dropout_rate']
    data_fraction = cfg['data_fraction']
    activation_name = cfg['activation']
    activation_fn = get_activation_fn(activation_name)

    # 数据加载
    train_loader = get_train_loader(data_fraction, batch_size)
    print(f"\n>>> Running {exp_name}: data={data_fraction*100:.0f}%, latent_dim={latent_dim}, "
          f"dropout={use_dropout}, activation={activation_name}")

    # 模型、优化器
    model = VAE(latent_dim, use_dropout, dropout_rate, activation_fn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录损失
    losses = []

    for epoch in range(1, epochs+1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        losses.append(loss)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    # 保存损失曲线
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss - {exp_name}')
    plt.savefig(os.path.join(exp_dir, 'loss_curve.png'))
    plt.close()

    # 保存最终生成的图像和编辑示例（使用测试集风格）
    model.eval()
    with torch.no_grad():
        # 随机生成 64 张图像
        z = torch.randn(64, latent_dim).to(device)
        samples = model.decoder(z).cpu()
        save_image(samples, os.path.join(exp_dir, 'generated.png'), nrow=8)

        # 隐变量编辑：沿第 0、5、10 维线性插值
        base_z = torch.zeros(1, latent_dim).to(device)
        for dim in [0, 5, 10]:
            if dim >= latent_dim:
                continue
            edited = []
            for step in np.linspace(-2.5, 2.5, 11):
                z_edit = base_z.clone()
                z_edit[0, dim] = step
                img = model.decoder(z_edit).cpu()
                edited.append(img)
            grid = torch.cat(edited, dim=0)
            save_image(grid, os.path.join(exp_dir, f'edit_dim{dim}.png'), nrow=11)

        # 重建对比（前 16 个测试样本）
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon, _, _ = model(data)
        comparison = torch.cat([data.cpu(), recon.cpu()])
        save_image(comparison, os.path.join(exp_dir, 'reconstruction.png'), nrow=16)

    # 返回最终损失
    return losses[-1]

# ======================= 主控函数 =======================
def run_all_experiments():
    """依次运行所有配置，生成汇总报告"""
    results_dir = 'experiment_results'
    os.makedirs(results_dir, exist_ok=True)

    summary = []  # 存储 (exp_name, final_loss)

    for exp in EXPERIMENTS:
        # 合并默认配置
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(exp)
        exp_name = exp['name']
        final_loss = run_experiment(cfg, exp_name, results_dir)
        summary.append((exp_name, final_loss))

    # 保存汇总 CSV
    csv_path = os.path.join(results_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Final Loss'])
        writer.writerows(summary)

    # 绘制所有损失曲线的对比图（单独保存每个实验已做，这里画一个总览）
    plt.figure(figsize=(10, 6))
    for exp_name, _ in summary:
        curve_path = os.path.join(results_dir, exp_name, 'loss_curve.png')
        if os.path.exists(curve_path):
            # 读取损失数据（可从图片无法读取，我们可以重新从保存的npy或从csv读取，但简单起见：再次从各实验的loss列表记录？）
            # 更简单：在每个实验中同时保存 loss list 为 npy，这里就不做了，只打印汇总表
            pass
    # 改为打印汇总表
    print("\n========== Experiment Summary ==========")
    for exp_name, loss in summary:
        print(f"{exp_name:20s} : Final Loss = {loss:.4f}")
    print(f"Detailed results saved in '{results_dir}'")
    print("CSV summary saved at:", csv_path)

if __name__ == "__main__":
    run_all_experiments()
