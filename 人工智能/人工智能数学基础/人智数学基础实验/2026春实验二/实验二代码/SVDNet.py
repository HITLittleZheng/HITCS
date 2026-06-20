# DATA_ROOT = "D:\\study\\AI_Math\\experiment\\2\\SVDNet-for-Pedestrian-Retrieval-master\\market"  # 请修改为您的路径

"""
SVDNet for Pedestrian Retrieval (PyTorch CPU Implementation)
使用完整 Market-1501 训练集 + MobileNetV2 骨干网络 + 独立验证集
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models

#  配置 
DATA_ROOT = r"D:\\study\\AI_Math\\experiment\\2\\SVDNet-for-Pedestrian-Retrieval-master\\market"  # 请修改
BACKBONE = "mobilenet_v2"          # 使用轻量级网络，CPU 友好
EIGEN_DIM = 512                    # Eigenlayer 输出维度
BATCH_SIZE = 32
EPOCHS_STEP0 = 20                  # Step0 训练轮数
NUM_RRI = 7                        # RRI 迭代次数（参考 ResNet-50）
RESTRAINT_EPOCHS = 5               # 每次 Restraint 轮数
RELAXATION_EPOCHS = 5              # 每次 Relaxation 轮数
LR = 0.001
USE_EIGEN_OUTPUT = True            # 测试时使用 Eigenlayer 输出
VAL_SPLIT_RATIO = 0.2              # 从训练集中划分 20% 作为验证集

device = torch.device('cpu')
print(f"Using device: {device}")

#  数据集类 
class Market1501(Dataset):
    """Market-1501 数据集，从文件名解析 ID"""
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root = os.path.join(root_dir, mode)
        self.transform = transform
        self.mode = mode
        self.samples = []  # list of (img_path, pid)

        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        for fname in os.listdir(self.root):
            if not fname.lower().endswith(img_extensions):
                continue
            pid_str = fname.split('_')[0]
            try:
                pid = int(pid_str)
            except ValueError:
                continue

            if mode == 'train':
                if pid >= 0:
                    self.samples.append((os.path.join(self.root, fname), pid))
            else:
                if pid >= 0:
                    self.samples.append((os.path.join(self.root, fname), pid))

        if mode == 'train':
            unique_pids = sorted(set(pid for _, pid in self.samples))
            self.pid_to_label = {pid: idx for idx, pid in enumerate(unique_pids)}
            self.samples = [(path, self.pid_to_label[pid]) for path, pid in self.samples]
            self.num_classes = len(unique_pids)
        else:
            self.num_classes = None

        print(f"Loaded {len(self.samples)} images from {mode} set.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, pid


#  模型定义 
class SVDNet(nn.Module):
    def __init__(self, backbone='mobilenet_v2', num_classes=751, eigen_dim=512):
        super(SVDNet, self).__init__()
        self.eigen_dim = eigen_dim

        if backbone == 'mobilenet_v2':
            mobilenet = models.mobilenet_v2(pretrained=True)
            self.features = mobilenet.features  # 输出通道 1280
            self.fc_in_dim = 1280
        else:
            raise ValueError("Unsupported backbone. Use 'mobilenet_v2'.")

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.eigenlayer = nn.Linear(self.fc_in_dim, eigen_dim, bias=False)
        self.classifier = nn.Linear(eigen_dim, num_classes)

    def forward(self, x, return_feature=False):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.eigenlayer(x)
        if return_feature:
            return x
        return self.classifier(x)

    def get_feature(self, x):
        return self.forward(x, return_feature=True)


#  辅助函数 
def compute_S(W):
    """计算正交性度量 S(W) = sum(diag(G)) / sum(|G|), G = W^T W"""
    G = W.T @ W
    diag_sum = torch.sum(torch.abs(torch.diag(G)))
    total_sum = torch.sum(torch.abs(G))
    return (diag_sum / total_sum).item() if total_sum > 0 else 1.0


def decorrelate(eigenlayer):
    """SVD 替换: W <- U * S"""
    with torch.no_grad():
        W = eigenlayer.weight.data  # shape (out_dim, in_dim)
        m, n = W.shape
        # 使用 full_matrices=True 得到完整的 U (m x m) 和 Vh (n x n)
        U, S, Vh = torch.linalg.svd(W, full_matrices=True)
        # 构造矩形对角矩阵 S_mat: shape (m, n)
        S_mat = torch.zeros(m, n, device=W.device, dtype=W.dtype)
        min_dim = min(m, n)
        S_mat[:min_dim, :min_dim] = torch.diag(S[:min_dim])
        # 替换为 U @ S_mat，形状 (m, n)
        eigenlayer.weight.data = U @ S_mat


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc='Training', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / len(loader.dataset), 100. * correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Validation', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return running_loss / len(loader.dataset), 100. * correct / total


def train_step0(model, train_loader, val_loader, epochs, lr):
    print("\n===== Step 0 Training =====")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 2, gamma=0.1)
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "step0_best.pth")
    print("Step 0 finished. Best model saved as step0_best.pth")
    return model


def train_rri(model, train_loader, val_loader, num_rri, restraint_epochs, relaxation_epochs, lr):
    print("\n===== RRI Training =====")
    criterion = nn.CrossEntropyLoss()
    sw_history = []

    for rri_idx in range(1, num_rri + 1):
        print(f"\n--- RRI Iteration {rri_idx} ---")

        # 1. Decorrelation
        decorrelate(model.eigenlayer)
        sw = compute_S(model.eigenlayer.weight.data)
        sw_history.append(sw)
        print(f"  After decorrelation: S(W) = {sw:.6f}")

        # 2. Restraint (fix eigenlayer)
        for param in model.eigenlayer.parameters():
            param.requires_grad = False
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=restraint_epochs // 2, gamma=0.1)
        for ep in range(restraint_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(model, val_loader, criterion)
            scheduler.step()
            print(f"  Restraint epoch {ep+1}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")

        # 3. Relaxation (unfreeze eigenlayer)
        for param in model.eigenlayer.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=relaxation_epochs // 2, gamma=0.1)
        for ep in range(relaxation_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(model, val_loader, criterion)
            scheduler.step()
            print(f"  Relaxation epoch {ep+1}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")

        torch.save(model.state_dict(), f"rri_{rri_idx}_model.pth")

    print("RRI finished. Final model saved as final_model.pth")
    torch.save(model.state_dict(), "final_model.pth")
    print("S(W) history:", sw_history)
    return model


#  评估函数 
def extract_features(model, loader, use_eigen_output=True):
    model.eval()
    features, pids = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Extracting features', leave=False):
            inputs = inputs.to(device)
            if use_eigen_output:
                feat = model.get_feature(inputs)
            else:
                x = model.features(inputs)
                x = model.gap(x)
                feat = x.view(x.size(0), -1)
            features.append(feat.cpu())
            pids.extend(targets.cpu().numpy())
    features = torch.cat(features, dim=0).numpy()
    pids = np.array(pids)
    return features, pids


def evaluate(model, query_loader, gallery_loader, use_eigen_output=True):
    print("\n===== Evaluation =====")
    print("Extracting query features...")
    q_feat, q_pid = extract_features(model, query_loader, use_eigen_output)
    print("Extracting gallery features...")
    g_feat, g_pid = extract_features(model, gallery_loader, use_eigen_output)

    # L2 normalize
    q_feat = q_feat / (np.linalg.norm(q_feat, axis=1, keepdims=True) + 1e-12)
    g_feat = g_feat / (np.linalg.norm(g_feat, axis=1, keepdims=True) + 1e-12)

    # 使用 torch.cdist 加速距离计算
    q_t = torch.from_numpy(q_feat)
    g_t = torch.from_numpy(g_feat)
    dist_mat = torch.cdist(q_t, g_t, p=2).numpy()

    m, n = dist_mat.shape
    indices = np.argsort(dist_mat, axis=1)

    # CMC 和 mAP
    cmc = np.zeros(n)
    ap = np.zeros(m)
    for i in range(m):
        same_id = (g_pid == q_pid[i])
        if np.sum(same_id) == 0:
            continue
        rank = indices[i]
        match_pos = np.where(same_id[rank])[0]
        if len(match_pos) == 0:
            continue
        first_match = match_pos[0]
        cmc[first_match:] += 1
        rel = same_id[rank]
        rel_cumsum = np.cumsum(rel)
        ap[i] = np.sum(rel_cumsum * rel / (np.arange(1, len(rel) + 1))) / np.sum(rel)

    rank1 = cmc[0] / m
    mAP = np.mean(ap)
    print(f"Rank-1 Accuracy: {rank1 * 100:.2f}%")
    print(f"mAP: {mAP * 100:.2f}%")
    return rank1, mAP


#  主流程 
def main():
    # 检查数据集路径
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"数据集路径不存在: {DATA_ROOT}")
    for subdir in ['train', 'query', 'gallery']:
        if not os.path.exists(os.path.join(DATA_ROOT, subdir)):
            raise FileNotFoundError(f"缺少子文件夹: {subdir}")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集（完整）
    full_train_set = Market1501(DATA_ROOT, transform=transform_train, mode='train')
    num_classes = full_train_set.num_classes
    print(f"Number of training classes: {num_classes}")

    # 划分训练集和验证集
    val_size = int(len(full_train_set) * VAL_SPLIT_RATIO)
    train_size = len(full_train_set) - val_size
    train_subset, val_subset = random_split(full_train_set, [train_size, val_size])
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 加载 query 和 gallery
    query_set = Market1501(DATA_ROOT, transform=transform_test, mode='query')
    gallery_set = Market1501(DATA_ROOT, transform=transform_test, mode='gallery')
    query_loader = DataLoader(query_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    gallery_loader = DataLoader(gallery_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 创建模型
    model = SVDNet(backbone=BACKBONE, num_classes=num_classes, eigen_dim=EIGEN_DIM)
    model.to(device)

    # Step 0 训练
    model = train_step0(model, train_loader, val_loader, EPOCHS_STEP0, LR)
    model.load_state_dict(torch.load("step0_best.pth", map_location=device))

    # RRI 训练
    model = train_rri(model, train_loader, val_loader, NUM_RRI,
                      RESTRAINT_EPOCHS, RELAXATION_EPOCHS, LR)

    # 最终评估
    model.load_state_dict(torch.load("final_model.pth", map_location=device))
    evaluate(model, query_loader, gallery_loader, use_eigen_output=USE_EIGEN_OUTPUT)


if __name__ == "__main__":
    main()