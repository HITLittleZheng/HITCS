import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 1. 数据准备 ====================
# 为了适应 AlexNet，将 MNIST 图像从 28x28 放大到 224x224（标准 AlexNet 输入尺寸）
# 注意：放大后计算量增大，可适当减少 epochs 或使用更小的 batch_size
image_size = 224  # 标准 AlexNet 输入尺寸，也可以改为 64 以加速（但会偏离原始结构）

transform_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 全局均值标准差
])

transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载完整训练集和测试集
train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 不同训练数据量
data_sizes = [500, 1000, 3000, len(train_full)]
subsets = {}
for size in data_sizes:
    indices = np.random.choice(len(train_full), size, replace=False)
    subsets[size] = Subset(train_full, indices)

# ==================== 2. AlexNet 模型定义 ====================
class AlexNet(nn.Module):
    """针对 MNIST 改编的 AlexNet（输入通道=1，输出类别=10）"""
    def __init__(self, activation='relu', dropout_rate=0.5):
        super(AlexNet, self).__init__()
        self.activation_name = activation
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation")
        
        # 卷积层部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # 输入通道改为1
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应池化到固定大小 (6x6)，以便全连接层处理任意输入尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 全连接层部分
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            self.activation,
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            self.activation,
            nn.Linear(4096, 10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==================== 3. 训练与评估函数 ====================
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(train_loader), 100.0 * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return test_loss / len(test_loader), 100.0 * correct / total

def train_model(model, train_loader, epochs, lr, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_test_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    return best_test_acc

# ==================== 实验设置 ====================
base_epochs = 15       # 由于图像放大到224x224，训练较慢，可适当减少（例如10）
base_batch_size = 64   # 显存不足可降至32
base_lr = 0.001

# 实验5.1：不同激活函数（固定数据量1000，无Dropout？AlexNet自带Dropout，但为了公平，这里统一使用dropout_rate=0.5）
print("\n" + "="*60)
print("实验5.1：不同激活函数对比 (训练集大小=1000, epochs=15, lr=0.001, dropout=0.5)")
print("="*60)
act_functions = ['relu', 'tanh', 'sigmoid']
act_results = {}
for act in act_functions:
    print(f"\n>>> 训练激活函数: {act.upper()}")
    model = AlexNet(activation=act, dropout_rate=0.5)
    train_loader = DataLoader(subsets[1000], batch_size=base_batch_size, shuffle=True)
    best_acc = train_model(model, train_loader, base_epochs, base_lr, device)
    act_results[act] = best_acc
    print(f"最佳测试准确率: {best_acc:.2f}%")

# 实验5.2：Dropout 效果对比（固定数据量1000，ReLU，不同dropout率）
print("\n" + "="*60)
print("实验5.2：Dropout效果对比 (训练集大小=1000, ReLU, epochs=15, lr=0.001)")
print("="*60)
dropout_rates = [0.0, 0.3, 0.5]  # 0.0表示不使用dropout
dropout_results = {}
for dr in dropout_rates:
    print(f"\n>>> Dropout率: {dr}")
    model = AlexNet(activation='relu', dropout_rate=dr)
    train_loader = DataLoader(subsets[1000], batch_size=base_batch_size, shuffle=True)
    best_acc = train_model(model, train_loader, base_epochs, base_lr, device)
    dropout_results[dr] = best_acc
    print(f"最佳测试准确率: {best_acc:.2f}%")

# 实验5.3：不同数据量的影响（固定ReLU，dropout=0.5，学习率0.001）
print("\n" + "="*60)
print("实验5.3：不同训练数据量对比 (ReLU, dropout=0.5, epochs=15, lr=0.001)")
print("="*60)
data_results = {}
for size in data_sizes:
    print(f"\n>>> 训练集大小: {size}")
    model = AlexNet(activation='relu', dropout_rate=0.5)
    train_loader = DataLoader(subsets[size], batch_size=base_batch_size, shuffle=True)
    best_acc = train_model(model, train_loader, base_epochs, base_lr, device)
    data_results[size] = best_acc
    print(f"最佳测试准确率: {best_acc:.2f}%")

# 实验5.4：不同学习率的影响（固定数据量1000，ReLU，dropout=0.5）
print("\n" + "="*60)
print("实验5.4：不同学习率对比 (训练集大小=1000, ReLU, dropout=0.5, epochs=15)")
print("="*60)
lrs = [0.0001, 0.001, 0.01]
lr_results = {}
for lr in lrs:
    print(f"\n>>> 学习率: {lr}")
    model = AlexNet(activation='relu', dropout_rate=0.5)
    train_loader = DataLoader(subsets[1000], batch_size=base_batch_size, shuffle=True)
    best_acc = train_model(model, train_loader, base_epochs, lr, device)
    lr_results[lr] = best_acc
    print(f"最佳测试准确率: {best_acc:.2f}%")

# ==================== 5. 可视化与结论 ====================
plt.figure(figsize=(16, 12))

# 激活函数对比
plt.subplot(2, 2, 1)
acts = list(act_results.keys())
accs = list(act_results.values())
plt.bar(acts, accs, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Activation Function Comparison (1000 samples)')
plt.ylabel('Test Accuracy (%)')
for i, v in enumerate(accs):
    plt.text(i, v+0.5, f'{v:.1f}%', ha='center')
plt.ylim(0, 100)

# Dropout对比
plt.subplot(2, 2, 2)
drates = [str(d) for d in dropout_rates]
daccs = list(dropout_results.values())
plt.bar(drates, daccs, color='lightcoral')
plt.title('Dropout Effect (1000 samples, ReLU)')
plt.xlabel('Dropout rate')
plt.ylabel('Test Accuracy (%)')
for i, v in enumerate(daccs):
    plt.text(i, v+0.5, f'{v:.1f}%', ha='center')
plt.ylim(0, 100)

# 数据量对比
plt.subplot(2, 2, 3)
sizes = [str(s) for s in data_sizes]
saccs = list(data_results.values())
plt.plot(sizes, saccs, marker='o', linestyle='-', color='purple')
plt.title('Impact of Training Data Size')
plt.xlabel('Number of training samples')
plt.ylabel('Test Accuracy (%)')
plt.grid(True)
for i, v in enumerate(saccs):
    plt.text(i, v+0.5, f'{v:.1f}%', ha='center')

# 学习率对比
plt.subplot(2, 2, 4)
lrs_str = [str(l) for l in lrs]
laccs = list(lr_results.values())
plt.bar(lrs_str, laccs, color='goldenrod')
plt.title('Learning Rate Comparison (1000 samples)')
plt.xlabel('Learning rate')
plt.ylabel('Test Accuracy (%)')
for i, v in enumerate(laccs):
    plt.text(i, v+0.5, f'{v:.1f}%', ha='center')
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig('alexnet_mnist_experiments.png', dpi=150)
plt.show()
