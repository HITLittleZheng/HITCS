import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数
batch_size = 64
epochs = 15
learning_rate = 0.001

# 数据预处理：转换为Tensor并归一化到[0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 展平为784维向量
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义网络结构：784 -> 128 -> 64 -> 10
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)   # 输出logits，未经过softmax
        return x

# 将标签转换为one-hot编码
def to_one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels].to(device)

# 计算KL散度：对于one-hot真实分布，KL(P||Q) = -log(Q_correct)
def kl_divergence(probs, labels):
    # probs: softmax概率输出, labels: 原始标签(0-9)
    # 取出每个样本对应正确类别的概率
    correct_class_probs = probs[range(len(probs)), labels]
    # 添加微小项避免log(0)
    kl = -torch.log(correct_class_probs + 1e-8)
    return kl.mean().item()

# 训练一个实验
def train_experiment(loss_type, epochs, train_loader, test_loader):
    model = MLP().to(device)
    if loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()  # 内部包含LogSoftmax + NLLLoss
    elif loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError("loss_type must be 'ce' or 'mse'")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录每个epoch在测试集上的指标
    test_losses = []
    test_kls = []
    test_accuracies = []
    
    for epoch in range(1, epochs+1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            if loss_type == 'ce':
                loss = criterion(logits, labels)
            else:  # mse
                # 将标签转为one-hot
                one_hot_labels = to_one_hot(labels)
                # 对logits应用softmax得到概率分布
                probs = torch.softmax(logits, dim=1)
                loss = criterion(probs, one_hot_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        
        # 测试阶段：计算损失、KL散度、准确率
        model.eval()
        test_loss = 0.0
        test_kl = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)  # 概率分布
                
                # 计算损失
                if loss_type == 'ce':
                    loss = criterion(logits, labels)
                else:
                    one_hot_labels = to_one_hot(labels)
                    loss = criterion(probs, one_hot_labels)
                test_loss += loss.item() * images.size(0)
                
                # 计算KL散度
                kl_batch = kl_divergence(probs, labels)
                test_kl += kl_batch * images.size(0)
                
                # 计算准确率
                preds = torch.argmax(probs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        test_loss /= len(test_loader.dataset)
        test_kl /= len(test_loader.dataset)
        accuracy = 100.0 * correct / total
        
        test_losses.append(test_loss)
        test_kls.append(test_kl)
        test_accuracies.append(accuracy)
        
        print(f"{loss_type.upper()} | Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | KL: {test_kl:.4f} | Acc: {accuracy:.2f}%")
    
    return test_losses, test_kls, test_accuracies

# 运行两个实验
print("===== 实验组：交叉熵损失 =====")
ce_losses, ce_kls, ce_accs = train_experiment('ce', epochs, train_loader, test_loader)

print("\n===== 对照组：均方误差损失（one-hot） =====")
mse_losses, mse_kls, mse_accs = train_experiment('mse', epochs, train_loader, test_loader)

# 绘制三条曲线对比
epochs_range = range(1, epochs+1)

plt.figure(figsize=(15, 4))

# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(epochs_range, ce_losses, 'b-o', label='CrossEntropyLoss')
plt.plot(epochs_range, mse_losses, 'r-s', label='MSELoss (one-hot)')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Loss Curves')
plt.legend()
plt.grid(True)

# KL散度曲线
plt.subplot(1, 3, 2)
plt.plot(epochs_range, ce_kls, 'b-o', label='CrossEntropyLoss')
plt.plot(epochs_range, mse_kls, 'r-s', label='MSELoss (one-hot)')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.title('KL Divergence Curves')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 3, 3)
plt.plot(epochs_range, ce_accs, 'b-o', label='CrossEntropyLoss')
plt.plot(epochs_range, mse_accs, 'r-s', label='MSELoss (one-hot)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 最终性能比较
print("\n========== 最终性能比较 ==========")
print(f"交叉熵损失 - 最终测试准确率: {ce_accs[-1]:.2f}% | 最终KL散度: {ce_kls[-1]:.6f}")
print(f"均方误差损失 - 最终测试准确率: {mse_accs[-1]:.2f}% | 最终KL散度: {mse_kls[-1]:.6f}")