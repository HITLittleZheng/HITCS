import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.init import kaiming_uniform_, xavier_uniform_

class iris_dataset(Dataset):
    def __init__(self, path='./iris.data'):
        super().__init__()
        iris = np.loadtxt(path, dtype=str)
        iris_data = [iris[i].split(',')[:-1] for i in range(len(iris))]
        self.iris = torch.from_numpy(np.array(iris_data, dtype=float)).type(torch.float32)
        self.iris = self.iris / self.iris.norm(dim=-1, keepdim=True)

        label = [iris[i].split(',')[-1] for i in range(len(iris))]
        label_list = list(set(label))
        self.label_len = len(label_list)
        mapping = {label_list[i]: i for i in range(len(label_list))}
        self.label = [mapping[label[i]] for i in range(len(label))]

    def __len__(self):
        return self.iris.shape[0]

    def __getitem__(self, index):
        label = torch.zeros(self.label_len)
        label[self.label[index]] = 1
        return self.iris[index], label

    def get_splits(self, n_test=0.3):
        # determine sizes
        test_size = round(n_test * self.__len__())
        train_size = self.__len__() - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.relu = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, output_dim)
        xavier_uniform_(self.fc3.weight)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def data_processing():
    dataset = iris_dataset()
    train, test = dataset.get_splits()
    trainloader = DataLoader(train, batch_size=128, shuffle=True)
    testloader = DataLoader(test, batch_size=1, shuffle=False)
    return trainloader, testloader

def draw(train_losses, test_accuracies):
    plt.figure(figsize=(12, 5))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss varies with Epoch')
    plt.grid(True)
    plt.legend()

    # 绘制测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy',color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy varies with Epoch')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# 在你的训练循环中调用这个函数来绘制曲线
def train(epochs, trainloader, testloader):
    model = MLP(4, 16, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        test_accuracy = eval(model, testloader)
        test_accuracies.append(test_accuracy)
        print(f'Test Accuracy: {test_accuracy:.4f}')

    torch.save(model.state_dict(), 'checkpoint.pt')
    draw(train_losses, test_accuracies)

def eval(model, data):
    total = 0
    correct = 0
    for i, (inputs, labels) in enumerate(data):
        inputs = inputs.to(device)
        x = model(inputs)
        value, pred = torch.max(x,1)
        pred = pred.data.cpu()
        _, label = torch.max(labels,1)
        label = label.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == label)

    return correct*100./total

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainloader, testloader = data_processing()
    train(50, trainloader, testloader)
