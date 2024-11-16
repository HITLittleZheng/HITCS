import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class AlexNet(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*14*14
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 128*7*7
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def data_processing(str):
    if str == 'mnist':
        transform = transforms.ToTensor()  # 转换为张量
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        return trainloader, testloader
    elif str == 'cifar':
        transform = transforms.ToTensor()  # 转换为张量
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        return trainloader, testloader
    return None, None

def eval(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.to(device)
        x = model(images)
        value, pred = torch.max(x,1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)

    return correct*100./total

def train(model, learning_rate, epochs, trainloader, testloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    max_accuracy=0

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        for i, (images,labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_item = loss.item()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        accuracy = float(eval(model, testloader))
        test_accuracies.append(accuracy)
        print("Epoch %d accuracy: %f loss: %f" % (epoch, accuracy, loss_item))
        if accuracy > max_accuracy:
            best_model = model
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        print('Epoch:', epoch+1, "Accuracy :", accuracy, '%')
    torch.save(model.state_dict(), 'checkpoint_mnist.pt')
    draw(train_losses, test_accuracies)

    return best_model

def test(testloader):
    alexnet = AlexNet()
    alexnet.load_state_dict(torch.load('./checkpoint_mnist.pt'))
    plt.figure(figsize=(2,5))
    for i, (image, label) in enumerate(testloader):
        predict = torch.argmax(alexnet(image), axis=1)
        print((predict == label).sum() / label.shape[0])
        for j in range(10):
            plt.subplot(2, 5, j + 1)
            plt.imshow(image[j, 0], cmap='gray')
            plt.title(predict[j].item())
            plt.axis('off')
        plt.show()
        break

def draw(train_losses, test_accuracies):
    plt.figure(figsize=(12, 5))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss varies with Iteration')
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

if __name__  == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet().to(device)
    trainloader, testloader = data_processing('mnist')
    best_model = train(model=model, learning_rate=0.01, epochs=30, trainloader=trainloader, testloader=testloader)
    test(testloader)