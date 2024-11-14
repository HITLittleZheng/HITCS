import numpy as np
import matplotlib.pyplot as plt

cont = 800  # 样本点数量
a = 100  # 欲拟合函数参数1
w = 1  # 欲拟合函数参数2
m = 10  # 拟合使用的多项式阶数
Epoch = 10000  # 迭代次数
alpha = 1  # 学习率的初值
epsilon = 1  # 学习率的参数
flag = 0  # 0 表示不使用惩罚项 1 表示使用惩罚项
lamb = 1000  # 惩罚项的系数
rho = 0.2  # 累计梯度的参数


def get_data(a, w, cont, m):
    x = np.linspace(-3, 3, cont)
    y = a * np.sin(w*x) + np.random.normal(0, 10, cont)
    # y = 10*np.sin(np.pi*x) + 20*np.cos(x/2) + np.random.normal(0, 1, cont)
    input_x = x[:int(cont*2/3)]
    input_y = y[:int(cont*2/3)]
    # input_x = x
    # input_y = y
    W = np.zeros(m+1)
    return x, y, input_x, input_y, W


def predict(x, W):
    y_hat = x.dot(W)
    return y_hat


def get_loss(y, y_hat, W, flag):
    loss = 0
    for i in range(len(y)):
        loss += (y[i]-y_hat[i])**2
    loss /= len(y)*2
    if flag == 1:
        for i in range(len(W)):
            loss += lamb * (W[i]**2)
    # print(loss)
    return loss


def train(input_x, input_y, x, y, W):
    # preparation
    X1 = np.ones((len(input_x), 1))
    X2 = np.ones((len(x), 1))
    tmp1 = np.ones(len(input_x))
    tmp2 = np.ones(len(x))
    g = np.zeros(len(W))
    G = np.zeros(len(W))
    eta = np.zeros(len(W))

    for i in range(len(W)-1):
        tmp1 = tmp1 * input_x
        X1 = np.hstack((X1, tmp1.reshape(len(tmp1), 1)))
        tmp2 = tmp2 * x
        X2 = np.hstack((X2, tmp2.reshape(len(tmp2), 1)))
    # print(X)

    train_losses = []
    all_losses = []
    loss_x = []

    # train
    for epoch in range(Epoch):
        y_hat1 = predict(X1, W)
        loss1 = get_loss(input_y, y_hat1, W, flag)
        y_hat2 = predict(X2, W)
        loss2 = get_loss(y, y_hat2, W, flag)
        if flag == 0:
            G = G - (1-rho)*g**2 + rho*g**2
            g = (X1.T.dot(y_hat1-input_y))/len(input_y)
            G = G + (1-rho)*g**2
            eta = alpha/(np.sqrt(G)+epsilon)
            W = W - eta * g
        else:
            G = G - (1 - rho) * g ** 2 + rho * g ** 2
            g = (X1.T.dot(y_hat1-input_y))/len(input_y) + 2*lamb*W
            G = G + (1-rho)*g**2
            eta = alpha/(np.sqrt(G)+epsilon)
            W = W - eta * g
        if epoch % 100 == 0:
            print(W)
            print("loss={}".format(loss1))
            train_losses.append(loss1)
            all_losses.append(loss2)
            loss_x.append(epoch)

    return W, train_losses, all_losses, loss_x


if __name__ == "__main__":
    x, y, input_x, input_y, W = get_data(a, w, cont, m)
    W1, train_losses1, all_losses1, loss_x1 = train(input_x, input_y, x, y, W)

    X = np.ones((len(x), 1))
    tmp = np.ones(len(x))

    for i in range(len(W)-1):
        tmp = tmp * x
        X = np.hstack((X, tmp.reshape(len(tmp), 1)))

    y_hat1 = predict(X, W1)

    flag = 1
    W2, train_losses2, all_losses2, loss_x2 = train(input_x, input_y, x, y, W)
    y_hat2 = predict(X, W2)

    print("W1 = {}".format(W1))
    print("W2 = {}".format(W2))

    loss1 = get_loss(y, y_hat1, W, 0)
    loss2 = get_loss(y, y_hat2, W, 0)
    print("loss1={}".format(loss1))
    print("loss2={}".format(loss2))
    plt.plot(x, y, 'bo', label='real-data')
    plt.plot(x, y_hat1, 'r', label='predicted-data-unpunished')
    plt.plot(x, y_hat2, 'c', label='predicted-data-punished')
    plt.legend()
    plt.show()

    plt.plot(loss_x1, train_losses1, 'b', label='loss-unpunished on train set')
    # plt.plot(loss_x1, all_losses1, 'r', label='loss-unpunished on all set')
    plt.plot(loss_x2, train_losses2, 'c', label='loss-punished on train set')
    # plt.plot(loss_x2, all_losses2, 'yellow', label='loss-punished on all set')
    plt.legend()
    plt.show()