import operator
import pandas as pd
import numpy as np
import random

data = pd.read_csv('./Iris.csv').values
data = np.array(data)
data_1 = data[:, 1:-1]
len_x = len(data_1[0])
len_y = len(data)
len_train = 100

def train_valid_split(data,  len_train, len_x, len_y):#将数据集分为训练集和验证集
    len_test = len_y - len_train
    train_set = np.zeros([len_train, len_x])
    test_set = np.zeros([len_test, len_x])
    mark = np.zeros([len_y,1])
    answer = np.zeros([len_test, 1])
    label = np.zeros([len_train,1])
    m = 0
    for i in range(len_train):
        x = random.randint(0, 149)
        train_set[i] = data[x]
        mark[x] = x
        label[i] = x
    for j in range(len_y):
        if mark[j] == 0 and m < len_test:
            test_set[m] = data[j]
            answer[m] = j
            m += 1
    return np.array(train_set), np.array(test_set), label, answer

def Distance(train_set, test_set_i, k, label, data):
    Distance = []
    len0 = len(train_set)

    diff = np.tile(test_set_i, (len0, 1)) - train_set
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)
    distance = squaredDist ** 0.5

    sort = distance.argsort()

    classCount = {}
    for i in range(k):
        if data[label[sort[i]].astype(int),5] =='Iris-setosa':
            Labels = 1
        if data[label[sort[i]].astype(int),5] =='Iris-versicolor':
            Labels = 2
        if data[label[sort[i]].astype(int),5] =='Iris-virginica':
            Labels = 3
        classCount[Labels] = classCount.get(Labels, 0) + 1

    sortClass = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)

    return sortClass[0][0]



train_set, test_set, label, answer= train_valid_split(data_1, len_train, len_x, len_y)
k = 7
j = 0
sum = 0
for i in range(len_y - len_train):
    if data[answer[j].astype(int),5] =='Iris-setosa':
        Labels = 1
    if data[answer[j].astype(int),5] =='Iris-versicolor':
        Labels = 2
    if data[answer[j].astype(int),5] =='Iris-virginica':
        Labels = 3
    if Distance(train_set, test_set[i], k, label, data)== Labels:
        sum += 1
    print(f'第{i+1}个训练数据为{test_set[i]},标签为{Distance(train_set, test_set[i], k, label, data)},正确答案是{Labels}')
    j += 1
print(f'正确率为{(sum/(len_y - len_train))*100}%')

