import pandas as pd
import numpy as np
import random

#读取数据
data = pd.read_csv('./Iris.csv').values
data = np.array(data)
#截取特征参数及其编号
numbers = data[:, 1]
characters = data[:, 1:-1]
labels = data[:, -1]

len_y = len(characters)
len_x = len(characters[0])

def Distance(characters, len_x, len_y, centers, k):
    Distance = []
    for data in characters:
        diff = np.tile(data, (k,1)) - centers
        squaredDiff = diff ** 2
        squaredDist = np.sum(squaredDiff, axis=1)
        distance = squaredDist ** 0.5
        Distance.append(distance)
    Distance = np.array(Distance)
    return Distance


def Center(characters, centers, len_x, len_y, k):
    distance = Distance(characters, len_x, len_y, centers, k)
    min_mark = np.argmin(distance, axis=1)
    centers = pd.DataFrame(characters).groupby(min_mark).mean()
    centers = centers.values
    centers = np.array(centers)
    return centers

def K_means(numbers, characters, labels, k, len_x, len_y, epochs):
    centers = np.zeros([k, len_x])

    centers[0] = characters[random.randint(0, 49)]
    centers[1] = characters[random.randint(50, 99)]
    centers[2] = characters[random.randint(100, 149)]

    for j in range(epochs):
        centers = Center(characters, centers, len_x, len_y, k)

    centers = sorted(centers.tolist())
    distance = Distance(characters, len_x, len_y, centers, k)
    min_mark = np.argmin(distance, axis=1)
    return centers, min_mark, distance

epochs = 500
k = 3
centers, min_mark, distance = K_means(numbers, characters, labels, k, len_x, len_y, epochs)
for i in range(k):
    print(f'第{i+1}个族群的中心为{centers[i]}')
for i in range(len_y):
    print(f'第{i+1}组数据为{data[i]},属于族群{min_mark[i]+1},距离为{distance[i,min_mark[i]]}\n')