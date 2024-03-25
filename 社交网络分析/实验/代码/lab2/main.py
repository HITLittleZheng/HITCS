import networkx as nx
import matplotlib.pyplot as plt
# 生成一个含有20个节点、每个节点有4个邻居、以概率p=0.3随机化重连边的WS小世界网络
G = nx.random_graphs.watts_strogatz_graph(100, 4, 0.1)
# circular布局
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels = False, node_size = 30)
# 获取各节点与相应度值的字典
degree = dict(nx.degree(G))
# 平均度为所有节点度之和除以总节点数
print("平均度为：", sum(degree.values())/len(G))
# 最短路径长度
print("最短路径长度为：",nx.average_shortest_path_length(G))
# 平均集类系数
print("平均聚类系数为：",nx.average_clustering(G))
# 获取度分布,返回所有位于区间[0, dmax]的度值的频数列表
degreeDis = nx.degree_histogram(G)
x = range(len(degreeDis))  # 生成X轴序列，从1到最大度
y = [z / float(sum(degreeDis)) for z in degreeDis]  # 将频次转化为频率
plt.figure(figsize=(5.8, 5.2), dpi=150) # 调整显示参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.xlabel("Degree", size=14)  # Degree
plt.ylabel("Frequency", size=14)  # Frequency
plt.xticks(fontproperties='Times New Roman', size=13) # 字体样式
plt.yticks(fontproperties='Times New Roman', size=13) # 字体样式
plt.plot(x,y) # 折线图
plt.show() # 显示图像