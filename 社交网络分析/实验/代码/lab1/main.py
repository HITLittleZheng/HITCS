import pandas as pd
import networkx as nx

# 读取CSV文件，跳过首行提示信息
data = pd.read_csv('musae_facebook_edges.csv', skiprows=0)
data.columns = ['id_1', 'id_2']

# 创建无向图
G = nx.from_pandas_edgelist(data, 'id_1', 'id_2')

# 移除自环
G.remove_edges_from(nx.selfloop_edges(G))

# 计算节点数量n（不包括自环的节点）
n = G.number_of_nodes()

# 计算边的数量m（不包括自环）
m = G.number_of_edges()

# 计算网络密度d（边数不包括自环）
d = (2 * m) / (n * (n - 1))

# 计算网络的平均聚集系数（不包括自环）
average_clustering = nx.average_clustering(G)

# 输出结果
print(f'节点数量n = {n}')
print(f'边的数量m = {m}')
print(f'网络密度d = {d:.4f}')
print(f'网络平均聚集系数c = {average_clustering:.4f}')
