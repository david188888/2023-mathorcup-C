import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


df = pd.read_excel('../data.xlsx', sheet_name='Sheet1', header=0)
lista = [] # 存储每个场地的货物量
df1 = pd.read_excel('total_sum.xlsx', header=0)


total_sum = 0
for item in df['场地1'].unique():
    try:
        sum1 = df.loc[df['场地1'] == item]['货量'].sum()
        sum2 = df.loc[df['场地2'] == item]['货量'].sum()
        total_sum += sum1 + sum2
        lista.append(sum1 + sum2)
    except KeyError:
        sum1 = df.loc[df['场地1'] == item]['货量'].sum()
        total_sum += sum1
        lista.append(sum1)

print(lista)
list_sum = [] # 存储场地编号
list_first = [] # 存储第一个场地的货物量
index = 0
for i in df['场地1'].unique():
    list_first.append(i)


list_sum = list_first + list(df['场地2'].unique())


total_dict = dict(zip(list_first, lista))
# print(total_dict)

df_total_sum = pd.DataFrame.from_dict(total_dict, orient='index', columns=['货物量'])
# print(df_total_sum)
# df_total_sum.to_excel('total_sum.xlsx')


df['日期'] = pd.to_datetime(df['日期'])
df.index = df['日期']
df = df.sort_index()
df = df.drop(['日期'], axis=1)
# print(df.head(10))




# 读取 Excel 文件，假设文件名为 data.xlsx，第一列名为 Name，第二列名为 Age
df1 = pd.read_excel('../data.xlsx', usecols=['场地1', '场地2'])


# 去除重复数据
df = df.drop_duplicates()
# 将 Name 和 Age 列组合成一个数组，并将这些数组组合成一个大数组
result = df1.values.tolist()

print(len(result))






# 初始化一个有向图对象
DG = nx.DiGraph()
# 添加节点   传入列表
DG.add_nodes_from(list_sum)
print(f'输出图的全部节点：{DG.nodes}')
print(f'输出节点的数量：{DG.number_of_nodes()}')
# 添加边  传入列表  列表里每个元素是一个元组  元组里表示一个点指向另一个点的边
DG.add_edges_from(result)
print(f'输出图的全部边:{DG.edges}')
print(f'输出边的数量：{DG.number_of_edges()}')
# 计算每个节点的度
degrees = dict(DG.degree())

# 映射度数到颜色
color_map = []
for node in DG.nodes():
    degree = degrees[node]
    if degree <= 2:
        color_map.append('brown')
    elif degree <= 4:
        color_map.append('red')
    elif degree <= 6:
        color_map.append('brown')
    elif degree <= 8:
        color_map.append('brown')
    elif degree <= 10:
        color_map.append('red')
    else:
        color_map.append('brown')

# 绘制网络图
# 运用布局

pos = nx.random_layout(DG)
#透明度设置
nx.draw(DG, pos=pos, with_labels=True, node_size=100, width=0.6, node_color=color_map, font_size=10,font_color='white',edge_color='black')

# 绘制网络图
# 展示图片
plt.show()


