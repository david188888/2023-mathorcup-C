import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('result.xlsx')


#数据标准化
label_need=df.keys()[1:]
data1=df[label_need].values
[m,n]=data1.shape
data2=data1.copy().astype('float')
for j in range(0,n):
    data2[:,j]=data1[:,j]/np.sqrt(sum(np.square(data1[:,j])))

#计算加权重后的数据
w=[0.54545455,0.27272727,0.18181818]   #使用求权重的方法求得,参见文献1
R=data2*w

#计算最大最小值距离
r_max=np.max(R,axis=0)   #每个指标的最大值
r_min=np.min(R,axis=0)   #每个指标的最小值
d_z = np.sqrt(np.sum(np.square((R -np.tile(r_max,(m,1)))),axis=1))  #d+向量
d_f = np.sqrt(np.sum(np.square((R -np.tile(r_min,(m,1)))),axis=1))  #d-向量  

#计算得分
s=d_f/(d_z+d_f )
Score=100*s/max(s)
for i in range(0,len(Score)):
    print(f"第{i+1}条线路得分百分制为:{Score[i]}")
result = pd.DataFrame(Score)
result = result.set_index(result.index+1)
print(result)
result.to_excel('Topiccs.xlsx')

