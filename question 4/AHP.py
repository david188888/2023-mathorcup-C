import numpy as np
from numpy import linalg
# np.set_printoptions(precision=4)
'''算术平均法权重'''


matrix = np.array([[1,2,3,4],[1/2,1,3/2,2],[1/3,2/3,1,4/3],[1/4,1/2,3/4,1]])
def arithmetic_mean(a):
    n = len(a)
    b = sum(a)
    print('b:', b)
    # 归一化处理
    normal_a = a/b
    print("算术平均法权重-归一化处理：")
    print(normal_a)
    average_weight = []
    for i in range(n):
        s = sum(normal_a[i])
        print("第{}行求和 ".format(i+1), s)
        # 平均权重
        average_weight.append(s/n)
    # print(average_weight)
    print('算术平均法权重:')
    print(np.array(average_weight))
    return np.array(average_weight)


arithmetic_mean(matrix)

