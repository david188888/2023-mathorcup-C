import geatpy as ea
import numpy as np
import random


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self,):
        self.name = 'NSGA2算法'  # 初始化name（函数名称，可以随意设置）
        self.M = 3  # 优化目标个数（两个x)
        self.maxormins = [1, 1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        self.Dim = 2  # 初始化Dim（决策变量维数）
        self.varTypes = [0,0 ]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        self.lb = [0, 0]  # 决策变量下界(自定义个上下界搜索)
        self.ub = [79411, 1031]  # 决策变量上界
        self.lbin = [1, 1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        self.ubin = [1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, self.name, self.M, self.maxormins, self.Dim,
                            self.varTypes, self.lb, self.ub, self.lbin, self.ubin)
        

    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen # 得到决策变量矩阵
        # print(x)
        bool_col = x[:, 1].astype(bool)
        x[:,1] = bool_col
        print(x)
        a = 2381385
        n = len(x[:self.Dim])
        f1 = np.array([(a-np.sum(x[:,0]))/a])# 第一个目标函数
        f2 = np.array([np.count_nonzero(x[:,1] == 1)]) # 第二个目标函数
        x_mean = np.mean(x[:n])
        h = np.sqrt(1/n*(x[:n]-x_mean)**2)
        f3 = np.array([np.min(h)])
        # print(x)        print(f1)
        print(f2)
        print(f3)
        print(f1.dtype)
        print(f2.dtype)
        print(f3.dtype)

        pop.ObjV = np.array([f1, f2, f3])  # 把求得的目标函数值赋值给种群pop的ObjV
        print(pop.ObjV)
        pop.ObjV = np.transpose(pop.ObjV)  # 转置
        pop.ObjV = np.broadcast_to(pop.ObjV, (55, 3))  # 广播数组以使其形状为(50, 3)
        print(pop.ObjV)
        print(pop.ObjV.shape)
        print(pop.ObjV.dtype)
        pop.CV = -x ** 2 + 2.5 * x - 1.5  # 构建违反约束程度矩阵（需要转换为小于，反转一下）
        print(pop.CV)
        print(pop.ObjV.ndim)
        return pop.ObjV, pop.CV


    


# 实例化问题对象
problem = MyProblem()
# 构建算法
algorithm = ea.moea_NSGA2_templet(problem,
                                  # RI编码，种群个体50
                                  ea.Population(Encoding='RI', NIND=55),
                                  MAXGEN=200,  # 最大进化代数
                                  logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
# 求解

res = ea.optimize(algorithm, seed=1, verbose=False, drawing=1, outputMsg=True, drawLog=True, saveFlag=False, dirName='result',)

print(res)