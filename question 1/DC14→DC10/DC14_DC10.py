import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
import seaborn as sns
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.stats.diagnostic import unitroot_adf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import warnings
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


# 设置字体
plt.rcParams['font.sans-serif']=['SimHei']
df = pd.read_excel('data.xlsx', header=0)

"""数据预处理"""



data1 = df.loc[(df["场地1"] == "DC14") & (df['场地2'] == "DC10")]#DC14 -> DC10
# print(data1)
data1.index = pd.to_datetime(data1['日期'])
data1.drop(['日期', '场地1', '场地2'], axis=1, inplace=True)

# print(data1.columns)






"""数据可视化"""

data1.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title('DC14 -> DC10的趋势图')
sns.despine()
plt.show()


"""数据平稳性检验"""
#数据平稳性检测 因为只有平稳数据才能做时间序列分析
def judge_stationarity(data_sanya_one):
    dftest = ts.adfuller(data_sanya_one)
    print(dftest)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    stationarity = 1
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value 
        if dftest[0] > value:
                stationarity = 0
    print(dfoutput)
    print("是否平稳(1/0): %d" %(stationarity))
    return stationarity
stationarity = judge_stationarity(data1)



#若不平稳进行一阶差分
if stationarity == 0:
    data_diff = data1.diff()
    data_diff = data_diff.dropna()
    plt.figure()
    plt.plot(data_diff)
    plt.title('一阶差分')
    plt.show()
 
#再次进行平稳性检测
stationarity = judge_stationarity(data_diff)



#画ACF图和PACF图来确定p、q值
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
 
def draw_acf_pacf(ts,lags):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts,ax=ax1,lags=lags)  #lags 表示滞后的阶数，值为30，显示30阶的图像
    ax2 = f.add_subplot(212)
    plot_pacf(ts,ax=ax2,lags=lags,method='ywm')  #method='ywm'表示采用ywma的方法
    plt.subplots_adjust(hspace=0.5)
    plt.show()
draw_acf_pacf(data_diff,30)



#对模型p,q进行定阶
warnings.filterwarnings("ignore") # specify to ignore warning messages
from statsmodels.tsa.arima_model import ARIMA 
 
pmax = int(5)    #一般阶数不超过 length /10
qmax = int(5)
bic_matrix = []
for p in range(pmax +1):
    temp= []
    for q in range(qmax+1):
        try:
            temp.append(ARIMA(data1, (p, 1, q)).fit().bic)
        except:
            temp.append(None)
        bic_matrix.append(temp)
 
bic_matrix = pd.DataFrame(bic_matrix)   #将其转换成Dataframe 数据结构
bic_matrix = bic_matrix.fillna(0)   #将空值填充为0
p,q = bic_matrix.stack().idxmin()   #先使用stack 展平， 然后使用 idxmin 找出最小值的位置
print(u'BIC 最小的p值 和 q 值：%s,%s' %(p,q))  #  BIC 最小的p值 和 q 值：0,1

#根据定阶参数进行模型拟合
mod = sm.tsa.statespace.SARIMAX(data1,
                                order=(0, 1, 1),
                                seasonal_order=(2, 1, 2, 52),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()



#预测
predict_ts = results.predict(tpy='levels')  #tpy='levels'直接预测值，没有的话预测的是差值

myts = data1.loc[predict_ts.index.date].reset_index(drop=True)
 #取出预测的时间段的数据

 
predict_ts.plot(color='blue', label='Predict',figsize=(12,8))
 
myts.plot(color='red', label='Original',figsize=(12,8))
plt.legend(loc='best')
myts.plot(color='red', label='Original',figsize=(12,8))
print('RMSE:',np.sqrt(((np.array(predict_ts) - np.array(myts)) ** 2).mean()))
plt.legend(loc='best')
plt.title('RMSE: ' + str(np.sqrt(((np.array(predict_ts) - np.array(myts)) ** 2).mean())))
plt.show()





steps = 31
start_time = datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')
forecast_ts = results.forecast(start_time = start_time, steps=steps) 
fore = pd.DataFrame()
fore['date'] = pd.date_range(start=start_time , periods=steps, freq='D')
fore['result'] = pd.DataFrame(forecast_ts.values)
fore.index = pd.to_datetime(fore['date'])
fore.drop(['date'], axis=1, inplace=True)
print(fore)
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(fore.index,fore['result'],color='black', label='forecast')
plt.xlabel('2023-01-01 ~ 2023-01-31')
plt.ylabel('货量')


plt.legend(loc='best')
plt.show()
print(fore['result'])

# fore.to_csv('forecast_20steps.csv')