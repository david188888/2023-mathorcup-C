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
df = pd.read_excel('../data.xlsx', header=0)

"""数据预处理"""





data3 = df.loc[(df["场地1"] == "DC25") & (df['场地2'] == "DC62")]#DC25 -> DC
# print(data1)
data3.index = pd.to_datetime(data3['日期'])
data3.drop(['日期', '场地1', '场地2'], axis=1, inplace=True)

# print(data3.head(5))



# print(data3.shape)

"""数据可视化"""

data3.plot(figsize=(12,8),ylabel='货物量')
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title('DC25 -> DC62的趋势图')
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
stationarity = judge_stationarity(data3)



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
draw_acf_pacf(data3,30)



#根据定阶参数进行模型拟合
mod = sm.tsa.statespace.SARIMAX(data3,
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
myts = data3.loc[predict_ts.index.date].reset_index(drop=True) # 过滤没有预测的记录
 
predict_ts.plot(color='blue', label='Predict',figsize=(12,8))
 
myts.plot(color='red', label='Original',figsize=(12,8))
print('RMSE:',np.sqrt(((np.array(predict_ts) - np.array(myts)) ** 2).mean()))
plt.legend(loc='best')
plt.title('RMSE: ' + str(np.sqrt(((np.array(predict_ts) - np.array(myts)) ** 2).mean())))
plt.show()


# Plot results

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