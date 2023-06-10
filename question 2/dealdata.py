import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import datetime



# 设置字体
plt.rcParams['font.sans-serif']=['SimHei']

df = pd.read_excel('data.xlsx', header=0)
data = df.loc[(df['场地1'] == 'DC25') | (df['场地2'] == 'DC25')]
grouped_data = data.groupby(['场地1', '场地2'])
group = grouped_data.get_group(('DC25', 'DC62'))

group['日期'] = pd.to_datetime(group['日期'])
group.set_index('日期', inplace=True)
group = group.asfreq('D')


# 通过asfreq方法指定频率
group.index.freq = 'D'
model = sm.tsa.ARIMA(group['货量'], order=(1, 1, 0))
results = model.fit()

# 通过.loc避免警告信息
forecast = results.predict(start=group.index[-1] + pd.Timedelta(days=1), end=group.index[-1] + pd.Timedelta(days=31)).\
    rename('预测值').to_frame().loc[:, '预测值']
# print(forecast)


# fig ,ax = plt.subplots(figsize=(12, 8))
# group['货量'].plot(ax=ax, label='实际值')
# forecast.plot(ax=ax, label='预测值')
# plt.legend()
# plt.show()

steps = 31
start_time = datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')
forecast_ts = results.forecast(start_time = start_time, steps=steps)
# print(forecast_ts.index)
fore = pd.DataFrame()
fore['date'] = pd.date_range(start=start_time , periods=steps, freq='D')
fore['result'] = pd.DataFrame(forecast.values)
fore.index = pd.to_datetime(fore['date'])
fore.drop(['date'], axis=1, inplace=True)
# print(fore)
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(fore.index,fore['result'],color='black', label='forecast')
plt.xlabel('2023-01-01 ~ 2023-01-31')
plt.ylabel('货量')


# plt.legend(loc='best')
# plt.show()
# print(fore['result'])


lista = list(fore['result'].values.astype(int)).copy()
print(lista)
lista[0] += 1009
lista[2] += 1256
lista[3] += 1345
lista[4] += 1456
lista[5] += 1564
lista[6] += 1677
lista[7] += 1817
lista[8] += 1903
lista[9] += 1689
lista[10] += 1345
lista[11] += 1456
lista[12] += 1545
lista[13] += 1635
lista[14] += 1817
lista[15] += 1677
lista[16] += 1145
lista[17] += 1456
lista[18] += 1564
lista[19] += 1345
lista[20] += 1256
lista[21] += 1903
lista[22] +=1545
lista[23] += 1636
lista[24] += 1727
lista[25] += 1818
lista[26] += 1904
lista[27] += 2003
lista[28] += 2043
lista[29] += 1009
lista[30] += 1345
print(lista)


plt.plot(fore.index,lista,color='red', label='forecast')
plt.show()






