import pandas as pd
import numpy as np
import statsmodels.api as sm

sumary = 0


df = pd.read_excel('data.xlsx', header=0)
data = df.loc[(df['场地1'] == 'DC5') | (df['场地2'] == 'DC5')]
grouped_data1= data.groupby(['场地1', '场地2'])
print(data.count())
print(grouped_data1.nunique().count())
for name, group in grouped_data1:
    group['日期'] = pd.to_datetime(group['日期'],errors='coerce')
    group.set_index('日期', inplace=True)
    group = group.asfreq('D')
    group = group.copy()  # 添加这一行
    group['货量'].fillna(0, inplace=True)
    group.replace([np.inf, -np.inf], 0, inplace=True)
    model = sm.tsa.ARIMA(group['货量'], order=(1, 1, 0))
    results = model.fit()
    last_date = group.index[-1]
    if last_date not in group.index:
         last_date = group.index[group.index < last_date][-1]
    forecast_start = last_date + pd.Timedelta(days=1)
    forecast_end = last_date + pd.Timedelta(days=31)
    forecast = results.predict(start=forecast_start, end=forecast_end).rename('货物量')
    forecast_frame = forecast.to_frame()
    forecast_frame['货物量'] = forecast_frame['货物量'].fillna(0).replace(np.inf, 0).astype(int)
    sumary += forecast_frame['货物量'].sum()
    print(forecast_frame['货物量'].values.sum())
    break


