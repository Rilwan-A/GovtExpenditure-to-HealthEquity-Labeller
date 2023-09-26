# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:15:01 2023
@author: giselara based on Mario Filho's Detrending
https://forecastegy.com/posts/detrending-time-series-data-python/
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.signal import detrend
from statsmodels.tsa.tsatools import detrend as dtd


df_exp = pd.read_csv('data/ppi/data_expenditure_raw.csv', encoding='utf-8')


# define the set of columns to adjust
colYears = [col for col in df_exp.columns if col.isnumeric()]
ids_columns = [c for c in df_exp.columns if 'Code' in c or 'category' in c or 'Name' in c]
pop_columns = [c for c in df_exp.columns if 'population' in c and '202' not in c]
cpi_columns = [c for c in df_exp.columns if 'cpi' in c and '202' not in c]

# 1) adjust for population growth and inflation
for y in colYears:
    year = y[0:4]
    cpi = df_exp['cpi_2019']/df_exp['cpi_' + year]
    pop = df_exp['population_' + year]
    df_exp[year] = df_exp[y] * cpi / pop 


# define colYears for yearly adjusted values
colYears = [col for col in df_exp.columns if len(col)==4]

# melt to de-trend
df_exp = pd.melt(df_exp, id_vars = ids_columns, value_vars = colYears, ignore_index=True, col_level=0,
        var_name='spell', value_name='value')


#Box.test(diff(goog200), lag=10, type="Ljung-Box")
#model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
#model.fit(train)
#forecast = model.predict(n_periods=len(valid))
#forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])


# a loops considers mean and trend per indicator
for seriesCode, group in df_exp.groupby('seriesCode'):
    mean_value = group.value.mean()
    detrended = detrend(group.value, type = 'linear')
    detrended += mean_value
    df_exp.loc[group.index, 'value']

# reshape wide on years
df = pd.pivot_table(df_exp, index = ['seriesCode', 'seriesName', 'Area Code', 'category'], columns = 'spell', \
                        values = 'value').reset_index(drop = False) 

df = df.groupby('category').sum()
df['category'] = df.index
df.reset_index(drop=True, inplace=True)

# export to csv
df.to_csv('./data/ppi/data_expenditure_trend.csv', index=False)
