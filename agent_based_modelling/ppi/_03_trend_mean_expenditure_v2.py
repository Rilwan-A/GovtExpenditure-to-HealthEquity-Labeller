import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.signal import detrend

df_exp = pd.read_csv('data/ppi/data_expenditure_raw.csv', encoding='unicode_escape')

# Define columns
colYears = [col for col in df_exp.columns if col.isnumeric()]
ids_columns = [c for c in df_exp.columns if 'Code' in c or 'category' in c or 'Name' in c]
pop_columns = [c for c in df_exp.columns if 'population' in c and '202' not in c]
cpi_columns = [c for c in df_exp.columns if 'cpi' in c and '202' not in c]

# 1) Adjust for population growth and inflation
for y in colYears:
    year = y[0:4]
    cpi = df_exp['cpi_2019'] / df_exp['cpi_' + year]
    pop = df_exp['population_' + year]
    df_exp[year] = df_exp[y] * cpi / pop 

# Define colYears for yearly adjusted values
colYears = [col for col in df_exp.columns if len(col) == 4]

# Melt to prepare for detrending
df_exp_melted = pd.melt(df_exp, id_vars=ids_columns, value_vars=colYears, ignore_index=True,
                        var_name='spell', value_name='value')

# Detrend data from 2013-2017 and apply to 2018-2019
for seriesCode, group in df_exp_melted.groupby('seriesCode'):
    # Only detrend using data from 2013-2017
    train_data = group[group['spell'].astype(int) <= 2017]
    mean_value = train_data.value.mean()
    detrended = detrend(train_data.value, type='linear')
    trend = train_data.value - detrended
    
    # Apply detrending to 2018-2019 using the trend calculated from 2013-2017
    test_data = group[group['spell'].astype(int) > 2017]
    if not test_data.empty:
        detrended_test = test_data.value - trend.mean()
        df_exp_melted.loc[test_data.index, 'value'] = detrended_test + mean_value
    
    # Update 2013-2017 values
    df_exp_melted.loc[train_data.index, 'value'] = detrended + mean_value

# Reshape wide on years
df = pd.pivot_table(df_exp_melted, index=['seriesCode', 'seriesName', 'Area Code', 'category'], 
                    columns='spell', values='value').reset_index(drop=False)

# Export to csv
df.to_csv('./data/ppi/data_expenditure_finegrained_trend_2013_2019.csv', index=False)
