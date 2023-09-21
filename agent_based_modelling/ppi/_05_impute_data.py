import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
warnings.simplefilter("ignore")

home =  os.getcwd()[:-16]



df = pd.read_csv(home+'data/england/deprivation/clean_data/pipeline_indicators_sample_raw.csv')
colYears = [col for col in df.columns if col.isnumeric()]


# calculate proportion imputed
df['count_missing'] = df[colYears].isnull().sum(axis = 'columns')
df['count_spells'] = len(colYears) #df[colYears].notnull().sum(axis = 'columns')
proportion_imputed = df['count_missing'].sum() / df['count_spells'].sum()
print(proportion_imputed)

colYears = [col for col in df.columns if str(col).isnumeric()]
years = np.array([int(col) for col in df.columns if str(col).isnumeric()])
years_indices = df.columns.isin(colYears)


new_rows = []
for index, row in df.iterrows():
    
    observations = np.where(~row[colYears].isnull())[0]
    missing_values = np.where(row[colYears].isnull())[0]
    new_row = row.values.copy()
        
        
    vals = row[colYears].values.copy()

    x = years[observations]
    y = vals[observations]
    X = x.reshape(-1, 1)

    kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, y)

    x_pred = years.reshape(-1,1)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    
    vals[missing_values] = y_pred[missing_values]
    new_row[years_indices] = vals
        
    new_rows.append(new_row)
    
    if index%100==0:
        print(index)



dff = pd.DataFrame(new_rows, columns=df.columns)
dff.to_csv(home+'data/england/deprivation/clean_data/pipeline_indicators_imputed_raw.csv', index=False)


